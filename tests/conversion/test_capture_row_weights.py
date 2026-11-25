"""
Row-weighted Hessian capture (content-weighted expert calibration, EXL3_CALIBRATION_EXPERT_WEIGHTS) against
upstream's packed capture, which takes the layer's input without copying it: the weighting must not reach the
forward pass, must give the weighted mean it promises, and must never meet the all-expert fast path, which has no
per-token weights to apply.
"""

import types

import pytest
import torch

from exllamav3.modules import block_sparse_mlp as bsm
from exllamav3.modules.linear import Linear
from exllamav3.modules.quant.exl3_lib.hessian_packed import unpack

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")


def _fake_linear(n: int):
    fake = types.SimpleNamespace(qmap = "q", key = "k", in_features = n, device = torch.device("cuda:0"))
    fake.init_H_data = lambda init: Linear.init_H_data(fake, init)
    return fake


def test_weighted_capture_leaves_input_untouched():
    torch.manual_seed(0)
    fake = _fake_linear(128)
    x = torch.randn(64, 128, dtype = torch.half, device = "cuda:0")
    x0 = x.clone()
    rw = torch.rand(64, device = "cuda:0") * 3
    params = {"capture": {}, "capture_row_weight": rw}
    Linear.capture_H(fake, x, params)
    assert torch.equal(x, x0), "row weighting scaled the layer's input, which the forward pass reads next"

    hd = params["capture"]["q"]
    xw = x0.float() * rw.unsqueeze(1)
    H = unpack(hd["H"], 128)[:128, :128]
    torch.testing.assert_close(H, xw.T @ xw, rtol = 1e-3, atol = 1e-2)
    assert hd["count"] == pytest.approx((rw.double() ** 2).sum().item())


def test_content_weighting_excludes_affinity():
    fake = _fake_linear(128)
    x = torch.randn(8, 128, dtype = torch.half, device = "cuda:0")
    params = {"capture": {}, "capture_row_weight": torch.ones(8, device = "cuda:0"),
              "capture_row_w": torch.ones(8, device = "cuda:0")}
    with pytest.raises(AssertionError, match = "exclusive"):
        Linear.capture_H(fake, x, params)


def test_all_expert_fast_path_refuses_content_weights(monkeypatch):
    monkeypatch.setattr(bsm, "_cal_expert_w", {"labels": torch.zeros(1, 1, dtype = torch.int16)})
    fake = types.SimpleNamespace(key = "model.layers.0.mlp")
    y = torch.zeros(4, 8, dtype = torch.half, device = "cuda:0")
    with pytest.raises(AssertionError, match = "unweighted all-expert capture"):
        bsm.BlockSparseMLP._capture_all_experts(fake, y, torch.zeros(1, 1, dtype = torch.long), None, None, {})
