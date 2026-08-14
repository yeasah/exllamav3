"""
Inference backends. Each backend runs the full test set and hands per-row logits
[1, len, vocab] on the compute device to a callback, one row at a time, so full-model logits
never accumulate.

Effective bits-per-weight is computed to one convention across all formats, so the x-axis is an
apples-to-apples comparison:

  - layer bpw   = stored bits of the weight matrices (>= 2 dims) outside the embedding and the
                  output head, divided by their element count. Includes every storage overhead
                  of the format (scales, codebooks, sign flips, group metadata where they are
                  materialized as part of the tensor's storage).
  - excluded    = embeddings, norm weights, biases, and MoE router gates (tiny, always kept
                  high-precision, and represented inconsistently across formats).
  - head bpw    = the lm_head / output matrix; falls back to the embedding matrix's storage for
                  tied-embedding models.
  - vram_gb     = layer + head stored bytes.
"""

import os

import torch

from exllamav3.util.memory import free_mem
from exllamav3.util.progress import ProgressBar

# Substrings identifying MoE router gate matrices per naming scheme (2D but excluded: tiny and
# stored unquantized in some formats but quantized in others)
HF_ROUTER_KEYS = (".mlp.gate.weight", ".block_sparse_moe.gate.weight", ".router.weight", ".feed_forward.gate.weight")
GGUF_ROUTER_KEYS = ("ffn_gate_inp.weight",)


def apply_mult_noise(x: torch.Tensor, eps: float, gen: torch.Generator) -> torch.Tensor:
    n = torch.randn(x.shape, generator = gen, device = x.device, dtype = torch.float)
    return (x.float() * (1.0 + eps * n)).to(x.dtype)


def fake_quantize(x: torch.Tensor, bits: int, granularity: str = "per_row") -> torch.Tensor:
    """Round x to a uniform bits-bit grid and immediately dequantize, in place of dtype/storage
    changes - lets a sweep characterize sensitivity to a given precision level (is there a knee,
    where) without committing to a real packed format or a real kernel. "per_row" scales each row
    (e.g. one embedding vector) by its own min/max: the floor for quantization done reasonably,
    not the pathological case a single tensor-wide scale would give."""
    levels = 2 ** bits - 1
    xf = x.float()
    if granularity == "per_row":
        lo = xf.amin(dim = -1, keepdim = True)
        hi = xf.amax(dim = -1, keepdim = True)
    elif granularity == "per_tensor":
        lo = xf.amin()
        hi = xf.amax()
    else:
        raise ValueError(f"Unknown granularity: {granularity}")
    scale = (hi - lo).clamp_min(1e-12) / levels
    q = ((xf - lo) / scale).round().clamp(0, levels)
    return (q * scale + lo).to(x.dtype)


class Exl3Backend:
    """Module-streamed exllamav3 pass (model_diff style): one module resident at a time"""

    def __init__(self, source: str, max_len: int, device: torch.device, options: dict):
        from exllamav3 import Config, Model
        self.device = device
        self.config = Config.from_directory(source)
        self.config.override_dynamic_seq_len(max_len)
        self.model = Model.from_config(self.config)
        self.info = None
        # Simulated embedding precision, for characterizing sensitivity independent of any real
        # storage format: {"bits": int, "granularity": "per_row"|"per_tensor"} or None to leave
        # the checkpoint's real (always fp16) embedding untouched
        self.embed_quant = options.get("embed_quant")

    def run(self, ids: torch.Tensor, callback, noise_eps: float = None):
        from exllamav3.modules import Embedding, Linear
        modules = self.model.modules
        states = list(ids.split(1))
        gen = torch.Generator(device = self.device)
        gen.manual_seed(1)

        sum_bits = sum_numel = head_bits = head_numel = embed_bits = embed_numel = 0
        with ProgressBar("Streaming", len(modules)) as pb:
            for idx, module in enumerate(modules):
                self.config.stc.begin_deferred_load()
                module.load(self.device if not module.caps.get("prefer_cpu") else "cpu")
                self.config.stc.end_deferred_load()

                if self.embed_quant and isinstance(module, Embedding):
                    module.embedding.weight.data = fake_quantize(
                        module.embedding.weight.data,
                        self.embed_quant["bits"],
                        self.embed_quant.get("granularity", "per_row"),
                    )

                # Storage info while the module is resident. Biases are excluded from the
                # convention; the MoE routing gate is not a Linear in exllamav3 and is thus
                # excluded structurally. Embedding shares Linear's get_tensors()/weights_numel()
                # interface but is never quantized (no qmap), so it's tallied into its own
                # bucket rather than the layer bucket
                if self.info is None:
                    stack = [module]
                    while stack:
                        m = stack.pop()
                        if isinstance(m, (Linear, Embedding)):
                            bits = 8 * sum(
                                t.element_size() * t.numel()
                                for k, t in m.get_tensors().items()
                                if not k.endswith(".bias")
                            )
                            if isinstance(m, Embedding):
                                embed_bits += bits
                                embed_numel += m.weights_numel()
                            elif m.key.endswith("lm_head"):
                                # A tied model's head module may have loaded a genuinely
                                # separate on-disk tensor -- this project's own EXL3
                                # quantizer writes one for every tied model regardless,
                                # see TODO.md #2 -- or fallen back to aliasing the
                                # embedding's storage via alt_key. `used_alt_key`, set by
                                # Linear.load() itself, is the ground truth for which
                                # happened. It is not equivalent to checking whether a bare
                                # "lm_head" key exists in storage: this codebase's
                                # checkpoints only ever store suffixed keys
                                # ("lm_head.trellis", "lm_head.weight", ...), so a bare-key
                                # check is always false, silently dead code -- every model,
                                # tied or not, fell through to the embed-bpw fallback below
                                # regardless of what actually got loaded onto the device.
                                if not getattr(m, "used_alt_key", False):
                                    head_bits += bits
                                    head_numel += m.weights_numel()
                            else:
                                sum_bits += bits
                                sum_numel += m.weights_numel()
                        stack.extend(getattr(m, "modules", []))

                logits_layer = idx == len(modules) - 1
                for r in range(len(states)):
                    # Hash-MoE layers (DeepSeek-V4) route by token id; provide the row's ids
                    params = {"input_ids": ids[r:r + 1]}
                    x = module.prepare_for_device(states[r], params)
                    x = module.forward(x, params)
                    if noise_eps and idx < len(modules) - 2 and x.is_floating_point():
                        x = apply_mult_noise(x, noise_eps, gen)
                    if logits_layer:
                        callback(r, x)
                        states[r] = None
                    else:
                        states[r] = x
                    del x

                module.unload()
                self.config.stc.close()
                free_mem()
                pb.update(idx + 1)

        tied_head = head_numel == 0
        if tied_head:
            # Tied embeddings: the head is served by the embedding matrix, so its bytes must
            # not also be added under embed_bits below (same storage, not two allocations)
            head_bits, head_numel = embed_bits, embed_numel
        self.info = {
            "bpw_layer": sum_bits / max(sum_numel, 1),
            "bpw_head": head_bits / max(head_numel, 1),
            "bpw_embed": embed_bits / max(embed_numel, 1),
            "vram_gb": (sum_bits + head_bits + (0 if tied_head else embed_bits)) / 8 / 1024 ** 3,
        }

    def close(self):
        self.model.unload()
        free_mem()


class TransformersBackend:
    """
    HF backend. Default: full load via accelerate device_map. With options: {streaming: true},
    the model is instead built as a weightless meta skeleton and each big submodule's weights
    are materialized from the safetensors shards just in time by forward hooks, then returned to
    meta - so peak VRAM is one decoder layer plus activations, and models far larger than the
    GPU (or system RAM) can serve as the reference. All rows are batched into a single forward,
    so each weight is read from disk exactly once per pass, and the head is applied per row so
    full [rows, len, vocab] logits never materialize. The model's own forward computes masks and
    rope, so no per-architecture layer-call knowledge is needed.

    Noise (for the self-noise floor) is injected with forward hooks on the decoder layer list,
    located as the largest ModuleList of same-type children; these compose with the streaming
    hooks unchanged.
    """

    def __init__(self, source: str, max_len: int, device: torch.device, options: dict):
        from transformers import AutoModelForCausalLM
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32} \
            [options.get("dtype", "bfloat16")]
        self.device = device
        self.dtype = dtype
        self.streaming = options.get("streaming", False)
        self.shard_handles = {}

        if self.streaming:
            import json
            import struct
            from accelerate import init_empty_weights
            from transformers import AutoConfig
            trc = options.get("trust_remote_code", True)
            config = AutoConfig.from_pretrained(source, trust_remote_code = trc)

            # Quantized checkpoints are dequantized on the fly by _get_tensor; the skeleton is
            # built unquantized (plain Linear modules), so the quantization config must not
            # reach from_config. Supported: compressed-tensors pack-quantized (AWQ-style int
            # groups), compressed-tensors mixed-precision / nvfp4-pack-quantized /
            # float-quantized (llm-compressor FP8-channel + NVFP4 mixes), and ModelOpt mixed
            # FP8 / NVFP4. Evaluation is weight-only in all cases: the deployed stacks'
            # activation quant (static or dynamic fp8/fp4) and fp8 kv cache are not emulated,
            # so results are the checkpoint's weight fidelity under bf16 compute
            self.ct_bits = None
            self.dequant_scaled = False
            qcfg = getattr(config, "quantization_config", None)
            if qcfg is not None:
                def q_get(key, default = None):
                    return qcfg.get(key, default) if isinstance(qcfg, dict) else getattr(qcfg, key, default)
                qm = q_get("quant_method")
                if qm == "compressed-tensors":
                    fmt = q_get("format")
                    assert fmt in ("pack-quantized", "mixed-precision", "nvfp4-pack-quantized", "float-quantized"), \
                        f"Unsupported compressed-tensors format: {fmt}"
                    # ct_bits only drives the int pack-quantized decode; the actual scheme per
                    # tensor is decided by its sidecars (weight_global_scale -> nvfp4,
                    # weight_scale next to a plain weight -> fp8-channel)
                    self.ct_bits = 4
                    for g in (q_get("config_groups", {}) or {}).values():
                        w = (g.get("weights") if isinstance(g, dict) else getattr(g, "weights", None)) or {}
                        if isinstance(w, dict) and w.get("type", "int") == "int":
                            self.ct_bits = w.get("num_bits", 4)
                    self.dequant_scaled = True
                elif qm == "modelopt":
                    self.dequant_scaled = True
                elif qm == "fp8":
                    # AutoFP8/DeepSeek-style: fp8-e4m3 weights with blockwise (typically
                    # 128x128) weight_scale_inv, unquantized modules stay plain bf16
                    self.dequant_scaled = True
                elif qm == "mxfp4":
                    # gpt-oss-style: MoE expert tensors stored as <name>_blocks (packed fp4
                    # e2m1 pairs) + <name>_scales (e8m0); resolved per-tensor by sidecar
                    pass
                else:
                    raise ValueError(f"Streaming does not support this quantization_config: {qm}")
                delattr(config, "quantization_config")

            # include_buffers = False keeps non-persistent buffers (rope inv_freq etc., which
            # are not in the shards) materialized with their init values
            with init_empty_weights(include_buffers = False):
                try:
                    self.model = AutoModelForCausalLM.from_config(config, dtype = dtype, trust_remote_code = trc)
                except ValueError:
                    # Multimodal wrappers not registered under CausalLM (Mistral3/Mistral4):
                    # the ConditionalGeneration class computes text-only logits fine, and the
                    # never-forwarded vision tower simply stays on meta
                    from transformers import AutoModelForImageTextToText
                    self.model = AutoModelForImageTextToText.from_config(config, dtype = dtype, trust_remote_code = trc)
            self.model.eval()

            # The streamed head skips the CausalLM wrapper's forward, so post-head transforms
            # it would apply must be replicated (gemma-style logit softcapping)
            text_config = getattr(config, "text_config", None) or config
            self.logit_softcap = (
                getattr(text_config, "final_logit_softcapping", None)
                or getattr(config, "final_logit_softcapping", None)
            )

            # Shard index: module-side tensor name -> (file, checkpoint tensor name). Checkpoint
            # layouts may differ from the module tree: transformers v5 declares per-model
            # conversions (renamings, and per-expert -> fused-experts merges) which
            # from_pretrained applies during load. Replicate that here: renamings fold into the
            # index; merge-type converters are kept and resolved on demand in _get_tensor
            import re
            from safetensors import safe_open
            self.source = source
            index_file = os.path.join(source, "model.safetensors.index.json")
            if os.path.exists(index_file):
                with open(index_file) as f:
                    raw_map = json.load(f)["weight_map"]
            else:
                single = os.path.join(source, "model.safetensors")
                with safe_open(single, framework = "pt", device = "cpu") as f:
                    raw_map = {k: "model.safetensors" for k in f.keys()}

            # Stored size per checkpoint tensor (from the shard headers), for exact bpw
            # accounting straight off the storage format
            self.tensor_nbytes = {}
            for fn in set(raw_map.values()):
                with open(os.path.join(source, fn), "rb") as f:
                    hdr_len = struct.unpack("<Q", f.read(8))[0]
                    hdr = json.loads(f.read(hdr_len))
                for k, v in hdr.items():
                    if k != "__metadata__":
                        a, b = v["data_offsets"]
                        self.tensor_nbytes[k] = b - a

            # Conversion mappings may be registered under the composite model_type or, for
            # ConditionalGeneration wrappers whose CausalLM class is the inner text model, under
            # text_config.model_type (e.g. qwen3_5's model.language_model.* strip lives under
            # qwen3_5_text only)
            model_types = [config.model_type]
            if getattr(config, "text_config", None) is not None:
                tmt = getattr(config.text_config, "model_type", None)
                if tmt and tmt not in model_types:
                    model_types.append(tmt)

            renames = []          # (source substring, target substring)
            self.converters = []  # (target pattern, [source patterns], [ops])
            try:
                from transformers.conversion_mapping import get_checkpoint_conversion_mapping
                for conv in [c for mt in model_types for c in (get_checkpoint_conversion_mapping(mt) or [])]:
                    sources = getattr(conv, "source_patterns", None)
                    targets = getattr(conv, "target_patterns", None)
                    ops = getattr(conv, "operations", None)
                    if not sources or not targets:
                        continue
                    if ops:
                        self.converters.append((targets[0], sources, ops))
                    else:
                        renames.append((sources[0], targets[0]))
            except ImportError:
                # Older transformers: fall back to the class-attr regex mapping
                for pat, repl in (getattr(type(self.model), "_checkpoint_conversion_mapping", None) or {}).items():
                    renames.append((pat, repl))

            # Renames are registered as aliases rather than replacements: some models' declared
            # renamings don't match their (vendored) module tree, so the original checkpoint
            # name stays resolvable either way
            self.tensor_index = {}
            for ck_name, fn in raw_map.items():
                self.tensor_index[ck_name] = (fn, ck_name)
                mod_name = ck_name
                for src, tgt in renames:
                    # Conversion sources are regex fragments; escaped literals ("mlp\.gate")
                    # must go through re.sub too, or they can never match anything (hy_v3's
                    # router/shared/bias renames are exactly this shape)
                    if src.startswith("^") or "(" in src or "\\" in src:
                        mod_name = re.sub(src, tgt, mod_name)
                    elif src in mod_name:
                        mod_name = mod_name.replace(src, tgt)
                if mod_name != ck_name:
                    self.tensor_index[mod_name] = (fn, ck_name)
                # Some conversion mappings (deepseek_v4) declare targets in the bare
                # base-model namespace and rely on the loader to add base_model_prefix;
                # alias the prefixed form so module-side lookups resolve. setdefault: real
                # top-level names (lm_head) must not be shadowed
                base_prefix = getattr(self.model, "base_model_prefix", "")
                if base_prefix:
                    self.tensor_index.setdefault(f"{base_prefix}.{mod_name}", (fn, ck_name))

            # Generic fallback for composite checkpoints whose model doesn't declare the
            # container strip in its conversion mapping: alias the text stack into the CausalLM
            # namespace (declared renames take precedence via setdefault)
            for ck_name, fn in raw_map.items():
                if ck_name.startswith("model.language_model."):
                    alias = "model." + ck_name[len("model.language_model."):]
                    self.tensor_index.setdefault(alias, (fn, ck_name))

            # Wrappers nesting a bare text model under language_model keep the text model's own
            # base prefix in the checkpoint ("language_model.model.*", old-style Mistral3/4);
            # from_pretrained strips it when nesting, so alias the stripped form of every
            # resolved name too
            for mod_name in list(self.tensor_index.keys()):
                if ".language_model.model." in mod_name:
                    self.tensor_index.setdefault(
                        mod_name.replace(".language_model.model.", ".language_model.", 1),
                        self.tensor_index[mod_name])

            # Real (init-valued) buffers up front on the compute device
            for m in self.model.modules():
                for k, b in m._buffers.items():
                    if b is not None and not b.is_meta:
                        m._buffers[k] = b.to(device)

            # Persistent buffers stored in the checkpoint (e.g. gemma4's per-layer scalars) are
            # not covered by parameter streaming; they are small, so load them up front. Missing
            # them is silent and catastrophic - the module keeps its init value
            for bn, _ in self.model.named_buffers():
                if bn in self.tensor_index:
                    t = self._read_shard(bn)
                    if t.is_floating_point() and t.dtype != torch.float32:
                        t = t.to(device, dtype)
                    else:
                        t = t.to(device)
                    owner_name, _, leaf = bn.rpartition(".")
                    owner = self.model.get_submodule(owner_name) if owner_name else self.model
                    owner._buffers[leaf] = t

            self.prefix = {id(m): n for n, m in self.model.named_modules()}
            # Tied-head fallback: the output embedding's weight is not in the shards when tied
            head = self.model.get_output_embeddings()
            embed = self.model.get_input_embeddings()
            self.head_weight_name = f"{self.prefix[id(head)]}.weight" if head is not None else None
            self.embed_weight_name = f"{self.prefix[id(embed)]}.weight"
        else:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    source,
                    dtype = dtype,
                    device_map = options.get("device_map", "auto"),
                    trust_remote_code = options.get("trust_remote_code", True),
                )
            except ValueError:
                from transformers import AutoModelForImageTextToText
                self.model = AutoModelForImageTextToText.from_pretrained(
                    source,
                    dtype = dtype,
                    device_map = options.get("device_map", "auto"),
                    trust_remote_code = options.get("trust_remote_code", True),
                )
            self.model.eval()

        # Biases and norms are 1D (excluded by the ndim test); router gates excluded by name.
        # In streaming mode, bits come from the checkpoint's actual stored bytes (including
        # quantization metadata like scales), so quantized formats report their true bitrate
        sum_bits = sum_numel = head_bits = head_numel = 0
        embed_bits = embed_numel = 0
        for name, p in self.model.named_parameters():
            if name.endswith(".bias") or name.endswith("_bias"):
                continue
            bits = None
            if self.streaming:
                bits = self._stored_bits(name)
            if bits is None:
                bits = p.numel() * p.element_size() * 8
            if "embed" in name:
                embed_bits += bits
                embed_numel += p.numel()
                continue
            if "lm_head" in name or "output" == name.split(".")[0]:
                head_bits += bits
                head_numel += p.numel()
            elif p.ndim >= 2 and not any(k in name for k in HF_ROUTER_KEYS):
                sum_bits += bits
                sum_numel += p.numel()
        tied_head = head_numel == 0
        if tied_head:
            # Tied embeddings: the head is served by the embedding matrix, so its bytes must
            # not also be added under embed_bits below (same storage, not two allocations)
            head_bits, head_numel = embed_bits, embed_numel
        self.info = {
            "bpw_layer": sum_bits / max(sum_numel, 1),
            "bpw_head": head_bits / max(head_numel, 1),
            "bpw_embed": embed_bits / max(embed_numel, 1),
            "vram_gb": (sum_bits + head_bits + (0 if tied_head else embed_bits)) / 8 / 1024 ** 3,
        }

    def _decoder_layers(self):
        best = None
        for m in self.model.modules():
            if isinstance(m, torch.nn.ModuleList) and len(m) >= 4:
                if len(set(type(c) for c in m)) == 1 and (best is None or len(m) > len(best)):
                    best = m
        if best is None:
            raise RuntimeError("Could not locate decoder layer list for noise injection")
        return best

    def _noise_hooks(self, noise_eps):
        gens = {}
        def hook(module, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            if out.device not in gens:
                g = torch.Generator(device = out.device)
                g.manual_seed(1)
                gens[out.device] = g
            out = apply_mult_noise(out, noise_eps, gens[out.device])
            if isinstance(output, tuple):
                return (out,) + output[1:]
            return out
        return [layer.register_forward_hook(hook) for layer in self._decoder_layers()]

    # ------ streaming machinery

    CT_SUFFIXES = ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape", "weight_g_idx",
                   "weight_global_scale", "input_global_scale")

    def _read_shard(self, name):
        fn, ck_name = self.tensor_index[name]
        fn = os.path.join(self.source, fn)
        if fn not in self.shard_handles:
            from safetensors import safe_open
            self.shard_handles[fn] = safe_open(fn, framework = "pt", device = "cpu")
        return self.shard_handles[fn].get_tensor(ck_name)

    def _nbytes(self, name):
        return self.tensor_nbytes[self.tensor_index[name][1]]

    def _dequant_ct(self, stem):
        """Dequantize a compressed-tensors pack-quantized weight: sequentially nibble-packed
        signed ints in int32 words, per-group scales, optional zero point"""
        bits = self.ct_bits or 4
        packed = self._read_shard(f"{stem}.weight_packed")          # int32 [out, in * bits/32]
        scale = self._read_shard(f"{stem}.weight_scale").float()    # [out, groups]
        shape = self._read_shard(f"{stem}.weight_shape").tolist()   # [out, in]
        shifts = torch.arange(0, 32, bits, dtype = torch.int32)
        mask = (1 << bits) - 1
        q = (packed.unsqueeze(-1) >> shifts) & mask
        q = q.flatten(1)[:, :shape[1]]
        q = q - (1 << (bits - 1))  # values are stored offset-binary (e.g. int4 as value + 8)
        zp_name = f"{stem}.weight_zero_point"
        group = shape[1] // scale.shape[1]
        if zp_name in self.tensor_index:
            zp = self._read_shard(zp_name)
            if zp.dtype == torch.int32:
                # Packed offset-binary like the weights, but the packing axis differs by
                # tensor: zero points are [out, groups] packed along whichever dim makes the
                # int32 count fit (observed: along out, unlike the weights)
                per_word = 32 // bits
                z = (zp.unsqueeze(-1) >> shifts) & mask
                if zp.shape[0] * per_word == shape[0]:      # packed along dim 0 (out)
                    zp = z.permute(0, 2, 1).reshape(-1, zp.shape[1])[:shape[0]]
                else:                                       # packed along dim 1 (groups)
                    zp = z.flatten(1)[:, :scale.shape[1]]
                zp = zp - (1 << (bits - 1))
            q = q - zp.repeat_interleave(group, dim = 1)
        return q.to(torch.float32) * scale.repeat_interleave(group, dim = 1)

    # fp4 e2m1 magnitudes for codes 0-7; codes 8-15 mirror negative (sign is the msb)
    NVFP4_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])

    def _scale_float(self, s):
        """Scale sidecar to fp32. E8M0 exponent bytes (DeepSeek-V4 ue8m0, stored as uint8 or
        torch.float8_e8m0fnu) decode as 2^(byte - 127); everything else casts directly."""
        e8m0 = getattr(torch, "float8_e8m0fnu", None)
        if s.dtype == torch.uint8 or (e8m0 is not None and s.dtype == e8m0):
            return (s.view(torch.uint8).float() - 127.0).exp2()
        return s.float()

    def _dequant_scaled(self, name):
        """Dequantize a weight stored under its plain name with a weight_scale sidecar:
        ModelOpt NVFP4 (uint8-packed e2m1 nibble pairs low-first, fp8-e4m3 scale per group,
        fp32 global scale) when weight_scale_2 exists, else FP8 (e4m3 values; per-tensor
        scalar for ModelOpt, [out, 1] channel scale for compressed-tensors float-quantized).
        FP8 with blockwise weight_scale_inv (DeepSeek-style 128x128 blocks; despite the name,
        dequantization MULTIPLIES by it) is handled first, as is the equivalent DeepSeek-V4
        "<stem>.scale" sidecar convention (fp8 128x128 grid on dense weights, [out, in/32]
        e8m0 grid on int8-packed fp4 expert weights; block shape derives from the grid).
        All conventions validated against the bf16 originals."""
        w = self._read_shard(name)
        if w.dtype == torch.int8:
            # DeepSeek-V4 fp4 experts: two e2m1 values per byte, low nibble first
            u8 = w.view(torch.uint8)
            lut = self.NVFP4_LUT.to(w.device)
            w = torch.stack([lut[(u8 & 0x0F).long()], lut[(u8 >> 4).long()]], dim = -1).flatten(1)
        si = None
        if f"{name}_scale_inv" in self.tensor_index:
            si = self._scale_float(self._read_shard(f"{name}_scale_inv"))
        elif name.endswith(".weight") and f"{name[:-len('.weight')]}.scale" in self.tensor_index:
            si = self._scale_float(self._read_shard(f"{name[:-len('.weight')]}.scale"))
        if si is not None:
            # Block grid of any rank: per-tensor scalar (Mistral-Small-4 dense weights),
            # per-expert [E, 1, 1] on fused 3D expert stacks, or the 2D 128x128 DeepSeek grid
            if si.ndim > 0:
                assert si.ndim == w.ndim, f"{name}: scale rank {si.ndim} vs weight rank {w.ndim}"
                for d in range(w.ndim):
                    if si.shape[d] != w.shape[d]:
                        bs = (w.shape[d] + si.shape[d] - 1) // si.shape[d]
                        si = si.repeat_interleave(bs, dim = d).narrow(d, 0, w.shape[d])
            return w.float() * si
        scale = self._read_shard(f"{name}_scale").float()
        if scale.ndim == 1:
            scale = scale.unsqueeze(1)
        if f"{name}_scale_2" in self.tensor_index:
            ws2 = self._read_shard(f"{name}_scale_2").float()
            q = torch.stack([(w & 0xF).long(), (w >> 4).long()], dim = -1).flatten(1)
            group = q.shape[1] // scale.shape[1]
            return self.NVFP4_LUT[q] * scale.repeat_interleave(group, dim = 1) * ws2
        return w.float() * scale

    def _dequant_ct_nvfp4(self, stem):
        """Dequantize a compressed-tensors nvfp4-pack-quantized weight: uint8-packed e2m1
        nibble pairs low-first, fp8-e4m3 scale per group, and a global scale the stored
        per-group scales are DIVIDED by (it maps them into fp8 range at quantization time)"""
        packed = self._read_shard(f"{stem}.weight_packed")
        scale = self._read_shard(f"{stem}.weight_scale").float()
        gs = self._read_shard(f"{stem}.weight_global_scale").float()
        q = torch.stack([(packed & 0xF).long(), (packed >> 4).long()], dim = -1).flatten(1)
        group = q.shape[1] // scale.shape[1]
        return self.NVFP4_LUT[q] * scale.repeat_interleave(group, dim = 1) / gs

    def _stored_bits(self, name):
        """Actual stored size in the checkpoint for a module-side tensor name, including any
        quantization metadata; None if the name cannot be resolved"""
        import re
        if name in self.tensor_index:
            total = 8 * self._nbytes(name)
            if name.endswith(".weight"):
                # ModelOpt-style scale sidecars stored next to the weight itself
                stem = name[:-len(".weight")]
                for s in ("weight_scale", "weight_scale_2", "weight_scale_inv", "input_scale", "input_global_scale", "scale"):
                    if f"{stem}.{s}" in self.tensor_index:
                        total += 8 * self._nbytes(f"{stem}.{s}")
            # Underscore sidecars on suffix-less fused tensors (fp8 expert stacks)
            for s in ("_scale", "_scale_inv"):
                if f"{name}{s}" in self.tensor_index:
                    total += 8 * self._nbytes(f"{name}{s}")
            return total
        if name.endswith(".weight"):
            stem = name[:-len(".weight")]
            bits = [8 * self._nbytes(f"{stem}.{s}") for s in self.CT_SUFFIXES if f"{stem}.{s}" in self.tensor_index]
            if bits:
                return sum(bits)
        if f"{name}_blocks" in self.tensor_index:
            return 8 * (self._nbytes(f"{name}_blocks") + self._nbytes(f"{name}_scales"))
        if name == self.head_weight_name and self.embed_weight_name in self.tensor_index:
            return 8 * self._nbytes(self.embed_weight_name)
        for target, sources, ops in self.converters:
            if not (name == target or name.endswith("." + target)):
                continue
            prefix = name[:-len(target)]
            total = 0
            for sp in sources:
                pat = re.compile(
                    "^" + re.escape(prefix + sp).replace(r"\*", r"\d+") + r"(_packed|_scale|_zero_point|_shape|_g_idx)?$"
                )
                total += sum(8 * self._nbytes(k) for k in self.tensor_index if pat.match(k))
            if total:
                return total
        fm = re.match(r"^(.*\.experts)\.(gate_up_proj|down_proj)$", name)
        if fm:
            prefix, kind = fm.group(1), fm.group(2)
            parts = ("down",) if kind == "down_proj" else ("gate", "up")
            pat = re.compile(
                "^" + re.escape(prefix) + r"\.\d+\.(" + "|".join(parts) +
                r")_proj\.weight(_packed|_scale|_scale_inv|_zero_point|_shape|_g_idx)?$")
            total = sum(8 * self._nbytes(k) for k in self.tensor_index if pat.match(k))
            if total:
                return total
        return None

    def _get_tensor(self, name):
        import re
        if name in self.tensor_index:
            # ModelOpt, ct-float-quantized and fp8 keep the quantized tensor under the plain
            # .weight name; the scale sidecar marks it as needing dequant. Fused fp8 expert
            # tensors (Mistral-Small-4 experts.gate_up_proj etc.) carry the same _scale_inv
            # sidecar without a .weight suffix
            if self.dequant_scaled and (
                    f"{name}_scale" in self.tensor_index or f"{name}_scale_inv" in self.tensor_index
                    or (name.endswith(".weight") and f"{name[:-len('.weight')]}.scale" in self.tensor_index)):
                return self._dequant_scaled(name)
            return self._read_shard(name)
        if name == self.head_weight_name and self.embed_weight_name in self.tensor_index:
            return self._read_shard(self.embed_weight_name)  # tied embeddings

        # gpt-oss-style MXFP4 (<name>_blocks + <name>_scales, e.g. mlp.experts.gate_up_proj):
        # transformers' own converter handles the fp4-pair unpack, e8m0 ldexp and the transpose
        # into module orientation
        if f"{name}_blocks" in self.tensor_index:
            from transformers.integrations.mxfp4 import convert_moe_packed_tensors
            return convert_moe_packed_tensors(
                self._read_shard(f"{name}_blocks"), self._read_shard(f"{name}_scales"))

        # compressed-tensors packed weight: nvfp4 when a global scale exists, int otherwise
        if name.endswith(".weight"):
            stem = name[:-len(".weight")]
            if f"{stem}.weight_packed" in self.tensor_index:
                if f"{stem}.weight_global_scale" in self.tensor_index:
                    return self._dequant_ct_nvfp4(stem)
                return self._dequant_ct(stem)

        # Per-expert checkpoints vs fused-expert module trees (mistral4 AWQ/compressed-tensors:
        # experts.{i}.{gate,up,down}_proj vs experts.gate_up_proj) with no declared conversion
        # mapping: stack the per-expert tensors, resolving each recursively so quantized
        # formats dequantize on the way in. Module orientation is (E, out, in), gate rows first
        fm = re.match(r"^(.*\.experts)\.(gate_up_proj|down_proj)$", name)
        if fm:
            prefix, kind = fm.group(1), fm.group(2)
            probe = "down" if kind == "down_proj" else "gate"
            e = 0
            while (f"{prefix}.{e}.{probe}_proj.weight" in self.tensor_index or
                   f"{prefix}.{e}.{probe}_proj.weight_packed" in self.tensor_index):
                e += 1
            if e:
                if kind == "down_proj":
                    return torch.stack([self._get_tensor(f"{prefix}.{i}.down_proj.weight") for i in range(e)])
                return torch.stack([
                    torch.cat([self._get_tensor(f"{prefix}.{i}.gate_proj.weight"),
                               self._get_tensor(f"{prefix}.{i}.up_proj.weight")], dim = 0)
                    for i in range(e)])

        # Merge-type conversion (fused MoE experts): gather the per-expert source tensors for
        # this prefix, stack each source group (MergeModulelist), then concatenate groups
        # (Concatenate) if the converter has more than one. Sources resolve recursively, so
        # quantized per-expert weights dequantize on the way in
        for target, sources, ops in self.converters:
            if not (name == target or name.endswith("." + target)):
                continue
            prefix = name[:-len(target)]
            merge_dim = next((getattr(o, "dim", 0) for o in ops if type(o).__name__ == "MergeModulelist"), 0)
            cat_dim = next((getattr(o, "dim", 1) for o in ops if type(o).__name__ == "Concatenate"), None)
            groups = []
            for sp in sources:
                base = prefix + sp
                if base.endswith(".weight"):
                    stem_re = re.compile(
                        "^" + re.escape(base[:-len(".weight")]).replace(r"\*", r"(\d+)") + r"\.weight(_packed)?$"
                    )
                    matched = {}
                    for key in self.tensor_index:
                        m = stem_re.match(key)
                        if m:
                            matched[int(m.group(1))] = key[:key.rindex(".weight")]
                    if not matched:
                        raise KeyError(f"No checkpoint tensors match {base}")
                    tensors = [self._get_tensor(matched[i] + ".weight") for i in sorted(matched)]
                else:
                    pat = re.compile("^" + re.escape(base).replace(r"\*", r"(\d+)") + "$")
                    matched = sorted(
                        (int(m.group(1)), key)
                        for key in self.tensor_index
                        if (m := pat.match(key))
                    )
                    if not matched:
                        raise KeyError(f"No checkpoint tensors match {base}")
                    tensors = [self._read_shard(k) for _, k in matched]
                groups.append(torch.stack(tensors, dim = merge_dim))
            return torch.cat(groups, dim = cat_dim) if cat_dim is not None and len(groups) > 1 else groups[0]

        raise KeyError(f"{name} not found in checkpoint shards")

    def _plain_fp32(self, name):
        """True for tensors stored fp32 and read verbatim (no dequant): these are exactly the
        ones from_pretrained's _keep_in_fp32_modules lists keep in fp32 (hc mixers, sinks,
        position biases, norms), so the streamed skeleton must too or mHC/routing amplify the
        bf16 rounding into a visible reference deviation."""
        if name not in self.tensor_index:
            return False
        if f"{name}_scale" in self.tensor_index or f"{name}_scale_inv" in self.tensor_index:
            return False
        if name.endswith(".weight") and f"{name[:-len('.weight')]}.scale" in self.tensor_index:
            return False
        return True

    def _materialize(self, module):
        prefix = self.prefix[id(module)]
        sd = {}
        for pn, _ in module.named_parameters():
            full = f"{prefix}.{pn}" if prefix else pn
            t = self._get_tensor(full)
            if t.is_floating_point() and not (t.dtype == torch.float32 and self._plain_fp32(full)):
                t = t.to(self.device, self.dtype)
            else:
                t = t.to(self.device)
            sd[pn] = t
        module.load_state_dict(sd, strict = False, assign = True)
        for pn, p in module.named_parameters():
            if p.is_meta:
                raise RuntimeError(f"No checkpoint tensor for {self.prefix[id(module)]}.{pn}")

    def _dematerialize(self, module):
        for sub in module.modules():
            for pn, p in list(sub._parameters.items()):
                if p is not None and not p.is_meta:
                    sub._parameters[pn] = torch.nn.Parameter(p.to("meta"), requires_grad = False)

    @torch.inference_mode()
    def _run_streaming(self, ids: torch.Tensor, callback, noise_eps: float = None):
        base = self.model.base_model
        layers = self._decoder_layers()
        embed = self.model.get_input_embeddings()
        head = self.model.get_output_embeddings()

        # Hook the embedding, every decoder layer, and every other weight-carrying module found
        # by walking the base tree - recursing into containers that hold the special modules
        # (multimodal wrappers nest the text stack one level down) instead of hooking them
        # wholesale, so per-layer streaming is preserved. The head is streamed manually below
        special = {id(embed)} | {id(m) for m in layers}
        def contains_special(m):
            return any(id(x) in special for x in m.modules())
        extra = []
        def walk(m):
            for child in m.children():
                if id(child) in special:
                    continue
                if not any(True for _ in child.parameters()):
                    continue
                if contains_special(child):
                    walk(child)
                else:
                    extra.append(child)
        walk(base)
        hook_modules = [embed] + list(layers) + extra

        pb_state = {"n": 0}
        pb = ProgressBar("Streaming", len(hook_modules) + 1)

        def pre_hook(module, args):
            self._materialize(module)

        def post_hook(module, args, output):
            self._dematerialize(module)
            pb_state["n"] += 1
            pb.update(pb_state["n"])

        hooks = []
        for m in hook_modules:
            hooks.append(m.register_forward_pre_hook(pre_hook))
            hooks.append(m.register_forward_hook(post_hook))
        if noise_eps:
            hooks += self._noise_hooks(noise_eps)

        try:
            with pb:
                # One batched pass: every weight is read exactly once. Activations are
                # rows x len x hidden, tiny next to the weights being streamed
                hidden = base(input_ids = ids.to(self.device), use_cache = False).last_hidden_state
                self._materialize(head)
                for r in range(ids.shape[0]):
                    logits = head(hidden[r:r + 1])
                    if self.logit_softcap:
                        logits = torch.tanh(logits / self.logit_softcap) * self.logit_softcap
                    callback(r, logits)
                self._dematerialize(head)
                pb.update(len(hook_modules) + 1)
        finally:
            for h in hooks:
                h.remove()

    @torch.inference_mode()
    def run(self, ids: torch.Tensor, callback, noise_eps: float = None):
        if self.streaming:
            return self._run_streaming(ids, callback, noise_eps)
        hooks = self._noise_hooks(noise_eps) if noise_eps else []
        try:
            with ProgressBar("Evaluating", ids.shape[0]) as pb:
                for r in range(ids.shape[0]):
                    row = ids[r:r + 1].to(self.model.device)
                    out = self.model(input_ids = row, use_cache = False)
                    callback(r, out.logits)
                    del out
                    pb.update(r + 1)
        finally:
            for h in hooks:
                h.remove()

    def close(self):
        del self.model
        self.shard_handles.clear()
        free_mem()


def gguf_shards(source: str) -> list:
    """All files of a split GGUF ("...-00001-of-00003.gguf" naming); [source] when unsplit"""
    import re
    m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", source)
    if not m:
        return [source]
    base, _, n = m.group(1), m.group(2), int(m.group(3))
    shards = [f"{base}-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)]
    missing = [s for s in shards if not os.path.exists(s)]
    assert not missing, f"Split GGUF is missing shards: {missing}"
    return shards


def gguf_storage_info(source: str) -> dict:
    """bpw/vram accounting over the full tensor table, spanning all shards of a split GGUF.
    Norms/biases are < 2 dims; router gates excluded by name; token_embd is tallied under its
    own bucket and also serves as the head fallback for tied models (not double-counted:
    head_is_fallback tracks whether output.weight has since overridden it), overridden by
    output.weight when present in any shard"""
    from gguf import GGUFReader
    sum_bits = sum_numel = head_bits = head_numel = embed_bits = embed_numel = 0
    head_is_fallback = True
    for shard in gguf_shards(source):
        reader = GGUFReader(shard)
        for t in reader.tensors:
            if t.name == "token_embd.weight":
                embed_bits += t.n_bytes * 8
                embed_numel += t.n_elements
                if head_is_fallback:
                    head_bits = t.n_bytes * 8
                    head_numel = t.n_elements
            elif t.name == "output.weight":
                head_bits = t.n_bytes * 8
                head_numel = t.n_elements
                head_is_fallback = False
            elif (
                t.name.endswith(".weight")
                and len(t.shape) >= 2
                and not any(k in t.name for k in GGUF_ROUTER_KEYS)
            ):
                sum_bits += t.n_bytes * 8
                sum_numel += t.n_elements
        del reader
    return {
        "bpw_layer": sum_bits / max(sum_numel, 1),
        "bpw_head": head_bits / max(head_numel, 1),
        "bpw_embed": embed_bits / max(embed_numel, 1),
        "vram_gb": (sum_bits + head_bits + (0 if head_is_fallback else embed_bits)) / 8 / 1024 ** 3,
    }


class LlamaCppBackend:
    """llama-cpp-python with logits_all; storage info from the GGUF tensor table"""

    def __init__(self, source: str, max_len: int, device: torch.device, options: dict):
        import llama_cpp
        from llama_cpp import Llama
        self.device = device
        self.info = gguf_storage_info(source)
        split_modes = {
            "layer": llama_cpp.LLAMA_SPLIT_MODE_LAYER,
            "row": llama_cpp.LLAMA_SPLIT_MODE_ROW,
            "none": llama_cpp.LLAMA_SPLIT_MODE_NONE,  # everything on main_gpu
        }
        self.model = Llama(
            model_path = source,
            logits_all = True,
            verbose = False,
            n_ctx = max_len,
            n_gpu_layers = options.get("n_gpu_layers", 999),
            split_mode = split_modes[options.get("split_mode", "layer")],
            main_gpu = options.get("main_gpu", 0),
        )

    def run(self, ids: torch.Tensor, callback, noise_eps: float = None):
        assert not noise_eps, "Noise injection not supported for llamacpp engine"
        with ProgressBar("Evaluating", ids.shape[0]) as pb:
            for r in range(ids.shape[0]):
                self.model.reset()
                self.model.eval(ids[r].tolist())
                logits = torch.from_numpy(self.model.scores).unsqueeze(0)
                logits = logits[:, :ids.shape[1]].to(self.device)
                callback(r, logits)
                pb.update(r + 1)

    def close(self):
        del self.model
        free_mem()


# ------ vLLM engine ------------------------------------------------------------------
#
# vLLM's public prompt_logprobs API is built for a UI's top-k display, not a dense
# [tokens, vocab] dump, and asking it for the full vocabulary (prompt_logprobs=-1) is
# expensive in a way that isn't just "a lot of Python objects":
#
#   >>> torch.topk(torch.randn(1024, 151936, device="cuda"), 151936, dim=-1)
#   peaks at ~7 GiB, transient, on top of the weights and KV cache -- vLLM's own
#   compute_topk_scores does exactly this per 1024-token chunk of scored prompt whenever
#   num_logprobs asks for (close to) the whole vocabulary, because k close to n makes
#   torch.topk fall back to something close to a full sort, workspace and all. That spike
#   happens *after* vLLM's memory profiler already decided how much of the GPU to give the
#   KV cache (the profiler's own warmup pass never asks for full-vocab logprobs), so it's
#   invisible to --gpu-memory-utilization and friends and shows up as an OOM inside
#   torch.topk that no amount of turning those knobs down reliably fixes -- only handing
#   KV cache *less* room works, and only by accident, by leaving more headroom for a spike
#   the profiler never saw coming.
#
# So this doesn't ask vLLM to do that sort at all. compute_topk_scores's *input* --
# `logits`, the [chunk, vocab] raw tensor computed once regardless of what width was
# requested -- is exactly what this needs, before the expensive part. Patching it to grab
# that argument and requesting prompt_logprobs=1 (not -1) means the real torch.topk vLLM
# still runs is a cheap top-1, and the tensor this wants was already sitting right there.
#
# Getting the per-request boundaries right without redoing vLLM's own request-splitting
# arithmetic (query_start_loc bookkeeping, chunked-prefill continuation, ...) is what
# max_num_seqs=1 buys: with only ever one request's prompt being scored at a time, every
# compute_topk_scores call in between "this row's prompt logprobs started" and "this row's
# prompt logprobs finished" belongs to that row, in position order, full stop. Slower than
# letting vLLM interleave requests for throughput, which is not a thing an eval harness
# needs from it.
#
# The other half of the row (the position after the last prompt token) never goes through
# prompt_logprobs at all -- vLLM only scores it if something asks it to generate, which is
# what max_tokens=1 is for. Its full-vocab distribution comes back through the ordinary
# sample-logprobs path instead, at the *requested* logprobs=-1: compute_topk_scores there
# always runs on a single row (one generated token), where the same full-width torch.topk
# costs ~0.6 GiB, not ~7 -- the row count is what made the prompt side expensive, not the
# vocab width by itself, so this path is left alone.
#
# Both hooks fire per row, not after the whole batch's generate() call returns, so this
# backend never holds more than one row's full-vocab tensor in memory at once -- at
# qbench's usual scale (10 rows, 2048 tokens, a 150-256k vocab) the alternative is tens of
# GB. EngineCoreOutput.update_from_output calls _update_sample_logprobs before
# _update_prompt_logprobs (the reverse of what the names' reading order suggests); rather
# than depend on that, whichever hook sees its half of a row arrive first stashes it and
# the other one finalizes.
#
# Three load-bearing pieces this depends on, worth knowing if a vLLM bump breaks it:
#  - VLLM_ENABLE_V1_MULTIPROCESSING=0 keeps EngineCore in this process, or none of the
#    patched code is ever the code the actual serving path runs (it lives in a subprocess
#    by default).
#  - logprobs_mode="raw_logits" (engine-level, ModelConfig) skips softmax and any
#    temperature/top-p processing, so what's captured is the model's literal pre-softmax
#    output -- matching every other backend's convention here.
#  - compute_topk_scores is patched as bound in vllm.v1.worker.gpu.sample.prompt_logprob
#    (where compute_prompt_logprobs_with_chunking looks it up via its own `from ... import`
#    binding), not in vllm.v1.worker.gpu.sample.logprob where it's defined -- that's what
#    keeps this from also intercepting (and truncating) the sample-logprobs path, which
#    imports the same function from its original module and must keep working at full
#    width. This is the V2 GPU model runner's code path specifically (the one actually
#    active in every environment this was built and tested against); the classic runner
#    (gpu_model_runner.py, use_v2_model_runner=False) computes prompt logprobs through
#    different code this does not patch, and would need the same fix applied there too.

_VLLM_HOOK_INSTALLED = False


def _install_vllm_prompt_logprob_hook():
    global _VLLM_HOOK_INSTALLED
    if _VLLM_HOOK_INSTALLED:
        return
    import vllm.v1.engine.logprobs as lp_mod
    import vllm.v1.worker.gpu.sample.prompt_logprob as plp_mod

    orig_from_new_request = lp_mod.LogprobsProcessor.from_new_request.__func__
    orig_update_prompt = lp_mod.LogprobsProcessor._update_prompt_logprobs
    orig_update_sample = lp_mod.LogprobsProcessor._update_sample_logprobs
    orig_topk_scores = plp_mod.compute_topk_scores

    def patched_from_new_request(cls, tokenizer, request):
        proc = orig_from_new_request(cls, tokenizer, request)
        # EngineCoreRequest.request_id is a "{parent}-{suffix}" child id; callers only
        # ever see the parent id (RequestOutput.request_id), which is what run() keys on.
        proc._qbench_req_id = request.request_id.split("-")[0]
        return proc

    def patched_topk_scores(logits, num_logprobs, *args, **kwargs):
        # Grab the real tensor before vLLM's own (now cheap, width=1) reduction of it.
        # No row is currently "in progress" the first time this fires for a given row (see
        # module docstring: max_num_seqs=1 makes this ordering safe), so plain appending is
        # enough -- nothing else can interleave chunks from a different request in between.
        if _VLLM_RUN_CTX.get("rows"):
            _VLLM_RAW_CHUNKS.append(logits.detach().to(device="cpu", dtype=torch.float32))
        return orig_topk_scores(logits, num_logprobs, *args, **kwargs)

    def patched_update_prompt(self, prompt_logprobs_tensors):
        rid = getattr(self, "_qbench_req_id", None)
        rows = _VLLM_RUN_CTX.get("rows")
        if rows is not None and rid in rows:
            dense = torch.cat(_VLLM_RAW_CHUNKS, dim=0) if _VLLM_RAW_CHUNKS else \
                torch.zeros((0, _VLLM_RUN_CTX["vocab"]))
            _VLLM_RAW_CHUNKS.clear()
            # update_from_output runs _update_sample_logprobs before this method, so the
            # last-row piece (below) is typically already waiting -- finalize and fire the
            # callback here rather than there.
            if rid in _VLLM_SAMPLE_PARTS:
                _finish_row(rid, dense, _VLLM_SAMPLE_PARTS.pop(rid))
            else:
                _VLLM_PROMPT_PARTS[rid] = dense
        return orig_update_prompt(self, prompt_logprobs_tensors)

    def patched_update_sample(self, logprobs_lists):
        result = orig_update_sample(self, logprobs_lists)
        rid = getattr(self, "_qbench_req_id", None)
        rows = _VLLM_RUN_CTX.get("rows")
        if rows is not None and rid in rows:
            vocab = _VLLM_RUN_CTX["vocab"]
            last = torch.full((1, vocab), float("-inf"))
            gen_lp = self.logprobs[-1] if self.logprobs else {}
            for tok_id, lp in (gen_lp or {}).items():
                if tok_id < vocab:
                    last[0, tok_id] = lp.logprob
            if rid in _VLLM_PROMPT_PARTS:
                _finish_row(rid, _VLLM_PROMPT_PARTS.pop(rid), last)
            else:
                _VLLM_SAMPLE_PARTS[rid] = last
        return result

    def _finish_row(rid, prompt_part, last_row):
        vocab = _VLLM_RUN_CTX["vocab"]
        length = _VLLM_RUN_CTX["length"]
        full = torch.cat([prompt_part, last_row], dim=0)
        if full.shape[0] < length:
            full = torch.cat([full, full.new_full((length - full.shape[0], vocab), float("-inf"))])
        full = full[:length].unsqueeze(0).to(_VLLM_RUN_CTX["device"])
        _VLLM_RUN_CTX["callback"](_VLLM_RUN_CTX["rows"][rid], full)

    lp_mod.LogprobsProcessor.from_new_request = classmethod(patched_from_new_request)
    lp_mod.LogprobsProcessor._update_prompt_logprobs = patched_update_prompt
    lp_mod.LogprobsProcessor._update_sample_logprobs = patched_update_sample
    plp_mod.compute_topk_scores = patched_topk_scores
    _VLLM_HOOK_INSTALLED = True


# State for whichever VllmBackend.run() call is currently in flight. Module-level because
# the patched methods above are bound to vLLM's own class/module, not to any VllmBackend
# instance; only one run() is ever active at a time (qbench runs engines sequentially, and
# within a run, max_num_seqs=1 keeps rows from interleaving), so this is a single slot
# rather than something keyed by backend instance.
_VLLM_RUN_CTX: dict = {}          # {"rows": {req_id: row_idx}, "vocab", "length", "device", "callback"}
_VLLM_RAW_CHUNKS: list = []       # raw [chunk, vocab] tensors for whichever row is currently scoring
_VLLM_PROMPT_PARTS: dict = {}     # req_id -> this row's assembled [length-1, vocab] tensor, once done
_VLLM_SAMPLE_PARTS: dict = {}     # req_id -> the [1, vocab] last-row piece, whichever hook sees it first


# Suffix -> logical module key, covering the on-disk conventions the engines this project
# actually compares understand: compressed-tensors (AWQ/GPTQ/FP8/NVFP4 via vLLM) and this
# project's own EXL3 plugin. Order doesn't matter (longest match isn't ambiguous here --
# no suffix is a prefix of another), but every one of them must be tried before the bare
# "weight" fallback. Duplicated here as plain strings rather than imported from
# vllm_exl3_plugin.format, so qbench keeps working standalone in exllamav3's own repo.
_CT_SUFFIXES = ("weight_packed", "weight_scale_inv", "weight_scale_2", "weight_scale",
                "weight_zero_point", "weight_shape", "weight_g_idx", "weight_global_scale",
                "input_global_scale", "input_scale", "scale")
_EXL3_SUFFIXES = ("trellis", "suh", "svh", "mcg", "mul1", "su", "sv")
# Classic GPTQ/AWQ (autogptq, autoawq, auto-round, ...), which predate compressed-tensors
# and name nothing the same way: an int-packed `qweight` plus `scales`/`qzeros`/`g_idx`
# sidecars, and no `weight`/`weight_shape` anywhere. Distinct enough from _CT_SUFFIXES
# that the two never collide ("scales" is not "scale", ".g_idx" is not ".weight_g_idx").
_GPTQ_SUFFIXES = ("qweight", "qzeros", "scales", "g_idx")
_KNOWN_SUFFIXES = _CT_SUFFIXES + _EXL3_SUFFIXES + _GPTQ_SUFFIXES + ("weight",)

#: Bits per stored element, for recovering an int-packed tensor's logical element count.
_ST_DTYPE_BITS = {"I64": 64, "U64": 64, "I32": 32, "U32": 32, "I16": 16, "U16": 16,
                  "I8": 8, "U8": 8, "F64": 64, "F32": 32, "F16": 16, "BF16": 16}


def _safetensors_headers(source: str) -> dict:
    """checkpoint tensor name -> (nbytes, shape, dtype, path, data_start), reading only
    shard headers. data_start is the absolute file offset of the tensor's own bytes, for
    the rare case (compressed-tensors' *.weight_shape) where a tensor's *contents*, not
    just its header shape, are needed -- those are always a couple of int values, cheap to
    seek-and-read directly without touching the (potentially huge) rest of the shard."""
    import json
    import struct
    index_file = os.path.join(source, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            files = set(json.load(f)["weight_map"].values())
    else:
        files = {"model.safetensors"}
    out = {}
    for fn in files:
        path = os.path.join(source, fn)
        with open(path, "rb") as f:
            hdr_len = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hdr_len))
            data_start = 8 + hdr_len
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            a, b = v["data_offsets"]
            out[k] = (b - a, v.get("shape", []), v.get("dtype"), path, data_start + a)
    return out


_ST_DTYPES = {"I64": "<q", "I32": "<i", "I16": "<h", "U64": "<Q", "U32": "<I", "U16": "<H"}


def _read_small_int_tensor(path: str, offset: int, nbytes: int, dtype: str) -> list:
    """Read a tiny integer tensor's actual values directly off disk (a targeted seek, not
    a full tensor load) -- used only for compressed-tensors' *.weight_shape sidecars,
    which store the true unpacked [out, in] as 2 int values rather than as their own
    on-disk shape."""
    import struct
    fmt = _ST_DTYPES.get(dtype)
    if fmt is None:
        return []
    n = nbytes // struct.calcsize(fmt)
    with open(path, "rb") as f:
        f.seek(offset)
        return list(struct.unpack(f"<{n}{fmt[-1]}", f.read(nbytes)))


def _read_config(source: str) -> dict:
    import json
    try:
        with open(os.path.join(source, "config.json")) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _tie_word_embeddings(config: dict) -> bool:
    """config.json's tie_word_embeddings, checked under text_config too for multimodal
    wrappers (mirrors TransformersBackend's own text_config-or-config fallback)."""
    text_config = config.get("text_config") or config
    return bool(text_config.get("tie_word_embeddings", config.get("tie_word_embeddings", False)))


def _quant_bits(config: dict) -> int | None:
    """Weight bit width from a GPTQ/AWQ-style quantization_config, which is the only place
    it is recorded -- unlike EXL3 (bit width recoverable from the trellis shape) or
    compressed-tensors (which ships an explicit weight_shape sidecar), an int-packed
    `qweight` cannot be un-packed to a logical element count without knowing it."""
    qc = config.get("quantization_config")
    if not isinstance(qc, dict):
        return None
    bits = qc.get("bits") or qc.get("w_bit") or qc.get("weight_bits")
    return int(bits) if isinstance(bits, (int, float)) and 1 <= bits <= 16 else None


def safetensors_storage_info(source: str) -> dict:
    """bpw/vram accounting straight off an HF-repo checkpoint's own stored bytes, for any
    vLLM-loadable safetensors model: plain bf16/fp16, compressed-tensors (AWQ/GPTQ/FP8/
    NVFP4), or EXL3 via this project's plugin. Same convention as gguf_storage_info:
    norms/biases excluded, router gates excluded by name, token embeddings tallied
    separately and used as the head's fallback for tied models.

    A tied model's on-disk lm_head tensor, if any, is never counted as separate storage,
    regardless of what the checkpoint physically contains: vLLM ties lm_head to the
    embedding and skips loading every lm_head.* weight outright (see this project's own
    EXL3Config.head_is_quantized), and this project's own EXL3 quantizer is known to write
    a full redundant lm_head anyway for every tied model it quantizes (TODO.md #2) -- bytes
    that are on disk but never make it into VRAM. Trusting the on-disk tensor here would
    silently overcount vram_gb by exactly that redundant head's size.

    A tensor whose suffix isn't recognized is bucketed under its own full name and
    treated as ndim < 2 (i.e. dropped), so an unknown format undercounts rather than
    crashing or silently merging into the wrong bucket -- quietly, which is a real hazard:
    a format whose weights are named in a way none of the suffix tables know reports
    bpw_layer=0.0 and a vram_gb covering only the embedding. Precise for what this project
    actually compares today; extend _CT_SUFFIXES/_EXL3_SUFFIXES/_GPTQ_SUFFIXES for a new
    one, and check bpw_layer is non-zero when first pointing this at an unfamiliar format.
    """
    config = _read_config(source)
    tied = _tie_word_embeddings(config)
    quant_bits = _quant_bits(config)
    headers = _safetensors_headers(source)
    TILE = 16  # EXL3 trellis is tile-granular on both dims (vllm_exl3_plugin/format.py)

    # module_key -> [bits, trellis_shape, plain_weight_shape, weight_shape_payload (the
    #                *values* in a compressed-tensors .weight_shape sidecar), qweight]
    modules: dict = {}
    for name, (nbytes, shape, dtype, path, offset) in headers.items():
        if name.endswith(".bias"):
            continue
        module_key, suffix = name, None
        for suf in _KNOWN_SUFFIXES:
            if name.endswith("." + suf):
                module_key, suffix = name[: -(len(suf) + 1)], suf
                break
        m = modules.setdefault(module_key, [0, None, None, None, None])
        m[0] += nbytes * 8
        if suffix == "trellis":
            m[1] = shape
        elif suffix == "weight":
            m[2] = shape
        elif suffix == "weight_shape":
            # Not the tensor's own on-disk shape (that's just "[2]") -- its *contents* are
            # the true unpacked [out, in] compressed-tensors reports for a packed weight.
            m[3] = _read_small_int_tensor(path, offset, nbytes, dtype)
        elif suffix == "qweight":
            m[4] = (shape, dtype)

    sum_bits = sum_numel = head_bits = head_numel = embed_bits = embed_numel = 0
    head_is_fallback = True
    for module_key, (bits, trellis_shape, weight_shape, packed_shape, qweight) in modules.items():
        if any(k in module_key for k in HF_ROUTER_KEYS):
            continue
        is_embed = module_key.endswith("embed_tokens") or module_key.endswith(".wte")
        is_head = module_key.endswith("lm_head") or module_key.split(".")[-1] == "output"
        if is_head and tied:
            continue  # never separately allocated; falls back to the embed bucket below

        if trellis_shape is not None:
            numel = trellis_shape[0] * TILE * trellis_shape[1] * TILE
        elif packed_shape:
            numel = 1
            for d in packed_shape:
                numel *= d
        elif qweight is not None and quant_bits:
            # GPTQ/AWQ: `qweight` holds in_features * out_features weights bit-packed into
            # a wider int, and the two formats disagree on which axis they pack along
            # (GPTQ [in/8, out], AWQ [in, out/8] at 4 bits). Total stored elements times
            # the packing factor is the same either way, so this needs no per-format
            # branch -- only the bit width, which is why quant_bits gates it.
            qshape, qdtype = qweight
            per_element = _ST_DTYPE_BITS.get(qdtype, 0) // quant_bits
            numel = 1
            for d in qshape:
                numel *= d
            numel *= per_element
        elif weight_shape is not None and len(weight_shape) >= 2:
            numel = 1
            for d in weight_shape:
                numel *= d
        else:
            numel = 0  # norm, bias-only, or something this function doesn't recognize

        if not numel:
            continue
        if is_embed:
            embed_bits += bits
            embed_numel += numel
            if head_is_fallback:
                head_bits, head_numel = bits, numel
        elif is_head:
            head_bits, head_numel = bits, numel
            head_is_fallback = False
        else:
            sum_bits += bits
            sum_numel += numel

    return {
        "bpw_layer": sum_bits / max(sum_numel, 1),
        "bpw_head": head_bits / max(head_numel, 1),
        "bpw_embed": embed_bits / max(embed_numel, 1),
        "vram_gb": (sum_bits + head_bits + (0 if head_is_fallback else embed_bits)) / 8 / 1024 ** 3,
    }


class VllmBackend:
    """vLLM engine, via the offline LLM API. Exists mainly to compare this project's own
    EXL3 vLLM plugin -- and other vLLM-servable quant formats (AWQ, GPTQ, FP8, ...) -- to
    exllamav3 native and the transformers reference under one methodology, on the actual
    served code path rather than a proxy for it.

    options: anything accepted by vllm.LLM (quantization, gpu_memory_utilization,
    enforce_eager, tensor_parallel_size, dtype, ...) is passed straight through, including
    max_num_seqs if a project genuinely wants concurrency back at the cost of the memory
    behavior described in the module docstring above -- default is 1 (rows scored strictly
    one at a time), which is what makes the prompt-logprobs capture correct and cheap.

    Noise injection (the self-noise-floor pass) is not implemented: unlike
    TransformersBackend's forward-hook approach, vLLM's decoder layers aren't at a
    predictable, engine-version-stable location. Don't put a vllm-engine model in the
    'reference' group with noise_floor left at its default (true), or set
    `noise_floor: false` in the project.
    """

    def __init__(self, source: str, max_len: int, device: torch.device, options: dict):
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        _install_vllm_prompt_logprob_hook()

        self.info = safetensors_storage_info(source)
        self.device = device

        from vllm import LLM
        llm_kwargs = dict(options)
        llm_kwargs.setdefault("gpu_memory_utilization", 0.85)
        llm_kwargs.setdefault("enforce_eager", True)
        llm_kwargs.setdefault("max_num_seqs", 1)
        self.llm = LLM(
            model=source,
            max_model_len=max_len,
            logprobs_mode="raw_logits",
            max_logprobs=-1,
            disable_log_stats=True,
            **llm_kwargs,
        )
        self.vocab_size = self.llm.llm_engine.model_config.get_vocab_size()

    def run(self, ids: torch.Tensor, callback, noise_eps: float = None):
        assert not noise_eps, "Noise injection not supported for the vllm engine"
        from vllm import SamplingParams

        rows = ids.shape[0]
        length = ids.shape[1]
        pb = ProgressBar("Evaluating", rows)
        done = {"n": 0}

        def wrapped_callback(r, logits):
            callback(r, logits)
            done["n"] += 1
            pb.update(done["n"])

        # Row index keyed by request_id ("0".."rows-1" in submission order -- vLLM's
        # offline API assigns ids that way for a single generate() call), so the hooks
        # above can fire wrapped_callback the moment each row finishes, rather than this
        # function collecting every row's full-vocab tensor before any of them run.
        _VLLM_RUN_CTX.update(rows={str(r): r for r in range(rows)}, vocab=self.vocab_size,
                              length=length, device=self.device, callback=wrapped_callback)
        _VLLM_PROMPT_PARTS.clear()
        _VLLM_SAMPLE_PARTS.clear()

        # prompt_logprobs=1, not -1: the module docstring above explains why asking for the
        # full vocabulary here would be actively harmful rather than just slow. logprobs=-1
        # (the *sample* side, one row at a time regardless) is the one that stays at -1.
        sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=-1, prompt_logprobs=1,
                             detokenize=False)
        try:
            with pb:
                self.llm.generate(prompts=[ids[r].tolist() for r in range(rows)],
                                   sampling_params=sp, use_tqdm=False)
        finally:
            _VLLM_RUN_CTX.clear()
            _VLLM_PROMPT_PARTS.clear()
            _VLLM_SAMPLE_PARTS.clear()

        assert done["n"] == rows, (
            f"vllm engine: only {done['n']}/{rows} rows produced logits -- a row generated "
            "0 tokens (EOS on its first position?) or the sample-logprobs hook didn't fire"
        )

    def close(self):
        """Release the GPU for the next model in the project.

        `del self.llm` alone frees almost nothing (measured: 8162 -> 8102 MiB, i.e. the
        entire KV cache reservation stays resident), because vLLM keeps the engine's
        worker alive behind module-level distributed/parallel state that a dropped
        reference does not touch. A project with more than one vllm-engine model would
        then fail on the *second* one -- either outright ("Free memory on device ... is
        less than desired GPU memory utilization") or, when it just barely fits, as a
        nondeterministic OOM later on depending on fragmentation.

        This is the same teardown sequence vLLM's own test suite uses between models
        (tests/conftest.py: `del self.model; cleanup_dist_env_and_memory()`), which is the
        supported way to run several engines back to back in one process.
        """
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        # Stops the worker itself; without it the model and KV cache stay referenced by
        # the still-running engine core regardless of what the caller drops.
        try:
            self.llm.llm_engine.engine_core.shutdown()
        except Exception as e:
            print(f" !! vllm engine: engine_core.shutdown() failed ({e}); "
                  f"continuing with distributed cleanup")
        del self.llm
        cleanup_dist_env_and_memory()
        free_mem()


ENGINES = {
    "exllamav3": Exl3Backend,
    "transformers": TransformersBackend,
    "llamacpp": LlamaCppBackend,
    "vllm": VllmBackend,
}


def open_backend(mspec: dict, max_len: int, device: torch.device):
    engine = mspec["engine"]
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine: {engine} (available: {', '.join(ENGINES)})")
    print(f" -- Loading ({engine}): {mspec['source']}")
    return ENGINES[engine](mspec["source"], max_len, device, mspec.get("options", {}))
