"""
KLD/ppl measurement against the cached reference logits.
"""

import math
import os

import torch

from exllamav3.util.measures import compute_kl_div, compute_target_log_probs

from .data import load_tensor, save_tensors

# One bf16 unit roundoff of multiplicative noise per element per layer. The measured floor is
# insensitive to the exact scale (it plateaus), so this needs no careful calibration.
BF16_ROUNDING_EPS = 2 ** -9

CONF_BUCKETS = [(0.00, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 0.95), (0.95, 1.01)]

# Bumped when the per-model measurement set changes; invalidates cached KLD results without
# invalidating the (expensive) cached reference logits. v3: bpw convention unified across
# engines (biases/router gates excluded, tied-head fallback). v4: per-token KLD vectors saved
# as a sidecar so histograms of (KLD - noise floor) can pair tokens across passes. v5: sidecars
# stored fp32 (fp16 flushed sub-6e-8 KLDs to zero, censoring the near-exact-reproduction tail
# that in-domain trace data surfaces).
METRICS_VERSION = 5
# The same for per-row results alone, so that a recall field leaves every other cached KLD result standing. v2: kept,
# bands and confident errors.
PER_ROW_VERSION = 2


#: Bands of how firmly the reference held an answer: name, lower and upper P
RECALL_BANDS = (("sure", 0.9, 1.01), ("firm", 0.7, 0.9), ("leaning", 0.5, 0.7), ("shaky", 0.3, 0.5), ("barely", 0.0, 0.3))


#: At or above this, a wrong token counts as the model stating something rather than leaning toward it
CONFIDENT_P = 0.7


def recall_kept(rows: list) -> float:
    """Of the correct answers the reference would give under sampling, the share the model still gives. Per question,
    so a gain on one cannot pay for a loss on another: a plain sum ratio rated 27B 3 bpw above 4 bpw."""
    return sum(min(math.exp(x["lp"]), math.exp(x["ref_lp"])) for x in rows) / sum(math.exp(x["ref_lp"]) for x in rows)


def first_wrong_p(logits: torch.Tensor, targets: torch.Tensor, vocab: int) -> float:
    """How sure the model is of what it chose at the first answer token it gets wrong; 0 if it gets none wrong.
    A low answer probability alone cannot tell a shrug from a confident wrong answer, and this can."""
    top, chosen = logits[..., :vocab].softmax(dim = -1).max(dim = -1)
    wrong = (chosen != targets).view(-1)
    return top.view(-1)[wrong.nonzero()[0]].item() if wrong.any() else 0.0


class DiffStats:
    """Accumulates ppl over the model's own logits, and (if a reference store is given) KLD vs
    the reference, per-token, bucketed by the reference's top-token probability."""

    def __init__(self, ids: torch.Tensor, ranges: list, vocab_size: int, ref_store: str | None,
                 per_row: bool = False):
        self.ids = ids
        # Per row: the scored tokens' total log-prob and whether each is the argmax, for this model
        # and (with a reference store) the reference -- for traces whose rows are short known answers
        # (recall_trace.py), where "does the quant still produce the reference's answer" is the metric
        self.per_row = per_row
        self.rows = {}
        self.ranges = ranges          # per row: (a, b) - score logits positions [a, b)
        self.vocab_size = vocab_size
        self.ref_store = ref_store
        self.logprob_sum = 0.0
        self.logprob_count = 0
        self.total_count = 0
        self.kl_toks = []
        self.conf_toks = []

    def __call__(self, r: int, logits: torch.Tensor, offset: int = 0):
        # logits cover positions [offset, offset + len): the whole row (offset 0), or just the
        # score range when the engine ran its head on that range only
        a, b = self.ranges[r]
        logits = logits[:, a - offset:b - offset, :].float()
        logits.clamp_(min = -200.0)

        # ppl on own logits. Non-finite positions (a model NaN-ing on some input, e.g. fp16
        # overflow in a backend) are excluded and counted rather than poisoning the aggregate
        targets = self.ids[r, a + 1:b].view(1, -1).to(logits.device)
        lp = compute_target_log_probs(logits[:, :-1, :], targets, min(self.vocab_size, logits.shape[-1]))
        finite = torch.isfinite(lp)
        self.logprob_sum += lp[finite].sum().item()
        self.logprob_count += finite.sum().item()
        self.total_count += targets.numel()
        if self.per_row:
            vs = min(self.vocab_size, logits.shape[-1])
            hit = bool((logits[:, :-1, :vs].argmax(dim = -1) == targets).all())
            self.rows[r] = {"lp": lp.sum().item(), "hit": hit, "wrong_p": first_wrong_p(logits[:, :-1, :], targets, vs)}

        # kld vs reference
        if self.ref_store:
            ref = load_tensor(os.path.join(self.ref_store, f"row_{r:06d}.safetensors"), "logits")
            ref = ref.to(logits.device).float()
            vs = min(self.vocab_size, logits.shape[-1], ref.shape[-1])
            kl = compute_kl_div(logits.squeeze(0), ref.squeeze(0), vs)
            self.kl_toks.append(kl.flatten().float().cpu())
            if self.per_row:
                ref_lp = compute_target_log_probs(ref[:, :-1, :], targets, vs)
                self.rows[r]["ref_lp"] = ref_lp.sum().item()
                self.rows[r]["ref_hit"] = bool((ref[:, :-1, :vs].argmax(dim = -1) == targets).all())
                self.rows[r]["ref_wrong_p"] = first_wrong_p(ref[:, :-1, :], targets, vs)
            del ref

    def load_conf(self):
        conf = load_tensor(os.path.join(self.ref_store, "conf.safetensors"), "conf")
        self.conf_toks = [conf.flatten().float()]

    def kl_vector(self) -> torch.Tensor | None:
        """Per-token KLD over all rows in test order (non-finite values included), for the
        (KLD - noise floor) histogram, which pairs tokens across passes"""
        return torch.cat(self.kl_toks) if self.kl_toks else None

    def results(self) -> dict:
        res = {}
        if self.total_count:
            nf_share = 1.0 - self.logprob_count / self.total_count
            if nf_share > 0.0:
                res["nonfinite_share"] = nf_share
        if self.logprob_count:
            res["ppl"] = math.exp(-self.logprob_sum / self.logprob_count)
        if self.per_row and self.rows:
            res["rows"] = [self.rows.get(r) for r in range(len(self.ranges))]
            got = [x for x in res["rows"] if x and "ref_hit" in x]
            known = [x for x in got if x["ref_hit"]]
            if known:
                # Over the rows the reference answers (argmax all the way): how many the model still does
                res["recall_retained"] = sum(x["hit"] for x in known) / len(known)
                res["recall_dlp"] = sum(x["lp"] - x["ref_lp"] for x in known) / len(known)
                res["recall_known_rows"] = len(known)
                # Damage is nothing like even across how firmly the reference held the answer, and no one
                # figure keeps that: on MiniCPM5-2B a 3 bpw quant keeps 98% at the top band and 28% at the bottom
                res["recall_bands"] = {}
                for name, lo, hi in RECALL_BANDS:
                    band = [x for x in known if lo <= math.exp(x["ref_lp"]) < hi]
                    if band:
                        res["recall_bands"][name] = {"rows": len(band), "retained": sum(x["hit"] for x in band) / len(band),
                                                     "kept": recall_kept(band)}
                res["recall_kept"] = recall_kept(known)
                res["recall_confident_errors"] = sum(x["wrong_p"] >= CONFIDENT_P for x in known) / len(known)
        if self.kl_toks:
            self.load_conf()
            kl = torch.cat(self.kl_toks)
            conf = torch.cat(self.conf_toks)
            assert kl.numel() == conf.numel()
            finite = torch.isfinite(kl)
            if finite.any():
                klf = kl[finite]
                res.update({
                    "kld": klf.mean().item(),
                    "kld_median": klf.median().item(),
                    "kld_p10": klf.quantile(0.1).item(),
                    "kld_p25": klf.quantile(0.25).item(),
                    "kld_p75": klf.quantile(0.75).item(),
                    "kld_p90": klf.quantile(0.9).item(),
                    "kld_buckets": [
                        {
                            "conf_range": [lo, min(hi, 1.0)],
                            "share": ((conf >= lo) & (conf < hi)).float().mean().item(),
                            "mean": kl[finite & (conf >= lo) & (conf < hi)].mean().item()
                                if (finite & (conf >= lo) & (conf < hi)).any() else None,
                            "median": kl[finite & (conf >= lo) & (conf < hi)].median().item()
                                if (finite & (conf >= lo) & (conf < hi)).any() else None,
                        }
                        for lo, hi in CONF_BUCKETS
                    ],
                })
        return res


def save_reference_row(store_dir: str, r: int, logits: torch.Tensor, rng: tuple, conf_rows: list,
                       offset: int = 0):
    logits = logits[:, rng[0] - offset:rng[1] - offset, :].float()
    logits.clamp_(min = -200.0)
    # reference top-token probability per position, for the confidence buckets
    l2 = logits.squeeze(0)
    confs = []
    for a in range(0, l2.shape[0], 256):
        c = l2[a:a + 256]
        confs.append((c.max(dim = -1).values - c.logsumexp(dim = -1)).exp().cpu())
    conf_rows.append(torch.cat(confs))
    save_tensors(
        os.path.join(store_dir, f"row_{r:06d}.safetensors"),
        {"logits": logits.half().cpu()},
    )


def print_stats(label: str, res: dict):
    if "ppl" not in res:
        print(f" -- {label:24}   INVALID (all logits non-finite; model broken under this backend)")
        return
    line = f" -- {label:24}   ppl {res['ppl']:9.4f}"
    if "kld" in res:
        line += f"   kld {res['kld']:.6f}   median {res['kld_median']:.6f}   p90 {res['kld_p90']:.6f}"
    if res.get("nonfinite_share"):
        line += f"   !! {100 * res['nonfinite_share']:.1f}% non-finite tokens excluded"
    if "recall_retained" in res:
        line += f"   recall {100 * res['recall_retained']:.1f}% of {res['recall_known_rows']} known, dlp {res['recall_dlp']:+.3f}"
    if "recall_kept" in res:      # a noise floor cached before these fields has only the two above
        line += f"   kept {100 * res['recall_kept']:.1f}%, confident errors {100 * res['recall_confident_errors']:.2f}%"
        line += "   bands " + " ".join(f"{k} {100 * v['kept']:.1f}/{v['rows']}" for k, v in res["recall_bands"].items())
    print(line)
    # Storage accounting has been silently wrong three times, always toward a plausible
    # number, so say it out loud rather than leaving it to be noticed in a plot later.
    if res.get("accounting_warning"):
        print(f"      !! storage accounting: {res['accounting_warning']}")
    if "kld_buckets" in res:
        for b in res["kld_buckets"]:
            if b["mean"] is None:
                continue
            lo, hi = b["conf_range"]
            print(
                f"      ref conf [{lo:4.2f}, {hi:4.2f}): {100 * b['share']:5.1f}% of tokens"
                f"   mean {b['mean']:.6f}   median {b['median']:.6f}"
            )
