import sys, os

import argparse
import glob
import json
import random
import zlib

import torch
from safetensors.torch import save_file

from exllamav3 import Generator, Job, model_init
from eval.qbench_prompts import DEFAULT_TEMPLATE_VARS
from sc_trace import CONVERSATIONS

"""
Conversational traces that put real documents where real use puts them: pasted into a user turn
under a short anchoring request, or returned as a tool result, optionally over several tool-call
rounds. The model's own answer is generated.

Eval mode writes one qbench-compatible trace per slice (<out_prefix>_<slice>.json); qbench scores
only the response positions -- in use the model reads a document, it never predicts one:

    python ctx_trace.py -m <model_dir> [model_init options] -o <out_prefix> --docs eval \\
        --code_glob '<repo>/**/*.py' --slices ctx_user,ctx_tool,ctx_ml,loop,self

Calibration mode packs a --cal_data file for convert.py: raw rows from the bundled default mix,
random-token rows, and packed conversational streams, in the proportions given by --shares:

    python ctx_trace.py -m <model_dir> -o <out_prefix> --docs cal --cal_out cal.safetensors \\
        --exclude_self <eval_prefix>_self.json

The trace should come from the unquantized model or a high-bitrate quant (e.g. 6 bpw), as with
sc_trace.py. Document sources are disjoint by purpose: calibration uses the bundled corpus and
Wikipedia row group 0 per language; eval uses openwebtext (end of the dataset), wikitext-2
validation, --code_glob source files and Wikipedia row group 1.
"""

ANCHORS = {
    "web": ["Summarize this.", "What are the main claims in this article?", "Give me the key points.",
            "What is the author arguing here, and is it convincing?", "Pull out any facts or numbers worth remembering."],
    "wiki": ["Summarize this.", "What are the most important facts here?", "Explain this to a curious 12-year-old.",
             "Write three quiz questions based on this text, with answers.", "What would be a good follow-up topic to read about?"],
    "technical": ["Explain this.", "What is this document about, and who is it for?", "Summarize the procedure described here.",
                  "What are the key technical points?", "Is anything here unclear or likely to trip people up?"],
    "code": ["What does this code do?", "Review this code for bugs or risky patterns.", "Explain this file to a new contributor.",
             "How would you write tests for this?", "Suggest one refactor that would make this clearer."],
}

# Anchors for non-English documents: half in the document's language, half asking for English
ML_ANCHORS = {
    "zh": ["请总结一下这篇文章。", "这段文字的要点是什么？", "用简单的话解释一下这段内容。"],
    "ja": ["この文章を要約してください。", "この記事の要点は何ですか？", "この内容をわかりやすく説明してください。"],
    "ko": ["이 글을 요약해 주세요.", "이 글의 핵심 내용은 무엇인가요?", "이 내용을 쉽게 설명해 주세요."],
    "es": ["Resume este texto.", "¿Cuáles son los puntos principales de este artículo?", "Explícame esto de forma sencilla."],
    "fr": ["Résume ce texte.", "Quels sont les points principaux de cet article ?", "Explique-moi cela simplement."],
    "de": ["Fasse diesen Text zusammen.", "Was sind die wichtigsten Punkte dieses Artikels?", "Erklär mir das bitte einfach."],
    "ru": ["Кратко перескажи этот текст.", "Какие основные мысли в этой статье?", "Объясни это простыми словами."],
    "pt": ["Resuma este texto.", "Quais são os pontos principais deste artigo?", "Explique isso de forma simples."],
    "it": ["Riassumi questo testo.", "Quali sono i punti principali di questo articolo?", "Spiegami questo in modo semplice."],
    "ar": ["لخّص هذا النص.", "ما هي النقاط الرئيسية في هذه المقالة؟", "اشرح لي هذا بطريقة بسيطة."],
    "hi": ["इस लेख का सारांश दीजिए।", "इस लेख के मुख्य बिंदु क्या हैं?", "इसे आसान शब्दों में समझाइए।"],
    "tr": ["Bu metni özetle.", "Bu makalenin ana noktaları neler?", "Bunu basitçe açıkla."],
    "vi": ["Hãy tóm tắt đoạn văn này.", "Những ý chính của bài viết này là gì?", "Giải thích điều này một cách đơn giản."],
}
ML_ANCHORS_EN = ["Summarize this in English.", "What is this article about? Answer in English.",
                 "Translate the first paragraph into English and summarize the rest."]
ML_LANGS = list(ML_ANCHORS)

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file from the workspace.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "fetch_url", "description": "Fetch a web page and return its text.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "search", "description": "Search an encyclopedia and return the best matching article.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]
TOOL_FOR = {"code": "read_file", "web": "fetch_url", "technical": "fetch_url", "wiki": "search", "ml": "search"}

TOOL_TASKS = {
    "code": ["Take a look at {ref} and tell me what it does.", "Is there anything wrong with {ref}?"],
    "web": ["Read {ref} and give me the gist.", "Can you check {ref} and tell me whether it's worth reading?"],
    "technical": ["Read {ref} and explain what it covers.", "Check {ref} and tell me the important parts."],
    "wiki": ["Look up {ref} and give me a short overview.", "Find out about {ref} and tell me the key facts."],
    "ml": ["Look up {ref} and give me a short overview.", "Find out about {ref} and tell me the key facts."],
}

LOOP_TASKS = [
    "I'm trying to get up to speed on a few things at once. Gather what you need and then give me a combined summary.",
    "Research this for me using your tools, then write up what you found in a few paragraphs.",
    "Check these sources one at a time and tell me how they relate to each other.",
]

CAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exllamav3", "conversion", "standard_cal_data")


# --- document sources ---------------------------------------------------------------------------

def detok_wikitext(t):
    # wikitext-2 raw keeps its tokenizer's spacing ("multi @-@ coloured", "Hu 's", "1 @,@ 000")
    for a, b in ((" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", "."), (" 's", "'s"), (" n't", "n't"),
                 (" ,", ","), (" .", "."), (" ;", ";"), (" :", ":"), ("( ", "("), (" )", ")")):
        t = t.replace(a, b)
    return t

def chunk_tokens(tokenizer, text, rng, lo, hi, code = False):
    # Pick a window of roughly lo..hi tokens, starting at a line (code) or sentence (prose) boundary
    ids = tokenizer.encode(text)[0]
    if ids.numel() < lo + 64:
        return None
    n = rng.randint(lo, min(hi, ids.numel() - 64))
    a = rng.randint(0, ids.numel() - n - 64)
    t = tokenizer.decode(ids[a:a + n + 64].unsqueeze(0))[0]
    if a > 0:
        seps = ("\n",) if code else (". ", "。", "\n")
        cands = [i for i in (t.find(s) for s in seps) if i >= 0]
        cut = min(cands) if cands else -1
        if cut < 0 or cut > len(t) // 4:
            return None
        t = t[cut + 1:] if code else t[cut + 1:].lstrip()
    end = t.rfind("\n") if code else max(t.rfind(". "), t.rfind("。"), t.rfind("\n"))
    return t[:end + 1].rstrip() if end > len(t) // 2 else t.rstrip()

def wiki_rowgroup(lang, rg, min_chars = 3000):
    # One parquet row group of a Wikipedia language edition, read straight from the Hub
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    path = f"datasets/wikimedia/wikipedia/20231101.{lang}/train-00000-of-" + \
           sorted(fs.ls(f"datasets/wikimedia/wikipedia/20231101.{lang}", detail = False))[0].rsplit("-of-", 1)[1]
    with fs.open(path) as f:
        t = pq.ParquetFile(f).read_row_group(rg, columns = ["title", "text"]).to_pylist()
    return [(r["title"], r["text"]) for r in t if len(r["text"]) >= min_chars]

def load_pool(kind, purpose, code_glob = None, lang = None):
    """(texts, refs) for one document kind and purpose."""
    from datasets import load_dataset
    if kind == "ml":
        arts = wiki_rowgroup(lang, 0 if purpose == "cal" else 1)
        return [a[1] for a in arts], [a[0] for a in arts]
    if purpose == "eval":
        if kind == "web":
            ds = load_dataset("parquet", split = "train",
                data_files = "hf://datasets/stas/openwebtext-10k@refs/convert/parquet/plain_text/train/*.parquet")
            pool = [t for t in ds["text"][-400:] if len(t) > 4000]      # qbench's rows come from the start
            return pool, [f"https://example.com/article/{i}" for i in range(len(pool))]
        if kind == "wiki":
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split = "validation")
            pool, cur = [], []
            for line in ds["text"]:
                if line.startswith(" = ") and not line.startswith(" = = ") and cur:
                    pool.append("".join(cur)); cur = []
                cur.append(line)
            pool = [detok_wikitext(p) for p in pool if len(p) > 4000]
            return pool, [p.strip().split("\n")[0].strip(" =") for p in pool]
        if kind == "code":
            assert code_glob, "--code_glob is required for eval code documents"
            files = [f for f in sorted(glob.glob(code_glob, recursive = True)) if 6000 < os.path.getsize(f) < 60000]
            root = os.path.commonpath(files) if files else ""
            return [open(f).read() for f in files], [os.path.relpath(f, os.path.dirname(root)) for f in files]
        raise ValueError(f"no eval source for kind {kind}")
    # calibration: the bundled corpus, as coherent documents
    if kind == "web":
        pool = [l for l in open(os.path.join(CAL_DIR, "c4.utf8"), encoding = "utf8").read().split("\n") if len(l) > 2500]
        return pool, [f"https://example.org/page/{i}" for i in range(len(pool))]
    if kind == "wiki":
        arts = [a[a.find("\n") + 1:] for a in open(os.path.join(CAL_DIR, "wiki.utf8"), encoding = "utf8").read().split("</doc>\n")]
        arts = [a for a in arts if len(a) > 3000]
        return arts, [a.strip().split("\n")[0][:60] for a in arts]
    if kind in ("code", "technical"):
        raw = open(os.path.join(CAL_DIR, f"{kind}.utf8"), encoding = "utf8").read()
        pool = [raw[i:i + 24000] for i in range(0, len(raw) - 24000, 24000)]
        name = "src/module_{}.py" if kind == "code" else "https://docs.example.org/guide/{}"
        return pool, [name.format(i) for i in range(len(pool))]
    raise ValueError(f"no cal source for kind {kind}")

def load_docs(kind, purpose, n, tokenizer, rng, lo, hi, code_glob = None):
    docs = []
    langs = ML_LANGS if kind == "ml" else [None]
    for li, lang in enumerate(langs):
        want = n // len(langs) + (1 if li < n % len(langs) else 0)
        if want == 0:
            continue
        pool, refs = load_pool(kind, purpose, code_glob, lang)
        idx = list(range(len(pool)))
        rng.shuffle(idx)
        got = 0
        for i in idx:
            c = chunk_tokens(tokenizer, pool[i], rng, lo, hi, code = kind == "code")
            if c:
                docs.append({"kind": kind, "lang": lang or "en", "text": c, "ref": refs[i]})
                got += 1
            if got >= want:
                break
    rng.shuffle(docs)
    return docs


# --- conversation builders: return (messages, tools) ready for generation ------------------------

def anchor_for(doc, rng):
    if doc["kind"] == "ml":
        return rng.choice(ML_ANCHORS[doc["lang"]]) if rng.random() < 0.5 else rng.choice(ML_ANCHORS_EN)
    return rng.choice(ANCHORS[doc["kind"]])

def conv_ctx_user(doc, rng):
    anchor = anchor_for(doc, rng)
    content = f"{anchor}\n\n{doc['text']}" if rng.random() < 0.6 else f"{doc['text']}\n\n{anchor}"
    return [{"role": "user", "content": content}], None

def tool_call(name, arg_value, call_id):
    key = {"read_file": "path", "fetch_url": "url", "search": "query"}[name]
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": {key: arg_value}}}]}

def conv_ctx_tool(doc, rng):
    task = rng.choice(TOOL_TASKS[doc["kind"]]).format(ref = doc["ref"])
    if doc["kind"] == "ml" and rng.random() < 0.5:
        task = f"{rng.choice(ML_ANCHORS[doc['lang']])} ({doc['ref']})"
    return [{"role": "user", "content": task},
            tool_call(TOOL_FOR[doc["kind"]], doc["ref"], "call_0"),
            {"role": "tool", "tool_call_id": "call_0", "content": doc["text"]}], TOOLS

def conv_loop(docs, rng):
    msgs = [{"role": "user", "content": rng.choice(LOOP_TASKS) + "\n\n" + "\n".join(f"- {d['ref']}" for d in docs)}]
    for k, d in enumerate(docs):
        msgs += [tool_call(TOOL_FOR[d["kind"]], d["ref"], f"call_{k}"),
                 {"role": "tool", "tool_call_id": f"call_{k}", "content": d["text"]}]
    return msgs, TOOLS


# --- generation -------------------------------------------------------------------------------

def probe_template_vars(tokenizer, template_vars):
    probe = [{"role": "user", "content": "hi"}]
    def works(v):
        try:
            tokenizer.hf_chat_template(probe, add_generation_prompt = True, **v); return True
        except Exception:
            return False
    while template_vars and not works(template_vars):
        for k in list(template_vars):
            if works({kk: v for kk, v in template_vars.items() if kk != k}):
                print(f" !! Chat template rejects {k} = {template_vars[k]!r}; using template default")
                del template_vars[k]; break
        else:
            return {}
    return template_vars

def generate(generator, config, tokenizer, convs, template_vars, max_new_tokens, seed):
    """convs: list of (meta, messages, tools). Returns trace rows."""
    pending = {}
    for i, (meta, msgs, tools) in enumerate(convs):
        kw = dict(template_vars, **({"tools": tools} if tools else {}))
        input_ids = tokenizer.hf_chat_template(msgs, add_generation_prompt = True, **kw)
        generator.enqueue(Job(input_ids = input_ids, max_new_tokens = max_new_tokens,
                              stop_conditions = config.eos_token_id_list, decode_special_tokens = True,
                              identifier = i, seed = zlib.crc32(f"{seed}|{i}".encode())))
        pending[i] = {"meta": meta, "input_ids": input_ids, "chunks": [], "eos_reason": None}
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if r["stage"] != "streaming":
                continue
            p = pending[r["identifier"]]
            if "token_ids" in r:
                p["chunks"].append(r["token_ids"])
            if r["eos"]:
                p["eos_reason"] = r.get("eos_reason")
                n = sum(c.shape[-1] for c in p["chunks"])
                print(f"    {r['identifier']:3} {p['meta']['slice']:9s} {p['meta'].get('lang', ''):3s} "
                      f"ctx {p['input_ids'].shape[-1]:5}  resp {n:5} ({p['eos_reason']})", flush = True)
    rows = []
    for i, p in pending.items():
        if not p["chunks"]:
            continue
        rows.append({**p["meta"], "eos_reason": p["eos_reason"],
                     "input_ids": p["input_ids"][0].tolist(),
                     "response_ids": torch.cat(p["chunks"], dim = -1)[0].tolist()})
    return rows


def build_convs(sl, n, docs, loop_docs, rng, self_pool):
    convs = []
    if sl in ("ctx_user", "ctx_tool", "ctx_ml"):
        pool = docs.pop(sl)
        for d in pool[:n]:
            builder = conv_ctx_user if (sl == "ctx_user" or (sl == "ctx_ml" and rng.random() < 0.5)) else conv_ctx_tool
            convs.append(({"slice": sl, "kind": d["kind"], "lang": d["lang"], "ref": d["ref"]}, *builder(d, rng)))
    elif sl == "loop":
        pool = list(loop_docs)
        while len(convs) < n and len(pool) >= 3:
            take = [pool.pop() for _ in range(rng.randint(2, 3))]
            convs.append(({"slice": sl, "kind": "+".join(d["kind"] for d in take), "lang": "+".join(d["lang"] for d in take),
                           "ref": [d["ref"] for d in take]}, *conv_loop(take, rng)))
    elif sl == "self":
        for c in rng.sample(self_pool, min(n, len(self_pool))):
            convs.append(({"slice": sl, "kind": "self", "conversation": c},
                          [{"role": "user", "content": CONVERSATIONS[c][0]}], None))
    return convs


@torch.inference_mode()
def main(args):
    model, config, cache, tokenizer = model_init.init(args)[:4]
    generator = Generator(model = model, cache = cache, tokenizer = tokenizer, max_chunk_size = 2048)
    tv = probe_template_vars(tokenizer, dict(DEFAULT_TEMPLATE_VARS, **args.template_vars))
    rng = random.Random(args.seed)
    purpose = args.docs
    lo, hi = args.doc_min, args.doc_max
    self_pool = list(range(len(CONVERSATIONS)))
    if args.exclude_self:
        used = {r["conversation"] for r in json.load(open(args.exclude_self))["rows"]}
        self_pool = [c for c in self_pool if c not in used]
        print(f" -- excluding {len(used)} own-voice prompts used by {args.exclude_self}")

    if purpose == "eval":
        slices = args.slices.split(",")
        n = args.n
        prose = ["web", "wiki", "code"]
        per = -(-n // 3)
        docs = {}
        if "ctx_user" in slices or "ctx_tool" in slices:
            pools = {k: load_docs(k, purpose, per * 2, tokenizer, rng, lo, hi, args.code_glob) for k in prose}
            docs["ctx_user"] = [d for k in prose for d in pools[k][:per]]
            docs["ctx_tool"] = [d for k in prose for d in pools[k][per:2 * per]]
        if "ctx_ml" in slices:
            docs["ctx_ml"] = load_docs("ml", purpose, n, tokenizer, rng, lo, hi)
        loop_docs = []
        if "loop" in slices:
            loop_docs = [d for k in prose for d in load_docs(k, purpose, per * 3, tokenizer, rng, lo // 2, hi // 2, args.code_glob)]
            rng.shuffle(loop_docs)
        for sl in slices:
            convs = build_convs(sl, n, docs, loop_docs, rng, self_pool)
            print(f" -- slice {sl}: {len(convs)} conversations", flush = True)
            rows = generate(generator, config, tokenizer, convs, tv, args.max_new_tokens, f"{args.seed}|{sl}")
            write_trace(f"{args.output}_{sl}.json", args, tokenizer, tv, rows, {"slice": sl, "docs": purpose})
        return

    # --- calibration: packed mix ---
    shares = dict(args.shares)
    assert abs(sum(shares.values()) - 1.0) < 1e-6, f"--shares must sum to 1, got {sum(shares.values())}"
    R, C = args.cal_rows, args.cal_cols
    rows_for = {k: int(round(v * R)) for k, v in shares.items()}
    rows_for["raw"] += R - sum(rows_for.values())                      # rounding remainder
    tok_budget = {k: rows_for[k] * C for k in ("ctx", "loop", "self")}
    print(f" -- calibration rows per slice: {rows_for}")

    # Documents: English prose/code kinds plus multilingual Wikipedia, ml_frac of the document slices
    kinds = ["web", "wiki", "technical", "code"]
    avg_ctx, avg_loop = 1300 + 450, 2 * 750 + 500                        # rough tokens per conversation
    n_ctx = int(tok_budget["ctx"] / avg_ctx * 1.25) + 1
    n_loop = int(tok_budget["loop"] / avg_loop * 1.25) + 1
    n_ml = int(round(args.ml_frac * n_ctx))
    ctx_docs = [d for k in kinds for d in load_docs(k, purpose, -(-(n_ctx - n_ml) // len(kinds)), tokenizer, rng, lo, hi)]
    ctx_docs += load_docs("ml", purpose, n_ml, tokenizer, rng, lo, hi)
    rng.shuffle(ctx_docs)
    n_loop_docs = n_loop * 3
    n_loop_ml = int(round(args.ml_frac * n_loop_docs))
    loop_docs = [d for k in kinds for d in load_docs(k, purpose, -(-(n_loop_docs - n_loop_ml) // len(kinds)), tokenizer, rng, lo // 2, hi // 2)]
    loop_docs += load_docs("ml", purpose, n_loop_ml, tokenizer, rng, lo // 2, hi // 2)
    rng.shuffle(loop_docs)

    convs = []
    for i, d in enumerate(ctx_docs):
        sl = "ctx_user" if i % 2 == 0 else "ctx_tool"
        b = conv_ctx_user if sl == "ctx_user" else conv_ctx_tool
        convs.append(({"slice": sl, "kind": d["kind"], "lang": d["lang"], "ref": d["ref"]}, *b(d, rng)))
    convs += build_convs("loop", n_loop, {}, loop_docs, rng, self_pool)
    n_self = int(tok_budget["self"] / 1400 * 1.25) + 1
    convs += build_convs("self", n_self, {}, [], rng, self_pool)
    print(f" -- generating {len(convs)} conversations", flush = True)
    rows = generate(generator, config, tokenizer, convs, tv, args.max_new_tokens, f"{args.seed}|cal")
    write_trace(f"{args.output}_cal.json", args, tokenizer, tv, rows, {"slice": "cal", "docs": purpose})

    # Pack: each conversational slice's streams into fixed-width rows, up to its row budget
    from exllamav3.conversion.calibration_data import get_default_calibration
    group = lambda r: "ctx" if r["slice"].startswith("ctx") else r["slice"]
    packed, comp = [], {}
    for g in ("ctx", "loop", "self"):
        rs = [r for r in rows if group(r) == g]
        rng.shuffle(rs)
        stream, ml_tok = [], 0
        for r in rs:
            seq = r["input_ids"] + r["response_ids"]
            stream += seq
            if r.get("lang", "en") not in ("en", "") and set(r["lang"].split("+")) != {"en"}:
                ml_tok += len(seq)
            if len(stream) >= rows_for[g] * C:
                break
        n_rows = min(rows_for[g], len(stream) // C)
        if n_rows < rows_for[g]:
            print(f" !! slice {g}: only {n_rows} of {rows_for[g]} rows of material; raw rows fill the rest")
            rows_for["raw"] += rows_for[g] - n_rows
        packed += [torch.tensor(stream[i * C:(i + 1) * C], dtype = torch.long).unsqueeze(0) for i in range(n_rows)]
        comp[g] = {"rows": n_rows, "ml_token_frac": round(ml_tok / max(1, len(stream)), 3)}
    default = get_default_calibration({"cal_rows": 250, "cal_cols": C}, tokenizer)
    n_default_random = 250 - sum(max(1, int(w / 135 * 250)) for w in (20, 20, 10, 10, 50, 5))
    text_rows, random_rows = default[:-n_default_random], default[-n_default_random:]
    step = len(text_rows) / max(1, rows_for["raw"])
    packed += [text_rows[int(i * step)] for i in range(rows_for["raw"])]      # every source, in proportion
    packed += random_rows[:rows_for["random"]]
    comp["raw"] = {"rows": rows_for["raw"]}; comp["random"] = {"rows": rows_for["random"]}
    order = list(range(len(packed)))
    rng.shuffle(order)
    out = torch.cat([packed[i] for i in order], dim = 0)
    assert out.shape == (R, C), out.shape
    save_file({"input_ids": out.contiguous()}, args.cal_out)
    json.dump({"shares": shares, "ml_frac": args.ml_frac, "composition": comp, "source_trace": f"{args.output}_cal.json"},
              open(os.path.splitext(args.cal_out)[0] + ".json", "w"), indent = 1)
    print(f" -- {R} x {C} calibration rows -> {args.cal_out}; composition {comp}")


def write_trace(path, args, tokenizer, tv, rows, meta):
    out = {"model": args.model_dir, "vocab_size": tokenizer.actual_vocab_size, "template_vars": tv,
           "meta": {**meta, "rows": len(rows),
                    "input_tokens": sum(len(r["input_ids"]) for r in rows),
                    "output_tokens": sum(len(r["response_ids"]) for r in rows)},
           "rows": rows}
    with open(path, "w") as f:
        json.dump(out, f)
    print(f" -- {path}: {out['meta']}", flush = True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev = False)
    model_init.add_args(parser, default_cache_size = 65536)
    parser.add_argument("-o", "--output", type = str, required = True, help = "Output prefix; writes <prefix>_<slice>.json")
    parser.add_argument("--docs", type = str, default = "eval", choices = ["eval", "cal"], help = "Document sources: held-out eval text, or the calibration corpus")
    parser.add_argument("--slices", type = str, default = "ctx_user,ctx_tool,ctx_ml,loop,self", help = "(eval) slices to write")
    parser.add_argument("--code_glob", type = str, default = None, help = "(eval) glob of source files for code documents")
    parser.add_argument("-n", type = int, default = 30, help = "(eval) conversations per slice")
    parser.add_argument("--cal_out", type = str, default = None, help = "(cal) packed calibration file for convert.py --cal_data")
    parser.add_argument("--cal_rows", type = int, default = 250)
    parser.add_argument("--cal_cols", type = int, default = 2048)
    parser.add_argument("--shares", type = json.loads, default = {"raw": 0.25, "ctx": 0.35, "loop": 0.25, "self": 0.10, "random": 0.05},
                        help = "(cal) row shares per slice, JSON")
    parser.add_argument("--ml_frac", type = float, default = 0.12, help = "(cal) fraction of documents drawn from non-English Wikipedia")
    parser.add_argument("--exclude_self", type = str, default = None, help = "Eval self-slice trace whose own-voice prompts must not be reused")
    parser.add_argument("--doc_min", type = int, default = 500, help = "Min document tokens")
    parser.add_argument("--doc_max", type = int, default = 1500, help = "Max document tokens")
    parser.add_argument("--max_new_tokens", type = int, default = 1536)
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("-tv", "--template_vars", type = json.loads, default = {})
    args = parser.parse_args()
    assert args.docs != "cal" or args.cal_out, "--docs cal needs --cal_out"
    main(args)
