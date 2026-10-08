import sys, os

import argparse
import glob
import json
import random
import zlib

import torch
from safetensors.torch import save_file

from exllamav3 import Config, Generator, Job, Tokenizer, model_init
from exllamav3.generator.sampler import ComboSampler
from eval.qbench_prompts import DEFAULT_TEMPLATE_VARS
from sc_trace import CONVERSATIONS

"""
Conversational traces that put real documents where real use puts them: pasted into a user turn
under a short anchoring request, or returned as a tool result, optionally over several tool-call
rounds. The model's own answer is generated.

Eval mode writes one qbench-compatible trace per slice (<out_prefix>_<slice>.json); qbench scores
only the response positions -- in use the model reads a document, it never predicts one:

    python ctx_trace.py -m <model_dir> [model_init options] -o <out_prefix> --docs eval \\
        --code_glob '<repo>/**/*.py' --slices ctx_user,ctx_tool,ctx_ml,loop,self,wild,swe

Eval conversations use a scaffolding pool disjoint from calibration's (anchors, tool names and
schemas, task wording) and vary the thinking settings per conversation; `wild` takes real first
user turns from WildChat-1M, stratified by language and length, and `swe` real agent sessions
from Open-SWE-Traces cut at a turn the model regenerates; both are eval-only.

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
    "diff": ["Review this patch.", "What do these changes do?", "Is there anything wrong with this diff?",
             "Write a changelog entry for these commits.", "Which of these changes is riskiest, and why?"],
    "log": ["Why is this build failing?", "Summarize the test failures here.", "What's the root cause of this error?",
            "Is anything in this log an actual problem, or is it noise?", "What should I fix first?"],
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
TOOL_FOR = {"code": "read_file", "diff": "read_file", "log": "read_file", "web": "fetch_url", "technical": "fetch_url", "wiki": "search", "ml": "search"}

TOOL_TASKS = {
    "code": ["Take a look at {ref} and tell me what it does.", "Is there anything wrong with {ref}?"],
    "diff": ["Review the changes in {ref}.", "Read {ref} and tell me what these commits change."],
    "log": ["Check {ref} and tell me why CI failed.", "Read {ref} and summarize what went wrong."],
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

# --- scaffolding pools -------------------------------------------------------------------------
# The constants above are the calibration pool. Eval traces use a disjoint pool below -- other
# phrasings, other tool names, argument names and descriptions, other task wording -- so an eval
# gain cannot come from fitting calibration's scaffolding. use_pool() rebinds the module names.

CAL_POOL = dict(ANCHORS = ANCHORS, ML_ANCHORS = ML_ANCHORS, ML_ANCHORS_EN = ML_ANCHORS_EN, TOOLS = TOOLS,
                TOOL_FOR = TOOL_FOR, TOOL_TASKS = TOOL_TASKS, LOOP_TASKS = LOOP_TASKS,
                TOOL_ARG = {"read_file": "path", "fetch_url": "url", "search": "query"})

EVAL_POOL = dict(
    ANCHORS = {
        "web": ["tl;dr?", "Can you break down what this piece is saying?", "Is this accurate? What's the gist?",
                "I don't have time to read this - what do I need to know?", "What's the angle of this article?"],
        "wiki": ["Give me the short version of this.", "What's notable here?", "Turn this into a few bullet points.",
                 "What would someone find surprising in this?", "Explain the background to this."],
        "technical": ["What does this say, in plain terms?", "Walk me through this.", "What should I take away from this?",
                      "Which part of this matters most in practice?", "Rewrite this as short instructions."],
        "code": ["Can you explain this code?", "Any problems with this?", "What would you change here?",
                 "What is the purpose of this module?", "Where would a bug most likely hide in this?"],
        "diff": ["Can you sanity-check this change?", "Summarize what changed here.", "Would you merge this? Why or why not?",
                 "What did these commits fix?", "Anything in this diff that looks like a regression?"],
        "log": ["My build broke - what happened?", "Which of these failures matter?", "Explain this error output to me.",
                "Is this flaky or a real bug?", "Where would you start debugging from this?"],
    },
    ML_ANCHORS = {
        "zh": ["这篇讲了什么？", "帮我概括一下重点。"], "ja": ["これは何について書かれていますか？", "ポイントを箇条書きにしてください。"],
        "ko": ["이 글은 무엇에 관한 건가요?", "핵심만 짧게 정리해 주세요."], "es": ["¿De qué trata esto?", "Hazme un resumen breve."],
        "fr": ["De quoi parle ce texte ?", "Fais-moi un résumé rapide."], "de": ["Worum geht es hier?", "Gib mir eine kurze Zusammenfassung."],
        "ru": ["О чём этот текст?", "Выдели главное в нескольких пунктах."], "pt": ["Sobre o que é este texto?", "Me dê um resumo rápido."],
        "it": ["Di cosa parla questo testo?", "Fammi un breve riassunto."], "ar": ["عمّ يتحدث هذا النص؟", "أعطني ملخصًا سريعًا."],
        "hi": ["यह लेख किस बारे में है?", "मुख्य बातें संक्षेप में बताइए।"], "tr": ["Bu yazı ne hakkında?", "Kısaca özetler misin?"],
        "vi": ["Bài này nói về điều gì?", "Tóm tắt nhanh giúp tôi."],
    },
    ML_ANCHORS_EN = ["Give me an English summary of this.", "I can't read this language - what does it say?"],
    TOOLS = [
        {"type": "function", "function": {"name": "open_document", "description": "Open a document from the user's files and return its contents.",
            "parameters": {"type": "object", "properties": {"filename": {"type": "string", "description": "Name of the file"}}, "required": ["filename"]}}},
        {"type": "function", "function": {"name": "browse", "description": "Load a URL in a headless browser and return the page text.",
            "parameters": {"type": "object", "properties": {"link": {"type": "string", "description": "Address to load"}}, "required": ["link"]}}},
        {"type": "function", "function": {"name": "kb_lookup", "description": "Look up a topic in the knowledge base.",
            "parameters": {"type": "object", "properties": {"topic": {"type": "string", "description": "Topic name"}}, "required": ["topic"]}}},
    ],
    TOOL_FOR = {"code": "open_document", "diff": "open_document", "log": "open_document", "web": "browse", "technical": "browse", "wiki": "kb_lookup", "ml": "kb_lookup"},
    TOOL_TASKS = {
        "code": ["Open {ref} - what's going on in there?", "I think {ref} has an issue, can you check?"],
        "diff": ["Open {ref} and tell me if these changes look right.", "What does the patch in {ref} do?"],
        "log": ["Open {ref} - why did this run fail?", "What's in {ref}? Anything I need to act on?"],
        "web": ["What's on {ref}?", "Pull up {ref} and summarize."],
        "technical": ["Pull up {ref} and walk me through it.", "What does {ref} cover?"],
        "wiki": ["What do we know about {ref}?", "Background on {ref}, please."],
        "ml": ["What do we know about {ref}?", "Background on {ref}, please."],
    },
    LOOP_TASKS = [
        "Go through these and tell me what I should know from each.",
        "Pull these up and write me a short briefing that covers all of them.",
        "Look at each of these, then tell me which one matters most and why.",
    ],
    TOOL_ARG = {"open_document": "filename", "browse": "link", "kb_lookup": "topic"},
)
TOOL_ARG = CAL_POOL["TOOL_ARG"]

# Eval conversations vary the template settings calibration fixes (thinking medium)
EVAL_TEMPLATE_VARIANTS = [{"enable_thinking": True, "reasoning_effort": "low"},
                          {"enable_thinking": True, "reasoning_effort": "medium"},
                          {"enable_thinking": True, "reasoning_effort": "xhigh"},
                          {"enable_thinking": False}]

def use_pool(purpose):
    global ANCHORS, ML_ANCHORS, ML_ANCHORS_EN, TOOLS, TOOL_FOR, TOOL_TASKS, LOOP_TASKS, TOOL_ARG
    P = EVAL_POOL if purpose == "eval" else CAL_POOL
    ANCHORS, ML_ANCHORS, ML_ANCHORS_EN, TOOLS = P["ANCHORS"], P["ML_ANCHORS"], P["ML_ANCHORS_EN"], P["TOOLS"]
    TOOL_FOR, TOOL_TASKS, LOOP_TASKS, TOOL_ARG = P["TOOL_FOR"], P["TOOL_TASKS"], P["LOOP_TASKS"], P["TOOL_ARG"]


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

# diff documents: real commits from CommitPackFT (MIT; each sample from a permissively licensed
# repository, kept only where its license field says so), rendered as `git log -p` shows them,
# a few commits of one language per document. Calibration and eval take disjoint repositories.
DIFF_LANGS = ["python", "javascript", "typescript", "c", "c++", "go", "rust", "java", "shell", "c#", "ruby", "php",
              "kotlin", "swift", "scala", "lua", "haskell", "r", "yaml", "json", "markdown", "html", "css", "sql",
              "makefile", "cmake", "dockerfile", "toml", "xml", "tex"]
DIFF_LICENSES = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc", "cc0-1.0", "unlicense", "mpl-2.0",
                 "epl-1.0", "artistic-2.0"}
DIFF_HEAD = 400                     # commits read from the head of each language's file

def render_commit(r):
    import difflib, hashlib
    a, b = r["old_contents"].splitlines(keepends = True), r["new_contents"].splitlines(keepends = True)
    hunks = list(difflib.unified_diff(a, b, f"a/{r['old_file']}", f"b/{r['new_file']}", n = 3))
    if not hunks or max(len(l) for l in hunks) > 400:      # empty, or minified/generated
        return None
    hunks = [l if l.endswith("\n") else l + "\n\\ No newline at end of file\n" for l in hunks]
    idx = lambda t: hashlib.sha1(f"blob {len(t.encode())}\0{t}".encode()).hexdigest()[:7]
    msg = "".join(f"    {l}\n" if l.strip() else "\n" for l in r["message"].rstrip().split("\n"))
    return (f"commit {r['commit']}\n\n{msg}\ndiff --git a/{r['old_file']} b/{r['new_file']}\n"
            f"index {idx(r['old_contents'])}..{idx(r['new_contents'])} 100644\n" + "".join(hunks))

_DIFF_POOLS = {}
def diff_pool(purpose):
    """(texts, refs): per language, runs of commits from one split's repositories, joined as a log."""
    if purpose in _DIFF_POOLS:
        return _DIFF_POOLS[purpose]
    import zlib
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    texts, refs = [], []
    for lang in DIFF_LANGS:
        commits = []
        with fs.open(f"datasets/bigcode/commitpackft/data/{lang}/data.jsonl") as f:
            for _ in range(DIFF_HEAD):
                line = f.readline()
                if not line:
                    break
                r = json.loads(line)
                repo = r["repos"].split(",")[0]
                if r["license"] not in DIFF_LICENSES or (zlib.crc32(repo.encode()) % 5 == 0) != (purpose == "eval"):
                    continue
                c = render_commit(r)
                if c and len(c) < 12000:
                    commits.append(c)
        for i in range(0, len(commits) - 5, 6):
            texts.append("\n".join(commits[i:i + 6])); refs.append(f"patches/{lang.replace('#', 'sharp').replace('+', 'p')}-{i // 6:03d}.patch")
    _DIFF_POOLS[purpose] = (texts, refs)
    return texts, refs

# log documents: build and test transcripts of real projects (quantization/stress/logcorpus.py,
# --log_corpus), clean and with injected faults; the corpus manifest splits projects into cal and
# eval. Long logs are cut into segments at line boundaries, at most LOG_SEGMENTS per project, so a
# few verbose test suites do not dominate; short ones (Go is terse) pool per ecosystem.
LOG_CORPUS = None
LOG_SEGMENTS, LOG_SEGMENT_CHARS = 12, 16000

def log_pool(purpose):
    assert LOG_CORPUS, "log documents need --log_corpus"
    with open(os.path.join(LOG_CORPUS, "manifest.json")) as f:
        man = [p for p in json.load(f) if p["split"] == purpose]
    texts, refs, short = [], [], {}
    for p in man:
        t = open(os.path.join(LOG_CORPUS, f"{p['eco']}__{p['name']}.log"), errors = "replace").read()
        if len(t) < LOG_SEGMENT_CHARS:
            short[p["eco"]] = short.get(p["eco"], "") + t + "\n"; continue
        segs, a = [], 0
        while a < len(t) - LOG_SEGMENT_CHARS // 4:
            b = t.find("\n", a + LOG_SEGMENT_CHARS); b = len(t) if b < 0 else b
            segs.append(t[a:b]); a = b + 1
        step = max(1, len(segs) // LOG_SEGMENTS)
        for k, sg in enumerate(segs[::step][:LOG_SEGMENTS]):
            texts.append(sg); refs.append(f"logs/{p['eco']}-{p['name']}-{k}.log")
    for eco, t in short.items():
        texts.append(t); refs.append(f"logs/{eco}-ci.log")
    return texts, refs

def load_pool(kind, purpose, code_glob = None, lang = None):
    """(texts, refs) for one document kind and purpose."""
    from datasets import load_dataset
    if kind == "diff":
        return diff_pool(purpose)
    if kind == "log":
        return log_pool(purpose)
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
            c = chunk_tokens(tokenizer, pool[i], rng, lo, hi, code = kind in ("code", "diff", "log"))
            if c:
                docs.append({"kind": kind, "lang": lang or "en", "text": c, "ref": refs[i]})
                got += 1
            if got >= want:
                break
    rng.shuffle(docs)
    return docs


WILD_LANG = {"English": "en", "Chinese": "zh", "Russian": "ru", "French": "fr", "Korean": "ko", "Spanish": "es",
             "Italian": "it", "Turkish": "tr", "German": "de", "Japanese": "ja", "Portuguese": "pt", "Arabic": "ar",
             "Hindi": "hi", "Vietnamese": "vi"}

def load_wildchat(n, rng, long_frac = 0.5, shares = None):
    """First user turns from WildChat-1M (ODC-BY; eval only, never calibration), deduplicated,
    non-toxic and unredacted, stratified by language and by length: 'long' turns carry pasted
    material (1500-24000 chars), 'short' ones are ordinary requests (80-1500). Only the user text
    is kept -- no metadata leaves this function."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    shares = shares or {"en": 0.45, "zh": 0.15, "ru": 0.07}
    rest = [l for l in WILD_LANG.values() if l not in shares]
    left = max(0.0, 1.0 - sum(shares.values()))
    shares = dict(shares, **{l: left / len(rest) for l in rest})
    want = {}
    for kind, frac in (("long", long_frac), ("short", 1 - long_frac)):
        for l, sh in shares.items():
            want[(l, kind)] = int(round(n * frac * sh))
    fs = HfFileSystem()
    files = sorted(fs.ls("datasets/allenai/WildChat-1M/data", detail = False), reverse = True)   # end of the dataset
    got, seen = {k: [] for k in want}, set()
    for f in files:
        pf = pq.ParquetFile(fs.open(f))
        for rg in range(pf.num_row_groups):
            for r in pf.read_row_group(rg, columns = ["conversation", "language", "toxic", "redacted"]).to_pylist():
                if r["toxic"] or r["redacted"] or not r["conversation"]:
                    continue
                first = r["conversation"][0]
                if first.get("role") != "user" or first.get("toxic") or first.get("redacted"):
                    continue
                lang = WILD_LANG.get(r["language"])
                text = first["content"] or ""
                kind = "long" if 1500 <= len(text) <= 24000 else "short" if 80 <= len(text) < 1500 else None
                if not lang or not kind:
                    continue
                key = " ".join(text.lower().split())[:160]
                if key in seen:
                    continue
                seen.add(key)
                if len(got[(lang, kind)]) < want[(lang, kind)] * 3:        # oversample, then draw
                    got[(lang, kind)].append(text)
            if all(len(got[k]) >= want[k] * 3 for k in want):
                break
        else:
            continue
        break
    out = []
    for (lang, kind), k in want.items():
        pool = got[(lang, kind)]
        take = rng.sample(pool, min(k, len(pool)))
        out += [{"lang": lang, "kind": kind, "text": t} for t in take]
        short = k - len(take)
        if short > 0:                                                         # fill from English
            extra = [t for t in got[("en", kind)] if t not in [o["text"] for o in out]]
            out += [{"lang": "en", "kind": kind, "text": t, "fill_for": lang} for t in rng.sample(extra, min(short, len(extra)))]
    rng.shuffle(out)
    return out[:n]


SWE_EVAL_GROUPS = ["openhands/minimax_m25/swe-rebench-v2", "sweagent/minimax_m25/swe-rebench-v2"]
# Calibration's agentic sessions: another harness (mini-swe-agent) and generator (Qwen3.8-27B), on
# the same multilingual task pool as eval -- the only pool with more than Python -- with every
# repository the eval slice uses excluded (--exclude_swe_from)
SWE_CAL_GROUPS = ["minisweagent/qwen38_27b/swe-rebench-v2"]
SWE_EXCLUDE = frozenset()                # eval: repositories to skip (--exclude_swe_from)

def load_swe(n, rng, tokenizer, template_vars, cap_tokens = 16000, min_turn = 5, groups = SWE_EVAL_GROUPS,
             min_tokens = 0, exclude_repos = frozenset()):
    """Real agent sessions from nvidia/Open-SWE-Traces (CC-BY-4.0), cut at an assistant turn: the
    context keeps the task, tool schemas, and the recorded agent's earlier actions and real tool
    output -- other-model text the model reads, as it would a subagent's -- and the turn itself is
    regenerated by the model under test, never scored from the recording. min_tokens: cut only
    where the context is at least this long (calibration windows); exclude_repos: repositories to
    skip (calibration excludes the eval slice's)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    per = [n // len(groups) + (1 if i < n % len(groups) else 0) for i in range(len(groups))]
    out = []
    for grp, want in zip(groups, per):
        path = sorted(fs.ls(f"datasets/nvidia/Open-SWE-Traces/data/{grp}", detail = False))[0]
        rows = pq.ParquetFile(fs.open(path)).read_row_group(0, columns = ["instance_id", "repo", "language", "messages", "tools", "resolved"]).to_pylist()
        rng.shuffle(rows)
        repos = set()
        for r in rows:
            if len(out) >= sum(per[:groups.index(grp) + 1]):
                break
            if r["repo"] in repos or r["repo"] in exclude_repos:    # one session per repo: spread, not depth
                continue
            tools = [json.loads(t) if isinstance(t, str) else t for t in r["tools"]]
            msgs = []
            for m in r["messages"]:
                m = {k: v for k, v in m.items() if k != "reasoning_content" and v is not None}
                for tc in m.get("tool_calls") or []:
                    f = tc.get("function", {})
                    if isinstance(f.get("arguments"), str):
                        try: f["arguments"] = json.loads(f["arguments"])
                        except Exception: pass
                m.setdefault("content", "")
                msgs.append(m)
            cand = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and i >= min_turn]
            if not cand:
                continue
            ctx_len = lambda i: tokenizer.hf_chat_template(msgs[:i], add_generation_prompt = True, tools = tools, **template_vars).shape[-1]
            lo, hi = 0, len(cand) - 1                # prefix length grows with i: binary search the cap
            if ctx_len(cand[0]) > cap_tokens:
                continue
            while lo < hi:
                mid = (lo + hi + 1) // 2
                lo, hi = (mid, hi) if ctx_len(cand[mid]) <= cap_tokens else (lo, mid - 1)
            first = 0
            if min_tokens:                           # and the floor, the same way
                if ctx_len(cand[lo]) < min_tokens:
                    continue
                a, b = 0, lo
                while a < b:
                    mid = (a + b) // 2
                    a, b = (mid + 1, b) if ctx_len(cand[mid]) < min_tokens else (a, mid)
                first = a
            cut = rng.choice(cand[first:lo + 1])
            repos.add(r["repo"])
            out.append({"harness": grp.split("/")[0], "group": grp, "repo": r["repo"], "language": r["language"], "resolved": r["resolved"],
                        "instance_id": r["instance_id"], "cut": cut, "messages": msgs[:cut], "tools": tools})
    rng.shuffle(out)
    return out


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
    key = TOOL_ARG[name]
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

# --sampling: exllamav3's own default (temperature 0.8, min-p 0.08) when unset, which is no model's
# recommendation. Keys are ComboSampler's: temperature, top_k, top_p, min_p, pres_p, freq_p, rep_p.
# pres_p/freq_p count generated tokens only (OpenAI semantics, as model cards assume): Job passes
# the sampler its generated count.
SAMPLING = None
# --sampling_profile: per-conversation sampling. {"source": ..., "modes": {name: {sampler args}},
# "rules": [{"slice": str | [str], "thinking": bool, "mode": name}, ...]}: the first rule whose given
# keys all match the conversation (its slice, and whether its chat template has thinking on)
# picks the mode. Mode args take OpenAI/vLLM names (presence_penalty, ...) or ComboSampler's.
PROFILE = None
SAMPLER_KEYS = {"presence_penalty": "pres_p", "frequency_penalty": "freq_p", "repetition_penalty": "rep_p"}

def load_profile(path):
    with open(path) as f:
        prof = json.load(f)
    assert prof.get("modes") and prof.get("rules"), f"{path}: a profile needs modes and rules"
    for rule in prof["rules"]:
        assert rule.get("mode") in prof["modes"], f"{path}: rule {rule} names no defined mode"
    return prof

def sampling_mode(meta, template_kw):
    thinking = bool(template_kw.get("enable_thinking", True))
    for rule in PROFILE["rules"]:
        sl = rule.get("slice")
        if sl is not None and meta["slice"] not in ([sl] if isinstance(sl, str) else sl):
            continue
        if rule.get("thinking") is not None and rule["thinking"] != thinking:
            continue
        return rule["mode"]
    raise ValueError(f"sampling profile has no rule for slice {meta['slice']}, thinking {thinking}")

def make_sampler(mode = None):
    if mode is not None:
        return ComboSampler(**{SAMPLER_KEYS.get(k, k): v for k, v in PROFILE["modes"][mode].items()})
    return ComboSampler(**SAMPLING) if SAMPLING is not None else None

# --backend vllm: for models exllamav3 cannot hold, such as a 27B in FP8 on two 16 GB cards.
# Conversations, template rendering and rows are unchanged; only sampling runs in vLLM, one engine
# per process (vLLM does not return all its VRAM when an engine closes). Per-row seeds and the
# sampling profile carry over; vLLM's presence/frequency penalties count generated tokens only,
# as ComboSampler's do here. Rows carry no stop token, as exllamav3's do.
VLLM_DEFAULTS = {"language_model_only": True, "max_model_len": 17408}
VLLM_KEYS = {"pres_p": "presence_penalty", "freq_p": "frequency_penalty", "rep_p": "repetition_penalty"}

class VllmBackend:
    def __init__(self, model_dir, engine_args, seed):
        # FlashInfer's top-p kernel takes workspace vLLM's memory profiling does not count; on a
        # card filled to the edge it fails in warmup. The native sampler is equivalent
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM
        kw = dict(VLLM_DEFAULTS, **engine_args)
        kw.setdefault("tensor_parallel_size", torch.cuda.device_count())
        self.max_len = kw["max_model_len"]
        self.llm = LLM(model = model_dir, seed = seed, **kw)

    def params(self, mode, max_new_tokens, prompt_len, stop_ids, seed):
        from vllm import SamplingParams
        if mode is not None:
            src = PROFILE["modes"][mode]
        elif SAMPLING is not None:
            src = SAMPLING
        else:
            src = {"temperature": 0.8, "min_p": 0.08}             # exllamav3's DefaultSampler
        kw = {VLLM_KEYS.get(k, k): v for k, v in src.items()}
        room = self.max_len - prompt_len
        assert room > 0, f"prompt of {prompt_len} tokens does not fit max_model_len {self.max_len}"
        return SamplingParams(**kw, max_tokens = min(max_new_tokens, room), stop_token_ids = stop_ids, seed = seed)

    def run(self, requests):
        """requests: [(input_ids list, SamplingParams)] -> [(response_ids, eos_reason)]"""
        outs = self.llm.generate([{"prompt_token_ids": ids} for ids, _ in requests], [sp for _, sp in requests])
        res = []
        for o, (_, sp) in zip(outs, requests):
            ids, fin = list(o.outputs[0].token_ids), o.outputs[0].finish_reason
            if fin == "stop" and ids and ids[-1] in sp.stop_token_ids:
                ids = ids[:-1]
            res.append((ids, "stop_token" if fin == "stop" else "max_new_tokens" if fin == "length" else fin))
        return res

def generate(generator, config, tokenizer, convs, template_vars, max_new_tokens, seed):
    """convs: list of (meta, messages, tools). Returns trace rows."""
    if isinstance(generator, VllmBackend):
        return generate_vllm(generator, config, tokenizer, convs, template_vars, max_new_tokens, seed)
    pending = {}
    for i, (meta, msgs, tools) in enumerate(convs):
        kw = dict(meta.get("template_vars") or template_vars, **({"tools": tools} if tools else {}))
        input_ids = tokenizer.hf_chat_template(msgs, add_generation_prompt = True, **kw)
        mode = sampling_mode(meta, kw) if PROFILE is not None else None
        generator.enqueue(Job(input_ids = input_ids, max_new_tokens = max_new_tokens,
                              stop_conditions = config.eos_token_id_list, decode_special_tokens = True,
                              identifier = i, seed = zlib.crc32(f"{seed}|{i}".encode()),
                              sampler = make_sampler(mode)))
        pending[i] = {"meta": meta, "input_ids": input_ids, "chunks": [], "eos_reason": None, "mode": mode}
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            # A failed job comes back as an "error" result, not an exception; skipping it would
            # silently drop the conversation (a poisoned CUDA context fails every job, and the
            # trace comes out empty with exit 0)
            if r["stage"] == "error":
                raise RuntimeError(f"generation failed for conversation {r['job'].identifier}") from r["error"]
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
                     **({"sampling_mode": p["mode"]} if p["mode"] is not None else {}),
                     "input_ids": p["input_ids"][0].tolist(),
                     "response_ids": torch.cat(p["chunks"], dim = -1)[0].tolist()})
    return rows


def generate_vllm(backend, config, tokenizer, convs, template_vars, max_new_tokens, seed):
    reqs, metas = [], []
    for i, (meta, msgs, tools) in enumerate(convs):
        kw = dict(meta.get("template_vars") or template_vars, **({"tools": tools} if tools else {}))
        input_ids = tokenizer.hf_chat_template(msgs, add_generation_prompt = True, **kw)[0].tolist()
        mode = sampling_mode(meta, kw) if PROFILE is not None else None
        reqs.append((input_ids, backend.params(mode, max_new_tokens, len(input_ids), config.eos_token_id_list,
                                               zlib.crc32(f"{seed}|{i}".encode()))))
        metas.append((meta, mode))
    rows = []
    for i, ((input_ids, _), (meta, mode), (resp, eos)) in enumerate(zip(reqs, metas, backend.run(reqs))):
        print(f"    {i:3} {meta['slice']:9s} {meta.get('lang', ''):3s} ctx {len(input_ids):5}  resp {len(resp):5} ({eos})", flush = True)
        if resp:
            rows.append({**meta, "eos_reason": eos, **({"sampling_mode": mode} if mode is not None else {}),
                         "input_ids": input_ids, "response_ids": resp})
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
    elif sl == "swe":
        for w in load_swe(n, rng, SWE_TOKENIZER, SWE_TV, exclude_repos = SWE_EXCLUDE):
            convs.append(({"slice": sl, "kind": w["harness"], "lang": "en", "repo": w["repo"], "code_language": w["language"],
                           "instance_id": w["instance_id"], "cut": w["cut"], "resolved": w["resolved"]}, w["messages"], w["tools"]))
    elif sl == "wild":
        for w in load_wildchat(n, rng):
            convs.append(({"slice": sl, "kind": w["kind"], "lang": w["lang"]},
                          [{"role": "user", "content": w["text"]}], None))
    elif sl == "self":
        for c in rng.sample(self_pool, min(n, len(self_pool))):
            convs.append(({"slice": sl, "kind": "self", "conversation": c},
                          [{"role": "user", "content": CONVERSATIONS[c][0]}], None))
    return convs


@torch.inference_mode()
def main(args):
    if args.repack:
        # Tokenizer only: the conversations are rebuilt to replay the original run's RNG state,
        # and their responses come from the trace instead of the model
        config = Config.from_directory(args.model_dir)
        tokenizer, generator = Tokenizer.from_config(config), None
    elif args.backend == "vllm":
        config = Config.from_directory(args.model_dir)
        tokenizer = Tokenizer.from_config(config)
        generator = VllmBackend(args.model_dir, args.vllm, args.seed)
    else:
        model, config, cache, tokenizer = model_init.init(args)[:4]
        generator = Generator(model = model, cache = cache, tokenizer = tokenizer, max_chunk_size = 2048)
    tv = probe_template_vars(tokenizer, dict(DEFAULT_TEMPLATE_VARS, **args.template_vars))
    rng = random.Random(args.seed)
    purpose = args.docs
    use_pool(purpose)
    global SWE_TOKENIZER, SWE_TV, SWE_EXCLUDE, LOG_CORPUS
    LOG_CORPUS = args.log_corpus
    SWE_TOKENIZER, SWE_TV = tokenizer, tv
    if purpose == "eval" and args.exclude_swe_from:      # the other direction: keep a calibration's repositories out of eval
        with open(args.exclude_swe_from) as f:
            SWE_EXCLUDE = frozenset(r["repo"] for r in json.load(f)["rows"] if r.get("repo"))
    variants = []
    for v in EVAL_TEMPLATE_VARIANTS:
        try:
            tokenizer.hf_chat_template([{"role": "user", "content": "hi"}], add_generation_prompt = True, **v)
            variants.append(v)
        except Exception:
            print(f" !! Chat template rejects {v}; dropped from eval variants")
    lo, hi = args.doc_min, args.doc_max
    self_pool = list(range(len(CONVERSATIONS)))
    if args.self_from:
        keep = {r["conversation"] for r in json.load(open(args.self_from))["rows"]}
        self_pool = [c for c in self_pool if c in keep]
        print(f" -- own-voice prompts restricted to the {len(self_pool)} used by {args.self_from}")
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
            if variants and not args.fixed_template:
                for meta, _, _ in convs:
                    meta["template_vars"] = rng.choice(variants)
            print(f" -- slice {sl}: {len(convs)} conversations", flush = True)
            rows = generate(generator, config, tokenizer, convs, tv, args.max_new_tokens, f"{args.seed}|{sl}")
            write_trace(f"{args.output}_{sl}.json", args, tokenizer, tv, rows, {"slice": sl, "docs": purpose})
        return

    # --- calibration: packed mix ---
    # --repack reuses an earlier pack's trace: the conversations are rebuilt with the shares and
    # ml_frac that generated it (which replays the original RNG state and is checked against the
    # trace's input ids), then packed at the new --shares. Generation over-provisions every slice,
    # so shrinking one draws on spare material; growing one past it falls back to raw rows, loudly
    src = json.load(open(args.repack)) if args.repack else None
    shares = dict(args.shares)
    assert abs(sum(shares.values()) - 1.0) < 1e-6, f"--shares must sum to 1, got {sum(shares.values())}"
    gen_shares = dict(src["shares"]) if src else shares
    ml_frac = src["ml_frac"] if src else args.ml_frac
    assert not src or args.ml_frac == ml_frac, \
        f"--repack cannot change ml_frac ({src['ml_frac']} in {args.repack}); the documents are fixed by the trace"
    R, C = (src.get("cal_rows", args.cal_rows), src.get("cal_cols", args.cal_cols)) if src else (args.cal_rows, args.cal_cols)
    def plan(sh, R):
        rf = {k: int(round(v * R)) for k, v in sh.items()}
        rf["raw"] += R - sum(rf.values())                               # rounding remainder
        return rf
    rows_for = plan(gen_shares, R)
    tok_budget = {k: rows_for[k] * C for k in ("ctx", "loop", "self")}
    print(f" -- calibration rows per slice: {rows_for}" + (f" (as generated; repacking from {args.repack})" if src else ""))

    # Documents: English prose/code kinds plus multilingual Wikipedia, ml_frac of the document slices
    # kind or kind:weight; documents per kind in proportion to weight (default 1), in each of ctx and loop
    from fractions import Fraction
    kw = {k.split(":")[0]: Fraction(k.split(":")[1]) if ":" in k else Fraction(1) for k in args.doc_kinds.split(",")}
    kinds, W = list(kw), sum(kw.values())
    per_kind = lambda total, k: -(-(total * kw[k]) // W)               # ceil; equal weights as before
    assert not src or src.get("doc_kinds", "web,wiki,technical,code") == args.doc_kinds, \
        f"--repack cannot change doc_kinds ({src.get('doc_kinds')} in {args.repack}); the documents are fixed by the trace"
    avg_ctx, avg_loop = 1300 + 450, 2 * 750 + 500                        # rough tokens per conversation
    n_ctx = int(tok_budget["ctx"] / avg_ctx * 1.25) + 1
    n_loop = int(tok_budget["loop"] / avg_loop * 1.25) + 1
    n_ml = int(round(ml_frac * n_ctx))
    ctx_docs = [d for k in kinds for d in load_docs(k, purpose, per_kind(n_ctx - n_ml, k), tokenizer, rng, lo, hi)]
    ctx_docs += load_docs("ml", purpose, n_ml, tokenizer, rng, lo, hi)
    rng.shuffle(ctx_docs)
    n_loop_docs = n_loop * 3
    n_loop_ml = int(round(ml_frac * n_loop_docs))
    loop_docs = [d for k in kinds for d in load_docs(k, purpose, per_kind(n_loop_docs - n_loop_ml, k), tokenizer, rng, lo // 2, hi // 2)]
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
    if rows_for.get("agent"):
        # Real agent sessions, regenerated at a cut, packed as one or two full windows each
        assert args.exclude_swe_from, "an agent share needs --exclude_swe_from (the eval swe trace) to keep its repositories out"
        with open(args.exclude_swe_from) as f:
            eval_repos = {r["repo"] for r in json.load(f)["rows"]}
        n_agent = int(-(-rows_for["agent"] // 2) * 1.3) + 1
        for w in load_swe(n_agent, rng, tokenizer, tv, groups = SWE_CAL_GROUPS, min_tokens = 2 * C, exclude_repos = eval_repos):
            convs.append(({"slice": "swe", "kind": "agent", "lang": "en", "repo": w["repo"], "code_language": w["language"],
                           "instance_id": w["instance_id"], "cut": w["cut"], "group": w["group"]}, w["messages"], w["tools"]))
    if src:
        source_trace = src["source_trace"]
        rows = json.load(open(source_trace))["rows"]
        assert len(rows) == len(convs), \
            f"{source_trace} has {len(rows)} conversations, the replay built {len(convs)}: different settings than generated it"
        for i, ((meta, msgs, tools), r) in enumerate(zip(convs, rows)):
            kw = dict(tv, **({"tools": tools} if tools else {}))
            ids = tokenizer.hf_chat_template(msgs, add_generation_prompt = True, **kw)[0].tolist()
            assert ids == r["input_ids"] and meta["slice"] == r["slice"], \
                f"conversation {i} does not match {source_trace}: the replay needs the original -m, -tv, --seed, --exclude_self and doc bounds"
        print(f" -- replayed {len(convs)} conversations, all matching {source_trace}")
        rows_for = plan(shares, R)
        print(f" -- repacked rows per slice: {rows_for}")
    else:
        print(f" -- generating {len(convs)} conversations", flush = True)
        rows = generate(generator, config, tokenizer, convs, tv, args.max_new_tokens, f"{args.seed}|cal")
        source_trace = f"{args.output}_cal.json"
        write_trace(source_trace, args, tokenizer, tv, rows, {"slice": "cal", "docs": purpose})

    # Pack: each conversational slice's streams into fixed-width rows, up to its row budget
    from exllamav3.conversion.calibration_data import get_default_calibration
    group = lambda r: "ctx" if r["slice"].startswith("ctx") else r["slice"]
    packed, comp = [], {}
    if rows_for.get("agent"):
        # One or two full windows per session, ending with the regenerated turn: room for the session
        # to show how it got there, without one long session filling the slice
        rs = [r for r in rows if r["slice"] == "swe"]
        rng.shuffle(rs)
        agent_rows, sessions, langs = [], 0, {}
        for r in rs:
            seq = r["input_ids"] + r["response_ids"]
            k = min(2, len(seq) // C, rows_for["agent"] - len(agent_rows))
            if k <= 0:
                continue
            agent_rows += [torch.tensor(seq[len(seq) - (k - i) * C:len(seq) - (k - i - 1) * C], dtype = torch.long).unsqueeze(0)
                           for i in range(k)]
            sessions += 1
            langs[r.get("code_language", "?")] = langs.get(r.get("code_language", "?"), 0) + k
            if len(agent_rows) >= rows_for["agent"]:
                break
        if len(agent_rows) < rows_for["agent"]:
            print(f" !! slice agent: only {len(agent_rows)} of {rows_for['agent']} rows of material; raw rows fill the rest")
            rows_for["raw"] += rows_for["agent"] - len(agent_rows)
        packed += agent_rows
        comp["agent"] = {"rows": len(agent_rows), "sessions": sessions, "code_languages": langs}
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
    json.dump({"shares": gen_shares, "ml_frac": ml_frac, "doc_kinds": args.doc_kinds, "cal_rows": R, "cal_cols": C, "composition": comp,
               "source_trace": source_trace,
               **({"packed_shares": shares, "repacked_from": args.repack} if src else {})},
              open(os.path.splitext(args.cal_out)[0] + ".manifest.json", "w"), indent = 1)
    print(f" -- {R} x {C} calibration rows -> {args.cal_out}; composition {comp}")


def write_trace(path, args, tokenizer, tv, rows, meta):
    out = {"model": args.model_dir, "vocab_size": tokenizer.actual_vocab_size, "template_vars": tv,
           "sampling": PROFILE or args.sampling or "exllamav3 DefaultSampler (temperature 0.8, min_p 0.08)",
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
    parser.add_argument("--backend", type = str, default = "exllamav3", choices = ["exllamav3", "vllm"],
                        help = "Generation backend. vllm loads -m as vLLM does (e.g. an FP8 checkpoint), one engine per process")
    parser.add_argument("--vllm", type = json.loads, default = {},
                        help = "vLLM engine arguments, JSON, over VLLM_DEFAULTS; Qwen3.8-27B FP8 on 2x16 GB: "
                               "'{\"max_num_batched_tokens\": 1024, \"enforce_eager\": true, \"gpu_memory_utilization\": 0.968}'")
    parser.add_argument("-o", "--output", type = str, required = True, help = "Output prefix; writes <prefix>_<slice>.json")
    parser.add_argument("--docs", type = str, default = "eval", choices = ["eval", "cal"], help = "Document sources: held-out eval text, or the calibration corpus")
    parser.add_argument("--slices", type = str, default = "ctx_user,ctx_tool,ctx_ml,loop,self,wild", help = "(eval) slices to write")
    parser.add_argument("--code_glob", type = str, default = None, help = "(eval) glob of source files for code documents")
    parser.add_argument("-n", type = int, default = 30, help = "(eval) conversations per slice")
    parser.add_argument("--cal_out", type = str, default = None, help = "(cal) packed calibration file for convert.py --cal_data")
    parser.add_argument("--cal_rows", type = int, default = 250)
    parser.add_argument("--cal_cols", type = int, default = 2048)
    parser.add_argument("--shares", type = json.loads, default = {"raw": 0.25, "ctx": 0.35, "loop": 0.25, "self": 0.10, "random": 0.05},
                        help = "(cal) row shares per slice, JSON")
    parser.add_argument("--repack", type = str, default = None, help = "(cal) manifest of an earlier pack: reuse its trace instead of generating, packed at --shares (needs the original -m, -tv, --seed and --exclude_self)")
    parser.add_argument("--exclude_swe_from", type = str, default = None,
                        help = "(cal) eval swe trace whose repositories the agent slice must not use; required with an agent share. "
                        "(eval) a calibration trace whose agent sessions' repositories the swe slice must not use")
    parser.add_argument("--doc_kinds", type = str, default = "web,wiki,technical,code",
                        help = "(cal) English document kinds for the ctx and loop slices, as kind or kind:weight (default 1); diff = CommitPackFT commits, "
                               "log = --log_corpus build and test transcripts")
    parser.add_argument("--log_corpus", type = str, default = None, help = "logcorpus.py output directory, for the log document kind")
    parser.add_argument("--ml_frac", type = float, default = 0.12, help = "(cal) fraction of documents drawn from non-English Wikipedia")
    parser.add_argument("--exclude_self", type = str, default = None, help = "Eval self-slice trace whose own-voice prompts must not be reused")
    parser.add_argument("--self_from", type = str, default = None, help = "Restrict own-voice prompts to those in this trace (e.g. the set calibration excluded)")
    parser.add_argument("--doc_min", type = int, default = 500, help = "Min document tokens")
    parser.add_argument("--doc_max", type = int, default = 1500, help = "Max document tokens")
    parser.add_argument("--max_new_tokens", type = int, default = 1536)
    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("-tv", "--template_vars", type = json.loads, default = {})
    parser.add_argument("--sampling", type = json.loads, default = None,
                        help = "ComboSampler arguments as JSON, e.g. '{\"temperature\": 1.0, \"top_k\": 20, \"top_p\": 0.95}' "
                               "(default: exllamav3's DefaultSampler). pres_p/freq_p count generated tokens only, as in vLLM")
    parser.add_argument("--sampling_profile", type = str, default = None,
                        help = "JSON file of sampling modes and the rules that pick one per conversation (slice, thinking); "
                               "each trace row records its mode")
    parser.add_argument("--fixed_template", action = "store_true", help = "(eval) use --template_vars for every conversation instead of varying thinking settings")
    args = parser.parse_args()
    assert args.docs != "cal" or args.cal_out, "--docs cal needs --cal_out"
    assert not args.repack or args.docs == "cal", "--repack applies to --docs cal"
    SAMPLING = args.sampling
    assert not (args.sampling and args.sampling_profile), "--sampling and --sampling_profile are exclusive"
    PROFILE = load_profile(args.sampling_profile) if args.sampling_profile else None
    main(args)
