"""
A qbench trace of closed-book recall: short factual questions whose answers are one entity, asked in the
model's chat template with thinking off, each followed by an answer the reference may or may not give.
qbench scores every row under teacher forcing (meta.per_row): the answer's total log-prob and whether
each answer token is the argmax, for the model and the reference. No generation, so a few thousand
questions cost about one ordinary slice.

Source: PopQA (Mallen et al. 2023, MIT), pinned to one commit. Each question carries its subject's
Wikipedia page views, the obscurity axis. Only relations with essentially one answer are kept. Every
question gets one row per answer alias (canonical first, up to --max_aliases): the analysis takes,
per question, the alias the reference finds most likely, so the surface form is the reference's own.

    python eval/recall_trace.py -m /models/hf/model -o eval_recall.json
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import csv
import hashlib
import json
import urllib.request

from exllamav3 import Config, Tokenizer

POPQA_URL = "https://raw.githubusercontent.com/AlexTMallen/adaptive-retrieval/5b56a22845de41170495f9fe0bc429e600df3f4f/data/popQA.tsv"
POPQA_SHA256 = "9a5227f41bff0e4c331d4a774d946b12f95307892b58f860a9606ef356e6089b"

# One answer per subject, or nearly: dropped are occupation, genre, color, religion and sport (vague or
# many-valued), screenwriter and producer (films often have several)
RELATIONS = ("capital", "capital of", "country", "place of birth", "father", "mother", "director", "author", "composer")

INSTRUCTION = "Answer with only the name, nothing else."


def popqa(cache_dir):
    path = os.path.join(cache_dir, "popQA.tsv")
    if not os.path.isfile(path):
        os.makedirs(cache_dir, exist_ok = True)
        urllib.request.urlretrieve(POPQA_URL, path + ".part")
        os.replace(path + ".part", path)
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert digest == POPQA_SHA256, f"{path}: sha256 {digest}, expected {POPQA_SHA256}"
    with open(path, newline = "") as f:
        return list(csv.DictReader(f, delimiter = "\t"))


def main(args):
    config = Config.from_directory(args.model_dir)
    tokenizer = Tokenizer.from_config(config)
    tv = json.loads(args.template_vars)
    rows, questions = [], 0
    for q in popqa(args.cache_dir):
        if q["prop"] not in RELATIONS:
            continue
        aliases = []
        for a in [q["obj"]] + json.loads(q["possible_answers"]):
            a = a.strip()
            if a and a not in aliases:
                aliases.append(a)
        msgs = [{"role": "user", "content": f"{q['question']} {INSTRUCTION}"}]
        prompt = tokenizer.hf_chat_template(msgs, add_generation_prompt = True, **tv)[0].tolist()
        for i, a in enumerate(aliases[:args.max_aliases]):
            answer = tokenizer.encode(a, add_bos = False)[0].tolist()
            rows.append({"input_ids": prompt, "response_ids": answer, "slice": "recall", "kind": q["prop"],
                         "lang": "en", "qid": int(q["id"]), "alias": i, "answer": a, "s_pop": int(q["s_pop"])})
        questions += 1
    out = {"model": args.model_dir, "vocab_size": tokenizer.actual_vocab_size, "template_vars": tv,
           "sampling": None,
           "meta": {"slice": "recall", "per_row": True, "rows": len(rows), "questions": questions,
                    "source": f"PopQA ({POPQA_URL}, MIT)", "relations": list(RELATIONS),
                    "instruction": INSTRUCTION, "max_aliases": args.max_aliases},
           "rows": rows}
    with open(args.output, "w") as f:
        json.dump(out, f)
    print(f" -- {args.output}: {questions} questions, {len(rows)} rows "
          f"({sum(len(r['input_ids']) + len(r['response_ids']) for r in rows) / len(rows):.0f} tokens per row)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = __doc__.split("\n\n")[0])
    parser.add_argument("-m", "--model_dir", required = True, help = "Model directory (its tokenizer and chat template)")
    parser.add_argument("-o", "--output", required = True, help = "Trace JSON for qbench's test_trace")
    parser.add_argument("-tv", "--template_vars", default = '{"enable_thinking": false}',
                        help = "Chat template variables, JSON (default: thinking off)")
    parser.add_argument("--max_aliases", type = int, default = 4, help = "Answer rows per question, canonical first")
    parser.add_argument("--cache_dir", default = os.path.expanduser("~/.cache/exllamav3/popqa"),
                        help = "Where the pinned PopQA file is kept")
    main(parser.parse_args())
