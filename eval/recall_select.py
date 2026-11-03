"""
A model's recall slice: the questions its reference clearly knows, out of a pool from recall_trace.py.

Retention over everything the reference answers mixes very different things. On MiniCPM5-2B a 3 bpw quant keeps 98% of what
the reference holds at P >= 0.9 and 28% of what it holds under 0.3, and a quarter of the low end flips between equivalent
conversions, so qbench reports retention by band of reference probability. This keeps, per question, the alias the
reference finds most likely, if the reference produces it. Numeric answers stay out (many spellings of one value). So that
easy relations cannot carry a band, confident and unconfident questions are capped apart: PopQA, whose capitals and countries
are the easy end, may fill at most --popqa_cap of either, and one relation at most --relation_cap.

Scoring every arm against a slice costs far more than the reference pass that chooses it, so this is where a thin slice
stops: under --min_questions confident ones (P >= --min_p) it writes nothing and exits 3, unless --allow_thin.

    python eval/recall_select.py -t recall_pool.json -r reference_results.json -o eval_recall.json
"""

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict

EXIT_THIN = 3


def reference_rows(path):
    """Per-row reference scores from a qbench result: any arm's rows carry them, and its Noise floor is always there."""
    for m in json.load(open(path)):
        rows = m.get("rows")
        if rows and "ref_lp" in (rows[0] or {}):
            return rows
    raise SystemExit(f"{path}: no per-row reference scores; was this a recall pool, with a noise floor?")


def spread_key(row):
    """Stable and unrelated to popularity, so that a cap thins a relation evenly."""
    return hashlib.sha256(f"{row['kind']}:{row['qid']}".encode()).digest()


def capped(rows, pool, popqa_cap, relation_cap):
    """`rows` under both caps. Each is a share of what is left, so its limit is found by lowering it until it holds."""
    ordered = sorted(rows, key = lambda r: spread_key(pool[r]))
    popqa = [r for r in ordered if "domain" not in pool[r]]
    others = [r for r in ordered if "domain" in pool[r]]
    keep = min(len(popqa), int(popqa_cap / (1 - popqa_cap) * len(others)))
    rows = others + popqa[:keep]
    by_relation = defaultdict(list)
    for r in rows:
        by_relation[pool[r]["kind"]].append(r)
    limit = max((len(v) for v in by_relation.values()), default = 0)
    while limit and limit > relation_cap * sum(min(len(v), limit) for v in by_relation.values()):
        limit -= 1
    return [r for v in by_relation.values() for r in v[:limit]]


def select(pool, ref, min_p, popqa_cap, relation_cap):
    by_question = defaultdict(list)
    for i, row in enumerate(pool):
        by_question[row["qid"]].append(i)
    floor = math.log(min_p)
    numeric = known = 0
    confident, shaky = [], []
    for rows in by_question.values():
        i = max(rows, key = lambda r: ref[r]["ref_lp"])
        if not ref[i]["ref_hit"]:
            continue
        known += 1
        if re.search(r"\d", str(pool[i]["answer"])):
            numeric += 1
        else:
            (confident if ref[i]["ref_lp"] >= floor else shaky).append(i)
    return capped(confident, pool, popqa_cap, relation_cap), capped(shaky, pool, popqa_cap, relation_cap), {
        "questions": len(by_question), "known": known, "numeric_known": numeric, "confident_uncapped": len(confident)}


def main(args):
    trace = json.load(open(args.trace))
    pool = trace["rows"]
    ref = reference_rows(args.reference)
    assert len(ref) == len(pool), f"{args.reference}: {len(ref)} rows, but the pool has {len(pool)}"
    confident, shaky, counts = select(pool, ref, args.min_p, args.popqa_cap, args.relation_cap)
    kept = sorted(confident + shaky)
    relations = Counter(pool[i]["kind"] for i in confident)
    domains = Counter(pool[i].get("domain", "popqa") for i in confident)
    print(f" -- {counts['questions']} questions, {counts['known']} known, {counts['numeric_known']} of those numeric, "
          f"{counts['confident_uncapped']} confident at P >= {args.min_p}; after caps {len(confident)} confident, {len(shaky)} below")
    print(f" -- confident by domain: {dict(domains.most_common())}")
    print(f" -- largest confident relations: {dict(relations.most_common(5))}")
    if len(confident) < args.min_questions and not args.allow_thin:
        print(f"!!! RECALL SLICE TOO THIN: {len(confident)} confident questions, {args.min_questions} required; nothing "
              "written. Grow the pool, or pass --allow_thin to score it anyway")
        sys.exit(EXIT_THIN)
    if len(confident) < args.min_questions:
        print(f"!! recall slice is thin: {len(confident)} confident questions, {args.min_questions} wanted; written because of --allow_thin")
    trace["rows"] = [pool[i] for i in kept]
    trace["meta"].update({"rows": len(kept), "questions": len(kept), "selection": {
        **counts, "confident": len(confident), "below": len(shaky), "min_p": args.min_p, "popqa_cap": args.popqa_cap,
        "relation_cap": args.relation_cap, "min_questions": args.min_questions, "thin": len(confident) < args.min_questions,
        "relations": dict(relations), "domains": dict(domains)}})
    with open(args.output, "w") as f:
        json.dump(trace, f)
    print(f" -- {args.output}: {len(kept)} questions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = __doc__.split("\n\n")[0])
    parser.add_argument("-t", "--trace", required = True, help = "Pool from recall_trace.py")
    parser.add_argument("-r", "--reference", required = True, help = "qbench results over that pool, with per-row scores")
    parser.add_argument("-o", "--output", required = True, help = "Trace JSON for qbench's test_trace")
    parser.add_argument("--min_p", type = float, default = 0.7, help = "Reference probability that counts as confident, toward --min_questions")
    parser.add_argument("--popqa_cap", type = float, default = 0.33, help = "Largest share PopQA may fill, of confident and of the rest")
    parser.add_argument("--relation_cap", type = float, default = 0.25, help = "Largest share one relation may fill, likewise")
    parser.add_argument("--min_questions", type = int, default = 300, help = "Fewer confident questions than this is a refusal")
    parser.add_argument("--allow_thin", action = "store_true", help = "Write a slice under --min_questions anyway")
    main(parser.parse_args())
