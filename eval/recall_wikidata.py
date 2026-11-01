"""
Closed-book facts across knowledge domains, from Wikidata (CC0) through the QLever endpoint, for recall_trace.py.

Each relation is a class, a property and a question template. A subject is kept when it has exactly one value of the
property; a question is dropped when its answer is spelled in its subject (a giveaway, not a memory). Popularity is the
subject's Wikipedia sitelink count. Writes one JSON object per question.

    python eval/recall_wikidata.py -o eval/recall_data/wikidata_facts.jsonl
"""

import argparse
import json
import random
import time
import urllib.parse
import urllib.request

ENDPOINT = "https://qlever.dev/api/wikidata"
PREFIX = ("PREFIX wdt: <http://www.wikidata.org/prop/direct/> PREFIX wd: <http://www.wikidata.org/entity/> "
          "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX wikibase: <http://wikiba.se/ontology#> ")

# name, domain, subject pattern on ?s, property, kind of value (item / literal / year), question template
RELATIONS = [
    ("painter",       "arts",      "?s wdt:P31 wd:Q3305213",  "P170", "item", "Who painted {s}?"),
    ("album_artist",  "arts",      "?s wdt:P31 wd:Q482994",   "P175", "item", "Who recorded the album {s}?"),
    ("book_author",   "literature", "?s wdt:P31 wd:Q7725634", "P50",  "item", "Who wrote {s}?"),
    ("element_symbol", "chemistry", "?s wdt:P31 wd:Q11344",   "P246", "literal", "What is the chemical symbol of {s}?"),
    ("genus_family",  "biology",   "?s wdt:P105 wd:Q34740 ; wdt:P171 ?v . ?v wdt:P105 wd:Q35409 . ?s wdt:P105 wd:Q34740",   "P171", "item", "What taxonomic family does the genus {s} belong to?"),
    ("gene_chromosome", "biology", "?s wdt:P31 wd:Q7187 ; wdt:P703 wd:Q15978631", "P1057", "item", "On which human chromosome is the {s} gene located?"),
    ("disease_cause", "medicine",  "?s wdt:P31/wdt:P279* wd:Q12136 ; wdt:P828 ?v . ?v wdt:P31 wd:Q16521 . ?s wdt:P31/wdt:P279* wd:Q12136", "P828", "item", "What pathogen causes {s}?"),
    ("moon_planet",   "astronomy", "?s wdt:P31/wdt:P279* wd:Q2537", "P397", "item", "What does the moon {s} orbit?"),
    ("star_constellation", "astronomy", "?s wdt:P31 wd:Q523", "P59",  "item", "In which constellation is the star {s}?"),
    ("battle_year",   "history",   "?s wdt:P31 wd:Q178561",   "P585", "year", "In what year was the {s} fought?"),
    ("treaty_year",   "history",   "?s wdt:P31 wd:Q131569",   "P585", "year", "In what year was the {s} signed?"),
    ("language_designer", "computing", "?s wdt:P31 wd:Q9143", "P287", "item", "Who designed the {s} programming language?"),
    ("advisor",       "academia",  "?s wdt:P31 wd:Q5 ; wdt:P106 wd:Q170790", "P184", "item", "Who was the doctoral advisor of the mathematician {s}?"),
    ("river_mouth",   "geography", "?s wdt:P31 wd:Q4022",     "P403", "item", "Into what body of water does the {s} flow?"),
    ("company_hq",    "business",  "?s wdt:P31 wd:Q4830453",  "P159", "item", "In what city is {s} headquartered?"),
    ("court_case",    "law",       "?s wdt:P31 wd:Q2334719",  "P4884", "item", "Which court decided {s}?"),
]


def query(sparql, attempts = 3):
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": PREFIX + sparql})
    req = urllib.request.Request(url, headers = {"Accept": "application/sparql-results+json", "User-Agent": "exl3-recall-eval/0.1"})
    for a in range(attempts):
        try:
            with urllib.request.urlopen(req, timeout = 90) as r:
                return json.load(r)["results"]["bindings"]
        except Exception as e:
            print(f"    !! {type(e).__name__}: {str(e)[:100]}; retrying")
            time.sleep(30 * (a + 1))
    raise RuntimeError("query failed")


def facts(rel, limit, per_relation, rng):
    name, domain, pattern, prop, kind, template = rel
    if kind == "item":
        value, label = "?v", "?v @en@rdfs:label ?o ."
    elif kind == "literal":
        value, label = "?o", ""
    else:
        value, label = "?d", "BIND(YEAR(?d) AS ?o)"
    rows = query(f"SELECT ?s ?sl ?o ?links WHERE {{ {pattern} ; @en@rdfs:label ?sl ; wikibase:sitelinks ?links ; wdt:{prop} {value} . {label} "
                 f"FILTER(?links >= 2) }} ORDER BY DESC(?links) LIMIT {limit}")
    subj = {}
    for r in rows:
        s = r["s"]["value"].rsplit("/", 1)[1]
        e = subj.setdefault(s, {"label": r["sl"]["value"], "links": int(r["links"]["value"]), "values": set()})
        e["values"].add(str(r["o"]["value"]))
    out = []
    for s, e in subj.items():
        if len(e["values"]) != 1:
            continue
        answer = next(iter(e["values"]))
        if answer.lower() in e["label"].lower() or e["label"].lower() in answer.lower() or answer[:1] == "Q" and answer[1:].isdigit():
            continue
        out.append({"qid": f"{name}:{s}", "relation": name, "domain": domain, "question": template.format(s = e["label"]),
                    "answer": answer.replace("human chromosome ", "").replace("human ", "").replace(" chromosome", "") if name == "gene_chromosome" else answer,
                    "links": e["links"]})
    out.sort(key = lambda x: -x["links"])
    # A near-constant answer is a guess, not a memory: no answer may fill more than 10% of a relation
    seen, cap, capped = {}, max(5, len(out) // 10), []
    for x in out:
        seen[x["answer"]] = seen.get(x["answer"], 0) + 1
        if seen[x["answer"]] <= cap and x["answer"] != "various artists":
            capped.append(x)
    out = capped
    # keep the popularity spread: every rank band, not just the head
    if len(out) > per_relation:
        step = len(out) / per_relation
        out = [out[int(i * step)] for i in range(per_relation)]
    return out, len(rows), len(subj)


def main(args):
    rng = random.Random(0)
    only = set(args.only.split(",")) if args.only else None
    with open(args.output, "a" if only else "w") as f:
        for rel in RELATIONS:
            if only and rel[0] not in only:
                continue
            try:
                out, n, ns = facts(rel, args.limit, args.per_relation, rng)
            except RuntimeError:
                print(f"{rel[0]:20} FAILED")
                continue
            for x in out:
                f.write(json.dumps(x, ensure_ascii = False) + "\n")
            f.flush()
            sample = "; ".join(f"{x['question']} -> {x['answer']} ({x['links']})" for x in (out[:1] + out[len(out) // 2:len(out) // 2 + 1]))
            print(f"{rel[0]:20} {n:6} rows, {ns:6} subjects, {len(out):5} kept | {sample[:150]}", flush = True)
            time.sleep(args.pause)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = __doc__.split("\n\n")[0])
    parser.add_argument("-o", "--output", required = True)
    parser.add_argument("--limit", type = int, default = 20000, help = "Rows per relation, most linked first")
    parser.add_argument("--per_relation", type = int, default = 1000, help = "Questions kept per relation, spread over popularity")
    parser.add_argument("--only", default = None, help = "Comma-separated relations to (re)collect, appended to --output")
    parser.add_argument("--pause", type = float, default = 5.0, help = "Seconds between queries")
    main(parser.parse_args())
