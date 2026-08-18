#!/usr/bin/env python3
"""What is in a qbench logit cache, and what is safe to delete?

    python eval/qbench_cache.py <cache-dir> [--project p.yaml ...] [--orphans]

Every cache entry is named by a hash of what produced it, which is what keeps the cache
correct and what makes it impossible to navigate: an 11 GiB reference pass and a 4 KB
results file look equally like `logits_<hex>_<hex>`. This prints the same directory with
names attached, biggest first.

Two independent sources of names, because neither covers everything:

  - the cache's own `manifest.json`, written by qbench.py as it fills the cache. Covers
    anything written since that was added, and survives the model being deleted.
  - `--project`, which re-derives the keys a project file implies. Covers entries older
    than the manifest, but only while the checkpoints still exist, since a model key
    includes a modification stamp of the source directory.

`--orphans` lists what neither source explains -- the entries to consider deleting when a
line of work is finished. It is deliberately a report rather than a `--delete` flag: the
expensive entries here are reference passes that can cost a re-download plus hours of
compute to recreate, and that decision should be typed by a human.
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def entry_size(path: str) -> int:
    if os.path.isdir(path):
        return sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fs in os.walk(path) for f in fs
        )
    return os.path.getsize(path)


def keys_from_project(project_file: str) -> dict:
    """Re-derive the cache keys a project implies: {key: description}."""
    import yaml

    from qbench.data import resolve_project_paths, sha_key, source_stamp
    from qbench.measure import BF16_ROUNDING_EPS, METRICS_VERSION

    with open(project_file, "r", encoding="utf8") as f:
        project = yaml.safe_load(f)
    resolve_project_paths(project, project_file)

    if project.get("test_trace"):
        return {}          # trace keys need the tokenized data; not worth loading here
    data_key = sha_key({"v": 1, "test_data": project["test_data"],
                        "tokenizer": project["tokenizer"]})

    def model_key(mspec, noise=False):
        options = {k: v for k, v in mspec.get("options", {}).items() if k != "streaming"}
        return sha_key({
            "v": 1, "engine": mspec["engine"], "source": mspec["source"],
            "options": options, "stamp": source_stamp(mspec["source"]),
            "noise": BF16_ROUNDING_EPS if noise else 0,
        })

    name = os.path.basename(project_file)
    out = {}
    refs = [m for m in project["models"] if m.get("group") == "reference"]
    out[data_key] = f"{name}: tokenized test set"
    if not refs:
        return out
    ref = refs[0]
    ref_key = model_key(ref)
    out[data_key] = f"{name}: tokenized test set"
    out[f"{data_key}_{ref_key}"] = f"{name}: reference logits ({ref.get('label')})"
    out[f"{data_key}_{ref_key}_self_m{METRICS_VERSION}"] = f"{name}: reference self-score"
    out[f"{data_key}_{ref_key}_{model_key(ref, noise=True)}_m{METRICS_VERSION}"] = \
        f"{name}: noise floor"
    for m in project["models"]:
        if m is ref:
            continue
        out[f"{data_key}_{ref_key}_{model_key(m)}_m{METRICS_VERSION}"] = \
            f"{name}: {m.get('label')} [{m.get('group')}]"
    return out


def describe(manifest: dict, derived: dict, key: str) -> str:
    if key in derived:
        return derived[key]
    e = manifest.get(key)
    if not e:
        return ""
    bits = [b for b in (e.get("project"), e.get("label"), e.get("role")) if b]
    src = e.get("repo") or e.get("source")
    if src and e.get("revision"):
        src = f"{src}@{e['revision']}"
    if src:
        bits.append(os.path.basename(str(src)))
    return ": ".join(bits[:2]) + (f" ({', '.join(bits[2:])})" if bits[2:] else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache", help="logit_cache dir from a project file (or its qbench/ subdir)")
    ap.add_argument("--project", action="append", default=[],
                    help="project YAML whose keys should be re-derived; repeatable")
    ap.add_argument("--orphans", action="store_true", help="only entries nothing explains")
    args = ap.parse_args()

    root = args.cache
    if os.path.isdir(os.path.join(root, "qbench")):
        root = os.path.join(root, "qbench")

    manifest = {}
    mf = os.path.join(root, "manifest.json")
    if os.path.exists(mf):
        with open(mf) as f:
            manifest = json.load(f)

    derived = {}
    for p in args.project:
        try:
            derived.update(keys_from_project(p))
        except Exception as exc:
            print(f" -- could not read {p}: {exc}", file=sys.stderr)

    rows = []
    for name in os.listdir(root):
        if name == "manifest.json":
            continue
        path = os.path.join(root, name)
        for prefix, suffix in (("logits_", ""), ("results_", ".json"),
                               ("kl_", ".safetensors"), ("tokens_", ".safetensors")):
            if name.startswith(prefix) and name.endswith(suffix):
                key = name[len(prefix):len(name) - len(suffix) if suffix else None]
                rows.append((entry_size(path), prefix.rstrip("_"), key, path))
                break

    rows.sort(reverse=True)
    total = named = 0
    for size, kind, key, path in rows:
        what = describe(manifest, derived, key)
        total += size
        if what:
            named += size
        if args.orphans and what:
            continue
        unit = f"{size / 2**30:7.2f} GiB" if size >= 2**30 else (
               f"{size / 2**20:7.1f} MiB" if size >= 2**20 else f"{size / 2**10:7.1f} KiB")
        print(f"{unit}  {kind:7s}  {what or '(unexplained)':58s}  {key}")
    print(f"\n{total / 2**30:.1f} GiB total, {named / 2**30:.1f} GiB explained, "
          f"{(total - named) / 2**30:.1f} GiB not")


if __name__ == "__main__":
    main()
