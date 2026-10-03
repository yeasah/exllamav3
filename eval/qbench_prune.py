"""
Evict reference logits a logit cache has not used since a given time; results, KL vectors and
tokenized data stay. Meant to run right after a successful bench series, with its start time:

    python eval/qbench_prune.py <cache dir> --unused-since 2026-10-03T14:00:00 [--dry-run]
"""
import argparse, datetime, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qbench.data import QCache

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache_dir", help = "The logit_cache dir of a qbench project (contains qbench/)")
    parser.add_argument("--unused-since", required = True, help = "ISO timestamp; logits last used before it are evicted")
    parser.add_argument("--dry-run", action = "store_true")
    args = parser.parse_args()
    evicted = QCache({"dir": args.cache_dir}).prune(datetime.datetime.fromisoformat(args.unused_since), args.dry_run)
    print(f" -- {'Would free' if args.dry_run else 'Freed'} {sum(s for _, s in evicted) / 1024**3:.1f} GB "
          f"in {len(evicted)} logit set(s)")
