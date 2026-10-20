"""Entity-resolution precision/recall against ground truth for one processed world.

    python scripts/eval_entity_resolution.py --data data/demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from argus.db import connect  # noqa: E402
from argus.evaluate.metrics import evaluate_er  # noqa: E402
from argus.evaluate.truth import Truth  # noqa: E402
from argus.paths import RunPaths  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/demo")
    a = ap.parse_args()
    p = RunPaths.at(a.data)
    con = connect(p.db, read_only=True)
    r = evaluate_er(con, Truth.load(p.root))
    for k in ("all_records", "person_records", "person_records_excl_burners", "org_records"):
        m = r[k]
        print(f"{k:30s} records={m['records']:5d} true_pairs={m['true_pairs']:6d} P={m['pairwise_precision']:.3f} R={m['pairwise_recall']:.3f} F1={m['pairwise_f1']:.3f} B3-F1={m['bcubed_f1']:.3f}")
    print("\nrecall by source pair:")
    for k, v in r["recall_by_source_pair"].items():
        print(f"  {k:20s} pairs={v['pairs']:5d} recall={v['recall']:.3f}")
    print("\nerror analysis:")
    print(json.dumps(r["error_analysis"], indent=2, default=str))


if __name__ == "__main__":
    main()
