"""Detection of planted networks (ring/member recall, decoy false positives) for one world.

    python scripts/eval_ring_detection.py --data data/demo [--threshold 45]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from argus.db import connect  # noqa: E402
from argus.evaluate.metrics import evaluate_detection  # noqa: E402
from argus.evaluate.truth import Truth  # noqa: E402
from argus.paths import RunPaths  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/demo")
    ap.add_argument("--threshold", type=float, default=45.0)
    a = ap.parse_args()
    p = RunPaths.at(a.data)
    con = connect(p.db, read_only=True)
    d = evaluate_detection(con, Truth.load(p.root), a.threshold)
    print(f"rings detected: {d['rings_detected']}/{d['rings_total']}   member recall: {d['ring_member_recall']:.1%}   (non-watch-listed: {d['non_watchlisted_member_recall']:.1%})")
    for r in d["rings"]:
        print(f"  {r['ring_id']} {r['typology']:18s} members={r['members']} alerted={r['members_alerted']} orgs_alerted={r['orgs_alerted']} community_flagged={r['community_flagged']}")
    print("alerts:", json.dumps(d["alerts"]))
    print("decoys:", json.dumps(d["decoys"]))
    print("victims:", json.dumps(d["victims"]))
    print("ranking:", json.dumps({k: d[k] for k in ("risk_score", "rule_score", "ml_score", "precision_at_k")}))


if __name__ == "__main__":
    main()
