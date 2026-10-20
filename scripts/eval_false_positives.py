"""False-positive rate on clean data.

Builds a world of the same size with *no* planted networks (decoys and benign
look-alikes stay), runs the pipeline with the model trained on ``--data`` and
reports how many people get alerted - every one of them is a false positive.

    python scripts/eval_false_positives.py --data data/demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from argus.db import connect  # noqa: E402
from argus.evaluate.metrics import clean_control_fpr  # noqa: E402
from argus.evaluate.report import build_clean_control  # noqa: E402
from argus.evaluate.truth import Truth  # noqa: E402
from argus.paths import RunPaths  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/demo")
    ap.add_argument("--threshold", type=float, default=45.0)
    a = ap.parse_args()
    p = RunPaths.at(a.data)
    if not p.model.exists():
        sys.exit(f"{p.model} missing - run the pipeline on {a.data} first")
    clean = build_clean_control(p.root, p.model)
    con = connect(RunPaths.at(clean).db, read_only=True)
    print(json.dumps(clean_control_fpr(con, Truth.load(clean), a.threshold), indent=2))
    top = con.execute("SELECT rank, display_name, risk_score, rule_score, ml_score, risk_driver, top_reason FROM risk_scores ORDER BY rank LIMIT 10").df()
    print("\nHighest-scoring entities in the clean world (all benign):")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
