"""Why Argus's gradient boosting is unstable on the ULB credit-card data.

Refits the GBM from :func:`argus.score.model.make_models` on one seed of the
random 60/40 split used by ``argus realworld`` and prints, per boosting stage,
test and train ROC-AUC and the range of raw decision values, plus the training
loss trajectory. A healthy fit has bounded raw scores and a falling training
loss; a diverging one does not.

Usage::

    argus realworld fetch --dataset creditcard
    python scripts/diagnose_gbm_creditcard.py --seed 4    # ~7 min single-core
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from argus.realworld.creditcard import FEATURES, load  # noqa: E402
from argus.score.model import make_models  # noqa: E402

STAGES = (1, 5, 10, 25, 50, 100, 150, 200, 250)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=4, help="split and model seed (argus realworld uses 0-4)")
    ap.add_argument("--data", default="data/external/creditcard/creditcard.csv")
    a = ap.parse_args()
    df = load(a.data)
    tr, te = train_test_split(df, train_size=0.6, stratify=df["Class"], random_state=a.seed)
    t = time.perf_counter()
    m = clone(make_models(a.seed)["gbm"]).fit(tr[FEATURES], tr["Class"])
    out: dict = {"seed": a.seed, "fit_seconds": round(time.perf_counter() - t, 1), "stages": {}}
    y_te, y_tr = te["Class"].to_numpy(), tr["Class"].to_numpy()
    for i, (d_te, d_tr) in enumerate(zip(m.staged_decision_function(te[FEATURES]), m.staged_decision_function(tr[FEATURES])), 1):
        if i in STAGES:
            out["stages"][i] = {
                "test_roc_auc": round(roc_auc_score(y_te, d_te.ravel()), 4),
                "test_auprc": round(average_precision_score(y_te, d_te.ravel()), 4),
                "train_roc_auc": round(roc_auc_score(y_tr, d_tr.ravel()), 4),
                "raw_min": float(d_te.min()),
                "raw_max": float(d_te.max()),
            }
    out["train_loss_first"] = float(m.train_score_[0])
    out["train_loss_last"] = float(m.train_score_[-1])
    out["train_loss_lowest_at_stage"] = int(np.argmin(m.train_score_)) + 1
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
