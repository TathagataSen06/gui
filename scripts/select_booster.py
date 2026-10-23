"""How Argus's gradient-boosting configuration was chosen, on synthetic data only.

The previous booster (``GradientBoostingClassifier``, no leaf penalty) diverged
on the real ULB credit-card data. To avoid tuning on that data, the replacement
was chosen with two synthetic checks and the rule below, and the card data was
only scored afterwards.

1. **Stability gate.** ``make_classification`` with 200,000 rows and 0.17%
   positives, three seeds. A setting passes if the largest absolute raw score
   over all boosting stages stays below 50 and the final training log-loss is
   not above the first stage's.
2. **Accuracy.** Out-of-fold average precision of the GBM under Argus's grouped
   cross-validation, on development worlds whose seeds are not benchmark seeds:
   320 people / seed 11 / 120 days (the test world), and 1,500 people with
   seeds 99 and 123.

Rule: among settings that pass the gate, take the highest mean development
average precision. (The rule was first written as "among settings with an L2
penalty"; it was widened to "settings that pass the gate" before any card
result was seen, because the gate tests stability directly.)

Usage::

    python scripts/select_booster.py --work data/select   # builds 3 worlds (~5 min), then ~5 min of fits

Set ``OMP_NUM_THREADS=1`` when other jobs share the machine: the histogram
booster's threads otherwise oversubscribe the CPU on these small fits.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argus.score.model as M  # noqa: E402
from argus.db import connect  # noqa: E402
from argus.generate.config import GeneratorConfig  # noqa: E402
from argus.generate.world import generate_world  # noqa: E402
from argus.graph import run as run_graph  # noqa: E402
from argus.ingest import ingest  # noqa: E402
from argus.resolve import run as run_resolve  # noqa: E402
from argus.score.features import build_features, calls_with_flagged, labels  # noqa: E402
from argus.score.pipeline import ScoreConfig  # noqa: E402
from argus.score.rules import apply_rules  # noqa: E402


def _hgb(l2: float, lr: float = 0.05) -> Callable[[int], Any]:
    return lambda s: HistGradientBoostingClassifier(
        max_iter=250, max_depth=3, learning_rate=lr, min_samples_leaf=5, l2_regularization=l2, early_stopping=False, random_state=s
    )


CANDIDATES: dict[str, Callable[[int], Any]] = {
    "previous (GradientBoostingClassifier)": lambda s: GradientBoostingClassifier(
        n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8, min_samples_leaf=5, random_state=s
    ),
    "hist, l2=0": _hgb(0.0),
    "hist, l2=0.1": _hgb(0.1),
    "hist, l2=0.3": _hgb(0.3),
    "hist, l2=1": _hgb(1.0),
    "hist, l2=1, lr=0.1": _hgb(1.0, 0.1),
}
DEV_WORLDS = {"seed11_320": dict(n_people=320, seed=11, days=120), "seed99_1500": dict(n_people=1500, seed=99), "seed123_1500": dict(n_people=1500, seed=123)}


def stability(make: Callable[[int], Any]) -> dict[str, Any]:
    runs = []
    for seed in range(3):
        X, y = make_classification(n_samples=200_000, n_features=30, n_informative=12, n_redundant=6, weights=[0.9983], flip_y=0.0005, class_sep=1.5, random_state=seed)
        m = make(seed).fit(X, y)
        raws, losses = [], []
        for d in m.staged_decision_function(X):
            d = d.ravel()
            raws.append(float(np.abs(d).max()))
            losses.append(float(log_loss(y, 1 / (1 + np.exp(-np.clip(d, -700, 700))), labels=[0, 1])))
        runs.append({"max_abs_raw": max(raws), "loss_first": losses[0], "loss_last": losses[-1]})
    return {"pass": all(r["max_abs_raw"] < 50 and r["loss_last"] <= r["loss_first"] for r in runs), "runs": runs}


def dev_world(work: Path, name: str) -> Path:
    root = work / name
    if not (root / "argus.duckdb").exists():
        generate_world(GeneratorConfig(**DEV_WORLDS[name]), root)
        ingest(root)
        run_resolve(root)
        run_graph(root)
    return root


def dev_ap(make: Callable[[int], Any], root: Path) -> float:
    con = connect(root / "argus.duckdb", read_only=True)
    try:
        cfg = ScoreConfig()
        X = build_features(con, cfg.features)
        rule1, _ = apply_rules(X)
        flagged = set(rule1[rule1 >= cfg.first_pass_flag].index) | set(X.index[X["watchlist_hit"] > 0])
        X["calls_with_flagged"] = calls_with_flagged(con, flagged).reindex(X.index).fillna(0.0)
        y, labelable = labels(con, X.index)
        groups = pd.Series(dict(con.execute("SELECT node_id, community_id FROM graph_nodes").fetchall())).reindex(X.index)
    finally:
        con.close()
    orig_make, orig_pi = M.make_models, M.permutation_importance
    M.make_models = lambda s: {"logreg": orig_make(s)["logreg"], "gbm": make(s)}
    M.permutation_importance = lambda est, X, y, **k: types.SimpleNamespace(importances_mean=np.zeros(X.shape[1]))  # does not affect AP
    try:
        return float(M.fit_and_score(X, y, labelable, groups, M.ModelConfig(model="gbm")).metrics["gbm"]["average_precision"])
    finally:
        M.make_models, M.permutation_importance = orig_make, orig_pi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="data/select", help="where the development worlds are built")
    a = ap.parse_args()
    roots = {n: dev_world(Path(a.work), n) for n in DEV_WORLDS}
    out: dict[str, Any] = {}
    for name, make in CANDIDATES.items():
        st = stability(make)
        aps = {w: round(dev_ap(make, r), 4) for w, r in roots.items()}
        out[name] = {"stable": st["pass"], "max_abs_raw": [round(r["max_abs_raw"], 1) for r in st["runs"]], "dev_ap": aps, "mean_dev_ap": round(float(np.mean(list(aps.values()))), 4)}
        print(name, json.dumps(out[name]), flush=True)
    chosen = max((n for n in out if out[n]["stable"]), key=lambda n: out[n]["mean_dev_ap"])
    print(json.dumps({"chosen": chosen}, indent=1))


if __name__ == "__main__":
    main()
