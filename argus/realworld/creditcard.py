"""Argus's supervised models on the ULB credit-card fraud data.

The data has no cardholder, merchant or device key, so only the transaction
scorer can be validated here: the same logistic regression and gradient
boosting configurations Argus uses on synthetic entities
(:func:`argus.score.model.make_models`), with no tuning for this dataset.

Two protocols, matching the two kinds of published numbers:

* ``out_of_time``: sort by ``Time``, train on the first 80% of rows and test on
  the last 20% (Fraud Dataset Benchmark, Grover et al.). Reported as ROC-AUC and
  AUPRC with bootstrap confidence intervals, plus precision and recall at fixed
  alert budgets.
* ``random_60_40``: stratified random 60% train / 40% test, repeated over five
  seeds (the training-set size used by Liu et al., ICDE 2020). Reported as
  AUPRC mean and standard deviation.

``Time`` is used only to order rows, never as a feature: in an out-of-time
split it would only teach the model which hour of the two days it is.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from ..score.model import make_models

N_ROWS = 284_807
N_FRAUD = 492
FEATURES = [f"V{i}" for i in range(1, 29)] + ["log_amount"]


@dataclass
class CreditCardConfig:
    seed: int = 0
    train_share: float = 0.8
    random_train_share: float = 0.6
    random_repeats: int = 5
    bootstrap: int = 1000
    alert_budgets: tuple[float, ...] = (0.001, 0.005)  # share of test transactions alerted
    models: tuple[str, ...] = ("logreg", "gbm")


def load(path: str | Path, expect: tuple[int, int] | None = (N_ROWS, N_FRAUD)) -> pd.DataFrame:
    """Read ``creditcard.csv`` and check it against the documented row and fraud counts.

    Some mirrors quote the label as ``'0'`` / ``'1'``; it is normalized to int.
    """
    df = pd.read_csv(path)
    missing = {"Time", "Amount", "Class", *FEATURES[:-1]} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    df["Class"] = df["Class"].astype(str).str.strip("'\" ").astype(int)
    if expect is not None and (len(df), int(df["Class"].sum())) != expect:
        raise ValueError(f"{path}: {len(df):,} rows / {int(df['Class'].sum())} frauds, expected {expect[0]:,} / {expect[1]}")
    df["log_amount"] = np.log1p(df["Amount"].clip(lower=0))
    return df


def bootstrap_ci(y: np.ndarray, s: np.ndarray, metric, n: int, seed: int) -> tuple[float, float]:  # type: ignore[no-untyped-def]
    """Percentile 95% interval of ``metric`` over ``n`` bootstrap resamples of the test set."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        vals.append(metric(y[idx], s[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def at_budget(y: np.ndarray, s: np.ndarray, share: float) -> dict[str, float]:
    """Precision and recall if the top ``share`` of transactions by score are alerted."""
    k = max(1, int(round(share * len(y))))
    top = np.argsort(-s, kind="stable")[:k]
    hits = int(y[top].sum())
    return {"alerts": k, "frauds_caught": hits, "precision": hits / k, "recall": hits / max(1, int(y.sum()))}


def out_of_time(df: pd.DataFrame, cfg: CreditCardConfig) -> dict[str, Any]:
    d = df.sort_values("Time", kind="stable").reset_index(drop=True)
    cut = int(cfg.train_share * len(d))
    tr, te = d.iloc[:cut], d.iloc[cut:]
    y_te = te["Class"].to_numpy()
    out: dict[str, Any] = {
        "protocol": f"sorted by Time; first {cfg.train_share:.0%} train, last {1 - cfg.train_share:.0%} test",
        "train_rows": len(tr), "train_frauds": int(tr["Class"].sum()),
        "test_rows": len(te), "test_frauds": int(y_te.sum()),
        "test_prevalence": float(y_te.mean()),
        "models": {},
    }
    for name in cfg.models:
        t = time.perf_counter()
        m = clone(make_models(cfg.seed)[name]).fit(tr[FEATURES], tr["Class"])
        s = m.predict_proba(te[FEATURES])[:, 1]
        out["models"][name] = {
            "roc_auc": float(roc_auc_score(y_te, s)),
            "roc_auc_ci95": bootstrap_ci(y_te, s, roc_auc_score, cfg.bootstrap, cfg.seed),
            "auprc": float(average_precision_score(y_te, s)),
            "auprc_ci95": bootstrap_ci(y_te, s, average_precision_score, cfg.bootstrap, cfg.seed),
            "budgets": {f"{b:.1%}": at_budget(y_te, s, b) for b in cfg.alert_budgets},
            "seconds": round(time.perf_counter() - t, 1),
        }
    return out


def random_split(df: pd.DataFrame, cfg: CreditCardConfig) -> dict[str, Any]:
    res: dict[str, list[float]] = {f"{n}_{k}": [] for n in cfg.models for k in ("auprc", "roc_auc")}
    for r in range(cfg.random_repeats):
        tr, te = train_test_split(df, train_size=cfg.random_train_share, stratify=df["Class"], random_state=cfg.seed + r)
        for name in cfg.models:
            m = clone(make_models(cfg.seed + r)[name]).fit(tr[FEATURES], tr["Class"])
            s = m.predict_proba(te[FEATURES])[:, 1]
            res[f"{name}_auprc"].append(float(average_precision_score(te["Class"], s)))
            res[f"{name}_roc_auc"].append(float(roc_auc_score(te["Class"], s)))
    return {
        "protocol": f"stratified random {cfg.random_train_share:.0%} train / {1 - cfg.random_train_share:.0%} test, {cfg.random_repeats} seeds",
        "train_rows": int(round(cfg.random_train_share * len(df))),
        "models": {
            n: {k: {"mean": float(np.mean(res[f"{n}_{k}"])), "std": float(np.std(res[f"{n}_{k}"])), "values": res[f"{n}_{k}"]} for k in ("auprc", "roc_auc")}
            for n in cfg.models
        },
    }


def run(path: str | Path, cfg: CreditCardConfig | None = None, expect: tuple[int, int] | None = (N_ROWS, N_FRAUD)) -> dict[str, Any]:
    cfg = cfg or CreditCardConfig()
    df = load(path, expect)
    return {
        "dataset": "creditcard",
        "rows": len(df), "frauds": int(df["Class"].sum()), "prevalence": float(df["Class"].mean()),
        "features": FEATURES,
        "out_of_time": out_of_time(df, cfg),
        "random_60_40": random_split(df, cfg),
    }
