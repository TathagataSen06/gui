"""Supervised risk models trained on historical case outcomes.

Design choices that matter for an honest benchmark:

* **Grouped cross-validation.** Members of one ring resemble each other and
  tend to sit in the same graph community. Folds are grouped by Louvain
  community so that a community is never split between train and test, which
  keeps most of a ring on one side (communities can still split a ring, see
  ``docs/FAILURE_MODES.md``).
* **Out-of-fold scores.** The score shown for a labelled entity is always the
  out-of-fold prediction, i.e. from a model that never saw that entity's label.
* **Explanations.** Logistic regression contributions are exact
  (``coef * standardized value`` in log-odds). Gradient boosting contributions
  use per-feature occlusion (replace a feature with its median and measure the
  change in log-odds), which is model-agnostic and deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOG_FEATURES = (
    "inflow_total", "outflow_total", "fan_in", "fan_out", "pass_through_amount", "large_round_transfers",
    "sub_threshold_cash", "cash_deposit_total", "wire_count", "wire_volume", "quasi_cash_spend",
    "young_account_inflow", "money_cycles", "cycle_volume", "call_partners", "calls_with_watchlisted",
    "calls_with_flagged", "burner_contacts", "pre_txn_coordination", "pre_cash_contacts",
    "after_hours_restricted", "access_denied", "after_hours_coloc", "exfil_emails", "degree", "strength",
    "controls_red_flag_orgs", "sub_threshold_branches",
)


@dataclass
class ModelConfig:
    n_splits: int = 5
    seed: int = 0
    model: str = "auto"  # "logreg" | "gbm" | "auto" (pick best out-of-fold average precision)


@dataclass
class ModelResult:
    chosen: str
    estimator: Any
    oof: pd.Series  # out-of-fold probability for labelled entities
    prob: pd.Series  # final probability for every entity
    contributions: pd.DataFrame  # per-entity, per-feature log-odds contributions
    metrics: dict[str, Any] = field(default_factory=dict)
    importances: pd.DataFrame | None = None


def transform(X: pd.DataFrame) -> pd.DataFrame:
    Xt = X.copy()
    for c in LOG_FEATURES:
        if c in Xt:
            Xt[c] = np.log1p(Xt[c].clip(lower=0))
    for c in ("betweenness", "pagerank"):
        if c in Xt:
            Xt[c] = np.log1p(Xt[c] * 1e4)
    return Xt


def make_models(seed: int) -> dict[str, Any]:
    return {
        "logreg": Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(C=0.3, max_iter=5000))]),
        "gbm": GradientBoostingClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8, min_samples_leaf=5, random_state=seed),
    }


def _folds(y: pd.Series, groups: pd.Series, cfg: ModelConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    n_pos = int(y.sum())
    k = max(2, min(cfg.n_splits, n_pos))
    try:
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=cfg.seed)
        folds = list(splitter.split(np.zeros(len(y)), y.values, groups.values))
        if all(y.values[te].sum() > 0 and y.values[tr].sum() > 0 for tr, te in folds):
            return folds
    except ValueError:
        pass
    return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg.seed).split(np.zeros(len(y)), y.values))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def precision_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    if k <= 0:
        return 0.0
    idx = np.argsort(-s, kind="mergesort")[:k]
    return float(y[idx].mean())


def ranking_metrics(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    k = int(y.sum())
    if k == 0 or k == len(y):
        return {"roc_auc": float("nan"), "average_precision": float("nan"), "precision_at_k": float("nan"), "k": k}
    return {
        "roc_auc": round(float(roc_auc_score(y, s)), 4),
        "average_precision": round(float(average_precision_score(y, s)), 4),
        "precision_at_k": round(precision_at_k(y, s, k), 4),
        "k": k,
    }


def explain_logreg(model: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    scaler: StandardScaler = model.named_steps["scale"]
    clf: LogisticRegression = model.named_steps["clf"]
    z = (X.values - scaler.mean_) / np.where(scaler.scale_ == 0, 1, scaler.scale_)
    return pd.DataFrame(z * clf.coef_[0], index=X.index, columns=X.columns)


def explain_occlusion(model: Any, X: pd.DataFrame, reference: pd.Series) -> pd.DataFrame:
    base = _logit(model.predict_proba(X.values)[:, 1])
    out = np.zeros(X.shape)
    for j, c in enumerate(X.columns):
        Xo = X.values.copy()
        Xo[:, j] = reference[c]
        out[:, j] = base - _logit(model.predict_proba(Xo)[:, 1])
    return pd.DataFrame(out, index=X.index, columns=X.columns)


def fit_and_score(X: pd.DataFrame, y: pd.Series, labelable: pd.Series, groups: pd.Series, cfg: ModelConfig | None = None, rule_score: pd.Series | None = None) -> ModelResult:
    """Cross-validate both models, choose one, and score every entity."""
    cfg = cfg or ModelConfig()
    Xt = transform(X)
    XL, yL, gL = Xt[labelable], y[labelable], groups[labelable].fillna("none")
    folds = _folds(yL, gL, cfg)
    models = make_models(cfg.seed)
    oof: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {"labelled_entities": int(len(yL)), "positives": int(yL.sum()), "folds": len(folds), "grouping": "louvain_community"}
    perm: dict[str, np.ndarray] = {}
    for name, m in models.items():
        pred = np.zeros(len(yL))
        pi = np.zeros(XL.shape[1])
        for tr, te in folds:
            mm = clone(m)
            mm.fit(XL.values[tr], yL.values[tr])
            pred[te] = mm.predict_proba(XL.values[te])[:, 1]
            # held-out permutation importance (drop in average precision on the test fold)
            r = permutation_importance(mm, XL.values[te], yL.values[te], scoring="average_precision", n_repeats=3, random_state=cfg.seed)
            pi += r.importances_mean / len(folds)
        oof[name] = pred
        perm[name] = pi
        metrics[name] = ranking_metrics(yL.values, pred)
    if rule_score is not None:
        metrics["rules"] = ranking_metrics(yL.values, rule_score[labelable].values)
    chosen = cfg.model if cfg.model in models else max(models, key=lambda n: (np.nan_to_num(metrics[n]["average_precision"]), n == "logreg"))
    final = clone(models[chosen]).fit(XL.values, yL.values)
    prob = pd.Series(final.predict_proba(Xt.values)[:, 1], index=Xt.index)
    oof_s = pd.Series(oof[chosen], index=XL.index)
    prob.loc[oof_s.index] = oof_s.values
    if chosen == "logreg":
        contrib = explain_logreg(final, Xt)
        imp = pd.DataFrame({"feature": Xt.columns, "coefficient": final.named_steps["clf"].coef_[0]})
        imp["importance"] = imp["coefficient"].abs()
    else:
        contrib = explain_occlusion(final, Xt, XL.median())
        imp = pd.DataFrame({"feature": Xt.columns, "importance": final.feature_importances_})
    imp["heldout_permutation_importance"] = perm[chosen]
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    metrics["chosen_model"] = chosen
    return ModelResult(chosen, final, oof_s, prob, contrib, metrics, imp)


def score_with_pretrained(X: pd.DataFrame, bundle: dict[str, Any]) -> ModelResult:
    """Apply a model trained on another world (used for the clean false-positive control)."""
    Xt = transform(X)[bundle["columns"]]
    est = bundle["estimator"]
    prob = pd.Series(est.predict_proba(Xt.values)[:, 1], index=Xt.index)
    if bundle["chosen"] == "logreg":
        contrib = explain_logreg(est, Xt)
    else:
        contrib = explain_occlusion(est, Xt, bundle["reference"])
    return ModelResult(bundle["chosen"], est, pd.Series(dtype=float), prob, contrib, {"pretrained": True, "chosen_model": bundle["chosen"]}, None)


def model_bundle(res: ModelResult, X: pd.DataFrame, labelable: pd.Series) -> dict[str, Any]:
    Xt = transform(X)
    return {"estimator": res.estimator, "chosen": res.chosen, "columns": list(Xt.columns), "reference": Xt[labelable].median(), "metrics": res.metrics}
