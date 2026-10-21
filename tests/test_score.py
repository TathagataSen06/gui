"""Risk scoring: rules, models, explanations, persisted scores."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from argus.score.features import FEATURE_DESCRIPTIONS
from argus.score.model import ModelConfig, explain_logreg, fit_and_score, transform
from argus.score.rules import RULES, apply_rules


def _blank(n: int = 1) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=[f"E{i}" for i in range(n)], columns=list(FEATURE_DESCRIPTIONS) + ["calls_with_flagged"]).loc[:, lambda d: ~d.columns.duplicated()]


def test_rules_fire_with_readable_reasons() -> None:
    X = _blank(2)
    X.loc["E0", ["sub_threshold_cash", "sub_threshold_branches", "cash_deposit_total"]] = [5, 3, 46_500]
    X.loc["E0", ["burner_contacts", "pre_cash_contacts"]] = [12, 4]
    score, hits = apply_rules(X)
    ids = {h["rule_id"] for h in hits["E0"]}
    assert {"R01", "R10", "R11"} <= ids
    assert score["E0"] == sum(h["points"] for h in hits["E0"])
    assert "5 cash deposits between $8,000 and $9,999 across 3 branch" in next(h["reason"] for h in hits["E0"] if h["rule_id"] == "R01")
    assert score["E1"] == 0 and hits["E1"] == []


def test_rule_score_is_capped() -> None:
    X = _blank(1)
    X.iloc[0] = 1e6  # every indicator at an extreme value
    X["round_amount_share"] = 1.0
    score, hits = apply_rules(X)
    assert score.iloc[0] == 100 and len(hits.iloc[0]) == len(RULES)


def test_model_learns_and_explains() -> None:
    rng = np.random.default_rng(0)
    n = 400
    X = _blank(n)
    y = pd.Series((rng.random(n) < 0.15).astype(int), index=X.index)
    X["burner_contacts"] = rng.poisson(1, n) + y * rng.poisson(10, n)
    X["sub_threshold_cash"] = y * rng.poisson(3, n)
    X["inflow_total"] = rng.lognormal(10, 1, n)
    groups = pd.Series([f"C{i % 40}" for i in range(n)], index=X.index)
    res = fit_and_score(X, y, pd.Series(True, index=X.index), groups, ModelConfig(seed=1))
    assert res.metrics["logreg"]["roc_auc"] > 0.9 and res.metrics["gbm"]["roc_auc"] > 0.9
    assert set(res.oof.index) == set(X.index)  # every labelled entity has an out-of-fold score
    top = res.importances.iloc[0]["feature"]
    assert top in {"burner_contacts", "sub_threshold_cash"}


def test_logreg_contributions_reconstruct_logit() -> None:
    rng = np.random.default_rng(2)
    X = _blank(200)
    X["fan_in"] = rng.poisson(3, 200)
    y = pd.Series((X["fan_in"] > 4).astype(int))
    from argus.score.model import make_models

    m = make_models(0)["logreg"].fit(transform(X).values, y.values)
    contrib = explain_logreg(m, transform(X))
    p = m.predict_proba(transform(X).values)[:, 1]
    logit = np.log(p / (1 - p))
    assert np.allclose(contrib.sum(axis=1).values + m.named_steps["clf"].intercept_[0], logit, atol=1e-6)


def test_scores_are_explained_and_ranked(con) -> None:  # type: ignore[no-untyped-def]
    df = con.execute("SELECT rank, risk_score, rule_score, ml_score, alert, rule_hits, ml_drivers, narrative FROM risk_scores ORDER BY rank").df()
    assert df["risk_score"].is_monotonic_decreasing
    assert df["risk_score"].between(0, 100).all() and df["ml_score"].between(0, 1).all()
    alerts = df[df["alert"]]
    assert len(alerts) > 0
    for hits, drivers, narrative in zip(alerts["rule_hits"], alerts["ml_drivers"], alerts["narrative"]):
        assert json.loads(hits) or json.loads(drivers)  # every alert lists what drove it
        assert "Recommended action" in narrative


def test_planted_members_rank_high(con, scored) -> None:  # type: ignore[no-untyped-def]
    from argus.evaluate.truth import Truth, entity_truth

    t = entity_truth(con, Truth.load(scored))
    top = con.execute("SELECT entity_id FROM risk_scores ORDER BY rank LIMIT 25").df()["entity_id"]
    assert t.loc[top, "suspicious"].mean() >= 0.6
