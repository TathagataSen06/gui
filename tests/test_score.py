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


def test_blend_candidate_and_later_window_predictions() -> None:
    rng = np.random.default_rng(3)
    n = 300
    X = _blank(n)
    y = pd.Series((rng.random(n) < 0.15).astype(int), index=X.index)
    X["burner_contacts"] = rng.poisson(1, n) + y * rng.poisson(8, n)
    X["inflow_total"] = rng.lognormal(10, 1, n)
    groups = pd.Series([f"C{i % 30}" for i in range(n)], index=X.index)
    lab = pd.Series(True, index=X.index)
    base = fit_and_score(X, y, lab, groups, ModelConfig(seed=1))
    assert {"logreg", "gbm", "blend"} <= set(base.metrics)
    same = fit_and_score(X, y, lab, groups, ModelConfig(seed=1), X_eval=X)
    assert np.allclose(base.prob.values, same.prob.values)  # predicting on the training window changes nothing
    later = X.assign(burner_contacts=rng.poisson(1, n))  # the signal is gone in the later window
    shifted = fit_and_score(X, y, lab, groups, ModelConfig(seed=1), X_eval=later)
    assert shifted.metrics[shifted.chosen]["average_precision"] < base.metrics[base.chosen]["average_precision"] - 0.3


def test_full_window_snapshot_reproduces_features(scored, tmp_path) -> None:  # type: ignore[no-untyped-def]
    import shutil
    from datetime import timedelta

    from argus.db import connect
    from argus.paths import RunPaths
    from argus.score.features import FeatureConfig
    from argus.score.snapshot import Window, features_with_contacts, snapshot_features, world_window

    shutil.copytree(scored, tmp_path / "world", ignore=shutil.ignore_patterns("*.wal"))  # other tests hold the shared one open
    db = RunPaths.at(tmp_path / "world").db
    con = connect(db)
    try:
        full = features_with_contacts(con, FeatureConfig(), 40.0)
        w = world_window(con)
        snap, groups = snapshot_features(con, db, w, FeatureConfig(), 40.0)
        early, _ = snapshot_features(con, db, Window(w.start, w.start + timedelta(days=30)), FeatureConfig(), 40.0)
    finally:
        con.close()
    pd.testing.assert_frame_equal(full, snap, check_exact=False, rtol=1e-9)
    assert groups.notna().any()
    assert early["inflow_total"].sum() < 0.5 * full["inflow_total"].sum()
    assert not list(db.parent.glob("argus-snapshot-*"))  # temporary copies are removed


def test_point_in_time_scoring_end_to_end(scored, tmp_path) -> None:  # type: ignore[no-untyped-def]
    import shutil

    import joblib

    from argus.evaluate import evaluate_run
    from argus.paths import RunPaths
    from argus.score import run
    from argus.score.pipeline import ScoreConfig

    root = tmp_path / "world"
    shutil.copytree(scored, root, ignore=shutil.ignore_patterns("*.wal"))
    st = run(root, ScoreConfig(point_in_time=True))
    pit = st["point_in_time"]
    assert pit["train"]["end"] == pit["score"]["start"] and pit["train"]["days"] == pit["score"]["days"] >= 28
    assert st["metrics"]["evaluated_on"].startswith("later window")
    bundle = joblib.load(RunPaths.at(root).model)
    assert bundle["point_in_time"]["window_days"] == pit["train"]["days"]
    d = evaluate_run(root)["detection"]
    assert d["scoring_window_days"][1] - d["scoring_window_days"][0] == pit["score"]["days"]
    assert 0 <= d["rings_active_in_scoring_window"] <= d["rings_total"]
    again = run(root, pretrained=RunPaths.at(root).model)  # a point-in-time model scores the latest window
    assert again["point_in_time"]["score"] == pit["score"]
