"""Evaluation layer: report structure and basic sanity of benchmark metrics."""

from __future__ import annotations

from pathlib import Path

from argus.evaluate import evaluate_run
from argus.evaluate.metrics import clean_control_fpr
from argus.evaluate.truth import Truth


def test_evaluate_run_writes_reports(scored: Path) -> None:
    r = evaluate_run(scored)
    assert (scored / "reports" / "evaluation.md").exists() and (scored / "reports" / "evaluation.json").exists()
    er = r["entity_resolution"]["person_records_excl_burners"]
    assert 0.9 <= er["pairwise_precision"] <= 1.0
    d = r["detection"]
    assert d["rings_total"] == 7 and 0 <= d["rings_detected"] <= 7
    assert d["risk_score"]["roc_auc"] > 0.8
    assert set(d["decoys"]) == {"cash_business", "freelancer", "landlord", "night_shift", "rosca"}
    assert "split_records_by_cause" in r["entity_resolution"]["error_analysis"]


def test_clean_control_metric_api(con, scored: Path) -> None:  # type: ignore[no-untyped-def]
    c = clean_control_fpr(con, Truth.load(scored), alert_threshold=45.0)
    assert 0 <= c["entity_false_positive_rate"] <= 1 and c["persons"] > 0
