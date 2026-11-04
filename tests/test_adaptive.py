"""Regression tests for personal baselines and delayed review feedback."""

from datetime import date, timedelta

import duckdb
import pandas as pd

from argus.evaluate.adaptive import SIGNALS, AdaptiveTriage
from argus.score.snapshot import Window

START = date(2026, 1, 1)


def _day(n: int) -> date:
    return START + timedelta(days=n)


def _init_cases(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE TABLE record_entity (record_id VARCHAR, entity_id VARCHAR)")
    con.execute("CREATE TABLE case_labels (case_id VARCHAR, subject_record_id VARCHAR, disposition VARCHAR, opened_date DATE, closed_date DATE)")


def _score(learner, con, end, values, rule_ids, eligible=None):
    X = pd.DataFrame(0.0, index=list(values), columns=list(SIGNALS))
    X["after_hours_restricted"] = pd.Series(values, dtype=float)
    eligible = set(values) if eligible is None else eligible
    hits = pd.Series({eid: [{"rule_id": rid} for rid in rule_ids.get(eid, [])] for eid in X.index})
    rule = pd.Series({eid: 45.0 for eid in X.index})
    return learner.score(con, Window(_day(end - 10), _day(end)), X, eligible, rule, hits)


def test_personal_change_uses_disjoint_prior_activity_and_direct_evidence() -> None:
    learner = AdaptiveTriage(lookback=10, step=5)
    support = {eid: ["R12"] for eid in ("stable", "changed", "sparse")}
    with duckdb.connect() as con:
        _init_cases(con)
        initial = {"stable": 10, "changed": 0, "sparse": 0, "unconfirmed": 0}
        _score(learner, con, 10, initial, support, set(initial) - {"sparse"})
        _score(learner, con, 15, initial, support)
        _, changes, stats = _score(learner, con, 20, initial, support)
        assert stats["profiled_entities"] == 0 and changes == {}
        _score(learner, con, 25, initial, support)
        scores, changes, stats = _score(learner, con, 30,
                                         {"stable": 11, "changed": 8, "sparse": 8, "unconfirmed": 8}, support)
        assert [w["end"] for w in stats["baseline_windows"]] == [_day(10).isoformat(), _day(20).isoformat()]
        assert stats["profiled_entities"] == 3
        assert scores.to_dict() == {"stable": 45, "changed": 51, "sparse": 45, "unconfirmed": 45}
        assert changes["changed"][0]["typical"] == 0 and changes["changed"][0]["observed"] == 8
        assert changes["changed"][0]["supporting_rule"] == "R12"
        for end in (40, 50, 60, 70, 80):
            scores, changes, _ = _score(learner, con, end, {eid: 10 for eid in initial}, support)
        assert scores["changed"] == 45 and "changed" not in changes


def test_case_feedback_waits_for_explicit_outcomes_and_uses_pre_case_evidence() -> None:
    learner = AdaptiveTriage(lookback=10, step=10)
    entities = [f"historical-{i}" for i in range(8)]
    values = dict.fromkeys([*entities, "next"], 0)
    previous = {eid: ["R12" if i < 3 else "R13"] for i, eid in enumerate(entities)}
    later = {eid: ["R05"] for eid in entities} | {"next": ["R12"]}
    with duckdb.connect() as con:
        _init_cases(con)
        con.executemany("INSERT INTO record_entity VALUES (?, ?)", [(eid, eid) for eid in entities])
        con.executemany("INSERT INTO case_labels VALUES (?, ?, ?, ?, ?)", [
            (eid, eid, "CONFIRMED_SUSPICIOUS" if i < 3 or i >= 6 else "CLOSED_NO_ACTION",
             _day(9) if i == 6 else _day(10), None if i == 7 else _day(20))
            for i, eid in enumerate(entities)
        ])
        _score(learner, con, 10, values, previous)
        scores, _, stats = _score(learner, con, 20, values, later, {"next"})
        assert scores["next"] == 45 and stats["rule_adjustments"] == {}
        assert stats["positive_feedback"] == stats["negative_feedback"] == 0
        scores, _, stats = _score(learner, con, 30, values, later, {"next"})
        assert stats["positive_feedback"] == stats["negative_feedback"] == 3
        assert stats["rule_adjustments"] == {"R12": 1.67, "R13": -1.67}
        assert scores["next"] == 46.67
        scores, _, stats = _score(learner, con, 70, values, later, {"next"})
        assert scores["next"] == 45 and stats["rule_adjustments"] == {}
