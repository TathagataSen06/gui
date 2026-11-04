"""Regression tests for event and case cutoffs and the review budget."""

from __future__ import annotations

import json
from datetime import date, timedelta

import duckdb
import pandas as pd

from argus.evaluate.replay import _candidates, _closed_labels, _queue, replay
from argus.score.features import labels


def test_case_dates_control_training_and_queue_eligibility() -> None:
    cutoff = date(2026, 2, 1)
    with duckdb.connect() as con:
        con.execute("CREATE TABLE entities AS SELECT id AS entity_id, 'person' AS entity_type FROM range(6) AS t(id)")
        con.execute("CREATE TABLE identifier_entity AS SELECT entity_id AS identifier, entity_id FROM entities")
        con.execute("CREATE TABLE event_actors AS SELECT identifier AS actor_id FROM identifier_entity WHERE entity_id < 5")
        con.execute("CREATE TABLE record_entity AS SELECT concat('bank:', entity_id) AS record_id, entity_id FROM entities")
        con.execute("CREATE TABLE case_labels (subject_record_id VARCHAR, disposition VARCHAR, opened_date DATE, closed_date DATE)")
        prior = cutoff - timedelta(days=5)
        con.executemany("INSERT INTO case_labels VALUES (?, ?, ?, ?)", [
            ("bank:0", "CONFIRMED_SUSPICIOUS", prior, prior + timedelta(days=1)),
            ("bank:1", "CLOSED_NO_ACTION", prior, prior + timedelta(days=1)),
            ("bank:2", "CONFIRMED_SUSPICIOUS", prior, cutoff),
            ("bank:3", "CONFIRMED_SUSPICIOUS", cutoff, cutoff + timedelta(days=1)),
        ])
        assert _closed_labels(con, prior, cutoff).to_dict() == {0: 1, 1: 0}
        assert _closed_labels(con, prior + timedelta(days=1), cutoff).empty
        assert _candidates(con, cutoff) == {3, 4}
        y, labelable = labels(con, pd.Index(range(6)), as_of=cutoff)
        assert y.to_dict() == {0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        assert labelable.to_dict() == {0: True, 1: True, 2: False, 3: True, 4: True, 5: True}


def test_queue_excludes_reviewed_and_breaks_ties_deterministically() -> None:
    scores = pd.Series({"Z": 50, "B": 70, "A": 70, "Y": 90, "C": 60, "D": 10})
    assert _queue(scores, {"Z", "B", "A", "C", "D"}, {"C"}, 45).index.tolist() == ["A", "B", "Z"]


def test_replay_persists_rounds_with_bounded_non_repeating_reviews(resolved) -> None:
    from argus.db import connect
    from argus.paths import RunPaths

    paths = RunPaths.at(resolved)
    with connect(paths.db, read_only=True) as con:
        report = replay(paths.root, lookback=30, step=30, budget=3, minimum=10, adaptive=True)
        assert con.execute("SELECT value FROM manifest WHERE key = 'window_start'").fetchone() == (report["world"]["start"],)
    assert json.loads((paths.reports / "replay.json").read_text()) == report
    assert (paths.reports / "replay.md").read_text().startswith("# Argus historical triage replay\n")
    assert [r["window"]["days"] for r in report["rounds"]] == [30, 30, 30, 30]
    assert report["rounds"][-1]["window"]["end"] == report["world"]["end"]
    assert report["rounds"][0]["hybrid_mode"] == "rules fallback"
    assert report["rounds"][0]["policies"]["rules"] == report["rounds"][0]["policies"]["hybrid"]
    assert report["rounds"][0]["adaptive"]["profiled_entities"] == 0
    assert report["rounds"][-1]["adaptive"]["profiled_entities"] > 0
    assert set(report["summary"]) == {"rules", "hybrid", "adaptive"}
    assert not list(paths.root.glob("argus-replay-*"))
    for policy, summary in report["summary"].items():
        seen = set()
        for tick in report["rounds"]:
            row = tick["policies"][policy]
            assert len(row["reviews"]) == min(row["candidates"], 3)
            for item in row["reviews"]:
                assert item["entity_id"] not in seen
                if policy == "adaptive":
                    assert "personal_changes" in item
                seen.add(item["entity_id"])
        assert len(seen) == summary["reviews"] == summary["true_positives"] + summary["false_positives"]
