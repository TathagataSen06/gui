"""Source masking and review-status regression tests."""

from datetime import date

import duckdb
import pandas as pd
import pytest

from argus.evaluate.outage import SourceOutage, audit_decisions, remove_events
from argus.score.snapshot import Window


def test_outage_masks_one_source_inside_local_date_bounds() -> None:
    outage = SourceOutage("bank", Window(date(2026, 3, 8), date(2026, 3, 9)))
    world = Window(date(2026, 3, 1), date(2026, 4, 1))
    outage.validate(world)
    assert not outage.overlaps(Window(world.start, outage.window.start))
    assert not outage.overlaps(Window(outage.window.end, world.end))
    assert outage.overlaps(world)
    with pytest.raises(ValueError, match="within the world"):
        outage.validate(Window(outage.window.end, world.end))
    with duckdb.connect() as con:
        con.execute("CREATE TABLE manifest AS SELECT 'timezone' AS key, 'America/New_York' AS value")
        con.execute("CREATE TABLE events (event_id VARCHAR, source VARCHAR, ts TIMESTAMP)")
        con.executemany("INSERT INTO events VALUES (?, ?, ?)", [
            ("before", "bank", "2026-03-08 04:59:59"),
            ("start", "bank", "2026-03-08 05:00:00"),
            ("last", "bank", "2026-03-09 03:59:59"),
            ("end", "bank", "2026-03-09 04:00:00"),
            ("telco", "telco", "2026-03-08 05:00:00"),
        ])
        con.execute("CREATE TABLE event_actors AS SELECT event_id FROM events")
        assert remove_events(con, outage) == 2
        for table in ("events", "event_actors"):
            assert con.execute(f"SELECT event_id FROM {table} ORDER BY event_id").fetchall() == [("before",), ("end",), ("telco",)]


def test_outage_audit_distinguishes_threshold_activity_budget_and_prior_reviews() -> None:
    scores = pd.Series({"selected": 70, "capacity": 50, "low": 40, "missing": 55, "prior": 60})
    hits = pd.Series({eid: [{"rule_id": "R02"}] for eid in scores.index})
    eligible = set(scores.index) - {"missing"}
    out = audit_decisions(scores.index.tolist(), scores.loc[["selected", "capacity"]], scores, hits, eligible, {"prior"}, 1)
    assert {eid: d["status"] for eid, d in out.items()} == {
        "selected": "selected", "capacity": "over_budget", "low": "below_threshold", "missing": "no_activity", "prior": "already_reviewed",
    }
    assert out["capacity"]["rank"] == 2 and out["missing"]["score"] is None
