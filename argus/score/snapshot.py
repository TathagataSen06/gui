"""Point-in-time feature snapshots.

A snapshot is the warehouse as it would have looked if only the events in
``[start, end)`` had happened. The graph stage (projection, centrality,
communities, money cycles) is re-run on those events and the feature matrix is
built from the result, so no feature can see activity outside the window.
Resolved entities, identity records, account reference data and the watch list
are kept; in the generated worlds the watch list predates the window.

Point-in-time scoring uses two snapshots of equal length: one ending when the
first case was opened (training: labels come from the cases opened afterwards)
and one ending at the end of the window (scoring: what an analyst sees today).
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

from ..db import connect
from ..graph.build import GraphConfig
from ..graph.pipeline import run_graph
from .features import FeatureConfig, build_features, calls_with_flagged
from .rules import apply_rules

MIN_WINDOW_DAYS = 28


@dataclass(frozen=True)
class Window:
    start: date
    end: date  # exclusive

    @property
    def days(self) -> int:
        return (self.end - self.start).days

    def to_dict(self) -> dict[str, str | int]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat(), "days": self.days}


def world_window(con: duckdb.DuckDBPyConnection) -> Window:
    m = dict(con.execute("SELECT key, value FROM manifest").fetchall())
    return Window(date.fromisoformat(m["window_start"]), date.fromisoformat(m["window_end"]))


def training_plan(con: duckdb.DuckDBPyConnection) -> tuple[Window, Window]:
    """(train, score) windows: train ends when the first case opens; score is the latest window of equal length."""
    w = world_window(con)
    first_case = con.execute("SELECT min(opened_date) FROM case_labels").fetchone()[0]
    if first_case is None:
        raise ValueError("point-in-time training needs dated case outcomes")
    train = Window(w.start, min(first_case, w.end))
    if train.days < MIN_WINDOW_DAYS:
        raise ValueError(f"only {train.days} days of history before the first case opened; need {MIN_WINDOW_DAYS}")
    return train, scoring_window(con, train.days)


def scoring_window(con: duckdb.DuckDBPyConnection, days: int) -> Window:
    w = world_window(con)
    return Window(max(w.start, w.end - timedelta(days=days)), w.end)


def _timezone(con: duckdb.DuckDBPyConnection) -> str:
    row = con.execute("SELECT value FROM manifest WHERE key = 'timezone'").fetchone()
    return row[0] if row else "UTC"


def _utc_midnight(d: date, tz: str) -> datetime:
    """Local midnight of ``d`` as a naive UTC timestamp (event ``ts`` is stored in UTC; window dates are local)."""
    return datetime.combine(d, datetime.min.time(), tzinfo=ZoneInfo(tz)).astimezone(timezone.utc).replace(tzinfo=None)


def features_with_contacts(con: duckdb.DuckDBPyConnection, features: FeatureConfig, first_pass_flag: float) -> pd.DataFrame:
    """Feature matrix plus the two-pass "calls with high-risk entities" feature."""
    X = build_features(con, features)
    rule1, _ = apply_rules(X)
    flagged = set(rule1[rule1 >= first_pass_flag].index) | set(X.index[X["watchlist_hit"] > 0])
    X["calls_with_flagged"] = calls_with_flagged(con, flagged).reindex(X.index).fillna(0.0)
    return X


def snapshot_features(
    con: duckdb.DuckDBPyConnection, db_path: Path, window: Window, features: FeatureConfig, first_pass_flag: float
) -> tuple[pd.DataFrame, pd.Series]:
    """Features and Louvain communities computed from the events inside ``window`` only.

    ``con`` is the open warehouse connection; it is checkpointed so the copy is complete.
    """
    con.execute("CHECKPOINT")
    with tempfile.TemporaryDirectory(prefix="argus-snapshot-", dir=db_path.parent) as tmp:
        copy = Path(tmp) / "snapshot.duckdb"
        shutil.copyfile(db_path, copy)
        scon = connect(copy)
        try:
            lo, hi = (_utc_midnight(d, _timezone(scon)) for d in (window.start, window.end))
            scon.execute("DELETE FROM events WHERE ts < ? OR ts >= ?", [lo, hi])
            scon.execute("DELETE FROM event_actors WHERE event_id NOT IN (SELECT event_id FROM events)")
            scon.execute("UPDATE manifest SET value = ? WHERE key = 'window_start'", [window.start.isoformat()])
            scon.execute("UPDATE manifest SET value = ? WHERE key = 'window_end'", [window.end.isoformat()])
            run_graph(scon, GraphConfig())
            X = features_with_contacts(scon, features, first_pass_flag)
            groups = pd.Series(dict(scon.execute("SELECT node_id, community_id FROM graph_nodes").fetchall())).reindex(X.index)
        finally:
            scon.close()
    return X, groups
