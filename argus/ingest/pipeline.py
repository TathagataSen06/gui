"""Run all registered adapters into a DuckDB warehouse."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

from ..db import connect
from ..paths import RunPaths
from ..schema import EVENT_TYPES, WAREHOUSE_DDL
from .base import EVENT_COLUMNS, RECORD_COLUMNS, SourceAdapter, registered_adapters
from .guard import check_manifest, enforce, synthetic_mask_events, synthetic_mask_records


@dataclass
class IngestReport:
    db_path: Path
    sources: dict[str, dict] = field(default_factory=dict)

    @property
    def total_events(self) -> int:
        return sum(s["events_loaded"] for s in self.sources.values())

    @property
    def total_records(self) -> int:
        return sum(s["records_loaded"] for s in self.sources.values())




def _validate_events(df: pd.DataFrame, adapter: SourceAdapter) -> pd.DataFrame:
    missing = set(EVENT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"adapter {adapter.name} produced events without columns {sorted(missing)}")
    if df.empty:
        return df
    ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    ok = ts.notna() & df["event_type"].isin(EVENT_TYPES) & df["actor_ids"].map(lambda a: isinstance(a, list) and len(a) > 0)
    ok &= df["actor_ids"].map(len) == df["actor_roles"].map(len)
    ok &= df["actor_ids"].map(lambda ids: all(":" in i and i.split(":", 1)[1] for i in ids))
    adapter.reject("schema_violation", int((~ok).sum()))
    df = df[ok].copy()
    df["ts"] = ts[ok].dt.tz_convert("UTC").dt.tz_localize(None)
    syn = synthetic_mask_events(df["actor_ids"])
    adapter.reject("non_synthetic_identifier", enforce(syn, f"{adapter.name} events"))
    return df[syn]


def _validate_records(df: pd.DataFrame, adapter: SourceAdapter) -> pd.DataFrame:
    missing = set(RECORD_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"adapter {adapter.name} produced records without columns {sorted(missing)}")
    if df.empty:
        return df
    df = df.drop_duplicates("record_id")
    syn = synthetic_mask_records(df)
    adapter.reject("non_synthetic_identifier", enforce(syn, f"{adapter.name} records"))
    return df[syn]


def ingest(root: str | Path, adapters: Iterable[str] | None = None, local_tz: str | None = None) -> IngestReport:
    """Load every available feed under ``<root>/raw`` into ``<root>/argus.duckdb``.

    The warehouse is rebuilt from scratch on every call (ingest is idempotent).
    """
    paths = RunPaths.at(root)
    manifest = check_manifest(paths.root)
    tz = local_tz or manifest.get("window", {}).get("timezone", "America/New_York")
    if paths.db.exists():
        paths.db.unlink()
    con = connect(paths.db)
    for ddl in WAREHOUSE_DDL:
        con.execute(ddl)
    con.execute("INSERT INTO manifest VALUES ('marker', ?), ('seed', ?), ('window_start', ?), ('window_end', ?), ('timezone', ?)", [manifest["marker"], str(manifest.get("seed")), manifest["window"]["start"], manifest["window"]["end"], tz])

    report = IngestReport(paths.db)
    registry = registered_adapters()
    names = list(adapters) if adapters is not None else list(registry)
    for name in names:
        adapter = registry[name](paths.raw)
        if hasattr(adapter, "local_tz"):
            adapter.local_tz = tz  # type: ignore[attr-defined]
        if not adapter.available():
            continue
        t0 = time.perf_counter()
        n_events = 0
        records_first = not getattr(adapter, "records_after_events", False)
        n_records = _load_records(con, adapter) if records_first else 0
        for batch in adapter.events():
            batch = _validate_events(batch, adapter)
            if batch.empty:
                continue
            con.register("_batch", batch[list(EVENT_COLUMNS)])
            con.execute("INSERT INTO events SELECT event_id, ts, source, event_type, actor_ids, actor_roles, attributes::JSON FROM _batch")
            con.unregister("_batch")
            n_events += len(batch)
        if not records_first:
            n_records = _load_records(con, adapter)
        labels = adapter.labels()
        if labels is not None and not labels.empty:
            con.register("_labels", labels)
            con.execute("INSERT INTO case_labels SELECT * FROM _labels")
            con.unregister("_labels")
        for tname, tdf in getattr(adapter, "reference_tables", lambda: {})().items():
            con.register("_ref", tdf)
            con.execute(f"CREATE OR REPLACE TABLE {tname} AS SELECT * FROM _ref")
            con.unregister("_ref")
        secs = time.perf_counter() - t0
        rejected = sum(adapter.rejects.values())
        con.execute(
            "INSERT INTO ingest_log VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [adapter.name, type(adapter).__name__, adapter.rows_read, n_events, n_records, rejected, json.dumps(adapter.rejects), secs],
        )
        report.sources[adapter.name] = {"rows_read": adapter.rows_read, "events_loaded": n_events, "records_loaded": n_records, "rejected": adapter.rejects, "seconds": round(secs, 2)}

    con.execute(
        """
        INSERT INTO event_actors
        SELECT event_id, UNNEST(actor_ids), UNNEST(actor_roles), UNNEST(generate_series(0, len(actor_ids) - 1))
        FROM events
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS ix_event_actors_actor ON event_actors(actor_id)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_event_actors_event ON event_actors(event_id)")
    con.close()
    return report


def _load_records(con: duckdb.DuckDBPyConnection, adapter: SourceAdapter) -> int:
    recs = _validate_records(adapter.records(), adapter)
    if recs.empty:
        return 0
    recs = recs[list(RECORD_COLUMNS)].copy()
    recs["owned_ids"] = recs["owned_ids"].map(lambda v: list(v) if isinstance(v, (list, tuple)) else [])
    con.register("_recs", recs)
    con.execute("INSERT INTO identity_records SELECT record_id, source, entity_type, name, dob, address, phone, email, owned_ids::VARCHAR[], attributes::JSON FROM _recs")
    con.unregister("_recs")
    return len(recs)
