"""Ingestion: common schema, double-entry collapse, synthetic guard, pluggable adapters."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest

from argus.db import connect
from argus.ingest import NonSyntheticDataError, SourceAdapter, ingest, register_adapter
from argus.ingest.base import _REGISTRY
from argus.ingest.dates import infer_format, parse_dates, parse_local_datetimes
from argus.ids import normalize_email, normalize_msisdn


def test_events_follow_common_schema(con) -> None:  # type: ignore[no-untyped-def]
    cols = [r[0] for r in con.execute("DESCRIBE events").fetchall()]
    assert cols[:6] == ["event_id", "ts", "source", "event_type", "actor_ids", "actor_roles"]
    srcs = {r[0] for r in con.execute("SELECT DISTINCT source FROM events").fetchall()}
    assert srcs == {"bank", "telco", "access", "email"}
    bad = con.execute("SELECT count(*) FROM events WHERE ts IS NULL OR len(actor_ids) = 0 OR len(actor_ids) <> len(actor_roles)").fetchone()[0]
    assert bad == 0
    kinds = {r[0] for r in con.execute("SELECT DISTINCT split_part(actor_id, ':', 1) FROM event_actors").fetchall()}
    assert kinds <= {"acct", "ext", "merchant", "branch", "msisdn", "card", "loc", "email"}


def test_double_entry_legs_collapse_to_one_transfer(con, world_dir: Path) -> None:  # type: ignore[no-untyped-def]
    tx = pd.concat(pd.read_csv(f, dtype=str) for f in sorted((world_dir / "raw/bank").glob("*/transactions_*.csv")))
    transfers = tx[tx["channel"].isin(["ACH", "WIRE", "P2P", "BOOK", "CHECK"])]
    n = con.execute("SELECT count(*) FROM events WHERE event_type = 'transfer'").fetchone()[0]
    assert n == transfers["end_to_end_id"].nunique()
    both = con.execute("SELECT count(*) FROM events WHERE event_type = 'transfer' AND CAST(attributes->>'legs' AS INT) = 2").fetchone()[0]
    assert both > 0


def test_timestamps_are_utc(con) -> None:  # type: ignore[no-untyped-def]
    # local wall-clock 03:00-06:00 payroll postings must land at 08:00-11:00 UTC in winter
    lo, hi = con.execute(
        "SELECT min(hour(ts)), max(hour(ts)) FROM events WHERE event_type='transfer' AND (attributes->>'narrative_in') LIKE '%PAYROLL%' AND month(ts) = 1"
    ).fetchone()
    assert 7 <= lo and hi <= 12


def test_ingest_log_and_records(con) -> None:  # type: ignore[no-untyped-def]
    log = con.execute("SELECT source, events_loaded, records_loaded FROM ingest_log").df().set_index("source")
    assert log.loc["bank", "events_loaded"] > 0 and log.loc["watchlist", "records_loaded"] > 0
    assert con.execute("SELECT count(*) FROM case_labels").fetchone()[0] > 0


def test_refuses_data_without_manifest(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    with pytest.raises(NonSyntheticDataError):
        ingest(tmp_path)


def test_refuses_real_looking_phone_numbers(world_dir: Path, tmp_path: Path) -> None:
    dst = tmp_path / "w"
    shutil.copytree(world_dir / "raw" / "telco", dst / "raw" / "telco")
    shutil.copy(world_dir / "manifest.json", dst / "manifest.json")
    subs = pd.read_csv(dst / "raw/telco/subscribers.csv", dtype=str)
    subs.loc[: len(subs) // 2, "msisdn"] = "12025551234"  # 555-1234 is outside the fictional 555-01XX block
    subs.to_csv(dst / "raw/telco/subscribers.csv", index=False)
    with pytest.raises(NonSyntheticDataError):
        ingest(dst, adapters=["telco"])


def test_new_feed_needs_only_an_adapter(world_dir: Path, tmp_path: Path) -> None:
    """A brand-new source plugs in without touching any downstream code."""

    @register_adapter
    class ParkingAdapter(SourceAdapter):
        name = "parking_test"

        def events(self) -> Iterator[pd.DataFrame]:
            df = pd.read_csv(self.root / "plates.csv")
            yield pd.DataFrame(
                {
                    "event_id": "parking:" + df["id"].astype(str),
                    "ts": pd.to_datetime(df["ts"], utc=True),
                    "source": "parking_test",
                    "event_type": "access_granted",
                    "actor_ids": [[f"card:{c}", f"loc:{g}"] for c, g in zip(df["card"], df["gate"])],
                    "actor_roles": [["cardholder", "reader"]] * len(df),
                    "attributes": [json.dumps({"zone": "TRANSIT"})] * len(df),
                }
            )

    try:
        dst = tmp_path / "w"
        (dst / "raw" / "parking_test").mkdir(parents=True)
        shutil.copy(world_dir / "manifest.json", dst / "manifest.json")
        pd.DataFrame({"id": [1, 2], "ts": ["2026-02-01T10:00:00Z", "2026-02-01T10:05:00Z"], "card": ["FC100-00001", "FC100-00002"], "gate": ["G1", "G1"]}).to_csv(dst / "raw/parking_test/plates.csv", index=False)
        report = ingest(dst, adapters=["parking_test"])
        assert report.sources["parking_test"]["events_loaded"] == 2
        c = connect(dst / "argus.duckdb", read_only=True)
        assert c.execute("SELECT count(*) FROM event_actors WHERE actor_id = 'loc:G1'").fetchone()[0] == 2
        c.close()
    finally:
        _REGISTRY.pop("parking_test", None)


def test_identifier_normalization() -> None:
    assert normalize_msisdn("(201) 555-0142") == normalize_msisdn("2015550142") == normalize_msisdn("+1 201 555 0142") == "+12015550142"
    assert normalize_msisdn("555-0142") is None
    assert normalize_email(" Bob.Smith+news@MailHaven.example ") == "bob.smith@mailhaven.example"
    assert normalize_email("not-an-email") is None


def test_date_format_inference() -> None:
    us = pd.Series(["03/04/1985", "12/31/1990", "07/15/2001"])
    assert infer_format(us.tolist()) == "%m/%d/%Y"
    eu = pd.Series(["31081966", "01021990"])
    assert parse_dates(eu).tolist() == ["1966-08-31", "1990-02-01"]
    t = parse_local_datetimes(pd.Series(["2026-07-01 09:00:00"]), "America/New_York")
    assert t.iloc[0].hour == 13  # EDT = UTC-4
