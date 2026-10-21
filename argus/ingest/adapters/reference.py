"""Watch-list and historical case-outcome feeds (no events)."""

from __future__ import annotations

from typing import Iterator

import pandas as pd

from ..base import SourceAdapter, attrs_json, none_if_blank, register_adapter
from ..dates import parse_dates


@register_adapter
class WatchlistAdapter(SourceAdapter):
    """Internal hot-list of subjects of interest (names, DOBs, aliases)."""

    name = "watchlist"

    def records(self) -> pd.DataFrame:
        wl = pd.read_csv(self.root / "watchlist.csv", dtype=str, keep_default_na=False)
        self.rows_read += len(wl)
        dob = parse_dates(wl["dob"])
        rows = []
        for (_, r), d in zip(wl.iterrows(), dob):
            rows.append(
                {
                    "record_id": f"watchlist:{r['entry_id']}",
                    "source": "watchlist",
                    "entity_type": "person",
                    "name": none_if_blank(r["full_name"]),
                    "dob": d,
                    "address": None,
                    "phone": None,
                    "email": None,
                    "owned_ids": [],
                    "attributes": {
                        "aliases": [a for a in r["aliases"].split(";") if a] or None,
                        "list_name": r["list_name"],
                        "reason_code": r["reason_code"],
                        "added_date": r["added_date"],
                    },
                }
            )
        out = pd.DataFrame(rows)
        out["attributes"] = attrs_json(out["attributes"].tolist())
        return out

    def events(self) -> Iterator[pd.DataFrame]:
        return iter(())


@register_adapter
class CaseOutcomeAdapter(SourceAdapter):
    """Dispositions of past investigations; used only as supervised labels."""

    name = "labels"

    def events(self) -> Iterator[pd.DataFrame]:
        return iter(())

    def labels(self) -> pd.DataFrame:
        df = pd.read_csv(self.root / "case_outcomes.csv", dtype=str, keep_default_na=False)
        self.rows_read += len(df)
        df["opened_date"] = pd.to_datetime(df["opened_date"]).dt.date
        df["closed_date"] = pd.to_datetime(df["closed_date"]).dt.date
        return df[["case_id", "subject_record_id", "disposition", "typology", "opened_date", "closed_date"]]
