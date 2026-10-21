"""Physical access-control (PACS) feed -> cardholder records + swipe events."""

from __future__ import annotations

import json
from typing import Iterator

import pandas as pd

from ...ids import normalize_email
from ..base import SourceAdapter, attrs_json, none_if_blank, register_adapter


@register_adapter
class AccessControlAdapter(SourceAdapter):
    name = "access"
    directory = "access_control"

    def records(self) -> pd.DataFrame:
        ch = pd.read_csv(self.root / "cardholders.csv", dtype=str, keep_default_na=False)
        self.rows_read += len(ch)
        rows = []
        for _, r in ch.iterrows():
            rows.append(
                {
                    "record_id": f"access:{r['card_number']}",
                    "source": "access",
                    "entity_type": "person",
                    "name": none_if_blank(r["cardholder_name"]),
                    "dob": None,
                    "address": None,
                    "phone": none_if_blank(r["mobile_phone"]),
                    "email": normalize_email(r["work_email"]),
                    "owned_ids": [f"card:{r['card_number']}"],
                    "attributes": {
                        "employee_id": r["employee_id"],
                        "company": r["company"],
                        "employer_name": r["company"] if r["card_type"] == "EMPLOYEE" else None,
                        "department": r["department"],
                        "card_type": r["card_type"],
                        "site_code": r["site_code"],
                        "issued_date": r["issued_date"],
                    },
                }
            )
        out = pd.DataFrame(rows)
        out["attributes"] = attrs_json(out["attributes"].tolist())
        return out

    def _readers(self) -> pd.DataFrame:
        return pd.read_csv(self.root / "readers.csv", dtype=str, keep_default_na=False)

    def reference_tables(self) -> dict[str, pd.DataFrame]:
        r = self._readers()
        r["lat"] = r["lat"].astype(float)
        r["lon"] = r["lon"].astype(float)
        return {"ref_readers": r}

    def events(self) -> Iterator[pd.DataFrame]:
        readers = self._readers().set_index("reader_id")
        zone = readers["zone"].to_dict()
        site_type = readers["site_type"].to_dict()
        site_name = readers["site_name"].to_dict()
        for f in sorted((self.root / "events").glob("access_*.jsonl")):
            recs = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.rows_read += len(recs)
            df = pd.DataFrame(recs)
            if "DenyReason" not in df:
                df["DenyReason"] = None
            ts = pd.to_datetime(df["EventTime"], format="%Y-%m-%dT%H:%M:%S.%fZ", errors="coerce", utc=True)
            known = df["ReaderId"].isin(zone.keys())
            ok = ts.notna() & known
            self.reject("unknown_reader_or_bad_time", int((~ok).sum()))
            df, ts = df[ok], ts[ok]
            granted = df["Result"] == "ACCESS_GRANTED"
            yield pd.DataFrame(
                {
                    "event_id": "access:" + df["EventId"],
                    "ts": ts,
                    "source": "access",
                    "event_type": ["access_granted" if g else "access_denied" for g in granted],
                    "actor_ids": [[f"card:{c}", f"loc:{r}"] for c, r in zip(df["CardNumber"], df["ReaderId"])],
                    "actor_roles": [["cardholder", "reader"]] * len(df),
                    "attributes": attrs_json(
                        [
                            {"reader_id": r, "site_code": s, "zone": zone[r], "site_type": site_type[r], "site_name": site_name[r], "direction": d, "deny_reason": dr}
                            for r, s, d, dr in zip(df["ReaderId"], df["SiteCode"], df["Direction"], df["DenyReason"])
                        ]
                    ),
                }
            )
