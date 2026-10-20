"""Carrier subscriber register + call-detail records -> identity records + call/SMS events."""

from __future__ import annotations

from typing import Iterator

import pandas as pd

from ...ids import normalize_email, normalize_msisdn
from ..base import SourceAdapter, attrs_json, none_if_blank, register_adapter
from ..dates import parse_dates


@register_adapter
class TelcoAdapter(SourceAdapter):
    name = "telco"

    def records(self) -> pd.DataFrame:
        subs = pd.read_csv(self.root / "subscribers.csv", dtype=str, keep_default_na=False)
        self.rows_read += len(subs)
        e164 = subs["msisdn"].map(normalize_msisdn)
        self.reject("bad_msisdn", int(e164.isna().sum()))
        subs = subs[e164.notna()].copy()
        subs["e164"] = e164[e164.notna()]
        dob = parse_dates(subs["subscriber_dob"])
        rows = []
        for (_, r), d in zip(subs.iterrows(), dob):
            rows.append(
                {
                    "record_id": f"telco:{r['e164']}",
                    "source": "telco",
                    "entity_type": "person",
                    "name": none_if_blank(r["subscriber_name"]),
                    "dob": d,
                    "address": none_if_blank(r["billing_address"]),
                    "phone": r["e164"],
                    "email": normalize_email(r["contact_email"]),
                    "owned_ids": [f"msisdn:{r['e164']}"],
                    "attributes": {
                        "carrier": r["carrier"],
                        "account_type": r["account_type"],
                        "activation_date": r["activation_date"],
                        "status": r["status"],
                        "dob_raw": r["subscriber_dob"],
                    },
                }
            )
        out = pd.DataFrame(rows)
        out["attributes"] = attrs_json(out["attributes"].tolist())
        return out

    def reference_tables(self) -> dict[str, pd.DataFrame]:
        p = self.root / "cell_sites.csv"
        return {"ref_cell_sites": pd.read_csv(p, dtype={"cell_id": str})} if p.exists() else {}

    def events(self) -> Iterator[pd.DataFrame]:
        for f in sorted((self.root / "cdr").glob("cdr_*.csv")):
            df = pd.read_csv(f, dtype=str, keep_default_na=False)
            self.rows_read += len(df)
            a = df["calling_number"].map(normalize_msisdn)
            b = df["called_number"].map(normalize_msisdn)
            ts = pd.to_datetime(df["start_time"], format="%Y%m%d%H%M%S", errors="coerce", utc=True)
            ok = a.notna() & b.notna() & ts.notna()
            self.reject("bad_number_or_time", int((~ok).sum()))
            df, a, b, ts = df[ok], a[ok], b[ok], ts[ok]
            sms = df["record_type"].str.startswith("SMS")
            yield pd.DataFrame(
                {
                    "event_id": "telco:" + df["cdr_id"],
                    "ts": ts,
                    "source": "telco",
                    "event_type": ["sms" if s else "call" for s in sms],
                    "actor_ids": [[f"msisdn:{x}", f"msisdn:{y}"] for x, y in zip(a, b)],
                    "actor_roles": [["caller", "callee"]] * len(df),
                    "attributes": attrs_json(
                        [
                            {"duration_sec": int(d), "cell_id": c, "call_result": r, "record_type": t, "cdr_id": i}
                            for d, c, r, t, i in zip(df["duration_sec"], df["cell_id"], df["call_result"], df["record_type"], df["cdr_id"])
                        ]
                    ),
                }
            )
