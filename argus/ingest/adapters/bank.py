"""Consortium core-banking feed -> identity records + transfer/card/cash events.

Handles per-bank file conventions (date formats, casing) by inference and
collapses double-entry ledger legs (DR on payer, CR on payee) into a single
``transfer`` event keyed by ``end_to_end_id``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from ..base import SourceAdapter, attrs_json, none_if_blank, register_adapter
from ..dates import DATETIME_CANDIDATES, infer_format, parse_dates, parse_local_datetimes

TRANSFER_CHANNELS = ("ACH", "WIRE", "P2P", "BOOK", "CHECK")


def _address(row: pd.Series) -> str | None:
    parts = [none_if_blank(row.get("street_address")), none_if_blank(row.get("city"))]
    st = " ".join(p for p in (none_if_blank(row.get("state")), none_if_blank(row.get("postal_code"))) if p)
    parts.append(st or None)
    joined = ", ".join(str(p) for p in parts if p)
    return joined or None


@register_adapter
class BankAdapter(SourceAdapter):
    name = "bank"
    local_tz = "America/New_York"

    def banks(self) -> list[Path]:
        return sorted(p for p in self.root.iterdir() if p.is_dir())

    def records(self) -> pd.DataFrame:
        frames = []
        for bdir in self.banks():
            code = bdir.name
            cust = pd.read_csv(bdir / "customers.csv", dtype=str, keep_default_na=False)
            accts = pd.read_csv(bdir / "accounts.csv", dtype=str, keep_default_na=False)
            self.rows_read += len(cust) + len(accts)
            owned = accts.groupby("customer_id")["account_number"].apply(lambda s: [f"acct:{code}:{a}" for a in sorted(s)]).to_dict()
            dob_fmt = infer_format(cust["date_of_birth"].tolist()) if "date_of_birth" in cust else None
            dob_iso = parse_dates(cust["date_of_birth"], dob_fmt)
            bo_fmt = infer_format(cust["beneficial_owner_dob"].tolist()) or dob_fmt
            bo_iso = parse_dates(cust["beneficial_owner_dob"], bo_fmt)
            rows = []
            for i, r in cust.iterrows():
                rid = f"bank:{code}:{r['customer_id']}"
                is_biz = r["customer_type"] == "BUSINESS"
                attrs = {
                    "bank_code": code,
                    "customer_id": r["customer_id"],
                    "customer_type": r["customer_type"],
                    "occupation": r.get("occupation"),
                    "employer_name": r.get("employer_name"),
                    "onboarding_date": r.get("onboarding_date"),
                    "kyc_risk_rating": r.get("kyc_risk_rating"),
                    "tax_reg_no": r.get("tax_reg_no"),
                    "incorporation_date": r.get("incorporation_date"),
                    "business_type": r.get("business_type"),
                    "dob_raw": r.get("date_of_birth"),
                    "postal_code": r.get("postal_code"),
                }
                rows.append(
                    {
                        "record_id": rid,
                        "source": "bank",
                        "entity_type": "org" if is_biz else "person",
                        "name": none_if_blank(r["full_name"]),
                        "dob": dob_iso.iloc[i] if not is_biz else None,
                        "address": _address(r),
                        "phone": none_if_blank(r.get("phone")),
                        "email": none_if_blank(r.get("email")),
                        "owned_ids": owned.get(r["customer_id"], []),
                        "attributes": attrs,
                    }
                )
                if is_biz and none_if_blank(r.get("beneficial_owner_name")):
                    rows.append(
                        {
                            "record_id": f"{rid}:bo",
                            "source": "bank",
                            "entity_type": "person",
                            "name": r["beneficial_owner_name"],
                            "dob": bo_iso.iloc[i],
                            "address": None,
                            "phone": None,
                            "email": None,
                            "owned_ids": [],
                            "attributes": {"bank_code": code, "role": "beneficial_owner", "controls": rid, "dob_raw": r.get("beneficial_owner_dob")},
                        }
                    )
            frames.append(pd.DataFrame(rows))
        out = pd.concat(frames, ignore_index=True) if frames else super().records()
        out["attributes"] = attrs_json(out["attributes"].tolist())
        return out

    def reference_tables(self) -> dict[str, pd.DataFrame]:
        frames = []
        for bdir in self.banks():
            a = pd.read_csv(bdir / "accounts.csv", dtype=str, keep_default_na=False)
            a.insert(0, "account_id", "acct:" + bdir.name + ":" + a["account_number"])
            a.insert(1, "bank_code", bdir.name)
            a["record_id"] = "bank:" + bdir.name + ":" + a["customer_id"]
            frames.append(a)
        if not frames:
            return {}
        df = pd.concat(frames, ignore_index=True)
        df["open_date"] = pd.to_datetime(df["open_date"]).dt.date
        return {"ref_accounts": df}

    def _ledger(self) -> pd.DataFrame:
        frames = []
        for bdir in self.banks():
            files = sorted(bdir.glob("transactions_*.csv"))
            for f in files:
                df = pd.read_csv(f, dtype={"account_number": str, "counterparty_account": str, "mcc": str, "merchant_id": str, "branch_id": str, "end_to_end_id": str, "txn_id": str}, keep_default_na=False)
                df["bank_code"] = bdir.name
                self.rows_read += len(df)
                fmt = infer_format(df["booking_datetime"].head(500).tolist(), DATETIME_CANDIDATES)
                df["ts"] = parse_local_datetimes(df["booking_datetime"], self.local_tz, fmt)
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        led = pd.concat(frames, ignore_index=True)
        bad = led["ts"].isna() | pd.to_numeric(led["amount"], errors="coerce").isna()
        self.reject("unparseable_timestamp_or_amount", int(bad.sum()))
        led = led[~bad].copy()
        led["amount"] = led["amount"].astype(float)
        led["acct_id"] = "acct:" + led["bank_code"] + ":" + led["account_number"]
        return led

    def events(self) -> Iterator[pd.DataFrame]:
        led = self._ledger()
        if led.empty:
            return
        # ---- single-leg events --------------------------------------------
        card = led[led["channel"] == "CARD"]
        yield pd.DataFrame(
            {
                "event_id": "bank:" + card["txn_id"],
                "ts": card["ts"],
                "source": "bank",
                "event_type": "card_purchase",
                "actor_ids": [[a, f"merchant:{m}"] for a, m in zip(card["acct_id"], card["merchant_id"])],
                "actor_roles": [["cardholder", "merchant"]] * len(card),
                "attributes": attrs_json(
                    [
                        {"amount": a, "mcc": m, "merchant_name": n, "narrative": nar, "txn_id": t, "bank_code": b}
                        for a, m, n, nar, t, b in zip(card["amount"], card["mcc"], card["merchant_name"], card["narrative"], card["txn_id"], card["bank_code"])
                    ]
                ),
            }
        )
        cash = led[led["channel"].isin(["CASH", "ATM"])]
        yield pd.DataFrame(
            {
                "event_id": "bank:" + cash["txn_id"],
                "ts": cash["ts"],
                "source": "bank",
                "event_type": ["cash_deposit" if d == "CR" else "cash_withdrawal" for d in cash["direction"]],
                "actor_ids": [[a, f"branch:{b}"] for a, b in zip(cash["acct_id"], cash["branch_id"])],
                "actor_roles": [["account", "branch"]] * len(cash),
                "attributes": attrs_json(
                    [
                        {"amount": a, "channel": c, "branch_id": b, "narrative": nar, "txn_id": t, "bank_code": bc}
                        for a, c, b, nar, t, bc in zip(cash["amount"], cash["channel"], cash["branch_id"], cash["narrative"], cash["txn_id"], cash["bank_code"])
                    ]
                ),
            }
        )
        # ---- transfers: collapse DR/CR legs by end_to_end_id ---------------
        tr = led[led["channel"].isin(TRANSFER_CHANNELS)]
        dr = tr[tr["direction"] == "DR"].set_index("end_to_end_id")
        cr = tr[tr["direction"] == "CR"].set_index("end_to_end_id")
        dup = dr.index.duplicated() | False
        self.reject("duplicate_debit_leg", int(dup.sum()))
        dr = dr[~dup]
        cr = cr[~cr.index.duplicated()]
        legs = dr.join(cr, how="outer", lsuffix="_dr", rsuffix="_cr")
        has_dr = legs["acct_id_dr"].notna()
        has_cr = legs["acct_id_cr"].notna()
        ts = legs["ts_dr"].where(has_dr, legs["ts_cr"])
        payer = legs["acct_id_dr"].where(has_dr, "ext:" + legs["counterparty_account_cr"].fillna("unknown"))
        payee = legs["acct_id_cr"].where(has_cr, "ext:" + legs["counterparty_account_dr"].fillna("unknown"))
        amount = legs["amount_dr"].where(has_dr, legs["amount_cr"])
        channel = legs["channel_dr"].where(has_dr, legs["channel_cr"])
        mismatch = has_dr & has_cr & ((legs["amount_dr"] - legs["amount_cr"]).abs() > 0.005)
        self.reject("leg_amount_mismatch", int(mismatch.sum()))
        attrs = []
        for e2e, a, ch, hd, hc, nd, nc, cpn_dr, cpn_cr, cpb_dr, cpb_cr, tdr, tcr, tsd, tsc in zip(
            legs.index, amount, channel, has_dr, has_cr, legs["narrative_dr"], legs["narrative_cr"],
            legs["counterparty_name_dr"], legs["counterparty_name_cr"], legs["counterparty_bank_dr"], legs["counterparty_bank_cr"],
            legs["txn_id_dr"], legs["txn_id_cr"], legs["ts_dr"], legs["ts_cr"],
        ):
            d = {"amount": a, "channel": ch, "end_to_end_id": e2e, "legs": int(hd) + int(hc)}
            if hd:
                d["narrative"] = nd
                d["payee_name"] = cpn_dr
                d["payee_bank"] = cpb_dr
                d["txn_id_dr"] = tdr
            if hc:
                d["narrative_in"] = nc
                d["payer_name"] = cpn_cr
                d["payer_bank"] = cpb_cr
                d["txn_id_cr"] = tcr
            if hd and hc:
                d["settlement_lag_sec"] = int((tsc - tsd).total_seconds())
            attrs.append(d)
        yield pd.DataFrame(
            {
                "event_id": "bank:" + legs.index.astype(str),
                "ts": ts.values,
                "source": "bank",
                "event_type": "transfer",
                "actor_ids": [[p, q] for p, q in zip(payer, payee)],
                "actor_roles": [["payer", "payee"]] * len(legs),
                "attributes": attrs_json(attrs),
            }
        )
