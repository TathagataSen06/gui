"""Orchestrates world generation and writes raw feeds + ground truth to disk."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from faker import Faker

from .. import SYNTHETIC_MARKER, __version__
from ..paths import RunPaths
from .config import GeneratorConfig, Rng
from .background import assign_background_holdings, generate_background
from .decoys import run_decoys, setup_decoys
from .holdings import assign_banking, assign_cards, assign_emails, assign_housing, assign_phones
from .identifiers import IdentifierFactory
from .identity import (
    email_truth,
    render_accounts,
    render_bank_customers,
    render_cardholders,
    render_case_outcomes,
    render_readers,
    render_subscribers,
    render_watchlist,
)
from .population import (
    assign_employment,
    build_billers,
    build_branches,
    build_cell_sites,
    build_employers,
    build_merchants,
    build_people,
    build_social_graph,
    build_third_places,
    build_towns,
)
from .scenarios import Ctx, run_scenarios, setup_scenarios
from .sources.access import SwipeLog, generate_benign_access
from .sources.bank import BankLedger, generate_benign_bank
from .sources.email import MailLog, generate_benign_email
from .sources.telco import CallLog, generate_benign_calls, generate_service_calls
from .state import Clock, World

NOTICE = (
    "SYNTHETIC DATA. Every person, organisation, account, phone number, e-mail address and event "
    "in this directory was fabricated by argus.generate for research and fraud-detection benchmarking. "
    "Any resemblance to real persons is coincidental."
)


@dataclass
class GenerationResult:
    paths: RunPaths
    stats: dict[str, Any]


def build_world(cfg: GeneratorConfig) -> World:
    """Create the in-memory world: population, holdings, activity, scenarios."""
    rng = Rng(cfg.seed)
    fake = Faker("en_US")
    fake.seed_instance(cfg.seed)
    w = World(cfg=cfg, rng=rng, fake=fake, ids=IdentifierFactory(rng.stream("holdings")), clock=Clock(cfg.start, cfg.days, cfg.timezone))

    build_towns(w)
    build_people(w)
    build_employers(w)
    build_third_places(w)
    build_cell_sites(w)
    build_branches(w)
    assign_employment(w)
    build_merchants(w)
    build_billers(w)
    build_social_graph(w)

    assign_banking(w)
    assign_phones(w)
    assign_emails(w)
    assign_cards(w)
    assign_housing(w)
    assign_background_holdings(w)

    decoys = setup_decoys(w)
    scenarios = setup_scenarios(w)

    ledger, calls, swipes, mail = BankLedger(w), CallLog(w), SwipeLog(w), MailLog(w)
    generate_benign_bank(w, ledger)
    generate_benign_calls(w, calls)
    generate_service_calls(w, calls)
    generate_benign_access(w, swipes)
    generate_benign_email(w, mail)
    generate_background(w, ledger, calls, swipes, mail)

    ctx = Ctx(w, ledger, calls, swipes, mail, rng.stream("scenarios"))
    run_decoys(Ctx(w, ledger, calls, swipes, mail, rng.stream("decoys")), decoys)
    run_scenarios(ctx, scenarios)

    w.bank_txn, w.cdr, w.access_events = ledger.t, calls.t, swipes.t
    return w


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------


def _write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
    return len(df)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(rows)


def _local(ts: pd.Series, tz: str) -> pd.Series:
    return pd.to_datetime(ts, unit="s", utc=True).dt.tz_convert(tz)


def _iso_local(ts: pd.Series, tz: str) -> tuple[np.ndarray, np.ndarray]:
    """Fast (ISO local wall-clock strings, UTC offsets in seconds) for epoch seconds."""
    loc = _local(ts, tz)
    naive = loc.dt.tz_localize(None).values.astype("datetime64[s]")
    offs = ((naive.astype(np.int64) - ts.values.astype(np.int64))).astype(np.int64)
    return np.datetime_as_string(naive, unit="s"), offs


def _fmt_bank_time(iso: np.ndarray, offs: np.ndarray, code: str) -> list[str]:
    if code == "FMB":  # 2026-03-04 14:22:10
        return [s.replace("T", " ") for s in iso]
    if code == "CFCU":  # 03/04/2026 14:22
        return [f"{s[5:7]}/{s[8:10]}/{s[0:4]} {s[11:16]}" for s in iso]
    # HVT: ISO 8601 with offset
    out = []
    for s, o in zip(iso, offs):
        sign = "-" if o < 0 else "+"
        o = abs(int(o))
        out.append(f"{s}{sign}{o // 3600:02d}:{(o % 3600) // 60:02d}")
    return out


def _in_window(w: World, df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Feeds only contain what happened inside the observation window."""
    return df[(df[col] >= int(w.clock.day_start[0])) & (df[col] < int(w.clock.end_epoch))]


def _guid(rng: np.random.Generator, n: int) -> list[str]:
    hexes = [f"{int(a):016x}{int(b):016x}" for a, b in zip(rng.integers(0, 2**63, n), rng.integers(0, 2**63, n))]
    return [f"{h[:8]}-{h[8:12]}-4{h[13:16]}-a{h[17:20]}-{h[20:32]}" for h in hexes]


def write_bank(w: World, raw: Path, planted: list[dict[str, str]], truth: list[dict[str, str]]) -> dict[str, int]:
    df = _in_window(w, w.bank_txn.to_frame(), "booking_ts")  # type: ignore[union-attr]
    df = df.sort_values(["booking_ts", "bank_code", "account_number", "direction", "amount", "end_to_end_id"], kind="mergesort").reset_index(drop=True)
    iso, offs = _iso_local(df["booking_ts"], w.cfg.timezone)
    df["value_date"] = [s[:10] for s in iso]
    df["_month"] = [s[0:4] + s[5:7] for s in iso]
    fmts = {"FMB": "", "CFCU": "", "HVT": ""}
    stats: dict[str, int] = {}
    customers, bank_truth = render_bank_customers(w)
    truth.extend(bank_truth)
    accounts = render_accounts(w)
    for code, fmt in fmts.items():
        sub = df[df["bank_code"] == code].copy()
        sub["txn_id"] = [f"{code}T{i:09d}" for i in range(1, len(sub) + 1)]
        sub["booking_datetime"] = _fmt_bank_time(iso[sub.index.values], offs[sub.index.values], code)
        for tag, tid in zip(sub["_tag"], sub["txn_id"]):
            if isinstance(tag, str) and tag:
                planted.append({"source": "bank", "native_id": tid, "ring_id": tag})
        cols = ["txn_id", "booking_datetime", "value_date", "account_number", "direction", "amount", "currency", "channel", "counterparty_account", "counterparty_name", "counterparty_bank", "mcc", "merchant_id", "merchant_name", "branch_id", "end_to_end_id", "narrative"]
        n = 0
        for month, part in sub.groupby("_month", sort=True):
            n += _write_csv(raw / "bank" / code / f"transactions_{month}.csv", part[cols])
        stats[f"bank_{code}_transactions"] = n
        stats[f"bank_{code}_customers"] = _write_csv(raw / "bank" / code / "customers.csv", customers[code])
        stats[f"bank_{code}_accounts"] = _write_csv(raw / "bank" / code / "accounts.csv", accounts[code])
    return stats


def write_telco(w: World, raw: Path, planted: list[dict[str, str]], truth: list[dict[str, str]]) -> dict[str, int]:
    df = _in_window(w, w.cdr.to_frame(), "start_ts").sort_values(["start_ts", "calling_number", "called_number"], kind="mergesort").reset_index(drop=True)  # type: ignore[union-attr]
    iso = np.datetime_as_string(df["start_ts"].values.astype("datetime64[s]"), unit="s")
    df["cdr_id"] = [f"NLW{i:010d}" for i in range(1, len(df) + 1)]
    df["start_time"] = [s[0:4] + s[5:7] + s[8:10] + s[11:13] + s[14:16] + s[17:19] for s in iso]
    df["_month"] = [s[0:4] + s[5:7] for s in iso]
    for tag, cid in zip(df["_tag"], df["cdr_id"]):
        if isinstance(tag, str) and tag:
            planted.append({"source": "telco", "native_id": cid, "ring_id": tag})
    cols = ["cdr_id", "record_type", "calling_number", "called_number", "start_time", "duration_sec", "cell_id", "call_result"]
    n = 0
    for month, part in df.groupby("_month", sort=True):
        n += _write_csv(raw / "telco" / "cdr" / f"cdr_{month}.csv", part[cols])
    subs, telco_truth = render_subscribers(w)
    truth.extend(telco_truth)
    return {
        "telco_cdr": n,
        "telco_subscribers": _write_csv(raw / "telco" / "subscribers.csv", subs),
        "telco_cell_sites": _write_csv(raw / "telco" / "cell_sites.csv", w.cell_sites),
    }


def write_access(w: World, raw: Path, planted: list[dict[str, str]], truth: list[dict[str, str]]) -> dict[str, int]:
    df = _in_window(w, w.access_events.to_frame(), "ts").sort_values(["ts", "card_number", "reader_id"], kind="mergesort").reset_index(drop=True)  # type: ignore[union-attr]
    df = df.rename(columns={"_tag": "tag"})
    iso = np.datetime_as_string(df["ts"].values.astype("datetime64[s]"), unit="s")
    rng = np.random.default_rng(w.cfg.seed + 7)
    ids = _guid(rng, len(df))
    ms = rng.integers(0, 1000, len(df))
    times = [f"{s}.{int(m):03d}Z" for s, m in zip(iso, ms)]
    months = [s[:7] for s in iso]
    sites = {r.reader_id: r.site_id for s in w.sites.values() for r in s.readers}
    by_month: dict[str, list[dict[str, Any]]] = {}
    for i, row in enumerate(df.itertuples(index=False)):
        rec: dict[str, Any] = {
            "EventId": ids[i],
            "EventTime": times[i],
            "CardNumber": row.card_number,
            "ReaderId": row.reader_id,
            "SiteCode": sites[row.reader_id],
            "Result": row.result,
            "Direction": row.direction,
        }
        if isinstance(row.deny_reason, str):
            rec["DenyReason"] = row.deny_reason
        by_month.setdefault(months[i], []).append(rec)
        if isinstance(row.tag, str) and row.tag:
            planted.append({"source": "access", "native_id": ids[i], "ring_id": row.tag})
    n = sum(_write_jsonl(raw / "access_control" / "events" / f"access_{m}.jsonl", recs) for m, recs in sorted(by_month.items()))
    holders, access_truth = render_cardholders(w)
    truth.extend(access_truth)
    return {
        "access_events": n,
        "access_cardholders": _write_csv(raw / "access_control" / "cardholders.csv", holders),
        "access_readers": _write_csv(raw / "access_control" / "readers.csv", render_readers(w)),
    }


def write_email(w: World, raw: Path, planted: list[dict[str, str]]) -> dict[str, int]:
    lo, hi = int(w.clock.day_start[0]), int(w.clock.end_epoch)
    recs = sorted((r for r in w.email_headers if lo <= r["ts"] < hi), key=lambda r: (r["ts"], r["message_id"]))
    tz = w.clock.tz
    by_month: dict[str, list[dict[str, Any]]] = {}
    for r in recs:
        dt = datetime.fromtimestamp(r["ts"], tz)
        out = {"message_id": r["message_id"], "date": format_datetime(dt), "from": r["from"], "to": r["to"]}
        if r["cc"]:
            out["cc"] = r["cc"]
        out["subject"] = r["subject"]
        out["size"] = r["size"]
        if r["attachments"]:
            out["attachments"] = r["attachments"]
        if "list_unsubscribe" in r:
            out["list_unsubscribe"] = r["list_unsubscribe"]
            out["precedence"] = r["precedence"]
        by_month.setdefault(dt.astimezone(timezone.utc).strftime("%Y-%m"), []).append(out)
        if r.get("_tag"):
            planted.append({"source": "email", "native_id": r["message_id"], "ring_id": r["_tag"]})
    n = sum(_write_jsonl(raw / "email" / f"headers_{m}.jsonl", v) for m, v in sorted(by_month.items()))
    return {"email_messages": n}


def write_ground_truth(w: World, gt: Path, planted: list[dict[str, str]], record_truth: list[dict[str, str]]) -> None:
    labels: dict[str, dict[str, Any]] = {p.person_id: {"person_id": p.person_id, "label": 0, "ring_id": "", "role": "", "decoy_id": "", "decoy_role": ""} for p in w.persons}
    for ring in w.rings:
        for m in ring["members"]:
            labels[m["person_id"]].update(label=1, ring_id=ring["ring_id"], role=m["role"])
        for v in ring.get("victims", []):
            labels[v].update(role="victim", ring_id=ring["ring_id"])
    for d in w.decoys:
        for m in d["members"]:
            labels[m["person_id"]].update(decoy_id=d["decoy_id"], decoy_role=f"{d['kind']}:{m['role']}")
    _write_csv(gt / "person_labels.csv", list(labels.values()))
    _write_csv(
        gt / "persons.csv",
        [
            {
                "person_id": p.person_id, "first": p.first, "middle": p.middle, "last": p.last, "suffix": p.suffix,
                "maiden_last": p.maiden_last, "sex": p.sex, "dob": p.dob.isoformat(), "household_id": p.household_id,
                "address": p.address.one_line(), "prior_address": p.prior_address.one_line() if p.prior_address else None,
                "employer_id": p.employer_id, "department": p.department, "occupation": p.occupation, "income_kind": p.income_kind,
                "phones": ";".join(p.phones), "personal_email": p.personal_email, "work_email": p.work_email,
                "accounts": ";".join(p.accounts), "cards": ";".join(p.cards), "traits": ";".join(sorted(p.traits)),
            }
            for p in w.persons
        ],
    )
    _write_csv(
        gt / "orgs.csv",
        [
            {"org_id": o.org_id, "name": o.name, "kind": o.kind, "reg_no": o.reg_no, "owners": ";".join(o.owners), "accounts": ";".join(o.accounts), "address": o.address.one_line() if o.address else None}
            for o in sorted(w.orgs.values(), key=lambda o: o.org_id)
        ],
    )
    _write_csv(gt / "accounts.csv", [{"account_key": a.key, "owner_kind": a.owner_kind, "owner_id": a.owner_id, "product": a.product, "open_date": a.open_date.isoformat()} for a in sorted(w.accounts.values(), key=lambda a: a.key)])
    _write_csv(gt / "phones.csv", [{"msisdn": m, "owner_id": o, "kind": "burner" if m in w.burner_owner else ("anonymous_prepaid" if m in w.anon_lines else "personal")} for m, o in sorted(w.phone_owner.items())])
    _write_csv(gt / "record_truth.csv", sorted(record_truth, key=lambda r: r["record_id"]))
    _write_csv(gt / "planted_events.csv", planted)
    (gt / "scenarios.json").write_text(json.dumps(w.rings, indent=2, default=str))
    (gt / "decoys.json").write_text(json.dumps(w.decoys, indent=2, default=str))


def generate_world(cfg: GeneratorConfig, out_dir: str | Path) -> GenerationResult:
    """Generate a synthetic world and write it under ``out_dir``.

    Returns the run paths and row counts per file family.
    """
    t0 = time.perf_counter()
    paths = RunPaths.at(out_dir).ensure()
    w = build_world(cfg)
    planted: list[dict[str, str]] = []
    stats: dict[str, Any] = {"people": len(w.persons), "orgs": len(w.orgs), "accounts": len(w.accounts), "rings": len(w.rings), "decoys": len(w.decoys)}
    truth: list[dict[str, str]] = []
    stats |= write_bank(w, paths.raw, planted, truth)
    stats |= write_telco(w, paths.raw, planted, truth)
    stats |= write_access(w, paths.raw, planted, truth)
    stats |= write_email(w, paths.raw, planted)
    wl_rows, wl_truth = render_watchlist(w)
    stats["watchlist_entries"] = _write_csv(paths.raw / "watchlist" / "watchlist.csv", wl_rows)
    stats["case_outcomes"] = _write_csv(paths.raw / "labels" / "case_outcomes.csv", render_case_outcomes(w))

    truth += email_truth(w) + wl_truth
    write_ground_truth(w, paths.ground_truth, planted, truth)

    manifest = {
        "marker": SYNTHETIC_MARKER,
        "synthetic": True,
        "notice": NOTICE,
        "generator": "argus.generate",
        "version": __version__,
        "seed": cfg.seed,
        "config": cfg.to_dict(),
        "window": {"start": cfg.start.isoformat(), "end": cfg.end.isoformat(), "timezone": cfg.timezone},
        "sources": {
            "bank": "raw/bank/<BANK>/{customers,accounts,transactions_YYYYMM}.csv",
            "telco": "raw/telco/{subscribers,cell_sites}.csv, raw/telco/cdr/cdr_YYYYMM.csv",
            "access_control": "raw/access_control/{cardholders,readers}.csv, raw/access_control/events/access_YYYY-MM.jsonl",
            "email": "raw/email/headers_YYYY-MM.jsonl",
            "watchlist": "raw/watchlist/watchlist.csv",
            "labels": "raw/labels/case_outcomes.csv",
        },
        "stats": stats,
    }
    paths.manifest.write_text(json.dumps(manifest, indent=2))
    (paths.raw / "README.txt").write_text(NOTICE + "\n")
    stats["seconds"] = round(time.perf_counter() - t0, 1)
    return GenerationResult(paths, stats)

