"""Render each source system's view of people and organisations (with noise).

The same person appears differently in every system: the bank stores the
legal name in its own casing convention and its own date format; the carrier
stores whatever the customer typed at the shop (often a nickname, often no
DOB); the access-control system stores ``LAST, FIRST M`` from HR; e-mail only
has display names. Each rendered record carries its hidden truth (who it
really is) in ``record_truth``.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np

from .model import Address, Org, Person
from .noise import (
    DOB_FORMATS,
    corrupt_dob,
    email_typo,
    format_person_name,
    format_phone,
    format_street,
    nickname,
    phone_digit_typo,
    typo,
)
from .reference import BANKS
from .state import World

_BANKS = {b.code: b for b in BANKS}
_BANK_PHONE_STYLE = {"FMB": "paren", "CFCU": "digits", "HVT": "dash"}
_BANK_ABBREV = {"FMB": 0.9, "CFCU": 0.0, "HVT": 0.5}
_CARRIERS = ("Northlink Wireless", "Vireo Mobile", "Cobalt Cellular")


def _addr_parts(w: World, a: Address, rng: np.random.Generator, abbrev_p: float, upper: bool = False) -> dict[str, str]:
    n = w.cfg.noise
    street = format_street(a.number, a.street, a.unit, rng, abbrev_p, n.unit_missing)
    if rng.random() < n.address_typo:
        street = typo(street, rng)
    z = a.zip
    if rng.random() < n.zip_typo:
        z = z[:3] + z[4] + z[3]
    city = a.city
    out = {"street_address": street, "city": city, "state": a.state, "postal_code": z}
    if upper:
        out = {k: v.upper() for k, v in out.items()}
    return out


def _person_name(w: World, p: Person, rng: np.random.Generator, style: str, nick_p: float, last: str | None = None) -> str:
    n = w.cfg.noise
    first = nickname(p.first, rng) if rng.random() < nick_p else p.first
    suffix = p.suffix if p.suffix and rng.random() >= n.drop_suffix else None
    name = format_person_name(first, p.middle, last or p.last, suffix, style)
    if rng.random() < n.name_typo:
        name = typo(name, rng)
    return name


def render_bank_customers(w: World) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    """Return per-bank customer rows and record-truth rows."""
    rng = w.rng.stream("noise")
    n = w.cfg.noise
    rows: dict[str, list[dict[str, Any]]] = {b.code: [] for b in BANKS}
    truth: list[dict[str, str]] = []
    for p in w.persons:
        for i, (bank_code, cust) in enumerate(sorted(p.customer_ids.items(), key=lambda kv: min(w.accounts[a].open_date for a in p.accounts if w.accounts[a].bank_code == kv[0]))):
            bank = _BANKS[bank_code]
            onboard = min(w.accounts[a].open_date for a in p.accounts if w.accounts[a].bank_code == bank_code)
            last = p.last
            older = i == 0 and len(p.customer_ids) > 1
            if p.maiden_last and ((older and rng.random() < n.maiden_name_in_older_record) or rng.random() < 0.12):
                last = p.maiden_last
            name = _person_name(w, p, rng, bank.name_style, 0.0, last=last)
            dob: str | None = corrupt_dob(p.dob, rng, n.dob_day_month_swap, n.dob_typo).strftime(bank.dob_format)
            if rng.random() < n.dob_missing_bank:
                dob = None
            addr = p.address
            if p.prior_address and i > 0 and rng.random() < n.stale_address * 0.5:
                addr = p.prior_address
            if p.prior_address and onboard < w.cfg.start - timedelta(days=500) and rng.random() < 0.25:
                addr = p.prior_address  # bank never got the change-of-address
            phone = None
            if p.phones and rng.random() >= n.kyc_phone_missing:
                ph = p.phones[0]
                if rng.random() < n.kyc_phone_typo:
                    ph = phone_digit_typo(ph, rng)
                phone = format_phone(ph, _BANK_PHONE_STYLE[bank_code])
            email = None
            if p.personal_email and rng.random() >= n.kyc_email_missing:
                email = email_typo(p.personal_email, rng) if rng.random() < n.kyc_email_typo else p.personal_email
                if bank_code == "CFCU":
                    email = email.upper()
            employer = w.orgs[p.employer_id].name if p.employer_id else ""
            row = {
                "customer_id": cust,
                "customer_type": "INDIVIDUAL",
                "full_name": name,
                "date_of_birth": dob,
                **_addr_parts(w, addr, rng, _BANK_ABBREV[bank_code], upper=bank_code == "CFCU"),
                "phone": phone,
                "email": email,
                "occupation": p.occupation if p.income_kind != "payroll" else (p.department or "Employee"),
                "employer_name": employer,
                "onboarding_date": onboard.isoformat(),
                "kyc_risk_rating": str(rng.choice(["LOW", "MEDIUM", "HIGH"], p=[0.82, 0.16, 0.02])),
                "tax_reg_no": None,
                "incorporation_date": None,
                "business_type": None,
                "beneficial_owner_name": None,
                "beneficial_owner_dob": None,
            }
            rows[bank_code].append(row)
            truth.append({"record_id": f"bank:{bank_code}:{cust}", "source": "bank", "entity_kind": "person", "true_id": p.person_id})
    for org in sorted(w.orgs.values(), key=lambda o: o.org_id):
        for bank_code, cust in sorted(org.customer_ids.items()):
            bank = _BANKS[bank_code]
            owner = w.person_by_id[org.owners[0]] if org.owners else None
            onboard = min(w.accounts[a].open_date for a in org.accounts if w.accounts[a].bank_code == bank_code)
            rating = {"cash_business": "MEDIUM", "front": "MEDIUM", "employer": "LOW"}.get(org.kind, "LOW")
            if rng.random() < 0.1:
                rating = "HIGH" if rating == "MEDIUM" else "MEDIUM"
            btype = {
                "employer": "Corporation", "shell": "Management consulting", "front": "Wholesale trade",
                "property_llc": "Lessors of residential buildings", "cash_business": "Retail / food service",
                "crypto_front": "Digital asset services",
            }.get(org.kind, "Other")
            if org.kind == "front" and any(k in org.name for k in ("Wash", "Laund", "Deli", "Pizz", "Salon", "Truck", "Vape")):
                btype = "Retail / food service"
            row = {
                "customer_id": cust,
                "customer_type": "BUSINESS",
                "full_name": org.name.upper() if bank_code == "CFCU" else org.name,
                "date_of_birth": None,
                **_addr_parts(w, org.address, rng, _BANK_ABBREV[bank_code], upper=bank_code == "CFCU"),  # type: ignore[arg-type]
                "phone": format_phone(org.phone, _BANK_PHONE_STYLE[bank_code]) if org.phone else None,
                "email": f"accounts@{org.domain}" if org.domain else None,
                "occupation": None,
                "employer_name": None,
                "onboarding_date": onboard.isoformat(),
                "kyc_risk_rating": rating,
                "tax_reg_no": org.reg_no,
                "incorporation_date": org.inc_date.isoformat() if org.inc_date else None,
                "business_type": btype,
                "beneficial_owner_name": _person_name(w, owner, rng, bank.name_style, 0.02) if owner else None,
                "beneficial_owner_dob": corrupt_dob(owner.dob, rng, n.dob_day_month_swap, n.dob_typo).strftime(bank.dob_format) if owner and rng.random() > 0.08 else None,
            }
            rows[bank_code].append(row)
            truth.append({"record_id": f"bank:{bank_code}:{cust}", "source": "bank", "entity_kind": "org", "true_id": org.org_id})
            if owner:
                truth.append({"record_id": f"bank:{bank_code}:{cust}:bo", "source": "bank", "entity_kind": "person", "true_id": owner.person_id})
    for code in rows:
        rows[code].sort(key=lambda r: r["customer_id"])
    return rows, truth


def render_accounts(w: World) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {b.code: [] for b in BANKS}
    for a in sorted(w.accounts.values(), key=lambda a: a.key):
        rows[a.bank_code].append(
            {
                "account_number": a.number,
                "customer_id": a.customer_id,
                "product": a.product,
                "open_date": a.open_date.isoformat(),
                "status": "OPEN",
                "currency": "USD",
                "routing_number": _BANKS[a.bank_code].routing,
            }
        )
    return rows


def render_subscribers(w: World) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rng = w.rng.stream("noise")
    n = w.cfg.noise
    rows: list[dict[str, Any]] = []
    truth: list[dict[str, str]] = []
    start = w.cfg.start
    for p in w.persons:
        for line in p.phones:
            style = str(rng.choice(["first_last", "full", "upper_last_comma_first", "lower"], p=[0.55, 0.15, 0.2, 0.1]))
            name = _person_name(w, p, rng, style, n.nickname_telco)
            if rng.random() < n.first_initial_only:
                name = format_person_name(p.first, None, p.last, None, "initial_last")
            addr = p.prior_address if (p.prior_address and rng.random() < n.stale_address) else p.address
            prepaid = line in p.prepaid_phones
            dob = None
            if not prepaid and rng.random() >= n.dob_missing_telco:
                dob = corrupt_dob(p.dob, rng, n.dob_day_month_swap, n.dob_typo).strftime(DOB_FORMATS["compact_eu"])
            parts = _addr_parts(w, addr, rng, 0.7)
            rows.append(
                {
                    "msisdn": line[1:],
                    "carrier": str(rng.choice(_CARRIERS, p=[0.55, 0.3, 0.15])),
                    "account_type": "PREPAID" if prepaid else "POSTPAID",
                    "subscriber_name": name,
                    "subscriber_dob": dob,
                    "billing_address": None if prepaid and rng.random() < 0.5 else f"{parts['street_address']}, {parts['city']} {parts['state']} {parts['postal_code']}",
                    "contact_email": p.personal_email if (p.personal_email and rng.random() < 0.45) else None,
                    "activation_date": (start - timedelta(days=int(rng.integers(30, 3650)))).isoformat(),
                    "status": "ACTIVE",
                }
            )
            truth.append({"record_id": f"telco:+{line[1:]}", "source": "telco", "entity_kind": "person", "true_id": p.person_id})
    for num in sorted(w.anon_lines):
        p = w.person_by_id[w.anon_lines[num]]
        rows.append(
            {
                "msisdn": num[1:],
                "carrier": str(rng.choice(_CARRIERS)),
                "account_type": "PREPAID",
                "subscriber_name": (nickname(p.first, rng) if rng.random() < 0.5 else p.first) if rng.random() < 0.35 else "",
                "subscriber_dob": None,
                "billing_address": None,
                "contact_email": None,
                "activation_date": (start - timedelta(days=int(rng.integers(30, 1500)))).isoformat(),
                "status": "ACTIVE",
            }
        )
        truth.append({"record_id": f"telco:+{num[1:]}", "source": "telco", "entity_kind": "anonymous_line", "true_id": p.person_id})
    for num in sorted(w.burner_owner):
        info = w.burner_info[num]
        rows.append(
            {
                "msisdn": num[1:],
                "carrier": str(rng.choice(_CARRIERS)),
                "account_type": "PREPAID",
                "subscriber_name": info["alias"] or "",
                "subscriber_dob": None,
                "billing_address": None,
                "contact_email": None,
                "activation_date": (start + timedelta(days=int(info["activation_day"]))).isoformat(),
                "status": "ACTIVE",
            }
        )
        truth.append({"record_id": f"telco:+{num[1:]}", "source": "telco", "entity_kind": "burner", "true_id": w.burner_owner[num]})
    rows.sort(key=lambda r: r["msisdn"])
    return rows, truth


def render_cardholders(w: World) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rng = w.rng.stream("noise")
    n = w.cfg.noise
    rows: list[dict[str, Any]] = []
    truth: list[dict[str, str]] = []
    for num in sorted(w.cards):
        c = w.cards[num]
        p = w.person_by_id[c.holder_id]
        site = w.sites[c.site_id]
        if c.kind == "employee":
            name = _person_name(w, p, rng, "last_comma_first_mi", n.nickname_access)
        else:
            name = _person_name(w, p, rng, "first_last", 0.2)
        mobile = format_phone(p.phones[0], "paren") if p.phones and rng.random() < n.cardholder_phone_present else None
        rows.append(
            {
                "card_number": num,
                "cardholder_name": name,
                "employee_id": c.employee_id,
                "company": c.company,
                "department": c.department,
                "work_email": p.work_email if c.kind == "employee" else (p.personal_email if rng.random() < 0.6 else None),
                "mobile_phone": mobile,
                "card_type": "EMPLOYEE" if c.kind == "employee" else "MEMBER",
                "site_code": site.site_id,
                "card_status": "ACTIVE",
                "issued_date": c.issued.isoformat(),
            }
        )
        truth.append({"record_id": f"access:{num}", "source": "access", "entity_kind": "person", "true_id": p.person_id})
    return rows, truth


def render_readers(w: World) -> list[dict[str, Any]]:
    rows = []
    for site in sorted(w.sites.values(), key=lambda s: s.site_id):
        for r in site.readers:
            rows.append(
                {
                    "reader_id": r.reader_id,
                    "reader_name": r.name,
                    "site_code": site.site_id,
                    "site_name": site.name,
                    "site_type": site.kind.upper(),
                    "zone": r.zone,
                    "address": site.address.one_line(),
                    "lat": r.lat,
                    "lon": r.lon,
                }
            )
    return rows


def email_truth(w: World) -> list[dict[str, str]]:
    seen: set[str] = set()
    out = []
    for rec in w.email_headers:
        for field in ("from",):
            seen.add(_addr(rec[field]))
        for a in rec["to"] + rec["cc"]:
            seen.add(_addr(a))
    for addr in sorted(seen):
        owner = w.email_owner.get(addr, "")
        if not owner:
            continue
        kind = "org" if owner in w.orgs else "person"
        out.append({"record_id": f"email:{addr.lower()}", "source": "email", "entity_kind": kind, "true_id": owner})
    return out


def _addr(header: str) -> str:
    if "<" in header:
        return header.rsplit("<", 1)[1].rstrip(">").strip()
    return header.strip()


def render_watchlist(w: World) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rng = w.rng.stream("watchlist")
    rows: list[dict[str, Any]] = []
    truth: list[dict[str, str]] = []
    leaders = {"organizer", "coordinator", "herder", "handler"}
    subjects: list[tuple[Person | None, str]] = []
    for ring in w.rings:
        for m in ring["members"]:
            if m["role"] in leaders:
                subjects.append((w.person_by_id[m["person_id"]], "INTERNAL_REFERRAL"))
    benign = [p for p in w.persons if p.person_id not in w.reserved]
    for i in sorted(rng.choice(len(benign), size=min(4, len(benign)), replace=False)):
        subjects.append((benign[int(i)], "ADVERSE_MEDIA"))
    for _ in range(4):
        subjects.append((None, "LE_REQUEST"))
    order = rng.permutation(len(subjects))
    subjects = [subjects[int(k)] for k in order]
    for i, (p, reason) in enumerate(subjects):
        eid = f"WL-{2024 + i % 3}-{1000 + i * 7:05d}"
        if p is None:
            name = f"{w.fake.first_name()} {w.fake.last_name()}"
            dob = date(int(rng.integers(1955, 2000)), int(rng.integers(1, 13)), int(rng.integers(1, 28))).isoformat()
            aliases = ""
        else:
            nick = nickname(p.first, rng)
            name = format_person_name(p.first if rng.random() < 0.7 else nick, p.middle if rng.random() < 0.4 else None, p.last, None, "full")
            if rng.random() < 0.15:
                name = typo(name, rng)
            dob = p.dob.isoformat() if rng.random() < 0.75 else None
            alias_list = [f"{nick} {p.last}"] if nick != p.first else []
            if p.maiden_last:
                alias_list.append(f"{p.first} {p.maiden_last}")
            aliases = ";".join(alias_list)
            truth.append({"record_id": f"watchlist:{eid}", "source": "watchlist", "entity_kind": "person", "true_id": p.person_id})
        rows.append(
            {
                "entry_id": eid,
                "full_name": name,
                "aliases": aliases,
                "dob": dob,
                "list_name": "INTERNAL_HOTLIST",
                "reason_code": reason,
                "added_date": (w.cfg.start - timedelta(days=int(rng.integers(10, 700)))).isoformat(),
            }
        )
    return rows, truth


def render_case_outcomes(w: World) -> list[dict[str, Any]]:
    """Historical investigation outcomes keyed by bank customer (training labels)."""
    rng = w.rng.stream("labels")
    rows: list[dict[str, Any]] = []
    k = 0

    def add(p: "Person | Org", disposition: str, typology: str) -> None:
        nonlocal k
        if not p.customer_ids:
            return
        k += 1
        bank, cust = sorted(p.customer_ids.items())[0]
        opened = w.cfg.start + timedelta(days=int(rng.integers(w.cfg.days // 2, w.cfg.days)))
        rows.append(
            {
                "case_id": f"CASE-{w.cfg.start.year}-{k:05d}",
                "subject_record_id": f"bank:{bank}:{cust}",
                "disposition": disposition,
                "typology": typology,
                "opened_date": opened.isoformat(),
                "closed_date": (opened + timedelta(days=int(rng.integers(20, 120)))).isoformat(),
            }
        )

    for ring in w.rings:
        for m in ring["members"]:
            add(w.person_by_id[m["person_id"]], "CONFIRMED_SUSPICIOUS", ring["typology"])
        for o in ring["orgs"]:
            add(w.orgs[o["org_id"]], "CONFIRMED_SUSPICIOUS", ring["typology"])
        for v in ring.get("victims", []):
            add(w.person_by_id[v], "CLOSED_VICTIM", ring["typology"])
    for d in w.decoys:
        for m in d["members"]:
            if rng.random() < 0.5:
                add(w.person_by_id[m["person_id"]], "CLOSED_NO_ACTION", "unfounded_alert")
        if d.get("org_id") and rng.random() < 0.5:
            add(w.orgs[d["org_id"]], "CLOSED_NO_ACTION", "unfounded_alert")
    return rows
