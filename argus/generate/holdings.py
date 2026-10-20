"""Who holds which identifiers: bank accounts, phone lines, e-mail addresses, access cards."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from .identifiers import slug
from .model import Account, Card, Org, Person
from .reference import BANKS, NICKNAMES, PERSONAL_EMAIL_DOMAINS, RESTRICTED_ACCESS_DEPTS
from .state import World

_BANK_BY_CODE = {b.code: b for b in BANKS}


def _pick_bank(rng: np.random.Generator, exclude: str | None = None) -> str:
    banks = [b for b in BANKS if b.code != exclude]
    p = np.array([b.market_share for b in banks])
    return banks[int(rng.choice(len(banks), p=p / p.sum()))].code


def open_account(
    w: World,
    owner_kind: str,
    owner_id: str,
    bank_code: str,
    product: str,
    open_date: date,
) -> Account:
    """Open an account (creating the customer relationship at that bank if needed)."""
    holder: Person | Org = w.person_by_id[owner_id] if owner_kind == "person" else w.orgs[owner_id]
    if bank_code not in holder.customer_ids:
        holder.customer_ids[bank_code] = w.ids.customer_id(_BANK_BY_CODE[bank_code].customer_prefix)
    number = w.ids.account_number(bank_code)
    acct = Account(
        key=f"{bank_code}:{number}",
        bank_code=bank_code,
        number=number,
        owner_kind=owner_kind,
        owner_id=owner_id,
        product=product,
        open_date=open_date,
        customer_id=holder.customer_ids[bank_code],
    )
    w.accounts[acct.key] = acct
    holder.accounts.append(acct.key)
    return acct


def assign_banking(w: World) -> None:
    rng = w.rng.stream("holdings")
    start = w.cfg.start
    for p in w.persons:
        if rng.random() < 0.05 and p.age_on(start) < 30:  # unbanked young adults
            continue
        bank = _pick_bank(rng)
        tenure = int(rng.integers(120, max(150, (p.age_on(start) - 17) * 365)))
        open_account(w, "person", p.person_id, bank, "CHECKING", start - timedelta(days=tenure))
        if rng.random() < 0.55:
            open_account(w, "person", p.person_id, bank, "SAVINGS", start - timedelta(days=int(rng.integers(60, tenure + 1))))
        if rng.random() < w.cfg.noise.second_bank_relationship:
            other = _pick_bank(rng, exclude=bank)
            open_account(w, "person", p.person_id, other, str(rng.choice(["CHECKING", "SAVINGS"])), start - timedelta(days=int(rng.integers(90, 3000))))
    for org in w.orgs.values():
        if org.kind == "employer" and (org.major or rng.random() < 0.7):
            open_account(w, "org", org.org_id, _pick_bank(rng), "BUSINESS_CHECKING", org.inc_date + timedelta(days=30))  # type: ignore[operator]
        elif org.kind == "employer":
            org.ext_account = w.ids.external_account()
            org.attrs["bank"] = "UNITED COMMERCE BANK"


def assign_phones(w: World) -> None:
    rng = w.rng.stream("holdings")
    for p in w.persons:
        if rng.random() < 0.97:
            line = w.ids.msisdn()
            p.phones.append(line)
            w.phone_owner[line] = p.person_id
            if rng.random() < 0.13:
                p.prepaid_phones.add(line)  # registered prepaid line (KYC-light)
        if p.phones and rng.random() < 0.07:
            line = w.ids.msisdn()
            p.phones.append(line)
            w.phone_owner[line] = p.person_id
    for org in w.orgs.values():
        if org.kind == "employer":
            org.phone = w.ids.msisdn()


def _local_part(p: Person, rng: np.random.Generator) -> str:
    first, last = p.first.lower(), p.last.lower()
    nick = NICKNAMES.get(first, (first,))[0]
    yy = f"{p.dob.year % 100:02d}"
    pattern = rng.choice(["first.last", "firstlast", "flast", "first_lastNN", "nickNN", "firstl", "lastfirst", "wordNN"], p=[0.24, 0.16, 0.14, 0.12, 0.1, 0.08, 0.06, 0.1])
    word = str(rng.choice(["sunny", "blue", "jersey", "runner", "coffee", "tiger", "gamer", "rocket", "maple", "pixel"]))
    return {
        "first.last": f"{first}.{last}",
        "firstlast": f"{first}{last}",
        "flast": f"{first[0]}{last}",
        "first_lastNN": f"{first}_{last}{yy}",
        "nickNN": f"{nick}{yy}",
        "firstl": f"{first}{last[0]}",
        "lastfirst": f"{last}.{first}",
        "wordNN": f"{word}{nick}{int(rng.integers(1, 999))}",
    }[str(pattern)]


def assign_emails(w: World) -> None:
    rng = w.rng.stream("holdings")
    for p in w.persons:
        if rng.random() < 0.96:
            addr = w.ids.email(slug(_local_part(p, rng), sep="."), str(rng.choice(PERSONAL_EMAIL_DOMAINS)))
            p.personal_email = addr
            w.email_owner[addr] = p.person_id
        if p.employer_id:
            org = w.orgs[p.employer_id]
            addr = w.ids.email(f"{slug(p.first, sep='')}.{slug(p.last, sep='')}", org.domain)  # type: ignore[arg-type]
            p.work_email = addr
            w.email_owner[addr] = p.person_id


def issue_card(w: World, person: Person, site_id: str, kind: str, issued: date, department: str | None = None) -> Card:
    site = w.sites[site_id]
    employer = w.orgs[site.employer_id] if site.employer_id else None
    card = Card(
        number=w.ids.access_card(site.facility_code),
        holder_id=person.person_id,
        site_id=site_id,
        employee_id=w.ids.employee_id(slug(employer.name, sep="")[:3].upper()) if employer else None,
        company=employer.name if employer else site.name,
        department=department,
        issued=issued,
        kind=kind,
    )
    w.cards[card.number] = card
    person.cards.append(card.number)
    return card


def assign_cards(w: World) -> None:
    rng = w.rng.stream("holdings")
    start = w.cfg.start
    gyms = [s for s in w.sites.values() if s.kind == "gym"]
    storage = [s for s in w.sites.values() if s.kind == "storage"]
    cowork = [s for s in w.sites.values() if s.kind == "coworking"]
    for p in w.persons:
        if p.employer_id and w.orgs[p.employer_id].major:
            site_id = w.orgs[p.employer_id].site_id
            issue_card(w, p, site_id, "employee", start - timedelta(days=int(rng.integers(30, 3000))), p.department)  # type: ignore[arg-type]
            if p.department in RESTRICTED_ACCESS_DEPTS:
                p.traits.add("restricted_access")
            if p.department == "Warehouse":
                p.traits.add("warehouse")
        age = p.age_on(start)
        if age < 60 and rng.random() < 0.16:
            issue_card(w, p, gyms[int(rng.integers(len(gyms)))].site_id, "member", start - timedelta(days=int(rng.integers(10, 900))))
            p.traits.add("gym")
        if rng.random() < 0.03:
            issue_card(w, p, storage[int(rng.integers(len(storage)))].site_id, "member", start - timedelta(days=int(rng.integers(10, 900))))
            p.traits.add("storage")
        if p.income_kind == "self" and rng.random() < 0.35:
            issue_card(w, p, cowork[int(rng.integers(len(cowork)))].site_id, "member", start - timedelta(days=int(rng.integers(10, 600))))
            p.traits.add("coworking")


def assign_housing(w: World) -> None:
    """Rent vs. mortgage; rent payees are property managers (external) by default.

    Landlord decoys later re-point some tenants' rent to their property LLC.
    """
    rng = w.rng.stream("holdings")
    pms = [o for o in w.orgs.values() if o.kind == "property_mgmt"]
    for members in w.households.values():
        head = w.person_by_id[members[0]]
        own = head.age_on(w.cfg.start) > 35 and rng.random() < 0.6
        for i, pid in enumerate(members):
            p = w.person_by_id[pid]
            if i == 0:
                p.housing = "own" if own else "rent"
                if own:
                    p.rent = round(float(rng.lognormal(np.log(2100), 0.3)), 2)  # mortgage P&I, not round
                else:
                    p.rent = float(round(rng.lognormal(np.log(1750), 0.25) / 25) * 25)  # rent is a round figure
                    p.landlord_account = pms[int(rng.integers(len(pms)))].org_id
            else:
                p.housing = "family"
