"""Benign behaviours that share surface features with the planted typologies.

Without these, every red-flag feature is almost perfectly separating and any
detector looks brilliant. Each behaviour below is common in real customer
populations and individually innocent:

=========================  ===================================================
behaviour                  mimics
=========================  ===================================================
anonymous prepaid line     burner phones (US prepaid SIMs need no ID)
one-off large cash         structuring (selling a car, a cash gift)
bank switching             mule accounts (new account + large inflow)
big-ticket round wires     layering (car purchase, tuition, home deposit)
family loans               round-tripping (lend $X, get $X back days later)
IT on-call call-outs       insider after-hours restricted access
self-forwarded files       data exfiltration (large attachments to own mailbox)
=========================  ===================================================
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np

from .holdings import _pick_bank, open_account
from .reference import HOUR_PROFILES
from .sources.bank import BankLedger
from .sources.email import MailLog
from .sources.access import SwipeLog
from .sources.telco import CallLog
from .state import World, sample_hours

ANON_PREPAID_SHARE = 0.07
LARGE_CASH_SHARE = 0.03
BANK_SWITCH_SHARE = 0.05
BIG_TICKET_SHARE = 0.06
FAMILY_LOAN_SHARE = 0.05
SELF_FORWARD_SHARE = 0.08
ONCALL_SHARE = 0.35  # of restricted-access staff


def assign_background_holdings(w: World) -> None:
    """Holdings needed before activity is generated (anonymous lines, switched banks)."""
    rng = w.rng.stream("background")
    for p in w.persons:
        if p.phones and rng.random() < ANON_PREPAID_SHARE:
            line = w.ids.msisdn()
            w.phone_owner[line] = p.person_id
            p.traits.add("anon_prepaid")
            w.anon_lines[line] = p.person_id
        if p.accounts and rng.random() < BANK_SWITCH_SHARE:
            current = w.accounts[p.accounts[0]].bank_code
            day = int(rng.integers(5, max(6, w.cfg.days - 20)))
            acct = open_account(w, "person", p.person_id, _pick_bank(rng, exclude=current), "CHECKING", w.cfg.start + timedelta(days=day))
            p.traits.add("bank_switch")
            w.bank_switch[p.person_id] = (acct.key, day)


def generate_background(w: World, L: BankLedger, C: CallLog, S: SwipeLog, M: MailLog) -> None:
    rng = w.rng.stream("background")
    clock = w.clock
    by_owner = {owner: line for line, owner in sorted(w.anon_lines.items())}
    ties: dict[str, list[str]] = {}
    for r in w.relationships:
        if r.kind in ("household", "friend"):
            ties.setdefault(r.a, []).append(r.b)
            ties.setdefault(r.b, []).append(r.a)

    for p in w.persons:
        # anonymous prepaid line: a share of the person's personal calls go through it
        line = by_owner.get(p.person_id)
        if line:
            contacts = [w.person_by_id[q] for q in ties.get(p.person_id, []) if w.person_by_id[q].phones]
            for _ in range(int(rng.poisson(0.25 * clock.days))) if contacts else ():
                q = contacts[int(rng.integers(len(contacts)))]
                d = int(rng.integers(0, clock.days))
                ts = int(clock.ts(d, sample_hours(rng, HOUR_PROFILES["social_call"], 1)[0]))
                sms = rng.random() < 0.5
                if rng.random() < 0.5:
                    C.add(ts, line, q.phones[0], sms, int(rng.lognormal(np.log(100), 0.8)), p)
                else:
                    C.add(ts, q.phones[0], line, sms, int(rng.lognormal(np.log(100), 0.8)), q)
        if not p.accounts:
            continue
        acct = w.accounts[p.accounts[0]]
        branches = w.branches[acct.bank_code]
        # one-off large cash deposits (car sale, cash gift, emptied safe)
        if rng.random() < LARGE_CASH_SHARE:
            for _ in range(int(rng.choice([1, 1, 2]))):
                d = int(rng.integers(0, clock.days))
                if clock.business[d]:
                    amt = float(rng.choice([float(rng.uniform(8_000, 9_999)), float(rng.uniform(10_000, 18_000)), float(rng.uniform(5_000, 8_000))]))
                    L.cash_deposit(int(clock.ts(d, int(rng.uniform(10, 16) * 3600))), acct.key, round(amt, -1), branches[int(rng.integers(len(branches)))])
        # bank switching: savings moved into a brand-new account at another bank
        if p.person_id in w.bank_switch:
            key, day = w.bank_switch[p.person_id]
            amt = round(float(rng.lognormal(np.log(14_000), 0.6)), -2)
            ts = int(clock.ts(min(clock.days - 1, day + int(rng.integers(1, 6))), int(rng.uniform(9, 20) * 3600)))
            L.transfer(ts, amt, "ACH", src=acct.key, dst=key, memo="TRANSFER TO NEW ACCOUNT")
            if p.income_kind == "payroll" and rng.random() < 0.6:
                org = w.orgs[p.employer_id]  # type: ignore[index]
                for d in range(day + 7, clock.days, 14):
                    pay = p.salary / 26 * 0.72
                    if org.accounts:
                        L.transfer(int(clock.ts(clock.prev_or_same_business_day(d), 3 * 3600)), pay, "ACH", src=org.accounts[0], dst=key, memo="PAYROLL PPD", lag=0)
        # big-ticket purchases: round wires to dealers, schools, escrow agents
        if rng.random() < BIG_TICKET_SHARE:
            kind = str(rng.choice(["AUTO DEALER", "UNIVERSITY BURSAR", "TITLE & ESCROW", "CONTRACTOR"]))
            amt = float(rng.choice([5_000, 7_500, 10_000, 12_500, 15_000, 20_000, 25_000, 30_000, 45_000]))
            d = int(rng.integers(0, clock.days))
            if clock.business[d]:
                name = f"{w.fake.last_name().upper()} {kind}"
                L.transfer(int(clock.ts(d, int(rng.uniform(10, 15) * 3600))), amt, "WIRE", src=acct.key, dst_ext=(w.ids.external_account(), name, "UNITED COMMERCE BANK"), memo=str(rng.choice(["DEPOSIT", "BALANCE DUE", "TUITION", "CLOSING FUNDS"])))
        # family loans that are repaid quickly (a benign 2-cycle)
        if rng.random() < FAMILY_LOAN_SHARE:
            fam = [w.person_by_id[q] for q in ties.get(p.person_id, []) if w.person_by_id[q].accounts]
            if fam:
                q = fam[int(rng.integers(len(fam)))]
                amt = float(rng.choice([1_000, 1_500, 2_000, 2_500, 3_000, 5_000]))
                d = int(rng.integers(0, clock.days - 14))
                L.transfer(int(clock.ts(d, 19 * 3600)), amt, "P2P", src=acct.key, dst=q.accounts[0], memo="loan")
                back = d + int(rng.integers(3, 10))
                L.transfer(int(clock.ts(back, 20 * 3600)), amt if rng.random() < 0.6 else amt * float(rng.uniform(0.85, 1.0)), "P2P", src=q.accounts[0], dst=acct.key, memo="paying you back")
                if p.phones and q.phones:
                    C.add(int(clock.ts(d, 18 * 3600)), p.phones[0], q.phones[0], False, int(rng.lognormal(np.log(240), 0.5)), p)
        # e-mailing yourself big files (photos, decks) from work
        if p.work_email and p.personal_email and rng.random() < SELF_FORWARD_SHARE:
            for _ in range(int(rng.integers(1, 4))):
                d = int(rng.choice(np.flatnonzero(clock.business)))
                ts = int(clock.ts(d, int(rng.uniform(11, 19) * 3600)))
                M.add(ts, p.work_email, [p.personal_email], subject=str(rng.choice(["fwd", "photos", "slides", "for home", ""])), attachments=M.random_attachments(1, size_scale=45.0, prefix=str(rng.choice(["IMG", "deck", "scan"]))))
        # IT / facilities on-call: after-hours visits to restricted rooms
        if "restricted_access" in p.traits and rng.random() < ONCALL_SHARE:
            site = w.sites[w.cards[next(c for c in p.cards if w.cards[c].kind == "employee")].site_id]
            room = site.reader("SERVER-ROOM-B1" if p.department == "Information Technology" else "RECORDS-ROOM-3F" if p.department == "Legal and Compliance" else "LOADING-DOCK-01")
            card = next(c for c in p.cards if w.cards[c].kind == "employee")
            for _ in range(int(rng.poisson(clock.days / 30))):
                d = int(rng.integers(0, clock.days))
                t = int(clock.ts(d, int(rng.choice([rng.uniform(20, 23.9), rng.uniform(0, 5.5)]) * 3600)))
                S.swipe(t, card, site.reader("TURNSTILE-01").reader_id)
                S.swipe(t + int(rng.integers(60, 300)), card, room.reader_id)
                if rng.random() < 0.1:
                    S.swipe(t + int(rng.integers(400, 900)), card, site.reader("RECORDS-ROOM-3F").reader_id, granted=False, reason="NO_PRIVILEGE")
                S.swipe(t + int(rng.integers(1800, 5400)), card, site.reader("TURNSTILE-01").reader_id, direction="OUT")
