"""Planted "hidden network" scenarios with known ground truth.

Each typology follows patterns documented in public AML/fraud guidance
(FATF typology reports, FinCEN advisories) and spans several feeds, so it is
only fully visible once the feeds are fused:

``shell_layering``     round-tripping through shell LLCs registered at one virtual
                       office; wires are preceded by coordinating calls (often from
                       prepaid burners), "invoice" e-mails and after-hours meetings.
``structuring``        smurfs make cash deposits just under the $10k reporting
                       threshold at different branches, then funnel to a front business.
``mule_funnel``        scam victims pay newly opened mule accounts that forward
                       85-92% within hours to a herder or a crypto on-ramp.
``insider_collusion``  an employee with restricted access makes after-hours visits
                       to a server/records room, mails large attachments to a personal
                       address and receives round "consulting" payments from an outsider.

Ring members also live ordinary lives (payroll, groceries, family calls), so
the signal has to be found inside normal behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

import numpy as np

from .holdings import issue_card, open_account
from .identifiers import slug
from .model import Org, Person, Relationship, Site
from .population import random_address
from .reference import CASH_BUSINESS_TYPES, SHELL_WORDS_A, SHELL_WORDS_B
from .sources.access import SwipeLog
from .sources.bank import BankLedger, crypto_merchant
from .sources.email import MailLog
from .sources.telco import CallLog
from .state import World


@dataclass
class Ctx:
    w: World
    L: BankLedger
    C: CallLog
    S: SwipeLog
    M: MailLog
    rng: np.random.Generator


def round_to(x: float, step: float) -> float:
    return float(max(step, round(x / step) * step))


# --------------------------------------------------------------------------
# Setup helpers (run before activity generation)
# --------------------------------------------------------------------------


def pick_people(w: World, rng: np.random.Generator, n: int, pred: Callable[[Person], bool]) -> list[Person]:
    pool = w.eligible(lambda p: bool(p.accounts) and bool(p.phones) and pred(p))
    if len(pool) < n:
        pool = w.eligible(lambda p: bool(p.accounts) and bool(p.phones))
    if len(pool) < n:
        raise RuntimeError("population too small to plant scenarios; increase n_people")
    idx = sorted(int(i) for i in rng.choice(len(pool), size=n, replace=False))
    chosen = [pool[i] for i in idx]
    for p in chosen:
        w.reserved.add(p.person_id)
    return chosen


def make_org(
    w: World,
    rng: np.random.Generator,
    kind: str,
    name: str,
    owners: list[Person],
    address: Any,
    inc_days_before: int,
    bank: str,
    acct_open_days_before: int,
) -> Org:
    prefix = f"ORG-{kind[:5].upper()}-"
    seq = sum(1 for k in w.orgs if k.startswith(prefix)) + 1
    org = Org(
        org_id=f"{prefix}{seq:03d}",
        name=name,
        kind=kind,
        reg_no=w.ids.reg_no(),
        address=address,
        inc_date=w.cfg.start - timedelta(days=inc_days_before),
        owners=[p.person_id for p in owners],
        domain=f"{slug(name)[:26].strip('-')}.example",
    )
    w.orgs[org.org_id] = org
    open_account(w, "org", org.org_id, bank, "BUSINESS_CHECKING", w.cfg.start - timedelta(days=acct_open_days_before))
    return org


def unique_name(w: World, rng: np.random.Generator, make: Callable[[], str]) -> str:
    existing = {o.name for o in w.orgs.values()}
    for _ in range(200):
        n = make()
        if n not in existing:
            return n
    raise RuntimeError("could not find unique org name")


def shell_name(w: World, rng: np.random.Generator) -> str:
    return unique_name(w, rng, lambda: f"{rng.choice(SHELL_WORDS_A)} {rng.choice(SHELL_WORDS_B)} {rng.choice(['LLC', 'LLC', 'Ltd', 'Inc'])}")


def burner(w: World, rng: np.random.Generator, owner: Person, day: int) -> str:
    num = w.ids.msisdn(metro=rng.random() < 0.6)
    w.burner_owner[num] = owner.person_id
    w.phone_owner[num] = owner.person_id
    alias = None
    if rng.random() < 0.4:  # registered under a throwaway name
        alias = f"{w.fake.first_name()} {w.fake.last_name()}"
    w.burner_info[num] = {"activation_day": max(0, day), "alias": alias}
    return num


def any_bank(rng: np.random.Generator) -> str:
    return str(rng.choice(["FMB", "CFCU", "HVT"]))


def ensure_card(w: World, p: Person, site: Site) -> str:
    for c in p.cards:
        if w.cards[c].site_id == site.site_id:
            return c
    return issue_card(w, p, site.site_id, "member", w.cfg.start - timedelta(days=30)).number


def add_ties(w: World, people: list[Person], kind: str) -> None:
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            a, b = sorted((people[i].person_id, people[j].person_id))
            w.relationships.append(Relationship(a, b, kind, 0.0, 0.0, 0.0))


# --------------------------------------------------------------------------
# Activity helpers
# --------------------------------------------------------------------------


def biz_ts(ctx: Ctx, day: int, h0: float = 9.0, h1: float = 16.5) -> int:
    clock = ctx.w.clock
    d = min(max(day, 0), clock.days - 1)
    if not clock.business[d]:
        d = clock.next_business_day(d)
    return int(clock.ts(d, int(ctx.rng.uniform(h0, h1) * 3600)))


def ring_contact(ctx: Ctx, ts: int, a: Person, b: Person, a_num: str | None, b_num: str | None, tag: str, sms_p: float = 0.35, opsec: float = 0.8) -> None:
    """A coordination call/SMS. With probability ``1-opsec`` a party slips and uses a personal phone."""
    rng = ctx.rng
    an = a_num if (a_num and rng.random() < opsec) else a.phones[0]
    bn = b_num if (b_num and rng.random() < opsec) else b.phones[0]
    sms = rng.random() < sms_p
    ctx.C.add(ts, an, bn, sms, int(rng.lognormal(np.log(75), 0.6)), a, tag=tag)


def meet(ctx: Ctx, people: list[Person], site: Site, day: int, hour: float, tag: str, room: str) -> None:
    """Co-located badge-ins of ``people`` at ``site`` within a few minutes."""
    rng = ctx.rng
    base = int(ctx.w.clock.ts(day, int(hour * 3600)))
    for p in people:
        card = ensure_card(ctx.w, p, site)
        t = base + int(rng.integers(-420, 420))
        entry = next((r for r in site.readers if r.zone in ("TRANSIT", "FACILITY")), site.readers[0])
        ctx.S.swipe(t, card, entry.reader_id, tag=tag)
        if room:
            ctx.S.swipe(t + int(rng.integers(40, 200)), card, site.reader(room).reader_id, tag=tag)


def window(ctx: Ctx, min_len: int, max_len: int) -> tuple[int, int]:
    days = ctx.w.clock.days
    length = int(ctx.rng.integers(min(min_len, days - 7), min(max_len, days - 2) + 1))
    start = int(ctx.rng.integers(0, max(1, days - length)))
    return start, start + length


# --------------------------------------------------------------------------
# Typologies
# --------------------------------------------------------------------------


class Scenario:
    """Base class: ``setup`` reserves people/holdings, ``run`` emits activity."""

    typology = ""

    def __init__(self, ring_id: str, w: World, rng: np.random.Generator) -> None:
        self.ring_id = ring_id
        self.w = w
        self.rng = rng
        self.members: list[tuple[Person, str]] = []
        self.orgs: list[Org] = []
        self.burners: dict[str, list[str]] = {}
        self.extra: dict[str, Any] = {}

    def setup(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def run(self, ctx: Ctx) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def truth(self) -> dict[str, Any]:
        return {
            "ring_id": self.ring_id,
            "typology": self.typology,
            "members": [{"person_id": p.person_id, "role": role} for p, role in self.members],
            "orgs": [{"org_id": o.org_id, "name": o.name, "kind": o.kind, "accounts": o.accounts} for o in self.orgs],
            "burners": self.burners,
            **self.extra,
        }

    def burner_for(self, p: Person) -> str | None:
        b = self.burners.get(p.person_id)
        return b[0] if b else None


class ShellLayering(Scenario):
    typology = "shell_layering"

    def setup(self) -> None:
        w, rng = self.w, self.rng
        org_pp = pick_people(w, rng, 1, lambda p: 35 <= p.age_on(w.cfg.start) <= 66)
        nominees = pick_people(w, rng, int(rng.integers(2, 4)), lambda p: 24 <= p.age_on(w.cfg.start) <= 72)
        accountant = pick_people(w, rng, 1, lambda p: p.department == "Finance" or p.income_kind == "self")
        organizer = org_pp[0]
        self.members = [(organizer, "organizer")] + [(n, "nominee") for n in nominees] + [(accountant[0], "facilitator")]
        town = int(rng.integers(len(w.towns)))
        agent = random_address(w, town, rng, unit_p=0.0)
        self.extra["registered_agent_address"] = f"{agent.number} {agent.street}, {agent.city}, {agent.state} {agent.zip}"
        front = make_org(w, rng, "front", unique_name(w, rng, lambda: f"{rng.choice(SHELL_WORDS_A)} Trading Co"), [organizer], random_address(w, organizer.town, rng, 0.3), int(rng.integers(700, 2200)), any_bank(rng), int(rng.integers(500, 1800)))
        self.orgs.append(front)
        shells: list[Org] = []
        n_shells = int(rng.integers(3, 6))
        for i in range(n_shells):
            nominee = nominees[i % len(nominees)]
            addr = random_address(w, town, rng, 0.0)
            addr.number, addr.street, addr.city, addr.state, addr.zip = agent.number, agent.street, agent.city, agent.state, agent.zip
            addr.unit = f"Suite {100 + int(rng.integers(1, 300))}"
            shells.append(make_org(w, rng, "shell", shell_name(w, rng), [nominee], addr, int(rng.integers(60, 420)), any_bank(rng), int(rng.integers(15, 160))))
        self.orgs += shells
        self.shells = shells
        self.front = front
        self.organizer, self.nominees, self.accountant = organizer, nominees, accountant[0]
        for p in [organizer] + nominees:
            self.burners[p.person_id] = [burner(w, rng, p, int(rng.integers(0, 20)))]
        if rng.random() < 0.5:
            self.burners[organizer.person_id].append(burner(w, rng, organizer, int(rng.integers(30, 90))))
        coworking = [s for s in w.sites.values() if s.kind == "coworking"]
        self.site = coworking[int(rng.integers(len(coworking)))]
        for p in [organizer] + nominees:
            ensure_card(w, p, self.site)
        add_ties(w, [organizer] + nominees + accountant, "ring")

    def controller(self, org: Org) -> Person:
        return self.w.person_by_id[org.owners[0]]

    @property
    def opsec(self) -> float:
        return float(self.extra.setdefault("opsec", round(float(self.rng.uniform(0.75, 0.97)), 2)))

    def run(self, ctx: Ctx) -> None:
        w, rng, tag = ctx.w, ctx.rng, self.ring_id
        s0, s1 = window(ctx, 90, 170)
        n_cycles = int(rng.integers(3, 7))
        starts = sorted(int(x) for x in rng.choice(np.arange(s0, max(s0 + n_cycles, s1 - 12)), size=n_cycles, replace=False))
        cycles = []
        foreign = [(w.ids.external_account(), f"{rng.choice(SHELL_WORDS_A).upper()} INTERNATIONAL FZE", "OFFSHORE COMMERCIAL BANK LTD") for _ in range(2)]
        for ci, day in enumerate(starts):
            amount = round_to(float(rng.uniform(40_000, 150_000)), 5_000)
            src = foreign[ci % 2]
            ts = biz_ts(ctx, day)
            ctx.L.transfer(ts, amount, "WIRE", src_ext=src, dst=self.front.accounts[0], memo=f"TRADE SETTLEMENT {int(rng.integers(1000, 9999))}", tag=tag)
            k = int(rng.integers(3, len(self.shells) + 1))
            path = [self.shells[i] for i in rng.choice(len(self.shells), size=k, replace=False)]
            hops: list[tuple[Org | Person, Org | Person]] = [(self.front, path[0])] + list(zip(path[:-1], path[1:]))
            end: Org | Person = self.front if rng.random() < 0.5 else self.organizer
            hops.append((path[-1], end))
            t = ts + int(rng.integers(3 * 3600, 30 * 3600))
            amt = amount
            for hi, (a, b) in enumerate(hops):
                amt = round_to(amt * (1 - float(rng.uniform(0.004, 0.022))), 250)
                day_t = int((t - w.clock.day_start[0]) // 86400)
                t = max(biz_ts(ctx, day_t), t)
                payer_person = self.organizer if isinstance(a, Org) and a is self.front else (a if isinstance(a, Person) else self.controller(a))
                payee_person = self.organizer if (isinstance(b, Org) and b is self.front) or isinstance(b, Person) else self.controller(b)
                # coordination: organizer phones the controller of the paying entity
                other = payer_person if payer_person is not self.organizer else payee_person
                if other is not self.organizer:
                    ring_contact(ctx, t - int(rng.integers(30 * 60, 4 * 3600)), self.organizer, other, self.burner_for(self.organizer), self.burner_for(other), tag, sms_p=0.25, opsec=self.opsec)
                # "invoice" from the receiving side a day before
                if isinstance(b, Org) and b is not self.front and payee_person.personal_email and payer_person.personal_email and rng.random() < 0.8:
                    ctx.M.add(t - int(rng.integers(12 * 3600, 30 * 3600)), payee_person.personal_email, [payer_person.personal_email], subject=f"Invoice INV-{w.cfg.start.year}-{int(rng.integers(100, 999)):03d} - {b.name}", attachments=ctx.M.random_attachments(1, prefix="invoice"), tag=tag)
                src_key = a.accounts[0] if isinstance(a, Org) else a.accounts[0]
                dst_key = b.accounts[0] if isinstance(b, Org) else b.accounts[-1]
                memo = str(rng.choice(["CONSULTING SERVICES", "MGMT FEE", "LOAN REPAYMENT", "INV SETTLEMENT", "ADVISORY Q" + str(1 + (day_t // 91))]))
                ctx.L.transfer(t, amt, "WIRE", src=src_key, dst=dst_key, memo=memo, tag=tag)
                if other is not self.organizer and rng.random() < 0.6:
                    ring_contact(ctx, t + int(rng.integers(600, 5400)), other, self.organizer, self.burner_for(other), self.burner_for(self.organizer), tag, sms_p=0.9, opsec=self.opsec)
                t += int(rng.integers(5 * 3600, 3 * 86400))
            cycles.append({"start_day": day, "placement": amount, "hops": len(hops), "path": [o.org_id for o in path]})
            # accountant reconciles by e-mail
            if self.accountant.personal_email and self.organizer.personal_email:
                ctx.M.add(t, self.accountant.personal_email, [self.organizer.personal_email], subject="Re: reconciliation", attachments=ctx.M.random_attachments(1, prefix="ledger"), tag=tag)
                ring_contact(ctx, t + 3600, self.accountant, self.organizer, None, self.burner_for(self.organizer), tag, sms_p=0.1, opsec=0.5)
        # after-hours meetings at the co-working space
        for _ in range(int(rng.integers(2, 5))):
            d = int(rng.integers(s0, s1))
            meet(ctx, [self.organizer] + self.nominees, self.site, d, float(rng.uniform(19.0, 21.5)), tag, "MEETING-ROOM-B")
        self.extra.update({"window_days": [s0, s1], "cycles": cycles, "meeting_site": self.site.site_id})


class Structuring(Scenario):
    typology = "structuring"

    def setup(self) -> None:
        w, rng = self.w, self.rng
        coord = pick_people(w, rng, 1, lambda p: 28 <= p.age_on(w.cfg.start) <= 62)[0]
        smurfs = pick_people(w, rng, int(rng.integers(4, 8)), lambda p: 19 <= p.age_on(w.cfg.start) <= 48)
        self.coord, self.smurfs = coord, smurfs
        self.members = [(coord, "coordinator")] + [(s, "smurf") for s in smurfs]
        name = unique_name(w, rng, lambda: f"{w.fake.last_name()}'s {rng.choice(CASH_BUSINESS_TYPES)} LLC")
        self.front = make_org(w, rng, "front", name, [coord], random_address(w, coord.town, rng, 0.0), int(rng.integers(400, 1500)), any_bank(rng), int(rng.integers(200, 1000)))
        self.orgs.append(self.front)
        self.burners[coord.person_id] = [burner(w, rng, coord, int(rng.integers(0, 30)))]
        add_ties(w, [coord] + smurfs, "ring")

    def run(self, ctx: Ctx) -> None:
        w, rng, tag = ctx.w, ctx.rng, self.ring_id
        s0, s1 = window(ctx, 70, 140)
        d = s0
        weeks = 0
        participation = {sm.person_id: float(rng.uniform(0.35, 0.8)) for sm in self.smurfs}
        while d < s1 - 3:
            for sm in self.smurfs:
                if rng.random() > participation[sm.person_id]:
                    continue
                acct = w.accounts[sm.accounts[0]]
                branches = list(w.branches[acct.bank_code])
                rng.shuffle(branches)
                n_dep = int(rng.choice([1, 1, 2, 3]))
                total = 0.0
                for k in range(n_dep):
                    day = min(w.clock.days - 1, d + k)
                    if not w.clock.business[day]:
                        day = w.clock.next_business_day(day)
                    amt = round_to(float(rng.uniform(8_150, 9_950)), float(rng.choice([10, 50, 100])))
                    ts = int(w.clock.ts(day, int(rng.uniform(9.5, 16.5) * 3600)))
                    ring_contact(ctx, ts - int(rng.integers(40 * 60, 3 * 3600)), self.coord, sm, self.burner_for(self.coord), None, tag, sms_p=0.5, opsec=0.75)
                    ctx.L.cash_deposit(ts, acct.key, amt, branches[k % len(branches)], tag=tag)
                    total += amt
                pay_day = min(w.clock.days - 1, d + n_dep + int(rng.integers(0, 3)))
                dst = self.front.accounts[0] if rng.random() < 0.7 else self.coord.accounts[0]
                ctx.L.transfer(biz_ts(ctx, pay_day, 8, 20), round_to(total * float(rng.uniform(0.9, 0.97)), 10), str(rng.choice(["P2P", "ACH"])), src=acct.key, dst=dst, memo=str(rng.choice(["LOAN REPAYMENT", "", "invoice", "supplies"])), tag=tag)
            # the coordinator also structures cash into the front directly
            for _ in range(int(rng.integers(1, 3))):
                day = min(w.clock.days - 1, d + int(rng.integers(0, 5)))
                br = w.branches[w.accounts[self.front.accounts[0]].bank_code]
                ctx.L.cash_deposit(biz_ts(ctx, day), self.front.accounts[0], round_to(float(rng.uniform(8_500, 9_900)), 100), br[int(rng.integers(len(br)))], tag=tag)
            d += int(rng.integers(9, 21))
            weeks += 1
        # the front pays the coordinator round "distributions"
        for m in range(s0 + 10, s1, 30):
            ctx.L.transfer(biz_ts(ctx, m), round_to(float(rng.uniform(15_000, 40_000)), 1_000), "BOOK" if w.accounts[self.front.accounts[0]].bank_code == w.accounts[self.coord.accounts[0]].bank_code else "ACH", src=self.front.accounts[0], dst=self.coord.accounts[0], memo="OWNER DRAW", tag=tag)
        self.extra.update({"window_days": [s0, s1], "deposit_rounds": weeks})


class MuleFunnel(Scenario):
    typology = "mule_funnel"

    def setup(self) -> None:
        w, rng = self.w, self.rng
        herder = pick_people(w, rng, 1, lambda p: 24 <= p.age_on(w.cfg.start) <= 52)[0]
        mules = pick_people(w, rng, int(rng.integers(5, 10)), lambda p: 19 <= p.age_on(w.cfg.start) <= 29 or p.income_kind in ("parents", "benefits"))
        self.herder, self.mules = herder, mules
        self.members = [(herder, "herder")] + [(m, "mule") for m in mules]
        # mule accounts are new: opened just before or during the window
        self.mule_accounts: dict[str, str] = {}
        for m in mules:
            bank = any_bank(rng)
            # the mule keeps an everyday primary account; the new one is used for the scheme
            acct = open_account(w, "person", m.person_id, bank, "CHECKING", w.cfg.start + timedelta(days=int(rng.integers(-60, 25))))
            self.mule_accounts[m.person_id] = acct.key
        name = unique_name(w, rng, lambda: f"{rng.choice(SHELL_WORDS_A)} Digital Assets LLC")
        self.collector = make_org(w, rng, "crypto_front", name, [herder], random_address(w, herder.town, rng, 0.6), int(rng.integers(30, 200)), any_bank(rng), int(rng.integers(10, 120)))
        self.orgs.append(self.collector)
        self.burners[herder.person_id] = [burner(w, rng, herder, 0), burner(w, rng, herder, int(rng.integers(20, 60)))]
        victims = [p for p in w.eligible(lambda p: bool(p.accounts) and p.age_on(w.cfg.start) >= 45)]
        k = min(len(victims), int(rng.integers(8, 16)))
        self.victims = [victims[int(i)] for i in sorted(rng.choice(len(victims), size=k, replace=False))]
        for v in self.victims:
            w.reserved.add(v.person_id)
        self.extra["victims"] = [v.person_id for v in self.victims]
        add_ties(w, [herder] + mules, "ring")

    def run(self, ctx: Ctx) -> None:
        w, rng, tag = ctx.w, ctx.rng, self.ring_id
        s0, s1 = window(ctx, 60, 120)
        crypto = crypto_merchant(w)
        recruiter = f"careers@{self.collector.domain}"
        w.email_owner[recruiter] = self.collector.org_id
        for m in self.mules:
            if m.personal_email:
                ctx.M.add(int(w.clock.ts(max(0, s0 - int(rng.integers(3, 15))), 10 * 3600)), recruiter, [m.personal_email], subject="Remote Payment Processing Assistant - Offer Letter", attachments=ctx.M.random_attachments(1, prefix="offer_letter"), tag=tag)
        ext_victims = [(w.ids.external_account(), f"{w.fake.first_name().upper()} {w.fake.last_name().upper()}", str(rng.choice(["NATIONAL CITY BANK NA", "UNITED COMMERCE BANK"]))) for _ in range(int(rng.integers(6, 14)))]
        hb = self.burners[self.herder.person_id]
        n_flows = 0
        for m in self.mules:
            acct = self.mule_accounts[m.person_id]
            a0 = int(rng.integers(s0, max(s0 + 1, s1 - 30)))
            a1 = min(s1, a0 + int(rng.integers(28, 80)))
            for week in range(a0, a1, 7):
                for _ in range(int(rng.integers(1, 4))):
                    day = min(w.clock.days - 2, week + int(rng.integers(0, 7)))
                    amt = float(rng.lognormal(np.log(1_800), 0.55))
                    ts = int(w.clock.ts(day, int(rng.uniform(8, 21) * 3600)))
                    if rng.random() < 0.55 and self.victims:
                        v = self.victims[int(rng.integers(len(self.victims)))]
                        if v.phones and rng.random() < 0.6:
                            ring_contact(ctx, ts - int(rng.integers(3600, 3 * 86400)), self.herder, v, hb[-1], None, tag, sms_p=0.3, opsec=0.97)
                        ctx.L.transfer(ts, amt, "P2P" if amt < 2_500 else str(rng.choice(["ACH", "WIRE"])), src=v.accounts[0], dst=acct, memo=str(rng.choice(["", "invoice", "loan", "gift", "deposit", "for the package"])), tag=tag)
                    else:
                        ctx.L.transfer(ts, amt, str(rng.choice(["P2P", "ACH", "WIRE"])), src_ext=ext_victims[int(rng.integers(len(ext_victims)))], dst=acct, memo=str(rng.choice(["", "invoice payment", "refund"])), tag=tag)
                    ring_contact(ctx, ts + int(rng.integers(600, 7200)), self.herder, m, hb[int(rng.integers(len(hb)))], None, tag, sms_p=0.85, opsec=0.95)
                    fwd = amt * float(rng.uniform(0.85, 0.92))
                    t2 = ts + int(rng.integers(2 * 3600, 36 * 3600))
                    if rng.random() < 0.6:
                        dst = self.collector.accounts[0] if rng.random() < 0.6 else self.herder.accounts[0]
                        ctx.L.transfer(t2, fwd, "P2P" if fwd < 2_500 else "WIRE", src=acct, dst=dst, memo="", tag=tag)
                    else:
                        ctx.L.card_batch(acct, np.array([t2], dtype=np.int64), np.array([fwd]), [crypto], tag=tag)
                    if rng.random() < 0.5:
                        ring_contact(ctx, t2 + int(rng.integers(120, 1800)), m, self.herder, None, hb[-1], tag, sms_p=0.95, opsec=0.9)
                    n_flows += 1
        # herder cashes out the collector via the crypto on-ramp and round transfers
        for d in range(s0 + 7, s1, 10):
            ctx.L.card_batch(self.collector.accounts[0], np.array([biz_ts(ctx, d, 0.5, 4.0)], dtype=np.int64), np.array([float(rng.uniform(3_000, 12_000))]), [crypto], tag=tag)
        self.extra.update({"window_days": [s0, s1], "victim_flows": n_flows})


class InsiderCollusion(Scenario):
    typology = "insider_collusion"

    def setup(self) -> None:
        w, rng = self.w, self.rng
        insider = pick_people(w, rng, 1, lambda p: "restricted_access" in p.traits and bool(p.cards) and bool(p.work_email) and bool(p.personal_email))[0]
        handler = pick_people(w, rng, 1, lambda p: 28 <= p.age_on(w.cfg.start) <= 62 and p.employer_id != insider.employer_id)[0]
        self.insider, self.handler = insider, handler
        self.members = [(insider, "insider"), (handler, "handler")]
        name = unique_name(w, rng, lambda: f"{rng.choice(SHELL_WORDS_A)} Consulting Group LLC")
        self.payco = make_org(w, rng, "shell", name, [handler], random_address(w, handler.town, rng, 0.9), int(rng.integers(90, 500)), any_bank(rng), int(rng.integers(30, 300)))
        self.orgs.append(self.payco)
        self.burners[handler.person_id] = [burner(w, rng, handler, int(rng.integers(0, 20)))]
        storage = [s for s in w.sites.values() if s.kind == "storage"]
        self.site = storage[int(rng.integers(len(storage)))]
        for p in (insider, handler):
            ensure_card(w, p, self.site)
        if not handler.personal_email:
            handler.personal_email = w.ids.email(f"{handler.first}{handler.last}", "quickmail.example")
            w.email_owner[handler.personal_email] = handler.person_id
        self.drop_box = w.ids.email(f"{rng.choice(['archive', 'files', 'backup', 'docs'])}{int(rng.integers(100, 999))}", "postbox.example")
        w.email_owner[self.drop_box] = handler.person_id
        self.extra["drop_box_email"] = self.drop_box
        add_ties(w, [insider, handler], "handler")

    def run(self, ctx: Ctx) -> None:
        w, rng, tag = ctx.w, ctx.rng, self.ring_id
        s0, s1 = window(ctx, 45, 80)
        card = next(c for c in self.insider.cards if w.cards[c].kind == "employee")
        site = w.sites[w.cards[card].site_id]
        target = "SERVER-ROOM-B1" if self.insider.department == "Information Technology" else ("RECORDS-ROOM-3F" if self.insider.department == "Legal and Compliance" else "LOADING-DOCK-01")
        visits = 0
        for week in range(s0, s1, 7):
            for _ in range(int(rng.integers(1, 3))):
                day = min(w.clock.days - 1, week + int(rng.integers(0, 6)))
                t = int(w.clock.ts(day, int(rng.uniform(20.0, 23.3) * 3600)))
                turn = site.reader("TURNSTILE-01").reader_id
                ctx.S.swipe(t, card, turn, tag=tag)
                ctx.S.swipe(t + int(rng.integers(120, 400)), card, site.reader(target).reader_id, tag=tag)
                if rng.random() < 0.45:
                    other = "RECORDS-ROOM-3F" if target != "RECORDS-ROOM-3F" else "SERVER-ROOM-B1"
                    ctx.S.swipe(t + int(rng.integers(500, 1500)), card, site.reader(other).reader_id, granted=False, reason="NO_PRIVILEGE", tag=tag)
                out = t + int(rng.integers(1800, 4200))
                ctx.S.swipe(out, card, turn, direction="OUT", tag=tag)
                big = ctx.M.random_attachments(int(rng.integers(1, 3)), size_scale=60.0, prefix=str(rng.choice(["export", "backup", "customer_list", "q_report"])))
                ctx.M.add(out + int(rng.integers(300, 3000)), self.insider.work_email, [self.insider.personal_email], subject=str(rng.choice(["", "fwd", "notes", "personal"])), attachments=big, tag=tag)  # type: ignore[arg-type,list-item]
                ctx.M.add(out + int(rng.integers(3600, 30000)), self.insider.personal_email, [self.drop_box], subject="", attachments=big, tag=tag)  # type: ignore[arg-type]
                ring_contact(ctx, out + int(rng.integers(600, 7200)), self.handler, self.insider, self.burner_for(self.handler), None, tag, sms_p=0.6, opsec=0.9)
                visits += 1
            if rng.random() < 0.6:
                meet(ctx, [self.insider, self.handler], self.site, min(w.clock.days - 1, week + 5), float(rng.uniform(10, 17)), tag, "BUILDING-B-DOOR")
        for d in range(s0 + 10, s1 + 14, 14):
            ctx.L.transfer(biz_ts(ctx, min(d, w.clock.days - 2)), round_to(float(rng.uniform(4_500, 7_500)), 500), "ACH", src=self.payco.accounts[0], dst=self.insider.accounts[0], memo="CONSULTING FEE", tag=tag)
        ctx.L.transfer(biz_ts(ctx, s0), round_to(float(rng.uniform(30_000, 60_000)), 5_000), "WIRE", src_ext=(w.ids.external_account(), "PRIVATE CLIENT", "OFFSHORE COMMERCIAL BANK LTD"), dst=self.payco.accounts[0], memo="RETAINER", tag=tag)
        self.extra.update({"window_days": [s0, s1], "after_hours_visits": visits, "target_reader": target})


TYPOLOGIES: dict[str, type[Scenario]] = {
    "shell_layering": ShellLayering,
    "structuring": Structuring,
    "mule_funnel": MuleFunnel,
    "insider_collusion": InsiderCollusion,
}


def setup_scenarios(w: World) -> list[Scenario]:
    rng = w.rng.stream("scenarios")
    out: list[Scenario] = []
    n = 0
    for typology in sorted(w.cfg.scenario_mix):
        for _ in range(int(w.cfg.scenario_mix[typology])):
            n += 1
            sc = TYPOLOGIES[typology](f"RING-{n:02d}", w, rng)
            sc.setup()
            out.append(sc)
    return out


def run_scenarios(ctx: Ctx, scenarios: list[Scenario]) -> None:
    for sc in scenarios:
        sc.run(ctx)
        ctx.w.rings.append(sc.truth())
