"""Benign look-alikes that stress a detector's false-positive rate.

Each decoy reproduces one surface feature of a typology while being entirely
legitimate, as investigators routinely encounter:

``rosca``          rotating savings club - friends pay a round sum to an organiser
                   monthly, who passes the pot to one member (fan-in/fan-out,
                   round amounts, cycles, coordinated calls)
``landlord``       a property LLC collecting round rents from many tenants
``cash_business``  a legitimately cash-intensive shop, occasionally depositing
                   just under $10k (not structured, just a busy weekend)
``freelancer``     many inbound payments from many distinct payers
``night_shift``    a warehouse crew badging in small groups after midnight

Decoy participants are labelled benign; their roles are saved to
``ground_truth/decoys.json`` so false positives can be attributed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .model import Person
from .population import random_address
from .reference import CASH_BUSINESS_TYPES
from .scenarios import Ctx, biz_ts, make_org, pick_people, ring_contact, round_to, unique_name
from .state import World


class Decoy:
    kind = ""

    def __init__(self, decoy_id: str, w: World, rng: np.random.Generator) -> None:
        self.decoy_id, self.w, self.rng = decoy_id, w, rng
        self.members: list[tuple[Person, str]] = []
        self.extra: dict[str, Any] = {}

    def truth(self) -> dict[str, Any]:
        return {"decoy_id": self.decoy_id, "kind": self.kind, "members": [{"person_id": p.person_id, "role": r} for p, r in self.members], **self.extra}

    def setup(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def run(self, ctx: Ctx) -> None:  # pragma: no cover
        raise NotImplementedError


class Rosca(Decoy):
    kind = "rosca"

    def setup(self) -> None:
        w, rng = self.w, self.rng
        org = pick_people(w, rng, 1, lambda p: p.age_on(w.cfg.start) >= 30)[0]
        friends = [r.b if r.a == org.person_id else r.a for r in w.relationships if org.person_id in (r.a, r.b) and r.kind in ("friend", "household")]
        pool = [w.person_by_id[f] for f in friends if f not in w.reserved and w.person_by_id[f].accounts and w.person_by_id[f].phones]
        need = int(rng.integers(6, 11)) - 1
        members = pool[:need]
        if len(members) < need:
            members += pick_people(w, rng, need - len(members), lambda p: p.town == org.town)
        for m in members:
            w.reserved.add(m.person_id)
        self.organizer, self.participants = org, members
        self.members = [(org, "organizer")] + [(m, "member") for m in members]
        self.contribution = float(rng.choice([200, 250, 500, 1000]))

    def run(self, ctx: Ctx) -> None:
        w, rng = ctx.w, ctx.rng
        everyone = [self.organizer] + self.participants
        for k, month0 in enumerate(range(0, w.clock.days - 5, 30)):
            for m in self.participants:
                d = min(w.clock.days - 1, month0 + int(rng.integers(0, 5)))
                ts = int(w.clock.ts(d, int(rng.uniform(8, 22) * 3600)))
                ring_contact(ctx, ts - int(rng.integers(3600, 86400)), self.organizer, m, None, None, "", sms_p=0.7)
                ctx.L.transfer(ts, self.contribution, "P2P", src=m.accounts[0], dst=self.organizer.accounts[0], memo=str(rng.choice(["sou-sou", "susu", "club", "monthly", ""])))
            recipient = everyone[k % len(everyone)]
            if recipient is not self.organizer:
                d = min(w.clock.days - 1, month0 + 6 + int(rng.integers(0, 3)))
                ctx.L.transfer(int(w.clock.ts(d, 19 * 3600)), self.contribution * len(self.participants), "P2P", src=self.organizer.accounts[0], dst=recipient.accounts[0], memo="pot")
        self.extra["contribution"] = self.contribution


class Landlord(Decoy):
    kind = "landlord"

    def setup(self) -> None:
        w, rng = self.w, self.rng
        owner = pick_people(w, rng, 1, lambda p: p.age_on(w.cfg.start) >= 40)[0]
        name = unique_name(w, rng, lambda: f"{rng.choice(['Maple', 'Cedar', 'Willow', 'Brookside', 'Hillcrest', 'Lakeview', 'Elmwood', 'Oak Hollow'])} {rng.choice(['Court', 'Gardens', 'Commons', 'Terrace'])} Properties LLC")
        self.llc = make_org(w, rng, "property_llc", name, [owner], random_address(w, owner.town, rng, 0.0), int(rng.integers(900, 4000)), w.accounts[owner.accounts[0]].bank_code, int(rng.integers(800, 3500)))
        tenants = [p for p in w.persons if p.housing == "rent" and p.person_id not in w.reserved and p.accounts]
        k = min(len(tenants), int(rng.integers(4, 10)))
        chosen = [tenants[int(i)] for i in sorted(rng.choice(len(tenants), size=k, replace=False))]
        for t in chosen:
            t.landlord_account = self.llc.accounts[0]
            t.rent = float(round(rng.uniform(1200, 2400) / 50) * 50)
        self.members = [(owner, "landlord")]
        self.owner = owner
        self.extra.update({"org_id": self.llc.org_id, "tenants": [t.person_id for t in chosen]})

    def run(self, ctx: Ctx) -> None:
        w, rng = ctx.w, ctx.rng
        for d in range(5, w.clock.days, 30):
            ctx.L.transfer(biz_ts(ctx, d), round_to(float(rng.uniform(2_000, 6_000)), 500), "BOOK" if w.accounts[self.llc.accounts[0]].bank_code == w.accounts[self.owner.accounts[0]].bank_code else "ACH", src=self.llc.accounts[0], dst=self.owner.accounts[0], memo="OWNER DISTRIBUTION")
            ctx.L.transfer(biz_ts(ctx, d + 8), float(rng.uniform(1_800, 3_900)), "ACH", src=self.llc.accounts[0], dst_ext=(w.orgs["ORG-EXT-06"].ext_account, "LAKESHORE HOME LOANS", "EXTERNAL - ACH NETWORK"), memo="COMMERCIAL MTG")  # type: ignore[arg-type]


class CashBusiness(Decoy):
    kind = "cash_business"

    def setup(self) -> None:
        w, rng = self.w, self.rng
        owner = pick_people(w, rng, 1, lambda p: 28 <= p.age_on(w.cfg.start) <= 68)[0]
        staff = pick_people(w, rng, int(rng.integers(2, 5)), lambda p: p.age_on(w.cfg.start) <= 40)
        name = unique_name(w, rng, lambda: f"{w.towns[owner.town].name} {rng.choice(CASH_BUSINESS_TYPES)} Inc")
        self.biz = make_org(w, rng, "cash_business", name, [owner], random_address(w, owner.town, rng, 0.0), int(rng.integers(900, 5000)), w.accounts[owner.accounts[0]].bank_code, int(rng.integers(800, 4000)))
        self.owner, self.staff = owner, staff
        self.members = [(owner, "owner")] + [(s, "employee") for s in staff]
        self.extra["org_id"] = self.biz.org_id

    def run(self, ctx: Ctx) -> None:
        w, rng = ctx.w, ctx.rng
        acct = self.biz.accounts[0]
        branches = w.branches[w.accounts[acct].bank_code]
        home_branch = branches[int(rng.integers(len(branches)))]
        for d in range(w.clock.days):
            if w.clock.business[d] and rng.random() < 0.55:
                amt = min(14_500.0, float(rng.lognormal(np.log(2_600), 0.6)))
                ctx.L.cash_deposit(int(w.clock.ts(d, int(rng.uniform(15, 17.5) * 3600))), acct, round(amt, 2), home_branch)
            if w.clock.business[d]:
                ctx.L.transfer(int(w.clock.ts(d, 4 * 3600)), float(rng.lognormal(np.log(1_400), 0.4)), "ACH", src_ext=(w.orgs["ORG-EXT-13"].ext_account, "CLEARPOINT MERCHANT SVCS", "EXTERNAL - ACH NETWORK"), dst=acct, memo="CARD SETTLEMENT")  # type: ignore[arg-type]
        for week in range(4, w.clock.days, 7):
            for s in self.staff:
                ctx.L.transfer(int(w.clock.ts(week, 18 * 3600)), round_to(float(rng.uniform(450, 900)), 50), "P2P", src=acct, dst=s.accounts[0], memo="wages")
            ctx.L.transfer(biz_ts(ctx, week), float(rng.lognormal(np.log(3_500), 0.4)), "ACH", src=acct, dst_ext=("EXT-SUPPLY-" + self.biz.org_id[-3:], "RESTAURANT DEPOT SUPPLY", "UNITED COMMERCE BANK"), memo="SUPPLIES")
            for s in self.staff:
                if rng.random() < 0.6:
                    ring_contact(ctx, int(w.clock.ts(week, int(rng.uniform(9, 20) * 3600))), self.owner, s, None, None, "", sms_p=0.5)


class Freelancer(Decoy):
    kind = "freelancer"

    def setup(self) -> None:
        w, rng = self.w, self.rng
        p = pick_people(w, rng, 1, lambda p: p.income_kind in ("self", "payroll") and 24 <= p.age_on(w.cfg.start) <= 60)[0]
        p.traits.add("freelancer")
        self.p = p
        self.members = [(p, "freelancer")]

    def run(self, ctx: Ctx) -> None:
        w, rng = ctx.w, ctx.rng
        clients = [q for q in w.persons if q.accounts and q.person_id != self.p.person_id]
        idx = rng.choice(len(clients), size=min(len(clients), int(rng.integers(15, 40))), replace=False)
        chosen = [clients[int(i)] for i in idx]
        for m0 in range(0, w.clock.days, 30):
            for _ in range(int(rng.integers(4, 12))):
                c = chosen[int(rng.integers(len(chosen)))]
                amt = float(rng.choice([150, 200, 250, 300, 400, 500, 750, 1000, 1200])) if rng.random() < 0.6 else float(rng.uniform(120, 1300))
                d = min(w.clock.days - 1, m0 + int(rng.integers(0, 30)))
                ctx.L.transfer(int(w.clock.ts(d, int(rng.uniform(9, 21) * 3600))), amt, "P2P", src=c.accounts[0], dst=self.p.accounts[0], memo=str(rng.choice(["photos", "session", "lessons", "design work", "invoice", ""])))
                if c.phones and self.p.phones and rng.random() < 0.5:
                    ring_contact(ctx, int(w.clock.ts(max(0, d - 2), 12 * 3600)), c, self.p, None, None, "", sms_p=0.4)
        self.extra["clients"] = len(chosen)


class NightShift(Decoy):
    kind = "night_shift"

    def setup(self) -> None:
        w, rng = self.w, self.rng
        crews: dict[str, list[Person]] = {}
        for p in w.persons:
            if p.department == "Warehouse" and p.cards and p.person_id not in w.reserved:
                crews.setdefault(p.employer_id or "", []).append(p)
        if not crews:
            return
        org_id = sorted(crews, key=lambda k: (-len(crews[k]), k))[0]
        crew = crews[org_id][: int(rng.integers(4, 8))]
        for p in crew:
            p.traits.add("night_shift")
            w.reserved.add(p.person_id)
        self.members = [(p, "night_worker") for p in crew]
        self.extra["employer_id"] = org_id

    def run(self, ctx: Ctx) -> None:
        # Behaviour is produced by the access-control generator via the trait.
        return


KINDS: dict[str, type[Decoy]] = {
    "rosca": Rosca,
    "landlord": Landlord,
    "cash_business": CashBusiness,
    "freelancer": Freelancer,
    "night_shift": NightShift,
}


def setup_decoys(w: World) -> list[Decoy]:
    rng = w.rng.stream("decoys")
    out: list[Decoy] = []
    n = 0
    for kind in sorted(w.cfg.decoy_mix):
        for _ in range(int(w.cfg.decoy_mix[kind])):
            n += 1
            d = KINDS[kind](f"DECOY-{n:02d}", w, rng)
            d.setup()
            out.append(d)
    return out


def run_decoys(ctx: Ctx, decoys: list[Decoy]) -> None:
    for d in decoys:
        d.run(ctx)
        ctx.w.decoys.append(d.truth())
