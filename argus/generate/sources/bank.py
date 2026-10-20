"""Consortium core-banking ledger (three fictional banks).

Realistic properties reproduced here:

* **double-entry**: a transfer between two consortium accounts is booked twice
  (a DR row on the payer account and a CR row on the payee account) sharing an
  ``end_to_end_id``; settlement lag depends on the rail (book, P2P, ACH, wire);
* ACH/payroll follow the business-day calendar (weekends and federal holidays);
* card spend follows per-person category preferences, local favourite
  merchants and hour-of-day profiles; amounts are log-normal;
* rent is a round number, mortgages and utilities are not; ATM withdrawals are
  round by construction (a classic false-positive source for "round amount" rules).
"""

from __future__ import annotations

import numpy as np

from ..reference import BANKS, HOUR_PROFILES, MERCHANT_CATEGORIES
from ..state import Table, World, sample_hours

COLUMNS = (
    "booking_ts",
    "bank_code",
    "account_number",
    "direction",
    "amount",
    "currency",
    "channel",
    "counterparty_account",
    "counterparty_name",
    "counterparty_bank",
    "mcc",
    "merchant_id",
    "merchant_name",
    "branch_id",
    "end_to_end_id",
    "narrative",
    "_tag",
)

_BANK_NAME = {b.code: b.name.upper() for b in BANKS}


class BankLedger:
    """Posting helpers that keep both legs of a transfer consistent."""

    def __init__(self, w: World) -> None:
        self.w = w
        self.t = Table(COLUMNS)
        self.rng = w.rng.stream("bank")

    # ---- naming ----------------------------------------------------------
    def holder_name(self, key: str) -> str:
        acct = self.w.accounts[key]
        if acct.owner_kind == "org":
            return self.w.orgs[acct.owner_id].name.upper()
        p = self.w.person_by_id[acct.owner_id]
        return f"{p.first} {p.last}".upper()

    def _settle_lag(self, channel: str, ts: int) -> int:
        if channel in ("BOOK", "CARD"):
            return 0
        if channel == "P2P":
            return int(self.rng.integers(1, 8))
        if channel == "WIRE":
            return int(self.rng.integers(600, 5400))
        if channel == "CHECK":
            return int(self.rng.integers(1, 3)) * 86400 + int(self.rng.integers(0, 7200))
        # ACH: next business day early morning
        day = int((ts - self.w.clock.day_start[0]) // 86400)
        day = max(0, min(day, self.w.clock.days - 1))
        nd = self.w.clock.next_business_day(day)
        return int(max(self.w.clock.ts(nd, int(self.rng.integers(4 * 3600, 7 * 3600))) - ts, 60))

    # ---- postings ----------------------------------------------------------
    def transfer(
        self,
        ts: int,
        amount: float,
        channel: str,
        src: str | None = None,
        dst: str | None = None,
        memo: str = "",
        src_ext: tuple[str, str, str] | None = None,
        dst_ext: tuple[str, str, str] | None = None,
        tag: str | None = None,
        lag: int | None = None,
    ) -> str:
        """Book a transfer. ``src``/``dst`` are consortium account keys; for an
        external side pass ``(account, name, bank)`` via ``src_ext``/``dst_ext``."""
        if (src is None) == (src_ext is None) or (dst is None) == (dst_ext is None):
            raise ValueError("each side needs exactly one of key/ext")
        amount = round(float(amount), 2)
        e2e = self.w.ids.txn_ref("E2E")
        ts = int(min(ts, self.w.clock.end_epoch - 7200))
        if src is not None:
            a = self.w.accounts[src]
            cp_acct, cp_name, cp_bank = (
                (self.w.accounts[dst].number, self.holder_name(dst), _BANK_NAME[self.w.accounts[dst].bank_code]) if dst else dst_ext  # type: ignore[misc]
            )
            self.t.add(
                booking_ts=ts, bank_code=a.bank_code, account_number=a.number, direction="DR", amount=amount, currency="USD",
                channel=channel, counterparty_account=cp_acct, counterparty_name=cp_name, counterparty_bank=cp_bank,
                end_to_end_id=e2e, narrative=_out_narrative(channel, cp_name, memo), _tag=tag,
            )
        if dst is not None:
            a = self.w.accounts[dst]
            cp_acct, cp_name, cp_bank = (
                (self.w.accounts[src].number, self.holder_name(src), _BANK_NAME[self.w.accounts[src].bank_code]) if src else src_ext  # type: ignore[misc]
            )
            same_bank = src is not None and self.w.accounts[src].bank_code == a.bank_code
            d = 0 if same_bank and channel in ("BOOK", "P2P") else (lag if lag is not None else self._settle_lag(channel, ts))
            self.t.add(
                booking_ts=int(min(ts + d, self.w.clock.end_epoch - 60)), bank_code=a.bank_code, account_number=a.number, direction="CR",
                amount=amount, currency="USD", channel=channel, counterparty_account=cp_acct, counterparty_name=cp_name,
                counterparty_bank=cp_bank, end_to_end_id=e2e, narrative=_in_narrative(channel, cp_name, memo), _tag=tag,
            )
        return e2e

    def cash_deposit(self, ts: int, key: str, amount: float, branch: str, tag: str | None = None) -> None:
        a = self.w.accounts[key]
        teller = int(self.rng.integers(1, 9))
        self.t.add(
            booking_ts=int(ts), bank_code=a.bank_code, account_number=a.number, direction="CR", amount=round(float(amount), 2),
            currency="USD", channel="CASH", branch_id=branch, end_to_end_id=self.w.ids.txn_ref("CSH"),
            narrative=f"CASH DEPOSIT {branch} TELLER {teller:02d}", _tag=tag,
        )

    def atm(self, ts: int, key: str, amount: float, atm_id: str, tag: str | None = None) -> None:
        a = self.w.accounts[key]
        self.t.add(
            booking_ts=int(ts), bank_code=a.bank_code, account_number=a.number, direction="DR", amount=round(float(amount), 2),
            currency="USD", channel="ATM", branch_id=atm_id, end_to_end_id=self.w.ids.txn_ref("ATM"),
            narrative=f"ATM WITHDRAWAL {atm_id}", _tag=tag,
        )

    def card_batch(self, key: str, ts: np.ndarray, amounts: np.ndarray, merchants: list, tag: str | None = None) -> None:
        a = self.w.accounts[key]
        keep = (ts >= self.w.clock.day_start[0]) & (ts < self.w.clock.end_epoch)  # outside the observation window
        if not keep.all():
            ts, amounts = ts[keep], amounts[keep]
            merchants = [m for m, k in zip(merchants, keep) if k]
        n = len(ts)
        if n == 0:
            return
        dates = [self.w.clock.dates[int((t - self.w.clock.day_start[0]) // 86400)].strftime("%m/%d") for t in ts]
        self.t.extend(
            booking_ts=ts.astype(np.int64),
            bank_code=a.bank_code,
            account_number=a.number,
            direction="DR",
            amount=np.round(amounts, 2),
            currency="USD",
            channel="CARD",
            counterparty_account=[m.merchant_id for m in merchants],
            counterparty_name=[m.name.upper() for m in merchants],
            counterparty_bank="CARD NETWORK",
            mcc=[m.mcc for m in merchants],
            merchant_id=[m.merchant_id for m in merchants],
            merchant_name=[m.name.upper() for m in merchants],
            end_to_end_id=[self.w.ids.txn_ref("CRD") for _ in range(n)],
            narrative=[
                f"{'POS DEBIT' if m.town is not None else 'CARD NOT PRESENT'} {d} {m.name.upper()[:22]} {m.city_label}"
                for m, d in zip(merchants, dates)
            ],
            _tag=tag,
        )


def _out_narrative(channel: str, cp: str, memo: str) -> str:
    memo = f" {memo}" if memo else ""
    return {
        "ACH": f"ACH DEBIT {cp[:28]}{memo}",
        "WIRE": f"WIRE OUT BNF:{cp[:30]}{' REF:' + memo.strip() if memo else ''}",
        "P2P": f"P2P PAYMENT TO {cp[:24]}{memo}",
        "BOOK": f"ONLINE TRANSFER TO {cp[:24]}{memo}",
        "CHECK": f"CHECK PAID{memo}",
    }.get(channel, f"{channel} {cp}{memo}")


def _in_narrative(channel: str, cp: str, memo: str) -> str:
    memo = f" {memo}" if memo else ""
    return {
        "ACH": f"ACH CREDIT {cp[:28]}{memo}",
        "WIRE": f"WIRE IN ORG:{cp[:30]}{' REF:' + memo.strip() if memo else ''}",
        "P2P": f"P2P PAYMENT FROM {cp[:24]}{memo}",
        "BOOK": f"ONLINE TRANSFER FROM {cp[:24]}{memo}",
        "CHECK": f"MOBILE CHECK DEPOSIT{memo}",
    }.get(channel, f"{channel} {cp}{memo}")


def ext_party(w: World, org_id: str) -> tuple[str, str, str]:
    o = w.orgs[org_id]
    return (o.ext_account or "", o.name.upper(), o.attrs.get("bank", "EXTERNAL BANK"))


# --------------------------------------------------------------------------
# Benign activity
# --------------------------------------------------------------------------


def _paydays(w: World, phase: int) -> list[int]:
    """Bi-weekly Friday paydays (moved earlier when Friday is a holiday)."""
    out = []
    first_friday = (4 - w.clock.weekday[0]) % 7 + 7 * phase
    for d in range(first_friday, w.clock.days, 14):
        out.append(w.clock.prev_or_same_business_day(d))
    return out


def _monthly_days(w: World, dom: int) -> list[int]:
    out = []
    for d, dt in enumerate(w.clock.dates):
        if dt.day == dom:
            out.append(d)
    return out


def generate_income(w: World, L: BankLedger) -> None:
    rng = L.rng
    ss = "ORG-EXT-11"
    ui = "ORG-EXT-12"
    for p in w.persons:
        if not p.accounts:
            continue
        acct = p.accounts[0]
        if p.income_kind == "payroll":
            org = w.orgs[p.employer_id]  # type: ignore[index]
            net = p.salary / 26 * float(rng.uniform(0.68, 0.78))
            for d in _paydays(w, int(rng.integers(0, 2)) if not org.major else 0):
                ts = int(w.clock.ts(d, int(rng.integers(2 * 3600, 5 * 3600))))
                amt = net * (float(rng.uniform(0.97, 1.12)) if p.department == "Warehouse" else 1.0)
                memo = f"PAYROLL PPD ID:{org.reg_no.replace('-', '')}"  # type: ignore[union-attr]
                if org.accounts:
                    L.transfer(ts, amt, "ACH", src=org.accounts[0], dst=acct, memo=memo, lag=0)
                else:
                    L.transfer(ts, amt, "ACH", src_ext=(org.ext_account, org.name.upper(), org.attrs.get("bank", "")), dst=acct, memo=memo)
        elif p.income_kind == "social_security":
            for d in [dd for dd in _monthly_days(w, 3)]:
                d = w.clock.prev_or_same_business_day(d)
                L.transfer(int(w.clock.ts(d, 3 * 3600)), p.salary / 12, "ACH", src_ext=ext_party(w, ss), dst=acct, memo="XXSOC SEC")
        elif p.income_kind == "benefits":
            for d in range(int(rng.integers(0, 7)), w.clock.days, 7):
                if w.clock.business[d]:
                    L.transfer(int(w.clock.ts(d, 4 * 3600)), p.salary / 52, "ACH", src_ext=ext_party(w, ui), dst=acct, memo="UI BENEFIT")
        elif p.income_kind == "parents":
            parent = (w.ids.external_account(), f"{w.fake.first_name().upper()} {p.last.upper()}", "EXTERNAL BANK")
            amt = float(rng.choice([300, 400, 500, 750]))
            for d in _monthly_days(w, int(rng.integers(1, 28))):
                L.transfer(int(w.clock.ts(d, int(rng.integers(9, 22)) * 3600)), amt, "P2P", src_ext=parent, dst=acct)
        elif p.income_kind == "self":
            monthly = p.salary / 12
            clients = [(w.ids.external_account(), w.fake.company().upper(), "EXTERNAL BANK") for _ in range(int(rng.integers(2, 7)))]
            for m0 in range(0, w.clock.days, 30):
                k = int(rng.integers(1, 5))
                parts = rng.dirichlet(np.ones(k)) * monthly
                for amt in parts:
                    d = min(w.clock.days - 1, m0 + int(rng.integers(0, 30)))
                    ch = str(rng.choice(["ACH", "P2P", "CHECK"], p=[0.5, 0.3, 0.2]))
                    client = clients[int(rng.integers(len(clients)))]
                    L.transfer(int(w.clock.ts(d, int(rng.integers(8, 20)) * 3600)), max(amt, 50.0), ch, src_ext=client, dst=acct, memo="INVOICE")


def generate_bills(w: World, L: BankLedger) -> None:
    rng = L.rng
    for p in w.persons:
        if not p.accounts:
            continue
        acct = p.accounts[0]
        rent_dom = int(rng.integers(1, 4))
        for d in _monthly_days(w, rent_dom):
            ts = int(w.clock.ts(d, int(rng.integers(7, 23)) * 3600))
            if p.housing == "rent" and p.landlord_account:
                if p.landlord_account in w.accounts:
                    L.transfer(ts, p.rent, str(rng.choice(["P2P", "ACH"])), src=acct, dst=p.landlord_account, memo="RENT")
                else:
                    L.transfer(ts, p.rent, "ACH", src=acct, dst_ext=ext_party(w, p.landlord_account), memo="RENT")
            elif p.housing == "own":
                servicer = "ORG-EXT-06" if int(p.person_id[1:]) % 2 else "ORG-EXT-07"
                L.transfer(ts, p.rent, "ACH", src=acct, dst_ext=ext_party(w, servicer), memo="MORTGAGE PMT")
        bills: list[tuple[str, float, float]] = [("ORG-EXT-03", float(rng.choice([45, 65, 85, 120])) + 0.99 + float(rng.choice([0, 10, 20])), 0.03)]
        if p.housing in ("rent", "own"):
            bills += [("ORG-EXT-00", float(rng.lognormal(np.log(115), 0.3)), 0.22), ("ORG-EXT-02", float(rng.lognormal(np.log(70), 0.4)), 0.35), ("ORG-EXT-04", 79.99, 0.0)]
        if rng.random() < 0.7:
            bills.append(("ORG-EXT-05", round(float(rng.lognormal(np.log(140), 0.3)), 2), 0.0))
        for org_id, base, var in bills:
            dom = int(rng.integers(5, 28))
            for d in _monthly_days(w, dom):
                month = w.clock.dates[d].month
                season = 1.0 + (0.45 if org_id == "ORG-EXT-02" and month in (1, 2, 12) else 0.0)
                amt = base * season * float(rng.lognormal(0, var)) if var else base
                bd = w.clock.prev_or_same_business_day(d)
                L.transfer(int(w.clock.ts(bd, int(rng.integers(1, 6)) * 3600)), amt, "ACH", src=acct, dst_ext=ext_party(w, org_id), memo="AUTOPAY")


def generate_card_spend(w: World, L: BankLedger) -> None:
    rng = L.rng
    cats = MERCHANT_CATEGORIES
    by_cat_town: dict[tuple[str, int | None], list] = {}
    for m in w.merchants:
        by_cat_town.setdefault((m.mcc, m.town), []).append(m)
    by_cat: dict[str, list] = {}
    for m in w.merchants:
        by_cat.setdefault(m.mcc, []).append(m)
    base_w = np.array([c.weight for c in cats])
    wk_factor = np.where(w.clock.weekday >= 5, 1.15, 0.95)
    for p in w.persons:
        if not p.accounts:
            continue
        lam = w.cfg.card_txn_rate * float(rng.lognormal(0, 0.4)) * (0.7 if p.income_kind in ("social_security", "benefits") else 1.0)
        counts = rng.poisson(lam * wk_factor)
        n = int(counts.sum())
        if n == 0:
            continue
        days = np.repeat(np.arange(w.clock.days), counts)
        prefs = base_w * rng.lognormal(0, 0.6, size=len(cats))
        cat_idx = rng.choice(len(cats), size=n, p=prefs / prefs.sum())
        favourites: dict[str, list] = {}
        for c in cats:
            local = by_cat_town.get((c.mcc, p.town)) or by_cat_town.get((c.mcc, None)) or by_cat.get(c.mcc, [])
            pool = list(local) + list(by_cat.get(c.mcc, []))
            if not pool:
                continue
            k = min(len(pool), int(rng.integers(1, 4)))
            favourites[c.mcc] = [pool[i] for i in rng.choice(len(pool), size=k, replace=False)]
        merchants, ts_list, amts = [], [], []
        secs = np.zeros(n, dtype=np.int64)
        for ci in np.unique(cat_idx):
            mask = cat_idx == ci
            c = cats[int(ci)]
            secs[mask] = sample_hours(rng, HOUR_PROFILES[c.hours], int(mask.sum()))
        for i in range(n):
            c = cats[int(cat_idx[i])]
            fav = favourites.get(c.mcc)
            if not fav:
                continue
            m = fav[int(rng.integers(len(fav)))]
            merchants.append(m)
            ts_list.append(int(w.clock.ts(int(days[i]), int(secs[i]))))
            amts.append(max(0.5, float(rng.lognormal(np.log(m.median_amount), m.sigma))))
        L.card_batch(p.accounts[0], np.array(ts_list, dtype=np.int64), np.array(amts), merchants)
    # Retail crypto buyers: legitimate use of the same quasi-cash merchant mules use.
    crypto = crypto_merchant(w)
    for p in w.persons:
        if not p.accounts or p.age_on(w.cfg.start) > 55 or rng.random() > 0.05:
            continue
        n = int(rng.poisson(w.clock.days / 30 * 1.2))
        if n:
            ts = w.clock.ts(rng.integers(0, w.clock.days, size=n), sample_hours(rng, HOUR_PROFILES["night"], n))
            L.card_batch(p.accounts[0], ts, rng.lognormal(np.log(250), 0.9, size=n), [crypto] * n)


def crypto_merchant(w: World):
    return next(m for m in w.merchants if m.mcc == "6051")


def generate_atm_and_savings(w: World, L: BankLedger) -> None:
    rng = L.rng
    for p in w.persons:
        if not p.accounts:
            continue
        acct = w.accounts[p.accounts[0]]
        branches = w.branches[acct.bank_code]
        n_atm = int(rng.poisson(w.clock.days / 30 * float(rng.choice([0.5, 1.5, 3.0]))))
        for _ in range(n_atm):
            d = int(rng.integers(0, w.clock.days))
            own = rng.random() < 0.8
            atm_id = f"{acct.bank_code}-ATM-{branches[int(rng.integers(len(branches)))][-3:]}" if own else f"NONBANK-ATM-{int(rng.integers(100, 999))}"
            L.atm(int(w.clock.ts(d, sample_hours(rng, HOUR_PROFILES["daytime"], 1)[0])), acct.key, float(rng.choice([40, 60, 80, 100, 120, 160, 200, 300, 400], p=[0.12, 0.16, 0.12, 0.2, 0.08, 0.06, 0.16, 0.06, 0.04])), atm_id)
        savings = [k for k in p.accounts if w.accounts[k].product == "SAVINGS" and w.accounts[k].bank_code == acct.bank_code]
        if savings and p.income_kind == "payroll":
            amt = float(rng.choice([100, 150, 200, 250, 300, 500, 750, 1000]))
            for d in _monthly_days(w, int(rng.integers(2, 27))):
                if rng.random() < 0.8:
                    L.transfer(int(w.clock.ts(d, int(rng.integers(7, 23)) * 3600)), amt, "BOOK", src=acct.key, dst=savings[0], memo="SAVINGS")
                if rng.random() < 0.12:
                    L.transfer(int(w.clock.ts(d, int(rng.integers(7, 23)) * 3600)), amt * float(rng.choice([1, 2])), "BOOK", src=savings[0], dst=acct.key)
        if p.income_kind == "payroll" and "warehouse" not in p.traits and p.department in ("Customer Support", "Sales") and rng.random() < 0.15:
            p.traits.add("cash_tips")
        if "cash_tips" in p.traits:
            for d in range(int(rng.integers(0, 7)), w.clock.days, 7):
                br = branches[int(rng.integers(len(branches)))]
                L.cash_deposit(int(w.clock.ts(d, int(rng.integers(10, 17)) * 3600)), acct.key, float(rng.uniform(120, 900)), br)


def generate_p2p(w: World, L: BankLedger) -> None:
    rng = L.rng
    months = w.clock.days / 30.4
    for r in w.relationships:
        if r.p2p_rate <= 0:
            continue
        a, b = w.person_by_id[r.a], w.person_by_id[r.b]
        if not a.accounts or not b.accounts:
            continue
        for _ in range(int(rng.poisson(r.p2p_rate * months))):
            payer, payee = (a, b) if rng.random() < 0.5 else (b, a)
            amt = float(rng.lognormal(np.log(42), 0.8))
            if rng.random() < 0.35:
                amt = max(5.0, round(amt / float(rng.choice([5, 10, 20, 50]))) * float(rng.choice([5, 10, 20, 50])))
            d = int(rng.integers(0, w.clock.days))
            memo = str(rng.choice(["", "dinner", "groceries", "utilities split", "tickets", "gas", "thanks!", "birthday gift", "uber"]))
            L.transfer(int(w.clock.ts(d, sample_hours(rng, HOUR_PROFILES["evening"], 1)[0])), amt, "P2P", src=payer.accounts[0], dst=payee.accounts[0], memo=memo)


def generate_business_activity(w: World, L: BankLedger) -> None:
    """Employer revenue and vendor payments so business accounts look alive."""
    rng = L.rng
    pool = [(w.ids.external_account(), w.fake.company().upper(), str(rng.choice(["UNITED COMMERCE BANK", "NATIONAL CITY BANK NA", "PACIFIC WESTERN BANK"]))) for _ in range(140)]
    for org in w.orgs.values():
        if org.kind != "employer" or not org.accounts:
            continue
        size = org.attrs["size"]
        acct = org.accounts[0]
        customers = [pool[i] for i in rng.choice(len(pool), size=6, replace=False)]
        vendors = [pool[i] for i in rng.choice(len(pool), size=8, replace=False)]
        for week in range(0, w.clock.days, 7):
            for _ in range(int(rng.poisson(2.5))):
                d = min(w.clock.days - 1, week + int(rng.integers(0, 5)))
                if not w.clock.business[d]:
                    continue
                c = customers[int(rng.integers(len(customers)))]
                L.transfer(int(w.clock.ts(d, int(rng.integers(9, 17)) * 3600)), float(rng.lognormal(np.log(size * 900), 0.5)), str(rng.choice(["WIRE", "ACH"])), src_ext=c, dst=acct, memo=f"INV {int(rng.integers(10000, 99999))}")
            for _ in range(int(rng.poisson(2.0))):
                d = min(w.clock.days - 1, week + int(rng.integers(0, 5)))
                if not w.clock.business[d]:
                    continue
                v = vendors[int(rng.integers(len(vendors)))]
                L.transfer(int(w.clock.ts(d, int(rng.integers(9, 17)) * 3600)), float(rng.lognormal(np.log(size * 300), 0.6)), "ACH", src=acct, dst_ext=v, memo=f"PO {int(rng.integers(1000, 9999))}")


def generate_benign_bank(w: World, L: BankLedger) -> None:
    generate_income(w, L)
    generate_bills(w, L)
    generate_card_spend(w, L)
    generate_atm_and_savings(w, L)
    generate_p2p(w, L)
    generate_business_activity(w, L)

