"""Mobile carrier call-detail records (CDRs).

One originating record per voice call or SMS (``MOC`` / ``SMS-MO``), stamped
with the serving cell. Called numbers are recorded *as dialled* (10-digit,
11-digit or E.164), as in real switch output. Calls to businesses go to
"service numbers" that have no subscriber record - hubs that a graph builder
must learn to discount.
"""

from __future__ import annotations

import numpy as np

from ..model import Person
from ..reference import HOUR_PROFILES
from ..state import Table, World, sample_hours

COLUMNS = (
    "start_ts",
    "record_type",
    "calling_number",
    "called_number",
    "duration_sec",
    "cell_id",
    "call_result",
    "_tag",
)


class CallLog:
    def __init__(self, w: World) -> None:
        self.w = w
        self.t = Table(COLUMNS)
        self.rng = w.rng.stream("telco")
        self.service_numbers = [w.ids.msisdn() for _ in range(220)]
        # popularity of service numbers is heavy-tailed (pizza place vs. dentist)
        sp = 1.0 / np.arange(1, len(self.service_numbers) + 1) ** 0.9
        self.service_p = sp / sp.sum()

    def cell_for(self, p: Person | None, ts: int) -> str:
        w = self.w
        if p is None:
            return str(w.cell_sites[int(self.rng.integers(len(w.cell_sites)))]["cell_id"])
        day = int((ts - w.clock.day_start[0]) // 86400)
        hour = int(((ts - w.clock.day_start[max(0, min(day, w.clock.days - 1))]) % 86400) // 3600)
        if p.employer_id and 0 <= day < w.clock.days and w.clock.business[day] and 9 <= hour < 17:
            org = w.orgs[p.employer_id]
            if org.site_id:
                return w.sites[org.site_id].cell_id or w.town_cells[p.town][0]
        cells = w.town_cells[p.town]
        return cells[int(self.rng.integers(len(cells)))]

    def add(
        self,
        ts: int,
        a_number: str,
        b_number: str,
        sms: bool,
        duration: int,
        caller: Person | None,
        tag: str | None = None,
        answered: bool = True,
    ) -> None:
        self.t.add(
            start_ts=int(ts),
            record_type="SMS-MO" if sms else "MOC",
            calling_number=a_number[1:],  # switch emits 1NXXNXXXXXX without '+'
            called_number=_as_dialled(b_number, self.rng),
            duration_sec=0 if sms or not answered else int(duration),
            cell_id=self.cell_for(caller, ts),
            call_result="DELIVERED" if sms else ("ANSWERED" if answered else str(self.rng.choice(["NO_ANSWER", "BUSY"], p=[0.8, 0.2]))),
            _tag=tag,
        )


def _as_dialled(e164: str, rng: np.random.Generator) -> str:
    u = rng.random()
    if u < 0.7:
        return e164[2:]
    if u < 0.95:
        return e164[1:]
    return e164


def _line(p: Person, rng: np.random.Generator) -> str | None:
    if not p.phones:
        return None
    if len(p.phones) > 1 and rng.random() < 0.25:
        return p.phones[1]
    return p.phones[0]


def generate_benign_calls(w: World, log: CallLog) -> None:
    rng = log.rng
    days = w.clock.days
    for r in w.relationships:
        a, b = w.person_by_id[r.a], w.person_by_id[r.b]
        if not a.phones or not b.phones:
            continue
        n = int(rng.poisson(r.call_rate * days))
        if n == 0:
            continue
        profile = HOUR_PROFILES["business_call" if r.kind == "coworker" else "social_call"]
        if r.kind == "coworker":
            biz_days = np.flatnonzero(w.clock.business)
            d = rng.choice(biz_days, size=n)
        else:
            d = rng.integers(0, days, size=n)
        secs = sample_hours(rng, profile, n)
        younger = min(a.age_on(w.cfg.start), b.age_on(w.cfg.start))
        sms_p = 0.65 if younger < 30 else (0.45 if younger < 50 else 0.2)
        median = {"household": 85, "friend": 150, "coworker": 170}.get(r.kind, 120)
        for i in range(n):
            caller, callee = (a, b) if rng.random() < 0.5 else (b, a)
            ts = int(w.clock.ts(int(d[i]), int(secs[i])))
            sms = rng.random() < sms_p
            dur = int(max(3, rng.lognormal(np.log(median), 0.9)))
            log.add(ts, _line(caller, rng), _line(callee, rng), sms, dur, caller, answered=rng.random() > 0.12)  # type: ignore[arg-type]


def generate_service_calls(w: World, log: CallLog) -> None:
    rng = log.rng
    for p in w.persons:
        if not p.phones:
            continue
        n = int(rng.poisson(0.12 * w.clock.days))
        if n == 0:
            continue
        favs = rng.choice(len(log.service_numbers), size=min(6, n), replace=False, p=log.service_p)
        d = rng.integers(0, w.clock.days, size=n)
        secs = sample_hours(rng, HOUR_PROFILES["office"], n)
        for i in range(n):
            num = log.service_numbers[int(favs[int(rng.integers(len(favs)))])]
            log.add(int(w.clock.ts(int(d[i]), int(secs[i]))), p.phones[0], num, False, int(max(10, rng.lognormal(np.log(120), 0.7))), p, answered=rng.random() > 0.2)
