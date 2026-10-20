"""Physical access-control (PACS) swipe log.

Modelled on enterprise PACS exports: turnstile/parking/floor/restricted
readers at corporate campuses plus membership readers at co-working spaces,
gyms and self-storage facilities. Includes remote/sick days, lunch exits,
anti-passback and privilege denials, and a warehouse night shift.
"""

from __future__ import annotations

import numpy as np

from ..model import Card, Person, Site
from ..state import Table, World

COLUMNS = ("ts", "card_number", "reader_id", "result", "direction", "deny_reason", "_tag")


class SwipeLog:
    def __init__(self, w: World) -> None:
        self.w = w
        self.t = Table(COLUMNS)
        self.rng = w.rng.stream("access")

    def swipe(self, ts: int, card: str, reader_id: str, direction: str = "IN", granted: bool = True, reason: str | None = None, tag: str | None = None) -> None:
        if not granted and reason is None:
            reason = str(self.rng.choice(["NO_PRIVILEGE", "ANTI_PASSBACK", "CARD_EXPIRED", "TIME_ZONE_VIOLATION"], p=[0.45, 0.3, 0.1, 0.15]))
        self.t.add(ts=int(ts), card_number=card, reader_id=reader_id, result="ACCESS_GRANTED" if granted else "ACCESS_DENIED", direction=direction, deny_reason=reason, _tag=tag)


def _employee_day(w: World, log: SwipeLog, p: Person, card: Card, site: Site, day: int) -> None:
    rng = log.rng
    night = "night_shift" in p.traits
    if night:
        arrive = int(rng.normal(21.75 * 3600, 900))
        shift = int(rng.normal(8.5 * 3600, 1200))
    elif p.department == "Warehouse":
        arrive = int(rng.normal(6.75 * 3600, 1200))
        shift = int(rng.normal(8.6 * 3600, 1500))
    else:
        arrive = int(rng.normal(8.6 * 3600, 2400))
        shift = int(rng.normal(8.7 * 3600, 2700))
    t0 = int(w.clock.ts(day, arrive))
    turn = site.reader("TURNSTILE-01" if rng.random() < 0.5 else "TURNSTILE-02").reader_id
    if rng.random() < 0.6:
        log.swipe(t0 - int(rng.integers(60, 240)), card.number, site.reader("PARKING-GATE-A").reader_id)
    log.swipe(t0, card.number, turn)
    if p.department in ("Warehouse",):
        floor = site.reader("WAREHOUSE-FLOOR")
    else:
        floor = site.readers[3 + (int(p.person_id[1:]) % 3)]
    log.swipe(t0 + int(rng.integers(60, 300)), card.number, floor.reader_id)
    if "restricted_access" in p.traits:
        for _ in range(int(rng.poisson(0.8))):
            room = site.reader("SERVER-ROOM-B1" if p.department == "Information Technology" else ("LOADING-DOCK-01" if p.department == "Facilities" else "RECORDS-ROOM-3F"))
            log.swipe(t0 + int(rng.integers(1800, max(1900, shift - 1800))), card.number, room.reader_id)
    elif rng.random() < 0.004:
        # wandering into a restricted door: denied
        log.swipe(t0 + int(rng.integers(1800, 20000)), card.number, site.reader("SERVER-ROOM-B1").reader_id, granted=False, reason="NO_PRIVILEGE")
    if p.department == "Warehouse" and rng.random() < 0.5:
        log.swipe(t0 + int(rng.integers(1800, max(1900, shift - 600))), card.number, site.reader("LOADING-DOCK-01").reader_id)
    if not night and rng.random() < 0.42:
        out = int(w.clock.ts(day, int(rng.normal(12.2 * 3600, 1500))))
        log.swipe(out, card.number, turn, direction="OUT")
        back = out + int(rng.integers(20 * 60, 70 * 60))
        log.swipe(back, card.number, turn, direction="IN", granted=rng.random() > 0.01)
    log.swipe(t0 + shift, card.number, turn, direction="OUT")


def generate_benign_access(w: World, log: SwipeLog) -> None:
    rng = log.rng
    biz = np.flatnonzero(w.clock.business)
    for p in w.persons:
        for cn in p.cards:
            card = w.cards[cn]
            site = w.sites[card.site_id]
            if card.kind == "employee":
                remote_p = 0.06 if p.department in ("Warehouse", "Facilities") else 0.22
                vacation_start = int(rng.integers(0, w.clock.days))
                vac = set(range(vacation_start, vacation_start + int(rng.choice([0, 5, 7, 10]))))
                for d in biz:
                    if d in vac or rng.random() < remote_p + 0.03:
                        continue
                    _employee_day(w, log, p, card, site, int(d))
            elif site.kind == "gym":
                per_week = float(rng.choice([0.5, 1, 2, 3, 4]))
                for d in range(w.clock.days):
                    if rng.random() < per_week / 7:
                        hour = float(rng.choice([6.0, 6.5, 7.0, 12.0, 17.5, 18.0, 18.5, 19.0, 19.5, 10.0]))
                        log.swipe(int(w.clock.ts(d, int(hour * 3600 + rng.integers(0, 1800)))), cn, site.readers[0].reader_id)
            elif site.kind == "storage":
                for d in range(w.clock.days):
                    if rng.random() < 1.5 / 30 * (2.0 if w.clock.weekday[d] >= 5 else 0.6):
                        ts = int(w.clock.ts(d, int(rng.integers(9, 18)) * 3600 + int(rng.integers(0, 3600))))
                        log.swipe(ts, cn, site.reader("GATE-KEYPAD").reader_id)
                        log.swipe(ts + int(rng.integers(60, 240)), cn, site.reader("BUILDING-B-DOOR").reader_id)
            elif site.kind == "coworking":
                for d in biz:
                    if rng.random() < 0.45:
                        ts = int(w.clock.ts(int(d), int(rng.normal(9.8 * 3600, 3000))))
                        log.swipe(ts, cn, site.reader("FRONT-DOOR").reader_id)
                        log.swipe(ts + int(rng.integers(30, 200)), cn, site.reader("PRIVATE-OFFICE-WING").reader_id)
                        if rng.random() < 0.25:
                            room = site.reader("MEETING-ROOM-A" if rng.random() < 0.5 else "MEETING-ROOM-B")
                            log.swipe(int(w.clock.ts(int(d), int(rng.integers(10, 16)) * 3600)), cn, room.reader_id)
