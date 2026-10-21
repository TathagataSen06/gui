"""Mutable generation state shared by the generator components."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from faker import Faker

from .config import GeneratorConfig, Rng
from .identifiers import IdentifierFactory
from .model import Account, Card, Merchant, Org, Person, Relationship, Site, Town


class Table:
    """Append-only columnar row buffer (much faster than lists of dicts)."""

    def __init__(self, columns: Iterable[str]) -> None:
        self.columns = list(columns)
        self._data: dict[str, list[Any]] = {c: [] for c in self.columns}

    def add(self, **row: Any) -> None:
        for c in self.columns:
            self._data[c].append(row.get(c))

    def extend(self, **cols: Any) -> None:
        """Append many rows at once; scalars are broadcast to the batch length."""
        n = max(len(v) for v in cols.values() if isinstance(v, (list, tuple, np.ndarray)))
        for c in self.columns:
            v = cols.get(c)
            if isinstance(v, (list, tuple, np.ndarray)):
                if len(v) != n:
                    raise ValueError(f"column {c} has length {len(v)} != {n}")
                self._data[c].extend(v.tolist() if isinstance(v, np.ndarray) else v)
            else:
                self._data[c].extend([v] * n)

    def __len__(self) -> int:
        return len(self._data[self.columns[0]])

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self._data, columns=self.columns)


def us_holidays(year: int) -> set[date]:
    """US federal holidays (observed) for ``year`` - drives ACH/payroll calendars."""

    def nth_weekday(month: int, weekday: int, n: int) -> date:
        d = date(year, month, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)

    def last_weekday(month: int, weekday: int) -> date:
        d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
        return d - timedelta(days=(d.weekday() - weekday) % 7)

    def observed(d: date) -> date:
        if d.weekday() == 5:
            return d - timedelta(days=1)
        if d.weekday() == 6:
            return d + timedelta(days=1)
        return d

    return {
        observed(date(year, 1, 1)),
        nth_weekday(1, 0, 3),
        nth_weekday(2, 0, 3),
        last_weekday(5, 0),
        observed(date(year, 6, 19)),
        observed(date(year, 7, 4)),
        nth_weekday(9, 0, 1),
        nth_weekday(10, 0, 2),
        observed(date(year, 11, 11)),
        nth_weekday(11, 3, 4),
        observed(date(year, 12, 25)),
    }


class Clock:
    """Maps (day index, local seconds-of-day) to UTC epoch seconds."""

    def __init__(self, start: date, days: int, tz: str) -> None:
        self.start = start
        self.days = days
        self.tz = ZoneInfo(tz)
        self.dates = [start + timedelta(days=i) for i in range(days)]
        self.day_start = np.array(
            [int(datetime(d.year, d.month, d.day, tzinfo=self.tz).timestamp()) for d in self.dates], dtype=np.int64
        )
        hol: set[date] = set()
        for y in {d.year for d in self.dates}:
            hol |= us_holidays(y)
        self.holidays = hol
        self.weekday = np.array([d.weekday() for d in self.dates])
        self.business = np.array([d.weekday() < 5 and d not in hol for d in self.dates])
        self.end_epoch = int(self.day_start[-1] + 86400)

    def ts(self, day: int | np.ndarray, local_seconds: int | float | np.ndarray) -> np.ndarray:
        return self.day_start[day] + np.asarray(local_seconds, dtype=np.int64)

    def day_of(self, d: date) -> int:
        return (d - self.start).days

    def next_business_day(self, day: int) -> int:
        d = day + 1
        while d < self.days and not self.business[d]:
            d += 1
        return min(d, self.days - 1)

    def prev_or_same_business_day(self, day: int) -> int:
        d = day
        while d > 0 and not self.business[d]:
            d -= 1
        return d

    def local_hour(self, epoch: int) -> int:
        return datetime.fromtimestamp(int(epoch), self.tz).hour


def sample_hours(rng: np.random.Generator, profile: tuple[float, ...], n: int) -> np.ndarray:
    """Sample ``n`` local seconds-of-day following an hourly weight profile."""
    p = np.asarray(profile, dtype=float)
    p = p / p.sum()
    hours = rng.choice(24, size=n, p=p)
    return hours * 3600 + rng.integers(0, 3600, size=n)


@dataclass
class World:
    """Everything the generator knows, including hidden ground truth."""

    cfg: GeneratorConfig
    rng: Rng
    fake: Faker
    ids: IdentifierFactory
    clock: Clock
    towns: list[Town] = field(default_factory=list)
    persons: list[Person] = field(default_factory=list)
    orgs: dict[str, Org] = field(default_factory=dict)
    accounts: dict[str, Account] = field(default_factory=dict)
    merchants: list[Merchant] = field(default_factory=list)
    sites: dict[str, Site] = field(default_factory=dict)
    cards: dict[str, Card] = field(default_factory=dict)
    relationships: list[Relationship] = field(default_factory=list)
    households: dict[str, list[str]] = field(default_factory=dict)
    branches: dict[str, list[str]] = field(default_factory=dict)  # bank_code -> branch ids
    cell_sites: list[dict[str, Any]] = field(default_factory=list)
    town_cells: dict[int, list[str]] = field(default_factory=dict)
    # truth bookkeeping
    rings: list[dict[str, Any]] = field(default_factory=list)
    decoys: list[dict[str, Any]] = field(default_factory=list)
    burner_owner: dict[str, str] = field(default_factory=dict)  # msisdn -> person_id
    burner_info: dict[str, dict[str, Any]] = field(default_factory=dict)  # msisdn -> activation, alias
    anon_lines: dict[str, str] = field(default_factory=dict)  # benign anonymous prepaid msisdn -> person_id
    bank_switch: dict[str, tuple[str, int]] = field(default_factory=dict)  # person_id -> (new account, day)
    phone_owner: dict[str, str] = field(default_factory=dict)  # msisdn -> person_id (all lines)
    email_owner: dict[str, str] = field(default_factory=dict)  # address -> person_id / org_id
    reserved: set[str] = field(default_factory=set)  # person ids taken by scenarios/decoys
    # raw feeds
    bank_txn: Table | None = None
    cdr: Table | None = None
    access_events: Table | None = None
    email_headers: list[dict[str, Any]] = field(default_factory=list)

    @property
    def person_by_id(self) -> dict[str, Person]:
        if not hasattr(self, "_pidx") or len(self._pidx) != len(self.persons):  # type: ignore[has-type]
            self._pidx = {p.person_id: p for p in self.persons}
        return self._pidx

    def primary_account(self, person: Person) -> str | None:
        return person.accounts[0] if person.accounts else None

    def eligible(self, pred) -> list[Person]:
        """Persons satisfying ``pred`` that are not yet reserved by a scenario/decoy."""
        return [p for p in self.persons if p.person_id not in self.reserved and pred(p)]
