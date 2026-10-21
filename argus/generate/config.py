"""Configuration for the synthetic world generator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NoiseConfig:
    """Probabilities of data-quality defects injected into identity records.

    Rates are per record (or per field occurrence) and were chosen to resemble
    the defect profile of real onboarding/subscriber data: most records are
    clean, but every field type has a long tail of errors.
    """

    name_typo: float = 0.06
    nickname_telco: float = 0.25
    nickname_email: float = 0.35
    nickname_access: float = 0.05
    first_initial_only: float = 0.04
    maiden_name_in_older_record: float = 0.45
    drop_suffix: float = 0.5
    dob_missing_telco: float = 0.30
    dob_missing_bank: float = 0.01
    dob_day_month_swap: float = 0.03
    dob_typo: float = 0.02
    stale_address: float = 0.6  # for people who moved: probability a non-bank source has the old address
    address_typo: float = 0.04
    zip_typo: float = 0.02
    unit_missing: float = 0.3
    kyc_phone_missing: float = 0.10
    kyc_phone_typo: float = 0.02
    kyc_email_missing: float = 0.15
    kyc_email_typo: float = 0.02
    display_name_missing: float = 0.15
    cardholder_phone_present: float = 0.5
    second_bank_relationship: float = 0.22

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


DEFAULT_SCENARIO_MIX: dict[str, int] = {
    "shell_layering": 2,
    "structuring": 2,
    "mule_funnel": 2,
    "insider_collusion": 1,
}

DEFAULT_DECOY_MIX: dict[str, int] = {
    "rosca": 2,  # rotating savings clubs: round-number fan-in/fan-out among friends
    "landlord": 6,  # property LLCs collecting round rent from many tenants
    "cash_business": 4,  # legitimately cash-intensive small businesses
    "freelancer": 8,  # many small inbound payments from many payers
    "night_shift": 1,  # a warehouse night crew (after-hours co-location)
}


@dataclass(frozen=True)
class GeneratorConfig:
    """Top-level knobs for :func:`argus.generate.world.generate_world`.

    All randomness flows from ``seed``; the same config always produces
    byte-identical output files.
    """

    n_people: int = 1500
    seed: int = 42
    start: date = date(2026, 1, 1)
    days: int = 182
    timezone: str = "America/New_York"
    scenario_mix: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_SCENARIO_MIX))
    decoy_mix: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_DECOY_MIX))
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    n_towns: int = 18
    n_major_employers: int = 6  # employers with a badge-controlled campus
    n_minor_employers: int = 30
    n_merchants: int = 260
    card_txn_rate: float = 0.85  # mean card purchases per person-day
    social_call_scale: float = 1.0  # multiplier on call/SMS rates
    work_email_rate: float = 1.6  # mean work e-mails sent per employee-workday

    def __post_init__(self) -> None:
        if self.n_people < 120:
            raise ValueError("n_people must be >= 120 so scenarios have room to be planted")
        if self.days < 28:
            raise ValueError("days must be >= 28")

    @property
    def end(self) -> date:
        from datetime import timedelta

        return self.start + timedelta(days=self.days)

    def with_overrides(self, **kwargs: Any) -> "GeneratorConfig":
        data = {**self.__dict__, **kwargs}
        return GeneratorConfig(**data)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["start"] = self.start.isoformat()
        return d


class Rng:
    """Named, independent random streams derived from one seed.

    Each component asks for its own stream (``rng.stream("telco")``) so that
    changing one component's consumption of randomness does not perturb the
    others - important for reproducible experiments.
    """

    _NAMES: tuple[str, ...] = (
        "population",
        "social",
        "holdings",
        "bank",
        "telco",
        "access",
        "email",
        "scenarios",
        "decoys",
        "noise",
        "watchlist",
        "labels",
        "background",
    )

    def __init__(self, seed: int) -> None:
        self.seed = seed
        children = np.random.SeedSequence(seed).spawn(len(self._NAMES))
        self._streams = {n: np.random.default_rng(s) for n, s in zip(self._NAMES, children)}

    def stream(self, name: str) -> np.random.Generator:
        return self._streams[name]
