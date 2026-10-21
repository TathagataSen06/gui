"""Factories for synthetic-but-realistic identifiers.

Every identifier is drawn from a namespace reserved for fiction or never
assigned in the real world, so generated data cannot collide with a real
person's phone, e-mail, bank routing number or tax ID:

* phone numbers: NANP ``NXX-555-01XX`` (the 555-0100..0199 block is reserved
  for fictional use in every area code)
* e-mail: RFC 2606 ``.example`` domains
* bank routing numbers: prefix ``99`` (not an assigned Federal Reserve symbol)
* business registration numbers: EIN-style with the never-issued ``00`` prefix
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np

from .reference import SYNTHETIC_EMAIL_SUFFIXES

FICTIONAL_PHONE_RE = re.compile(r"^\+1[2-9]\d{2}55501\d{2}$")


class IdentifierFactory:
    """Issues unique synthetic identifiers, deterministically from an RNG."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._used: set[str] = set()
        # Area codes: first digit 2-9, excluding N11 service codes.
        self._area_codes = [a for a in range(200, 1000) if a % 100 != 11 and (a // 10) % 10 != 9]
        # Weighted toward a few "metro" area codes, like a real regional population.
        self._metro_codes = [201, 551, 732, 908, 973, 862, 609, 856, 212, 646, 718, 917, 215, 267, 610, 484]

    def _unique(self, make) -> str:
        for _ in range(10_000):
            v = make()
            if v not in self._used:
                self._used.add(v)
                return v
        raise RuntimeError("identifier space exhausted")

    def msisdn(self, metro: bool = True) -> str:
        """E.164 fictional mobile number, e.g. ``+12015550142``."""

        def make() -> str:
            if metro and self._rng.random() < 0.85:
                area = int(self._rng.choice(self._metro_codes))
            else:
                area = int(self._rng.choice(self._area_codes))
            return f"+1{area}55501{int(self._rng.integers(0, 100)):02d}"

        return self._unique(make)

    def account_number(self, bank_code: str) -> str:
        length = {"FMB": 10, "CFCU": 12, "HVT": 11}.get(bank_code, 10)
        return self._unique(lambda: "".join(str(int(d)) for d in self._rng.integers(0, 10, length)).lstrip("0").rjust(length, "7"))

    def customer_id(self, prefix: str) -> str:
        return self._unique(lambda: f"{prefix}{int(self._rng.integers(100000, 9999999)):07d}")

    def reg_no(self) -> str:
        """EIN-shaped business tax ID with the unassigned ``00`` prefix."""
        return self._unique(lambda: f"00-{int(self._rng.integers(0, 9_999_999)):07d}")

    def access_card(self, facility_code: int) -> str:
        """Wiegand-style ``FC<facility>-<card>`` badge number."""
        return self._unique(lambda: f"FC{facility_code:03d}-{int(self._rng.integers(1, 65535)):05d}")

    def employee_id(self, employer_prefix: str) -> str:
        return self._unique(lambda: f"{employer_prefix}{int(self._rng.integers(10000, 99999))}")

    def merchant_id(self) -> str:
        return self._unique(lambda: f"MID{int(self._rng.integers(10**8, 10**9))}")

    def external_account(self) -> str:
        return self._unique(lambda: f"EXT{int(self._rng.integers(10**9, 10**10))}")

    def txn_ref(self, prefix: str) -> str:
        return self._unique(lambda: f"{prefix}{int(self._rng.integers(10**11, 10**12))}")

    def claim(self, value: str) -> str:
        """Reserve an externally constructed identifier (e.g. an e-mail)."""
        if value in self._used:
            raise ValueError(f"identifier {value!r} already issued")
        self._used.add(value)
        return value

    def email(self, local: str, domain: str) -> str:
        base = slug(local, sep=".") or "user"
        candidate = f"{base}@{domain}"
        n = 1
        while candidate in self._used:
            n += 1
            candidate = f"{base}{n if n > 2 else int(self._rng.integers(10, 99))}@{domain}"
        self._used.add(candidate)
        return candidate


def slug(text: str, sep: str = "-") -> str:
    """ASCII-fold and slugify ``text`` (``"O'Brien & Sons"`` -> ``"obrien-sons"``)."""
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    t = re.sub(r"[^A-Za-z0-9]+", " ", t.replace("'", "")).strip().lower()
    return sep.join(t.split())


def is_fictional_phone(e164: str) -> bool:
    """True if ``e164`` is in the NANP 555-01XX fictional block."""
    return bool(FICTIONAL_PHONE_RE.match(e164))


def is_synthetic_email(address: str) -> bool:
    a = address.strip().lower()
    return any(a.endswith(sfx) for sfx in SYNTHETIC_EMAIL_SUFFIXES)
