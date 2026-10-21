"""Identifier normalization and synthetic-namespace checks (shared by ingest/resolve)."""

from __future__ import annotations

import re

_DIGITS = re.compile(r"\D+")
FICTIONAL_PHONE_RE = re.compile(r"^\+1[2-9]\d{2}55501\d{2}$")
SYNTHETIC_EMAIL_SUFFIXES: tuple[str, ...] = (".example", ".test", ".invalid", "@example.com", "@example.org", "@example.net")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize_msisdn(raw: object) -> str | None:
    """Normalize a NANP phone number in any common layout to E.164 (``+1NXXNXXXXXX``).

    >>> normalize_msisdn("(201) 555-0142")
    '+12015550142'
    >>> normalize_msisdn("1-201-555-0142")
    '+12015550142'
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return None
    d = _DIGITS.sub("", s)
    if len(d) == 10:
        return "+1" + d
    if len(d) == 11 and d.startswith("1"):
        return "+" + d
    return None


def normalize_email(raw: object) -> str | None:
    """Lower-case, trim and drop ``+tag`` sub-addressing; ``None`` if malformed."""
    if raw is None:
        return None
    s = str(raw).strip().strip("<>").lower()
    if not s or s == "nan" or not _EMAIL_RE.match(s):
        return None
    local, domain = s.split("@", 1)
    local = local.split("+", 1)[0]
    return f"{local}@{domain}"


def is_fictional_phone(e164: str) -> bool:
    return bool(FICTIONAL_PHONE_RE.match(e164))


def is_synthetic_email(address: str) -> bool:
    a = address.strip().lower()
    return any(a.endswith(sfx) for sfx in SYNTHETIC_EMAIL_SUFFIXES)


def identifier_is_synthetic(identifier: str) -> bool:
    """True if a typed identifier lies in a reserved/fictional namespace.

    Only identifier kinds that could point at a real person are checked
    (phone numbers and e-mail addresses); account numbers, cards and readers
    are meaningless outside the generated world.
    """
    kind, _, value = identifier.partition(":")
    if kind == "msisdn":
        return is_fictional_phone(value)
    if kind == "email":
        return is_synthetic_email(value)
    return True
