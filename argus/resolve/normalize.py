"""Field normalization for entity resolution.

Ingestion has already standardized *formats* (dates to ISO, phone numbers to
E.164 where owned). Normalization here removes *semantic* variation that
source systems introduce: name order (``LAST, FIRST``), titles and suffixes,
nicknames, initials, address abbreviations and units.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache

import jellyfish

from ..generate.reference import DIRECTIONALS, NAME_SUFFIXES, NAME_TITLES, NICKNAME_TO_FORMAL, STREET_SUFFIX_ABBREV, UNIT_DESIGNATORS

_ABBREV_CANON: dict[str, str] = {}
for _full, _ab in STREET_SUFFIX_ABBREV.items():
    _ABBREV_CANON[_full] = _ab
    _ABBREV_CANON[_ab] = _ab
for _full, _ab in DIRECTIONALS.items():
    _ABBREV_CANON[_full] = _ab
    _ABBREV_CANON[_ab] = _ab
_UNIT_WORDS = set(UNIT_DESIGNATORS) | set(UNIT_DESIGNATORS.values()) | {"apt", "ste", "unit", "fl", "rm", "bldg", "suite", "apartment"}
_ORG_SUFFIXES = {"llc", "inc", "ltd", "co", "corp", "corporation", "company", "incorporated", "limited", "plc", "lp", "llp", "group", "and", "the"}
_ZIP = re.compile(r"\b(\d{5})(?:-\d{4})?\b")


def ascii_fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


@dataclass(frozen=True)
class NameParts:
    first: str | None
    middle: tuple[str, ...]
    last: str | None
    suffix: str | None
    first_forms: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_initial(self) -> bool:
        return bool(self.first) and len(self.first) == 1  # type: ignore[arg-type]

    @property
    def full(self) -> str:
        return " ".join(p for p in (self.first, self.last) if p)

    @property
    def canonical_full(self) -> str:
        first = min(self.first_forms) if self.first_forms else (self.first or "")
        return " ".join(p for p in (first, self.last) if p)


@lru_cache(maxsize=200_000)
def first_name_forms(first: str) -> frozenset[str]:
    """The given name plus every formal name it can be a nickname of."""
    return frozenset({first, *NICKNAME_TO_FORMAL.get(first, ())})


def _clean_token(t: str) -> str:
    return t.replace("'", "").replace("-", "").replace(".", "")


@lru_cache(maxsize=200_000)
def parse_name(raw: str | None) -> NameParts:
    """Parse a person name in any common layout.

    >>> parse_name("SMITH JR, ROBERT J").last, parse_name("SMITH JR, ROBERT J").suffix
    ('smith', 'jr')
    >>> sorted(parse_name("Bob Smith").first_forms)
    ['bob', 'robert']
    """
    if not raw or not str(raw).strip():
        return NameParts(None, (), None, None, frozenset())
    s = ascii_fold(str(raw)).lower()
    s = re.sub(r"[^a-z,\s'\-.]", " ", s)
    if "," in s:
        last_part, rest = s.split(",", 1)
        tokens = rest.split() + ["|"] + last_part.split()
    else:
        tokens = s.split()
    tokens = [_clean_token(t) if t != "|" else t for t in tokens]
    suffix = None
    kept: list[str] = []
    for t in tokens:
        if t in NAME_TITLES:
            continue
        if t in NAME_SUFFIXES:
            suffix = t
            continue
        if t:
            kept.append(t)
    if "|" in kept:
        i = kept.index("|")
        given, family = kept[:i], kept[i + 1 :]
        if not given and family:
            given, family = family[:1], family[1:]
        first = given[0] if given else None
        middle = tuple(given[1:])
        last = family[-1] if family else None
        if len(family) > 1:  # compound surname "de la cruz" -> "delacruz"
            last = "".join(family)
    else:
        if not kept:
            return NameParts(None, (), None, suffix, frozenset())
        first = kept[0]
        last = kept[-1] if len(kept) > 1 else None
        middle = tuple(kept[1:-1])
    if first and last and len(first) > 1 and len(last) == 1 and not middle:
        # "Smith J" style (rare): treat the single letter as first initial
        first, last = last, first
    forms = first_name_forms(first) if first and len(first) > 1 else frozenset({first} if first else ())
    return NameParts(first, middle, last, suffix, forms)


def metaphone(s: str | None) -> str:
    if not s:
        return ""
    try:
        return jellyfish.metaphone(s)
    except Exception:  # pragma: no cover - defensive against odd unicode
        return s[:4]


@dataclass(frozen=True)
class AddressParts:
    number: str | None
    street: str | None  # standardized street name tokens (no number/unit)
    unit: str | None
    zip: str | None
    city: str | None

    @property
    def key(self) -> str | None:
        if not self.number or not self.street:
            return None
        return f"{self.number} {self.street}"


@lru_cache(maxsize=200_000)
def parse_address(raw: str | None) -> AddressParts:
    """Standardize a free-text US address line.

    >>> parse_address("12 North Maple Street Apt. 4B, Lake Paul, NJ 08812").key
    '12 n maple st'
    """
    if not raw or not str(raw).strip():
        return AddressParts(None, None, None, None, None)
    s = ascii_fold(str(raw)).lower()
    zips = _ZIP.findall(s)
    zip_code = zips[-1] if zips else None
    parts = [p.strip() for p in s.split(",")]
    line = parts[0]
    city = parts[1] if len(parts) > 2 else None
    line = line.replace("#", " # ")
    tokens = re.sub(r"[^a-z0-9#\s]", " ", line).split()
    number = tokens[0] if tokens and tokens[0][0].isdigit() else None
    rest = tokens[1:] if number else tokens
    street_toks: list[str] = []
    unit = None
    i = 0
    while i < len(rest):
        t = rest[i]
        if t in _UNIT_WORDS or t == "#":
            unit = "".join(rest[i + 1 :]) or None
            break
        street_toks.append(_ABBREV_CANON.get(t, t))
        i += 1
    return AddressParts(number, " ".join(street_toks) or None, unit, zip_code, re.sub(r"[^a-z ]", "", city).strip() if city else None)


@lru_cache(maxsize=50_000)
def normalize_org_name(raw: str | None) -> str | None:
    """``"Bluefin Holdings, LLC"`` -> ``"bluefin holdings"``."""
    if not raw:
        return None
    s = ascii_fold(str(raw)).lower()
    s = re.sub(r"[^a-z0-9& ]", " ", s.replace("&", " and "))
    toks = [t for t in s.split() if t not in _ORG_SUFFIXES]
    return " ".join(toks) or None


def parse_iso(d: str | None) -> date | None:
    if not d:
        return None
    try:
        return date.fromisoformat(str(d)[:10])
    except ValueError:
        return None


def dob_relation(a: date | None, b: date | None) -> str:
    """Compare two dates of birth: exact | swap | near | diff | null."""
    if a is None or b is None:
        return "null"
    if a == b:
        return "exact"
    if a.year == b.year and a.month == b.day and a.day == b.month:
        return "swap"
    same = (a.year == b.year) + (a.month == b.month) + (a.day == b.day)
    if same == 2:
        if a.year != b.year and abs(a.year - b.year) in (1, 10):
            return "near"
        if a.day != b.day and (abs(a.day - b.day) in (1, 10) or str(a.day)[-1] == str(b.day)[-1]):
            return "near"
        if a.month != b.month and abs(a.month - b.month) == 1:
            return "near"
    return "diff"


def dobs_compatible(a: date, b: date) -> bool:
    return dob_relation(a, b) in ("exact", "swap", "near")
