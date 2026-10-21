"""Data-quality defect injection.

Real identity data is messy in characteristic ways: keyboard typos, nicknames,
initials, transposed day/month, stale addresses, inconsistent abbreviations,
missing fields and inconsistent phone formatting. Each function here models
one of those defects and takes an explicit RNG for reproducibility.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from .reference import KEY_NEIGHBOURS, NICKNAMES, OCR_CONFUSIONS, STREET_SUFFIX_ABBREV, UNIT_DESIGNATORS


def typo(text: str, rng: np.random.Generator) -> str:
    """Apply one realistic typing error (substitution, transposition, deletion, insertion)."""
    if len(text) < 3:
        return text
    letters = [i for i, c in enumerate(text) if c.isalpha()]
    if len(letters) < 2:
        return text
    # Errors are rarer on the first character (people check the start of names).
    i = int(rng.choice(letters[1:]))
    kind = rng.choice(["sub", "swap", "del", "dup", "ocr"], p=[0.38, 0.25, 0.2, 0.12, 0.05])
    c = text[i]
    lower = c.lower()
    if kind == "sub" and lower in KEY_NEIGHBOURS:
        r = str(rng.choice(list(KEY_NEIGHBOURS[lower])))
        r = r.upper() if c.isupper() else r
        return text[:i] + r + text[i + 1 :]
    if kind == "swap" and i + 1 < len(text) and text[i + 1].isalpha():
        return text[:i] + text[i + 1] + text[i] + text[i + 2 :]
    if kind == "del":
        return text[:i] + text[i + 1 :]
    if kind == "dup":
        return text[:i] + c + text[i:]
    if kind == "ocr" and lower in OCR_CONFUSIONS:
        return text[:i] + OCR_CONFUSIONS[lower] + text[i + 1 :]
    return text[:i] + text[i + 1 :]


def nickname(first: str, rng: np.random.Generator) -> str:
    """Return a nickname for ``first`` if one exists, else ``first`` unchanged."""
    nicks = NICKNAMES.get(first.lower())
    if not nicks:
        return first
    n = str(rng.choice(list(nicks)))
    return n.capitalize() if first[:1].isupper() else n


def format_person_name(
    first: str,
    middle: str | None,
    last: str,
    suffix: str | None,
    style: str,
) -> str:
    """Render a name in one of the layouts used by different source systems."""
    mi = f"{middle[0]}." if middle else ""
    sfx = f" {suffix}" if suffix else ""
    if style == "title_middle_initial":
        return " ".join(p for p in (first, mi, last) if p) + sfx
    if style == "upper_last_comma_first":
        return f"{last}{sfx}, {first}{' ' + middle if middle else ''}".upper()
    if style == "title_first_last":
        return f"{first} {last}{sfx}"
    if style == "full":
        return " ".join(p for p in (first, middle, last) if p) + sfx
    if style == "last_comma_first_mi":  # access-control cardholder convention
        return f"{last.upper()}, {first.upper()}{' ' + middle[0].upper() if middle else ''}"
    if style == "first_last":
        return f"{first} {last}"
    if style == "initial_last":
        return f"{first[0]}. {last}"
    if style == "lower":
        return f"{first} {last}".lower()
    if style == "first_only":
        return first
    raise ValueError(f"unknown name style {style!r}")


DOB_FORMATS: dict[str, str] = {
    "iso": "%Y-%m-%d",
    "us": "%m/%d/%Y",
    "eu": "%d/%m/%Y",
    "compact_eu": "%d%m%Y",
    "compact_iso": "%Y%m%d",
    "dmy_mon": "%d-%b-%Y",
}


def corrupt_dob(d: date, rng: np.random.Generator, swap_p: float, typo_p: float) -> date:
    """Return ``d`` possibly with day/month transposed or a digit/year error."""
    u = rng.random()
    if u < swap_p and d.day <= 12 and d.day != d.month:
        return date(d.year, d.day, d.month)
    if u < swap_p + typo_p:
        kind = rng.choice(["year", "day"])
        try:
            if kind == "year":
                return d.replace(year=d.year + int(rng.choice([-1, 1, -10, 10])))
            nd = d.day + int(rng.choice([-1, 1, 10, -10]))
            return d.replace(day=nd) if 1 <= nd <= 28 else d
        except ValueError:
            return d
    return d


def format_street(number: str, street: str, unit: str | None, rng: np.random.Generator, abbreviate_p: float, unit_missing_p: float) -> str:
    """Render a street line with inconsistent abbreviation / casing / unit formatting."""
    words = street.split()
    out: list[str] = []
    for w in words:
        lw = w.lower()
        if lw in STREET_SUFFIX_ABBREV and rng.random() < abbreviate_p:
            ab = STREET_SUFFIX_ABBREV[lw]
            out.append(ab.capitalize() + ("." if rng.random() < 0.3 else ""))
        else:
            out.append(w)
    line = f"{number} {' '.join(out)}"
    if unit and rng.random() >= unit_missing_p:
        kind, val = unit.split(" ", 1) if " " in unit else ("Apt.", unit)
        k = kind.lower().strip(".")
        style = rng.choice(["keep", "abbr", "hash"])
        if style == "hash":
            line += f" #{val}"
        elif style == "abbr":
            line += f" {UNIT_DESIGNATORS.get(k, k).capitalize()} {val}"
        else:
            line += f", {kind} {val}"
    return line


def format_phone(e164: str, style: str) -> str:
    """Render an E.164 NANP number in a source-specific layout."""
    d = e164[2:]  # strip +1
    a, x, n = d[:3], d[3:6], d[6:]
    return {
        "e164": e164,
        "paren": f"({a}) {x}-{n}",
        "dash": f"{a}-{x}-{n}",
        "dot": f"{a}.{x}.{n}",
        "digits": d,
        "one_dash": f"1-{a}-{x}-{n}",
        "intl": f"+1 {a} {x} {n}",
    }[style]


def phone_digit_typo(e164: str, rng: np.random.Generator) -> str:
    """Transpose or mistype one of the last two digits (a common keying error).

    Only the final two digits are touched so the number stays inside the
    fictional 555-01XX block (any other edit could land on a real number).
    """
    d = list(e164)
    if rng.random() < 0.5 and d[-1] != d[-2]:
        d[-1], d[-2] = d[-2], d[-1]
    else:
        i = len(d) - 1 - int(rng.integers(0, 2))
        d[i] = str((int(d[i]) + int(rng.choice([1, 9]))) % 10)
    return "".join(d)


def email_typo(address: str, rng: np.random.Generator) -> str:
    local, _, domain = address.partition("@")
    if rng.random() < 0.5:
        return typo(local, rng) + "@" + domain
    return local + "@" + typo(domain.split(".")[0], rng) + "." + ".".join(domain.split(".")[1:])
