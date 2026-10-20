"""Date/time format inference for heterogeneous feeds.

Source systems rarely document their date formats and a single consortium
feed can mix several. :func:`infer_format` tries candidate ``strptime``
patterns on a sample and picks the one that parses the most values; for
ambiguous day/month layouts a single value with a component > 12 settles it.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

DATE_CANDIDATES: tuple[str, ...] = ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%d-%b-%Y", "%d%m%Y", "%Y%m%d", "%m-%d-%Y", "%d.%m.%Y")
DATETIME_CANDIDATES: tuple[str, ...] = (
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%d %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%Y%m%d%H%M%S",
    "%a, %d %b %Y %H:%M:%S %z",
)


def infer_format(values: Iterable[object], candidates: tuple[str, ...] = DATE_CANDIDATES, sample: int = 2000) -> str | None:
    """Return the candidate format that parses the largest share of ``values``."""
    s = pd.Series([v for v in values if isinstance(v, str) and v.strip()][:sample], dtype="object")
    if s.empty:
        return None
    best, best_ok = None, -1
    for fmt in candidates:
        ok = int(pd.to_datetime(s, format=fmt, errors="coerce").notna().sum())
        if ok > best_ok:
            best, best_ok = fmt, ok
    return best if best_ok > 0 else None


def parse_dates(values: pd.Series, fmt: str | None = None) -> pd.Series:
    """Parse to ISO ``YYYY-MM-DD`` strings (``None`` where unparseable)."""
    fmt = fmt or infer_format(values.tolist())
    if fmt is None:
        return pd.Series([None] * len(values), index=values.index, dtype="object")
    parsed = pd.to_datetime(values, format=fmt, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), None)


def parse_local_datetimes(values: pd.Series, tz: str, fmt: str | None = None) -> pd.Series:
    """Parse wall-clock or offset-bearing timestamps to tz-aware UTC."""
    fmt = fmt or infer_format(values.tolist(), DATETIME_CANDIDATES)
    if fmt is None:
        raise ValueError("could not infer timestamp format")
    if "%z" in fmt:
        return pd.to_datetime(values, format=fmt, errors="coerce", utc=True)
    naive = pd.to_datetime(values, format=fmt, errors="coerce")
    return naive.dt.tz_localize(tz, ambiguous=False, nonexistent="shift_forward").dt.tz_convert("UTC")
