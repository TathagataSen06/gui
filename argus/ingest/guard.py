"""Synthetic-data guard.

The platform is a benchmark, not a surveillance tool. Ingestion therefore
refuses to load a data directory unless

1. it carries a manifest written by :mod:`argus.generate` declaring the data
   synthetic, and
2. every phone number and e-mail address it contains lies in a namespace
   reserved for fiction (NANP 555-01XX; RFC 2606 ``.example``/``.test``).

Rows that violate (2) are dropped; if more than a tiny fraction violate it the
whole ingest aborts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .. import SYNTHETIC_MARKER
from ..ids import identifier_is_synthetic, is_fictional_phone, is_synthetic_email, normalize_email, normalize_msisdn

MAX_NON_SYNTHETIC_SHARE = 0.001


class NonSyntheticDataError(RuntimeError):
    """Raised when input does not look like argus-generated synthetic data."""


def check_manifest(root: Path) -> dict:
    path = Path(root) / "manifest.json"
    if not path.exists():
        raise NonSyntheticDataError(f"{path} not found: argus only ingests data produced by `argus generate`.")
    manifest = json.loads(path.read_text())
    if manifest.get("marker") != SYNTHETIC_MARKER or manifest.get("synthetic") is not True:
        raise NonSyntheticDataError("manifest does not declare argus synthetic data; refusing to ingest.")
    return manifest


def synthetic_mask_events(actor_ids: pd.Series) -> pd.Series:
    return actor_ids.map(lambda ids: all(identifier_is_synthetic(i) for i in ids))


def synthetic_mask_records(df: pd.DataFrame) -> pd.Series:
    def ok(row: pd.Series) -> bool:
        if not all(identifier_is_synthetic(i) for i in row["owned_ids"]):
            return False
        ph = normalize_msisdn(row["phone"]) if row["phone"] else None
        if ph and not is_fictional_phone(ph):
            return False
        em = normalize_email(row["email"]) if row["email"] else None
        if em and not is_synthetic_email(em):
            return False
        return True

    if df.empty:
        return pd.Series([], dtype=bool)
    return df.apply(ok, axis=1)


def enforce(mask: pd.Series, what: str) -> int:
    """Return number of rejected rows; abort if the share is implausible for synthetic data."""
    bad = int((~mask).sum())
    if len(mask) and bad / len(mask) > MAX_NON_SYNTHETIC_SHARE:
        raise NonSyntheticDataError(
            f"{bad} of {len(mask)} {what} reference phone numbers or e-mail addresses outside the reserved "
            "fictional namespaces. argus is not designed for real-world personal data; aborting."
        )
    return bad
