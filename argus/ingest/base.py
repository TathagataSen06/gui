"""Source-adapter contract and registry.

Adding a new feed means writing one class::

    @register_adapter
    class MyFeedAdapter(SourceAdapter):
        name = "myfeed"
        def records(self) -> pd.DataFrame: ...   # identity records (optional)
        def events(self) -> Iterator[pd.DataFrame]: ...  # event batches

and dropping its raw files under ``raw/myfeed/``. Nothing downstream changes:
the warehouse, entity resolution, graph and scoring layers only ever see the
common schema (:mod:`argus.schema`).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

#: Columns every adapter's ``events()`` batches must provide.
EVENT_COLUMNS: tuple[str, ...] = ("event_id", "ts", "source", "event_type", "actor_ids", "actor_roles", "attributes")

#: Columns every adapter's ``records()`` frame must provide.
RECORD_COLUMNS: tuple[str, ...] = ("record_id", "source", "entity_type", "name", "dob", "address", "phone", "email", "owned_ids", "attributes")


class SourceAdapter(ABC):
    """Maps one raw feed into identity records and common-schema events."""

    #: Short source name; also the default raw sub-directory.
    name: str = ""
    #: Raw sub-directory (defaults to ``name``).
    directory: str | None = None

    def __init__(self, raw_root: Path) -> None:
        self.raw_root = Path(raw_root)
        self.rejects: dict[str, int] = {}
        self.rows_read = 0

    @property
    def root(self) -> Path:
        return self.raw_root / (self.directory or self.name)

    def available(self) -> bool:
        return self.root.exists()

    def reject(self, reason: str, n: int = 1) -> None:
        if n:
            self.rejects[reason] = self.rejects.get(reason, 0) + n

    def records(self) -> pd.DataFrame:
        """Identity records described by this source (may be empty)."""
        return pd.DataFrame(columns=list(RECORD_COLUMNS))

    @abstractmethod
    def events(self) -> Iterator[pd.DataFrame]:
        """Yield event batches with :data:`EVENT_COLUMNS` (``ts`` tz-aware UTC)."""

    def labels(self) -> pd.DataFrame | None:
        """Optional case-outcome labels (only the labels adapter provides these)."""
        return None


_REGISTRY: dict[str, type[SourceAdapter]] = {}


def register_adapter(cls: type[SourceAdapter]) -> type[SourceAdapter]:
    """Class decorator adding an adapter to the registry."""
    if not cls.name:
        raise ValueError("adapter must define a name")
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ValueError(f"adapter {cls.name!r} already registered")
    _REGISTRY[cls.name] = cls
    return cls


def registered_adapters() -> dict[str, type[SourceAdapter]]:
    from . import adapters  # noqa: F401  (import side effect: registration)

    return dict(sorted(_REGISTRY.items()))


def attrs_json(rows: list[dict[str, Any]]) -> list[str]:
    """Serialize attribute dicts, dropping null/empty values."""
    out = []
    for r in rows:
        clean = {k: v for k, v in r.items() if v is not None and v == v and v != ""}
        out.append(json.dumps(clean, sort_keys=True, default=str, separators=(",", ":")))
    return out


def none_if_blank(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and v != v:
        return None
    if isinstance(v, str) and not v.strip():
        return None
    return v
