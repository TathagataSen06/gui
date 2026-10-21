"""Common, source-agnostic data model.

Every feed is mapped into two record kinds:

* :class:`Event` - something that happened at a point in time
  (``timestamp, source, event_type, actor_ids, attributes``). Actors are
  *typed identifiers* (``acct:...``, ``msisdn:...``, ``card:...``,
  ``email:...``) rather than people: who stands behind an identifier is the
  job of entity resolution, not ingestion.
* :class:`IdentityRecord` - a source system's description of a party (a bank
  KYC profile, a telco subscriber, an access-card holder, an e-mail display
  name, a watch-list entry). Records *own* identifiers and carry the noisy
  descriptive fields used for matching.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

# Identifier namespaces. Prefix -> human label.
ID_TYPES: dict[str, str] = {
    "acct": "bank account",
    "merchant": "card merchant",
    "ext": "external account (other institution)",
    "msisdn": "phone number",
    "card": "access card",
    "email": "e-mail address",
    "loc": "location / reader",
    "branch": "bank branch / ATM",
    "cell": "cell site",
}

EVENT_TYPES: frozenset[str] = frozenset(
    {
        "transfer",
        "card_purchase",
        "cash_deposit",
        "cash_withdrawal",
        "call",
        "sms",
        "access_granted",
        "access_denied",
        "email",
    }
)


def typed_id(kind: str, value: str) -> str:
    """Build a typed identifier such as ``msisdn:+12015550142``."""
    if kind not in ID_TYPES:
        raise ValueError(f"unknown identifier type {kind!r}")
    if not value:
        raise ValueError("identifier value must be non-empty")
    return f"{kind}:{value}"


def id_kind(identifier: str) -> str:
    """Return the namespace of a typed identifier (``'acct'`` for ``acct:...``)."""
    return identifier.split(":", 1)[0]


@dataclass(frozen=True)
class Event:
    """One normalized event. ``timestamp`` is always timezone-aware UTC.

    ``actor_ids`` is ordered: the initiator (payer, caller, sender, card
    holder) comes first, then counterparties. ``actor_roles`` runs parallel to
    ``actor_ids`` and names each actor's role.
    """

    event_id: str
    timestamp: datetime
    source: str
    event_type: str
    actor_ids: tuple[str, ...]
    actor_roles: tuple[str, ...]
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("Event.timestamp must be timezone-aware")
        if self.timestamp.utcoffset() != timezone.utc.utcoffset(None):
            object.__setattr__(self, "timestamp", self.timestamp.astimezone(timezone.utc))
        if self.event_type not in EVENT_TYPES:
            raise ValueError(f"unknown event_type {self.event_type!r}")
        if len(self.actor_ids) != len(self.actor_roles):
            raise ValueError("actor_ids and actor_roles must have equal length")
        if not self.actor_ids:
            raise ValueError("an event needs at least one actor")


@dataclass(frozen=True)
class IdentityRecord:
    """A party as described by one source system (fields are raw/noisy)."""

    record_id: str
    source: str
    entity_type: str  # "person" | "org"
    name: str | None = None
    dob: str | None = None
    address: str | None = None
    phone: str | None = None
    email: str | None = None
    owned_ids: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.entity_type not in ("person", "org"):
            raise ValueError(f"entity_type must be person|org, got {self.entity_type!r}")


def to_json(obj: Mapping[str, Any]) -> str:
    """Deterministic JSON serialization used for ``attributes`` columns."""
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


# DuckDB DDL for the normalized warehouse.
WAREHOUSE_DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS manifest (key VARCHAR PRIMARY KEY, value VARCHAR)
    """,
    """
    CREATE TABLE IF NOT EXISTS identity_records (
        record_id   VARCHAR PRIMARY KEY,
        source      VARCHAR NOT NULL,
        entity_type VARCHAR NOT NULL,
        name        VARCHAR,
        dob         VARCHAR,
        address     VARCHAR,
        phone       VARCHAR,
        email       VARCHAR,
        owned_ids   VARCHAR[],
        attributes  JSON
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        event_id    VARCHAR PRIMARY KEY,
        ts          TIMESTAMP NOT NULL,   -- UTC
        source      VARCHAR NOT NULL,
        event_type  VARCHAR NOT NULL,
        actor_ids   VARCHAR[] NOT NULL,
        actor_roles VARCHAR[] NOT NULL,
        attributes  JSON
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS event_actors (
        event_id VARCHAR NOT NULL,
        actor_id VARCHAR NOT NULL,
        role     VARCHAR NOT NULL,
        ordinal  INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS case_labels (
        case_id           VARCHAR PRIMARY KEY,
        subject_record_id VARCHAR NOT NULL,
        disposition       VARCHAR NOT NULL,
        typology          VARCHAR,
        opened_date       DATE,
        closed_date       DATE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ingest_log (
        source        VARCHAR,
        adapter       VARCHAR,
        rows_read     BIGINT,
        events_loaded BIGINT,
        records_loaded BIGINT,
        rows_rejected BIGINT,
        reject_reasons JSON,
        seconds       DOUBLE
    )
    """,
)
