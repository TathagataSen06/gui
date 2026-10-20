"""Load identity records from the warehouse and normalize them for matching."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date

import duckdb

from ..ids import normalize_email, normalize_msisdn
from .normalize import AddressParts, NameParts, normalize_org_name, parse_address, parse_iso, parse_name


@dataclass
class Rec:
    """A normalized identity record."""

    idx: int
    record_id: str
    source: str
    entity_type: str
    name_raw: str | None
    name: NameParts
    aliases: list[NameParts]
    dob: date | None
    addr: AddressParts
    phones: frozenset[str]
    emails: frozenset[str]
    employer: str | None
    org_name: str | None
    reg_no: str | None
    owned_ids: tuple[str, ...]
    attrs: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Concatenated canonical text used by the embedding model."""
        if self.entity_type == "org":
            return f"{self.org_name or ''} {self.addr.key or ''} {self.addr.zip or ''}"
        return f"{self.name.canonical_full} {self.addr.key or ''} {self.addr.zip or ''}".strip()


def load_records(con: duckdb.DuckDBPyConnection) -> list[Rec]:
    rows = con.execute(
        "SELECT record_id, source, entity_type, name, dob, address, phone, email, owned_ids, attributes FROM identity_records ORDER BY record_id"
    ).fetchall()
    out: list[Rec] = []
    for i, (rid, src, etype, name, dob, address, phone, email, owned, attrs) in enumerate(rows):
        a = json.loads(attrs) if attrs else {}
        owned = tuple(owned or ())
        phones = {p for p in (normalize_msisdn(phone),) if p} | {o.split(":", 1)[1] for o in owned if o.startswith("msisdn:")}
        emails = {e for e in (normalize_email(email),) if e} | {o.split(":", 1)[1] for o in owned if o.startswith("email:")}
        aliases = [parse_name(x) for x in (a.get("aliases") or [])]
        if src == "email" and a.get("display_names"):
            aliases += [parse_name(x) for x in a["display_names"][1:]]
        out.append(
            Rec(
                idx=i,
                record_id=rid,
                source=src,
                entity_type=etype,
                name_raw=name,
                name=parse_name(name) if etype == "person" else parse_name(None),
                aliases=aliases,
                dob=parse_iso(dob),
                addr=parse_address(address),
                phones=frozenset(phones),
                emails=frozenset(emails),
                employer=normalize_org_name(a.get("employer_name")),
                org_name=normalize_org_name(name) if etype == "org" else None,
                reg_no=a.get("tax_reg_no"),
                owned_ids=owned,
                attrs=a,
            )
        )
    return out
