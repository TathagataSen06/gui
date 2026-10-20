"""End-to-end entity resolution over the warehouse's identity records."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from ..db import connect
from ..paths import RunPaths
from .blocking import block
from .clustering import cluster_pairs
from .embedding import CharNgramEmbedder
from .matching import FellegiSunterScorer, MatchConfig, PairScore
from .normalize import NameParts, normalize_org_name
from .records import Rec, load_records

SOURCE_PRIORITY = {"bank": 0, "access": 1, "telco": 2, "watchlist": 3, "email": 4}


@dataclass
class ResolutionResult:
    record_entity: dict[str, str]
    entities: pd.DataFrame
    candidate_pairs: set[tuple[str, str]]
    stats: dict[str, Any] = field(default_factory=dict)


def entity_id(kind: str, members: list[str]) -> str:
    digest = hashlib.sha1(min(members).encode()).hexdigest()[:10].upper()
    return f"{'PER' if kind == 'person' else 'ORG'}-{digest}"


def _display_name(n: NameParts) -> str | None:
    if not n.first and not n.last:
        return None
    mid = f" {n.middle[0][0].upper()}." if n.middle else ""
    sfx = f" {n.suffix.capitalize()}." if n.suffix else ""
    return f"{(n.first or '').capitalize()}{mid} {(n.last or '').capitalize()}{sfx}".strip()


def smart_title(s: str) -> str:
    """Title-case an all-caps name without mangling apostrophes or corporate suffixes."""
    t = re.sub(r"[A-Za-z]+('[A-Za-z]+)?", lambda m: m.group(0)[0].upper() + m.group(0)[1:].lower(), s)
    for suf in ("LLC", "LP", "LLP", "Inc", "Ltd", "Co"):
        t = re.sub(rf"\b{suf}\b", suf, t, flags=re.IGNORECASE)
    return t


def _consensus_name(named: list[Rec]) -> NameParts:
    """Majority vote over records (typos lose to the spelling most sources agree on)."""
    votes = Counter((min(r.name.first_forms) if r.name.first_forms else r.name.first, r.name.last) for r in named)
    best = max(votes.items(), key=lambda kv: (kv[1], -min(SOURCE_PRIORITY.get(r.source, 9) for r in named if (min(r.name.first_forms) if r.name.first_forms else r.name.first, r.name.last) == kv[0])))[0]
    pool = [r for r in named if (min(r.name.first_forms) if r.name.first_forms else r.name.first, r.name.last) == best]
    with_middle = [r for r in pool if r.name.middle and len(r.name.first or "") > 1]
    return (with_middle or pool)[0].name


def _canonical(members: list[Rec]) -> dict[str, Any]:
    ordered = sorted(members, key=lambda r: (SOURCE_PRIORITY.get(r.source, 9), r.record_id))
    etype = ordered[0].entity_type
    if etype == "org":
        named = [r for r in ordered if r.name_raw]
        name = named[0].name_raw if named else None
        if name and name.isupper():
            name = smart_title(name)
        if not name:
            emails = sorted({e for r in ordered for e in r.emails})
            name = emails[0] if emails else ordered[0].record_id
    else:
        named = [r for r in ordered if r.name.first and r.name.last and len(r.name.first) > 1]
        name = _display_name(_consensus_name(named)) if named else None
        if name is None:
            alias = next((r.name_raw for r in ordered if r.name_raw), None)
            phones = sorted({p for r in ordered for p in r.phones})
            emails = sorted({e for r in ordered for e in r.emails})
            name = alias or (f"Unregistered line {phones[0]}" if phones else (emails[0] if emails else ordered[0].record_id))
    dobs = Counter(r.dob for r in ordered if r.dob)
    addr = next((r.attrs.get("_address_raw") for r in ordered if r.attrs.get("_address_raw")), None)
    return {
        "entity_type": etype,
        "display_name": name,
        "dob": dobs.most_common(1)[0][0].isoformat() if dobs else None,
        "address": addr,
        "n_records": len(members),
        "sources": sorted({r.source for r in members}),
        "record_ids": sorted(r.record_id for r in members),
    }


def resolve(con: duckdb.DuckDBPyConnection, cfg: MatchConfig | None = None) -> ResolutionResult:
    """Block, score, cluster and persist entities. Returns the mapping and stats."""
    cfg = cfg or MatchConfig()
    t0 = time.perf_counter()
    records = load_records(con)
    raw_addr = dict(con.execute("SELECT record_id, address FROM identity_records").fetchall())
    for r in records:
        r.attrs["_address_raw"] = raw_addr.get(r.record_id)
    emb = CharNgramEmbedder(seed=cfg.seed).embed([r.text for r in records]) if cfg.use_embeddings and len(records) > 3 else None
    blocking = block(records)
    scorer = FellegiSunterScorer(records, cfg, emb)
    scored: list[PairScore] = [scorer.score(i, j) for i, j in sorted(blocking.pairs)]
    links = [(s.i, s.j, max(s.prob, 0.95) if s.rule else s.prob) for s in scored if s.is_match]
    uf = cluster_pairs(len(records), [r.dob if r.entity_type == "person" else None for r in records], links)
    clusters = uf.clusters()

    record_entity: dict[str, str] = {}
    ent_rows = []
    for members_idx in clusters.values():
        members = [records[i] for i in members_idx]
        eid = entity_id(members[0].entity_type, [m.record_id for m in members])
        for m in members:
            record_entity[m.record_id] = eid
        ent_rows.append({"entity_id": eid, **_canonical(members)})
    entities = pd.DataFrame(ent_rows).sort_values("entity_id").reset_index(drop=True)

    pair_rows = []
    for s in scored:
        if s.prob >= 0.05 or s.rule or s.is_match:
            a, b = records[s.i], records[s.j]
            pair_rows.append(
                {
                    "record_a": a.record_id,
                    "record_b": b.record_id,
                    "match_weight": round(s.weight, 3),
                    "match_probability": round(s.prob, 5),
                    "linked": bool(s.is_match),
                    "same_entity": record_entity[a.record_id] == record_entity[b.record_id],
                    "evidence": json.dumps(s.evidence(), sort_keys=True),
                }
            )
    pairs_df = pd.DataFrame(pair_rows, columns=["record_a", "record_b", "match_weight", "match_probability", "linked", "same_entity", "evidence"])

    ident_rows = [{"identifier": oid, "entity_id": record_entity[r.record_id], "record_id": r.record_id} for r in records for oid in r.owned_ids]
    ident_df = pd.DataFrame(ident_rows, columns=["identifier", "entity_id", "record_id"]).drop_duplicates()

    link_rows = []
    org_by_name: dict[str, str] = {}
    for r in records:
        if r.entity_type == "org" and r.org_name:
            org_by_name.setdefault(r.org_name, record_entity[r.record_id])
    for r in records:
        if r.attrs.get("role") == "beneficial_owner" and r.attrs.get("controls") in record_entity:
            link_rows.append({"src_entity": record_entity[r.record_id], "dst_entity": record_entity[r.attrs["controls"]], "relation": "controls", "evidence": f"beneficial owner declared on {r.attrs['controls']}"})
        emp = normalize_org_name(r.attrs.get("employer_name")) if r.entity_type == "person" else None
        if emp and emp in org_by_name:
            link_rows.append({"src_entity": record_entity[r.record_id], "dst_entity": org_by_name[emp], "relation": "employed_by", "evidence": f"employer on {r.record_id}"})
    links_df = pd.DataFrame(link_rows, columns=["src_entity", "dst_entity", "relation", "evidence"]).drop_duplicates(["src_entity", "dst_entity", "relation"])

    stats = {
        "records": len(records),
        "entities": int(len(entities)),
        "candidate_pairs": len(blocking.pairs),
        "blocking_key_pairs": blocking.key_counts,
        "dropped_oversized_blocks": blocking.dropped_blocks,
        "linked_pairs": len(links),
        "rule_links": sum(1 for s in scored if s.rule),
        "cannot_link_refusals": uf.refused,
        "scorer": scorer.describe(),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    _persist(con, entities, record_entity, pairs_df, ident_df, links_df, stats)
    cand = {tuple(sorted((records[i].record_id, records[j].record_id))) for i, j in blocking.pairs}
    return ResolutionResult(record_entity, entities, cand, stats)  # type: ignore[arg-type]


def _persist(con, entities, record_entity, pairs_df, ident_df, links_df, stats) -> None:  # type: ignore[no-untyped-def]
    ent = entities.copy()
    con.register("_ent", ent)
    con.execute("CREATE OR REPLACE TABLE entities AS SELECT entity_id, entity_type, display_name, dob, address, n_records, sources::VARCHAR[] AS sources, record_ids::VARCHAR[] AS record_ids FROM _ent")
    con.unregister("_ent")
    re_df = pd.DataFrame(sorted(record_entity.items()), columns=["record_id", "entity_id"])
    for name, df in (("record_entity", re_df), ("er_pairs", pairs_df), ("identifier_entity", ident_df), ("entity_links", links_df)):
        con.register("_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _df")
        con.unregister("_df")
    con.execute("CREATE TABLE IF NOT EXISTS run_meta (stage VARCHAR, payload JSON)")
    con.execute("DELETE FROM run_meta WHERE stage = 'resolve'")
    con.execute("INSERT INTO run_meta VALUES ('resolve', ?)", [json.dumps(stats, default=str)])


def run(root: str | Path, cfg: MatchConfig | None = None) -> ResolutionResult:
    paths = RunPaths.at(root)
    con = connect(paths.db)
    try:
        return resolve(con, cfg)
    finally:
        con.close()


__all__ = ["resolve", "run", "ResolutionResult", "MatchConfig"]
