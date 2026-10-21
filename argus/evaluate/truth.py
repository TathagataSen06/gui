"""Ground-truth loading and entity <-> truth alignment (evaluation only).

Nothing outside :mod:`argus.evaluate` reads ``ground_truth/``; the pipeline
only ever sees the raw feeds and the historical case outcomes.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from ..paths import RunPaths


@dataclass
class Truth:
    record_truth: dict[str, str]  # record_id -> true person/org id
    record_kind: dict[str, str]  # record_id -> person | org | burner
    person_labels: pd.DataFrame
    rings: list[dict]
    decoys: list[dict]
    suspicious_ids: set[str] = field(default_factory=set)  # persons + orgs in planted rings
    victims: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, root: str | Path) -> "Truth":
        gt = RunPaths.at(root).ground_truth
        rt = pd.read_csv(gt / "record_truth.csv", dtype=str)
        labels = pd.read_csv(gt / "person_labels.csv", dtype=str, keep_default_na=False)
        rings = json.loads((gt / "scenarios.json").read_text())
        decoys = json.loads((gt / "decoys.json").read_text())
        sus = {m["person_id"] for r in rings for m in r["members"]} | {o["org_id"] for r in rings for o in r["orgs"]}
        vic = {v for r in rings for v in r.get("victims", [])}
        return cls(dict(zip(rt["record_id"], rt["true_id"])), dict(zip(rt["record_id"], rt["entity_kind"])), labels, rings, decoys, sus, vic)


def entity_truth(con: duckdb.DuckDBPyConnection, truth: Truth) -> pd.DataFrame:
    """Per predicted entity: majority true id, purity, and whether it is suspicious.

    An entity is *suspicious* if any of its records belongs to a planted ring
    member or ring organisation (so an over-merged entity that swallowed a
    ring member still counts as a hit - the analyst would find them).
    """
    re = con.execute("SELECT record_id, entity_id FROM record_entity").fetchall()
    by_ent: dict[str, list[str]] = defaultdict(list)
    for r, e in re:
        by_ent[e].append(r)
    rows = []
    for e, recs in by_ent.items():
        ids = [truth.record_truth.get(r) for r in recs if truth.record_truth.get(r)]
        c = Counter(ids)
        major, n = c.most_common(1)[0] if c else (None, 0)
        rows.append(
            {
                "entity_id": e,
                "true_id": major,
                "purity": n / len(ids) if ids else 1.0,
                "true_ids": sorted(c),
                "suspicious": any(i in truth.suspicious_ids for i in c),
                "victim": any(i in truth.victims for i in c),
                "phone_only": all(truth.record_kind.get(r) in ("burner", "anonymous_line") for r in recs),
            }
        )
    return pd.DataFrame(rows).set_index("entity_id")


def person_entities(con: duckdb.DuckDBPyConnection, truth: Truth) -> dict[str, list[str]]:
    """true person/org id -> predicted entities holding any of its records."""
    out: dict[str, set[str]] = defaultdict(set)
    for r, e in con.execute("SELECT record_id, entity_id FROM record_entity").fetchall():
        t = truth.record_truth.get(r)
        if t:
            out[t].add(e)
    return {k: sorted(v) for k, v in out.items()}
