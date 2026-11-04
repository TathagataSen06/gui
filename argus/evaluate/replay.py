"""Chronological, capacity-limited triage on an Argus-generated world."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import duckdb
import pandas as pd

from .. import SYNTHETIC_MARKER
from ..db import connect
from ..ingest.guard import NonSyntheticDataError, check_manifest
from ..paths import RunPaths
from ..score.model import explain_logreg, make_models, transform
from ..score.narrative import top_drivers
from ..score.pipeline import ScoreConfig, blend_risk
from ..score.rules import apply_rules
from ..score.snapshot import Window, features_with_contacts, snapshot_warehouse
from .truth import Truth, entity_truth

MIN_CLASS_CASES = 3
POLICIES = {"rules": "Rules only", "hybrid": "Model and rules"}


def _closed_labels(con: duckdb.DuckDBPyConnection, training_end: date, cutoff: date) -> pd.Series:
    """Only explicit outcomes for cases opened after the training features were captured."""
    rows = con.execute(
        """
        SELECT re.entity_id, max(CASE WHEN c.disposition = 'CONFIRMED_SUSPICIOUS' THEN 1 ELSE 0 END) AS label
        FROM case_labels c JOIN record_entity re ON re.record_id = c.subject_record_id
        WHERE c.opened_date >= ? AND c.closed_date >= c.opened_date AND c.closed_date < ?
          AND c.disposition IN ('CONFIRMED_SUSPICIOUS', 'CLOSED_NO_ACTION', 'CLOSED_VICTIM')
        GROUP BY re.entity_id ORDER BY re.entity_id
        """,
        [training_end, cutoff],
    ).fetchall()
    return pd.Series(dict(rows), dtype=int)


def _candidates(con: duckdb.DuckDBPyConnection, cutoff: date) -> set[str]:
    """Entities with activity in the snapshot and no case opened before the cutoff."""
    return {r[0] for r in con.execute(
        """
        SELECT DISTINCT ie.entity_id
        FROM event_actors ea JOIN identifier_entity ie ON ie.identifier = ea.actor_id
        JOIN entities e ON e.entity_id = ie.entity_id
        WHERE e.entity_type IN ('person', 'org')
        EXCEPT
        SELECT re.entity_id FROM case_labels c JOIN record_entity re ON re.record_id = c.subject_record_id
        WHERE c.opened_date < ?
        """,
        [cutoff],
    ).fetchall()}


def _queue(scores: pd.Series, eligible: set[str], reviewed: set[str], minimum: float) -> pd.Series:
    """Rank unseen eligible entities. Entity ID is a deterministic tie-breaker."""
    qualified = scores[scores.index.isin(eligible - reviewed) & (scores >= minimum)]
    return qualified.sort_index().sort_values(ascending=False, kind="stable")


def _review_rounds(
    con: duckdb.DuckDBPyConnection, db: Path, world: Window, lookback: int, step: int, budget: int, minimum: float,
) -> tuple[Window, list[dict[str, Any]]]:
    cfg = ScoreConfig()
    train = Window(world.start, world.start + timedelta(days=lookback))
    offsets = [*range(lookback, world.days, step), world.days]
    reviewed: dict[str, set[str]] = {p: set() for p in POLICIES}
    rounds = []
    train_features = None
    for offset in offsets:
        cutoff = world.start + timedelta(days=offset)
        window = Window(cutoff - timedelta(days=lookback), cutoff)
        with snapshot_warehouse(con, db, window) as snapshot:
            X = features_with_contacts(snapshot, cfg.features, cfg.first_pass_flag)
            eligible = _candidates(snapshot, cutoff)
        Xt = transform(X)
        if train_features is None:
            train_features = Xt
        rule, hits = apply_rules(X)
        labels = _closed_labels(con, train.end, cutoff)
        positives = int(labels.sum())
        negatives = len(labels) - positives
        hybrid = rule
        contributions = None
        prob = None
        if min(positives, negatives) >= MIN_CLASS_CASES:
            model = make_models(cfg.model.seed)["logreg"].fit(train_features.loc[labels.index].values, labels.values)
            prob = pd.Series(model.predict_proba(Xt.values)[:, 1], index=X.index)
            hybrid, model_drives = blend_risk(rule, prob, cfg)
            contributions = explain_logreg(model, Xt.loc[model_drives])
        policies = {}
        for policy, scores in (("rules", rule), ("hybrid", hybrid)):
            queue = _queue(scores, eligible, reviewed[policy], minimum)
            entries = []
            for eid, score in queue.head(budget).items():
                drivers = []
                if policy == "hybrid" and contributions is not None and eid in contributions.index:
                    drivers = top_drivers(contributions.loc[eid], X.loc[eid], k=3)
                entries.append({
                    "entity_id": eid, "score": float(score), "rule_score": float(rule[eid]),
                    "model_probability": float(prob[eid]) if policy == "hybrid" and prob is not None else None,
                    "rule_hits": hits[eid], "model_drivers": drivers,
                })
                reviewed[policy].add(eid)
            policies[policy] = {"candidates": len(queue), "over_budget": max(0, len(queue) - budget), "reviews": entries}
        rounds.append({
            "window": window.to_dict(), "eligible": len(eligible), "positive_labels": positives,
            "negative_labels": negatives, "hybrid_mode": "logreg" if contributions is not None else "rules fallback",
            "policies": policies,
        })
    return train, rounds


def _evaluate(rounds: list[dict[str, Any]], truth: Truth, aligned: pd.DataFrame, start: date) -> dict[str, Any]:
    """Grade decisions after all queues have been built. Truth never affects ranking."""
    ring_members = {r["ring_id"]: {m["person_id"] for m in r["members"]} for r in truth.rings}
    onsets = {r["ring_id"]: (r.get("window_days") or [0])[0] for r in truth.rings}
    first_seen: dict[str, int] = {}
    for ring in truth.rings:
        rid = ring["ring_id"]
        for tid in ring_members[rid] | {o["org_id"] for o in ring["orgs"]}:
            first_seen[tid] = min(first_seen.get(tid, onsets[rid]), onsets[rid])
    matches = aligned["true_ids"].to_dict()
    summary = {}
    for policy in POLICIES:
        detected_on: dict[str, int] = {}
        found: set[str] = set()
        reviewed = tp = victim_reviews = 0
        for tick in rounds:
            day = (date.fromisoformat(tick["window"]["end"]) - start).days
            result = tick["policies"][policy]
            for entry in result["reviews"]:
                ids = set(matches.get(entry["entity_id"], []))
                detected = {tid for tid in ids if tid in first_seen and first_seen[tid] < day}
                entry["matches_started_ring"] = bool(detected)
                entry["contains_scam_victim"] = bool(ids & truth.victims)
                found.update(detected)
            result["true_positives"] = sum(e["matches_started_ring"] for e in result["reviews"])
            result["false_positives"] = len(result["reviews"]) - result["true_positives"]
            result["victim_reviews"] = sum(e["contains_scam_victim"] for e in result["reviews"])
            reviewed += len(result["reviews"])
            tp += result["true_positives"]
            victim_reviews += result["victim_reviews"]
            for rid, members in ring_members.items():
                if members and day > onsets[rid] and 2 * len(found & members) >= len(members):
                    detected_on.setdefault(rid, day)
        rings = []
        for ring in truth.rings:
            rid = ring["ring_id"]
            rings.append({
                "ring_id": rid, "typology": ring["typology"], "members": len(ring_members[rid]),
                "members_found": len(found & ring_members[rid]),
                "days_to_half_coverage": detected_on[rid] - onsets[rid] if rid in detected_on else None,
            })
        summary[policy] = {
            "reviews": reviewed, "true_positives": tp, "false_positives": reviewed - tp,
            "review_precision": round(tp / reviewed, 4) if reviewed else None,
            "victim_reviews": victim_reviews, "rings_detected": len(detected_on), "rings_total": len(rings), "rings": rings,
        }
    return summary


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Argus historical triage replay", "",
        f"Synthetic world seed {report['seed']}. Review budget {report['budget']} per round, "
        f"minimum score {report['minimum_score']:g}, {report['lookback_days']}-day activity windows.", "",
        "| Policy | Reviews | Hits | False reviews | Precision | Reviews involving victims | Rings with ≥50% coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, title in POLICIES.items():
        s = report["summary"][policy]
        precision = "—" if s["review_precision"] is None else f"{s['review_precision']:.1%}"
        lines.append(f"| {title} | {s['reviews']} | {s['true_positives']} | {s['false_positives']} | {precision} | "
                     f"{s['victim_reviews']} | {s['rings_detected']} / {s['rings_total']} |")
    lines += ["", "## Review rounds", "",
              "| Cutoff (exclusive) | Known positive / negative labels | Hybrid mode | Rules hits / reviews | Hybrid hits / reviews | Rules / hybrid over budget |",
              "|---|---:|---|---:|---:|---:|"]
    for tick in report["rounds"]:
        r, h = (tick["policies"][p] for p in POLICIES)
        lines.append(f"| {tick['window']['end']} | {tick['positive_labels']} / {tick['negative_labels']} | {tick['hybrid_mode']} | "
                     f"{r['true_positives']} / {len(r['reviews'])} | {h['true_positives']} / {len(h['reviews'])} | "
                     f"{r['over_budget']} / {h['over_budget']} |")
    lines += ["", "## Ring coverage", "",
              "| Ring | Typology | Rules members found | Hybrid members found | Rules days to half | Hybrid days to half |",
              "|---|---|---:|---:|---:|---:|"]
    for r, h in zip(*(report["summary"][p]["rings"] for p in POLICIES), strict=True):
        rd, hd = ("—" if s["days_to_half_coverage"] is None else s["days_to_half_coverage"] for s in (r, h))
        lines.append(f"| {r['ring_id']} | {r['typology']} | {r['members_found']} / {r['members']} | "
                     f"{h['members_found']} / {h['members']} | {rd} | {hd} |")
    lines += [
        "", "Each policy reviews an eligible resolved entity at most once. Candidates need activity in the window "
        "and no case opened before the cutoff. A review counts as a hit if its records link to a planted ring "
        "whose scheduled activity has started. Victim counts include reviews that also link to a ring.",
        "", "The model uses a fixed initial feature window and explicit outcomes closed before each cutoff. "
        f"The hybrid uses the rules queue until at least {MIN_CLASS_CASES} outcomes are available in each class. "
        "Identity links and reference data are held fixed from the full warehouse. The replay models event time, "
        "with no arrival-delay simulation or analyst feedback into later training.",
    ]
    return "\n".join(lines) + "\n"


def replay(root: str | Path, *, lookback: int = 30, step: int = 30, budget: int = 10, minimum: float = 45.0) -> dict[str, Any]:
    """Run the replay on a resolved world and write its JSON and Markdown reports."""
    if min(lookback, step, budget) < 1 or not 0 <= minimum <= 100:
        raise ValueError("lookback, step and budget must be positive and minimum score must be between 0 and 100")
    paths = RunPaths.at(root)
    manifest = check_manifest(paths.root)
    if not paths.db.is_file():
        raise FileNotFoundError(f"{paths.db} missing. Run argus ingest and argus resolve first.")
    with TemporaryDirectory(prefix="argus-replay-", dir=paths.root) as tmp:
        base = Path(tmp) / "replay.duckdb"
        with connect(base) as con:
            world = _copy_warehouse(con, paths.db, manifest, lookback)
            train, rounds = _review_rounds(con, base, world, lookback, step, budget, minimum)
            truth = Truth.load(paths.root)
            summary = _evaluate(rounds, truth, entity_truth(con, truth), world.start)
    report = {
        "seed": manifest["seed"], "world": world.to_dict(), "training_window": train.to_dict(),
        "lookback_days": lookback, "step_days": step, "budget": budget, "minimum_score": minimum,
        "rounds": rounds, "summary": summary,
    }
    paths.reports.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".replay-", dir=paths.reports) as tmp:
        staging = Path(tmp)
        (staging / "replay.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (staging / "replay.md").write_text(render_markdown(report), encoding="utf-8")
        for name in ("replay.json", "replay.md"):
            (staging / name).replace(paths.reports / name)
    return report


def _copy_warehouse(
    con: duckdb.DuckDBPyConnection, path: Path, manifest: dict[str, Any], lookback: int,
) -> Window:
    """Copy the source under a shared lock so the dashboard can remain open."""
    source = str(path).replace("'", "''")
    con.execute(f"ATTACH '{source}' AS source_world (READ_ONLY)")
    try:
        stored = dict(con.execute("SELECT key, value FROM source_world.manifest").fetchall())
        if stored.get("marker") != SYNTHETIC_MARKER:
            raise NonSyntheticDataError("the warehouse does not declare Argus synthetic data")
        if not con.execute("""
            SELECT count(*) FROM information_schema.tables
            WHERE table_catalog = 'source_world' AND table_name = 'identifier_entity'
            """).fetchone()[0]:
            raise ValueError("run argus resolve before replaying this warehouse")
        world = Window(date.fromisoformat(stored["window_start"]), date.fromisoformat(stored["window_end"]))
        if (stored.get("seed") != str(manifest["seed"]) or world.start.isoformat() != manifest["window"]["start"]
                or world.end.isoformat() != manifest["window"]["end"]):
            raise ValueError("warehouse and world manifest disagree. Run argus ingest and argus resolve again.")
        if lookback >= world.days:
            raise ValueError(f"lookback must be shorter than the {world.days}-day world")
        con.execute("COPY FROM DATABASE source_world TO replay")
    finally:
        con.execute("DETACH source_world")
    return world
