"""Chronological, capacity-limited triage on an Argus-generated world."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
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
from .adaptive import AdaptiveTriage
from .outage import SourceOutage, audit_decisions, compare_rounds, remove_events, render_outage
from .truth import Truth, entity_truth

MIN_CLASS_CASES = 3
POLICIES = {"rules": "Rules only", "hybrid": "Model and rules", "adaptive": "Adaptive rules"}


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
    adaptive: bool, *, outage: SourceOutage | None = None, watched: dict[str, list[str]] | None = None,
) -> tuple[Window, list[dict[str, Any]]]:
    cfg = ScoreConfig()
    train = Window(world.start, world.start + timedelta(days=lookback))
    offsets = [*range(lookback, world.days, step), world.days]
    learner = AdaptiveTriage(lookback, step) if adaptive else None
    audit_policy = "adaptive" if adaptive else "rules"
    reviewed: dict[str, set[str]] = defaultdict(set)
    rounds = []
    train_features = None
    for offset in offsets:
        cutoff = world.start + timedelta(days=offset)
        window = Window(cutoff - timedelta(days=lookback), cutoff)
        affected = outage is not None and outage.overlaps(window)
        with snapshot_warehouse(con, db, window, sealed=True) as snapshot:
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
        scores_by_policy = {"rules": rule, "hybrid": hybrid}
        changes = {}
        adaptive_meta = {}
        if learner is not None:
            scores_by_policy["adaptive"], changes, adaptive_meta = learner.score(
                con, window, X, eligible, rule, hits, complete=not affected,
            )
        policies = {}
        audited = {}
        for policy, scores in scores_by_policy.items():
            queue = _queue(scores, eligible, reviewed[policy], minimum)
            if watched is not None and policy == audit_policy:
                audited = audit_decisions(watched[cutoff.isoformat()], queue, scores, hits, eligible, reviewed[policy], budget)
            entries = []
            for eid, score in queue.head(budget).items():
                drivers = []
                if policy == "hybrid" and contributions is not None and eid in contributions.index:
                    drivers = top_drivers(contributions.loc[eid], X.loc[eid], k=3)
                entry = {
                    "entity_id": eid, "score": float(score), "rule_score": float(rule[eid]),
                    "model_probability": float(prob[eid]) if policy == "hybrid" and prob is not None else None,
                    "rule_hits": hits[eid], "model_drivers": drivers,
                }
                if policy == "adaptive":
                    entry["personal_changes"] = changes.get(eid, [])
                entries.append(entry)
                reviewed[policy].add(eid)
            policies[policy] = {"candidates": len(queue), "over_budget": max(0, len(queue) - budget), "reviews": entries}
        tick = {
            "window": window.to_dict(), "eligible": len(eligible), "positive_labels": positives,
            "negative_labels": negatives, "hybrid_mode": "logreg" if contributions is not None else "rules fallback",
            "policies": policies,
        }
        if learner is not None:
            tick["adaptive"] = adaptive_meta
        if outage is not None:
            tick["outage_overlap"] = affected
            tick["audited"] = audited
        rounds.append(tick)
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
    for policy in rounds[0]["policies"]:
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
    policies = list(report["summary"])
    lines = [
        "# Argus historical triage replay", "",
        f"Synthetic world seed {report['seed']}. Review budget {report['budget']} per round, "
        f"minimum score {report['minimum_score']:g}, {report['lookback_days']}-day activity windows.", "",
        "| Policy | Reviews | Hits | False reviews | Precision | Reviews involving victims | Rings with ≥50% coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in policies:
        s = report["summary"][policy]
        precision = "—" if s["review_precision"] is None else f"{s['review_precision']:.1%}"
        lines.append(f"| {POLICIES[policy]} | {s['reviews']} | {s['true_positives']} | {s['false_positives']} | {precision} | "
                     f"{s['victim_reviews']} | {s['rings_detected']} / {s['rings_total']} |")
    lines += ["", "## Review rounds", "", "| Cutoff (exclusive) | Known positive / negative labels | Hybrid mode | "
              + " | ".join(f"{POLICIES[p]} hits / reviews / overflow" for p in policies) + " |",
              "|---|---:|---|" + "---:|" * len(policies)]
    for tick in report["rounds"]:
        cells = [f"{r['true_positives']} / {len(r['reviews'])} / {r['over_budget']}" for r in tick["policies"].values()]
        lines.append(f"| {tick['window']['end']} | {tick['positive_labels']} / {tick['negative_labels']} | {tick['hybrid_mode']} | "
                     + " | ".join(cells) + " |")
    lines += ["", "## Ring coverage", "",
              "| Ring | Typology | " + " | ".join(f"{POLICIES[p]} members found (days to half)" for p in policies) + " |",
              "|---|---|" + "---:|" * len(policies)]
    for rows in zip(*(report["summary"][p]["rings"] for p in policies), strict=True):
        r = rows[0]
        cells = [f"{s['members_found']} / {s['members']} ({s['days_to_half_coverage'] or '—'})" for s in rows]
        lines.append(f"| {r['ring_id']} | {r['typology']} | " + " | ".join(cells) + " |")
    if "adaptive" in policies:
        lines += ["", "## Adaptation by cutoff", "",
                  "| Cutoff (exclusive) | Entities with enough history | Entities with a supported change | Confirmed / other feedback | Rules adjusted |",
                  "|---|---:|---:|---:|---:|"]
        for tick in report["rounds"]:
            a = tick["adaptive"]
            lines.append(f"| {tick['window']['end']} | {a['profiled_entities']} | {a['entities_with_changes']} | "
                         f"{a['positive_feedback']} / {a['negative_feedback']} | {len(a['rule_adjustments'])} |")
        lines += ["", "Adaptive rules compare recent activity to each entity's own earlier windows. Direct rule evidence "
                  "is required to boost a change. Case feedback updates rule weights only after an explicit disposition "
                  "is available. The report's JSON includes the baseline values and rule changes."]
    if audit := report.get("outage"):
        lines += render_outage(audit)
    lines += [
        "", "Each policy reviews an eligible resolved entity at most once. Candidates need activity in the window "
        "and no case opened before the cutoff. A review counts as a hit if its records link to a planted ring "
        "whose scheduled activity has started. Victim counts include reviews that also link to a ring.",
        "", "The model uses a fixed initial feature window and explicit outcomes closed before each cutoff. "
        f"The hybrid uses the rules queue until at least {MIN_CLASS_CASES} outcomes are available in each class. "
        "Identity links and reference data are held fixed from the full warehouse. The replay uses event timestamps "
        "and the world's historical case outcomes. The simulated review decisions do not generate new case outcomes.",
    ]
    return "\n".join(lines) + "\n"


def replay(
    root: str | Path, *, lookback: int = 30, step: int = 30, budget: int = 10, minimum: float = 45.0,
    adaptive: bool = False, outage: SourceOutage | None = None,
) -> dict[str, Any]:
    """Run the replay on a resolved world and write its JSON and Markdown reports."""
    if min(lookback, step, budget) < 1 or not 0 <= minimum <= 100:
        raise ValueError("lookback, step and budget must be positive and minimum score must be between 0 and 100")
    paths = RunPaths.at(root)
    manifest = check_manifest(paths.root)
    if not paths.db.is_file():
        raise FileNotFoundError(f"{paths.db} missing. Run argus ingest and argus resolve first.")
    audit = None
    with TemporaryDirectory(prefix="argus-replay-") as tmp:
        base = Path(tmp) / "replay.duckdb"
        with connect(base) as con:
            world = _copy_warehouse(con, paths.db, manifest, lookback)
            con.execute("CHECKPOINT replay")
        if outage is not None:
            outage.validate(world)
            shadow_db = Path(tmp) / "outage.duckdb"
            shutil.copyfile(base, shadow_db)
            with connect(shadow_db) as con:
                removed = remove_events(con, outage)
                con.execute("CHECKPOINT outage")
        # Close the writer before copying this immutable base into each window.
        with connect(base, read_only=True) as con:
            train, rounds = _review_rounds(con, base, world, lookback, step, budget, minimum, adaptive)
            truth = Truth.load(paths.root)
            aligned = entity_truth(con, truth)
            summary = _evaluate(rounds, truth, aligned, world.start)
        if outage is not None:
            policy = "adaptive" if adaptive else "rules"
            watched = {
                tick["window"]["end"]: [r["entity_id"] for r in tick["policies"][policy]["reviews"]] for tick in rounds
            }
            with connect(shadow_db, read_only=True) as con:
                _, shadow = _review_rounds(con, shadow_db, world, lookback, step, budget, minimum, adaptive,
                                           outage=outage, watched=watched)
            shadow_summary = _evaluate(shadow, truth, aligned, world.start)
            audit = {
                "source": outage.source, "window": outage.window.to_dict(), "removed_events": removed, "policy": policy,
                "full_feed": summary[policy], "with_outage": shadow_summary[policy],
                "rounds": compare_rounds(policy, rounds, shadow),
            }
    report = {
        "seed": manifest["seed"], "world": world.to_dict(), "training_window": train.to_dict(),
        "lookback_days": lookback, "step_days": step, "budget": budget, "minimum_score": minimum,
        "rounds": rounds, "summary": summary,
    }
    if audit is not None:
        report["outage"] = audit
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
