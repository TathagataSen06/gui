"""Paired replay audit for a known gap in one synthetic activity feed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import duckdb
import pandas as pd

from ..score.snapshot import Window, utc_bounds

OUTAGE_SOURCES = ("bank", "telco", "email", "access")
STATUS_LABELS = {
    "selected": "Still selected",
    "already_reviewed": "Reviewed in an earlier round",
    "no_activity": "No eligible activity observed",
    "below_threshold": "Below threshold",
    "over_budget": "Outside review budget",
}


@dataclass(frozen=True)
class SourceOutage:
    source: str
    window: Window

    def overlaps(self, window: Window) -> bool:
        return self.window.start < window.end and window.start < self.window.end

    def validate(self, world: Window) -> None:
        if self.source not in OUTAGE_SOURCES:
            raise ValueError(f"outage source must be one of {', '.join(OUTAGE_SOURCES)}")
        if not world.start <= self.window.start < self.window.end <= world.end:
            raise ValueError("outage dates must define a nonempty interval within the world")


def remove_events(con: duckdb.DuckDBPyConnection, outage: SourceOutage) -> int:
    """Hide events and their actor links in a private warehouse copy."""
    params = [outage.source, *utc_bounds(con, outage.window)]
    match = "source = ? AND ts >= ? AND ts < ?"
    count = con.execute(f"SELECT count(*) FROM events WHERE {match}", params).fetchone()[0]
    if not count:
        raise ValueError(f"no {outage.source} events in the requested outage interval")
    con.execute(f"DELETE FROM event_actors WHERE event_id IN (SELECT event_id FROM events WHERE {match})", params)
    con.execute(f"DELETE FROM events WHERE {match}", params)
    return int(count)


def audit_decisions(
    watched: list[str], queue: pd.Series, scores: pd.Series, hits: pd.Series,
    eligible: set[str], reviewed: set[str], budget: int,
) -> dict[str, dict[str, Any]]:
    """Account for baseline reviews before the outage run updates its own review history."""
    ranks = {eid: rank for rank, eid in enumerate(queue.index, 1)}
    out = {}
    for eid in watched:
        rank = ranks.get(eid)
        if rank is not None:
            status = "selected" if rank <= budget else "over_budget"
        elif eid in reviewed:
            status = "already_reviewed"
        elif eid not in eligible:
            status = "no_activity"
        else:
            status = "below_threshold"
        out[eid] = {
            "status": status,
            "score": float(scores[eid]) if eid in eligible else None,
            "rank": rank,
            "rules": sorted(h["rule_id"] for h in hits[eid]),
        }
    return out


def compare_rounds(
    policy: str, baseline: list[dict[str, Any]], shadow: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for full, partial in zip(baseline, shadow, strict=True):
        window = full["window"]
        before = full["policies"][policy]["reviews"]
        after = partial["policies"][policy]["reviews"]
        decisions = []
        for review in before:
            eid = review["entity_id"]
            altered = partial["audited"][eid]
            full_rules = {h["rule_id"] for h in review["rule_hits"]}
            remaining = set(altered["rules"])
            decisions.append({
                "entity_id": eid,
                "full_score": review["score"],
                "outage_score": altered["score"],
                "outage_rank": altered["rank"],
                "status": altered["status"],
                "rules_lost": sorted(full_rules - remaining),
                "rules_gained": sorted(remaining - full_rules),
            })
        seen = {r["entity_id"] for r in before}
        result.append({
            "window": window,
            "overlaps_outage": partial["outage_overlap"],
            "baseline_reviews": len(before),
            "outage_reviews": len(after),
            "retained": sum(d["status"] == "selected" for d in decisions),
            "new_reviews": [{"entity_id": r["entity_id"], "score": r["score"]} for r in after if r["entity_id"] not in seen],
            "decisions": decisions,
        })
    return result


def render_outage(audit: dict[str, Any]) -> list[str]:
    w = audit["window"]
    full, shadow = audit["full_feed"], audit["with_outage"]
    lines = [
        "", "## Feed outage audit", "",
        f"Withheld {audit['removed_events']:,} {audit['source']} events from {w['start']} to {w['end']} (end excluded). "
        f"Compared the {audit['policy']} policy at the same cutoff, threshold and review budget.", "",
        "| Feed condition | Reviews | Hits | False reviews | Rings with ≥50% coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for title, s in (("Full feed", full), ("With outage", shadow)):
        lines.append(f"| {title} | {s['reviews']} | {s['true_positives']} | {s['false_positives']} | {s['rings_detected']} / {s['rings_total']} |")
    lines += ["", "| Cutoff | Window overlaps outage | Full-feed reviews | Outage reviews | Kept at same cutoff | New at same cutoff |",
              "|---|---|---:|---:|---:|---:|"]
    for r in audit["rounds"]:
        overlap = "Yes" if r["overlaps_outage"] else "No"
        lines.append(f"| {r['window']['end']} | {overlap} | {r['baseline_reviews']} | {r['outage_reviews']} | "
                     f"{r['retained']} | {len(r['new_reviews'])} |")
    lines += ["", "The audit rebuilds features and graphs from the remaining activity. It holds resolved identities, "
              "reference records and case outcomes fixed. Adaptive replay pauses personal changes and skips collecting "
              "baseline and pre-case evidence from affected windows. Later rounds can still differ because the earlier "
              "review history and baseline inputs differ."]
    return lines
