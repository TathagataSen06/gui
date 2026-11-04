"""Personal behavior baselines and delayed feedback for synthetic replay."""

from __future__ import annotations

from collections import Counter, deque
from datetime import date, timedelta
from math import ceil
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from ..score.features import FEATURE_DESCRIPTIONS
from ..score.snapshot import Window

# Each feature needs a material increase and corroboration from a direct rule.
SIGNALS = {
    "fan_in": (5, "R04"),
    "pass_through_amount": (5_000, "R03"),
    "sub_threshold_cash": (2, "R01"),
    "money_cycles": (1, "R02"),
    "quasi_cash_spend": (5_000, "R07"),
    "after_hours_restricted": (3, "R12"),
    "exfil_emails": (2, "R13"),
}
DIRECT_RULES = {rule_id for _, rule_id in SIGNALS.values()} | {"R05", "R11", "R16"}
MIN_HISTORY = 2
MAX_HISTORY = 4
MIN_CASES_PER_CLASS = 3


class AdaptiveTriage:
    """Learn from preceding windows and explicit outcomes as a replay advances."""

    def __init__(self, lookback: int, step: int) -> None:
        self.history: deque[tuple[Window, pd.DataFrame]] = deque(maxlen=2 + MAX_HISTORY * ceil(lookback / step))
        self.feedback_days = MAX_HISTORY * lookback
        self.last_cutoff: date | None = None
        self.previous_rules: dict[str, set[str]] = {}
        self.pending: dict[str, tuple[str, set[str]]] = {}
        self.outcomes: dict[str, tuple[date, bool, set[str]]] = {}

    def _update_feedback(self, con: duckdb.DuckDBPyConnection, cutoff: date) -> None:
        if self.last_cutoff is None:
            return
        # Freeze the rule evidence that was available before each new case opened.
        opened = con.execute(
            """
            SELECT c.case_id, re.entity_id
            FROM case_labels c JOIN record_entity re ON re.record_id = c.subject_record_id
            WHERE c.opened_date >= ? AND c.opened_date < ?
            """,
            [self.last_cutoff, cutoff],
        ).fetchall()
        for case_id, eid in opened:
            if eid in self.previous_rules:
                self.pending[case_id] = (eid, self.previous_rules[eid])
        closed = con.execute(
            """
            SELECT case_id, closed_date, disposition
            FROM case_labels
            WHERE closed_date >= ? AND closed_date < ? AND closed_date >= opened_date
            ORDER BY closed_date, case_id
            """,
            [self.last_cutoff, cutoff],
        ).fetchall()
        for case_id, closed_on, disposition in closed:
            captured = self.pending.pop(case_id, None)
            if captured and disposition in {"CONFIRMED_SUSPICIOUS", "CLOSED_NO_ACTION", "CLOSED_VICTIM"}:
                eid, rules = captured
                self.outcomes[eid] = (closed_on, disposition == "CONFIRMED_SUSPICIOUS", rules)
        earliest = cutoff - timedelta(days=self.feedback_days)
        self.outcomes = {eid: outcome for eid, outcome in self.outcomes.items() if outcome[0] >= earliest}

    def _rule_adjustments(self) -> tuple[dict[str, float], int, int]:
        positives = sum(confirmed for _, confirmed, _ in self.outcomes.values())
        negatives = len(self.outcomes) - positives
        if min(positives, negatives) < MIN_CASES_PER_CLASS:
            return {}, positives, negatives
        total_hits: Counter[str] = Counter()
        confirmed_hits: Counter[str] = Counter()
        for _, confirmed, rules in self.outcomes.values():
            total_hits.update(rules)
            if confirmed:
                confirmed_hits.update(rules)
        base_rate = positives / len(self.outcomes)
        adjustments = {}
        for rid, count in sorted(total_hits.items()):
            if count >= MIN_CASES_PER_CLASS:
                # Shrink small samples toward the overall reviewed-case hit rate.
                smoothed = (confirmed_hits[rid] + 6 * base_rate) / (count + 6)
                adjustments[rid] = round(10 * (smoothed - base_rate), 2)
        return adjustments, positives, negatives

    def _baseline(self, window: Window) -> list[tuple[Window, pd.DataFrame]]:
        """Pick recent, disjoint windows that end before current activity starts."""
        compatible = []
        before = window.start
        for old_window, old in reversed(self.history):
            if old_window.end <= before:
                compatible.append((old_window, old))
                before = old_window.start
            if len(compatible) == MAX_HISTORY:
                break
        return compatible

    def score(
        self, con: duckdb.DuckDBPyConnection, window: Window, X: pd.DataFrame,
        eligible: set[str], rules: pd.Series, hits: pd.Series, *, complete: bool = True,
    ) -> tuple[pd.Series, dict[str, list[dict[str, Any]]], dict[str, Any]]:
        self._update_feedback(con, window.end)
        adjustments, positives, negatives = self._rule_adjustments()
        active = X.loc[X.index.isin(eligible), list(SIGNALS)].rename_axis("entity_id")
        active_rules = {eid: {h["rule_id"] for h in hits[eid]} & DIRECT_RULES for eid in active.index}
        feedback = pd.Series({eid: sum(adjustments.get(rid, 0) for rid in sorted(ids)) for eid, ids in active_rules.items()}, dtype=float).clip(-8, 8)
        compatible = self._baseline(window) if complete else []
        changes: dict[str, list[dict[str, Any]]] = {}
        profiled = 0
        if len(compatible) >= MIN_HISTORY:
            past = pd.concat([frame for _, frame in compatible], keys=range(len(compatible)))
            grouped = past.groupby(level="entity_id")
            median = grouped.median()
            counts = grouped.size().reindex(active.index, fill_value=0)
            deviation = past.sub(median, axis="index", level="entity_id").abs().groupby(level="entity_id").median()
            typical = median.reindex(active.index)
            mad = deviation.reindex(active.index)
            profiled = int((counts >= MIN_HISTORY).sum())
            for feature, (floor, rid) in SIGNALS.items():
                required = np.maximum(floor, np.maximum(0.75 * typical[feature], 3 * 1.4826 * mad[feature]))
                unusual = active.index[(counts >= MIN_HISTORY) & (active[feature] - typical[feature] >= required)]
                for eid in unusual:
                    if rid in active_rules[eid]:
                        changes.setdefault(eid, []).append({
                            "feature": feature, "label": FEATURE_DESCRIPTIONS[feature],
                            "observed": round(float(active.at[eid, feature]), 2),
                            "typical": round(float(typical.at[eid, feature]), 2),
                            "minimum_increase": round(float(required[eid]), 2),
                            "history_windows": int(counts[eid]), "supporting_rule": rid,
                        })
        bonus = pd.Series({eid: min(12, 6 * len(evidence)) for eid, evidence in changes.items()}, dtype=float)
        scores = (rules + bonus.reindex(rules.index, fill_value=0) + feedback.reindex(rules.index, fill_value=0)).clip(0, 100).round(2)
        if complete:
            self.history.append((window, active))
        self.previous_rules = active_rules if complete else {}
        self.last_cutoff = window.end
        return scores, changes, {
            "baseline_windows": [w.to_dict() for w, _ in reversed(compatible)],
            "profiled_entities": profiled, "entities_with_changes": len(changes),
            "positive_feedback": positives, "negative_feedback": negatives,
            "rule_adjustments": adjustments,
        }
