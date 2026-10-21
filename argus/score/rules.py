"""Explainable rule-based risk scorer.

Each rule mirrors a red-flag indicator an AML or insider-risk analyst would
recognise, awards points when it fires, and renders a human-readable reason
from the entity's feature values. The score is the capped sum of points, so
any score can be decomposed exactly into the rules that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


def _usd(x: float) -> str:
    return f"${x:,.0f}"


@dataclass(frozen=True)
class Rule:
    rule_id: str
    title: str
    points: int
    fires: Callable[[pd.Series], bool]
    reason: Callable[[pd.Series], str]
    typology: str


RULES: tuple[Rule, ...] = (
    Rule(
        "R01", "Structuring below the $10k reporting threshold", 30,
        lambda f: f["sub_threshold_cash"] >= 3,
        lambda f: f"{int(f['sub_threshold_cash'])} cash deposits between $8,000 and $9,999 across {int(f['sub_threshold_branches'])} branch(es); total cash {_usd(f['cash_deposit_total'])}",
        "structuring",
    ),
    Rule(
        "R02", "Circular payment path (round-tripping)", 25,
        lambda f: f["money_cycles"] >= 1,
        lambda f: f"Part of {int(f['money_cycles'])} value-preserving payment cycle(s) moving {_usd(f['cycle_volume'])}",
        "layering",
    ),
    Rule(
        "R03", "Rapid pass-through of funds", 20,
        lambda f: f["pass_through_ratio"] >= 0.7 and f["pass_through_amount"] >= 5_000,
        lambda f: f"{f['pass_through_ratio']:.0%} of non-payroll inflow ({_usd(f['pass_through_amount'])}) forwarded within 72h",
        "mule",
    ),
    Rule(
        "R04", "Funds received from many unrelated payers", 10,
        lambda f: f["fan_in"] >= 8 and f["inflow_total"] >= 10_000,
        lambda f: f"{int(f['fan_in'])} distinct non-payroll payers, total inflow {_usd(f['inflow_total'])}",
        "mule",
    ),
    Rule(
        "R05", "Repeated large round-amount transfers", 10,
        lambda f: f["large_round_transfers"] >= 3,
        lambda f: f"{int(f['large_round_transfers'])} transfers of $5,000+ in exact multiples of $250",
        "layering",
    ),
    Rule(
        "R06", "Controls shell-like companies", 15,
        lambda f: f["controls_red_flag_orgs"] >= 1,
        lambda f: f"Linked to {int(f['controls_red_flag_orgs'])} company(ies) incorporated <12 months ago or sharing a registered-agent address",
        "layering",
    ),
    Rule(
        "R07", "Quasi-cash / crypto off-ramp", 10,
        lambda f: f["quasi_cash_spend"] >= 5_000,
        lambda f: f"{_usd(f['quasi_cash_spend'])} spent at crypto / quasi-cash merchants",
        "mule",
    ),
    Rule(
        "R08", "Internal watch-list match", 20,
        lambda f: f["watchlist_hit"] >= 1,
        lambda f: "Resolved to an entry on the internal hot-list",
        "watchlist",
    ),
    Rule(
        "R09", "Frequent contact with high-risk entities", 10,
        lambda f: f["calls_with_watchlisted"] + f["calls_with_flagged"] >= 6,
        lambda f: f"{int(f['calls_with_watchlisted'] + f['calls_with_flagged'])} calls/SMS with watch-listed or high-risk entities",
        "network",
    ),
    Rule(
        "R10", "Contact with unregistered prepaid lines", 10,
        lambda f: f["burner_contacts"] >= 5,
        lambda f: f"{int(f['burner_contacts'])} calls/SMS with prepaid lines that have no verified identity",
        "network",
    ),
    Rule(
        "R11", "Communications coordinated with money movement", 15,
        lambda f: f["pre_txn_coordination"] + f["pre_cash_contacts"] >= 2,
        lambda f: f"{int(f['pre_txn_coordination'])} large transfers and {int(f['pre_cash_contacts'])} large cash deposits preceded within 4h by a call between the parties / with a prepaid line",
        "network",
    ),
    Rule(
        "R12", "After-hours access to restricted areas", 15,
        lambda f: f["after_hours_restricted"] >= 3,
        lambda f: f"{int(f['after_hours_restricted'])} after-hours entries to restricted zones, {int(f['access_denied'])} denied swipes",
        "insider",
    ),
    Rule(
        "R13", "Large attachments sent outside the organisation", 15,
        lambda f: f["exfil_emails"] >= 2,
        lambda f: f"{int(f['exfil_emails'])} e-mails with 5 MB+ attachments to other domains",
        "insider",
    ),
    Rule(
        "R14", "After-hours small-group meetings", 5,
        lambda f: f["after_hours_coloc"] >= 2,
        lambda f: f"{int(f['after_hours_coloc'])} after-hours co-locations with small groups",
        "network",
    ),
    Rule(
        "R15", "Member of a flagged money-flow community", 5,
        lambda f: f["community_flagged"] >= 1,
        lambda f: f"Community money-flow anomaly score {f['community_score']:.1f}",
        "network",
    ),
    Rule(
        "R16", "Young account with significant inflow", 10,
        lambda f: f["young_account_inflow"] >= 10_000,
        lambda f: f"{_usd(f['young_account_inflow'])} received into accounts opened within 90 days of the review window",
        "mule",
    ),
)


def apply_rules(X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return (score 0-100, list of fired-rule dicts) per entity."""
    scores, hits = [], []
    for _, f in X.iterrows():
        fired = []
        for r in RULES:
            if r.fires(f):
                fired.append({"rule_id": r.rule_id, "title": r.title, "points": r.points, "reason": r.reason(f), "typology": r.typology})
        scores.append(min(100, sum(h["points"] for h in fired)))
        hits.append(fired)
    return pd.Series(scores, index=X.index, dtype=float), pd.Series(hits, index=X.index, dtype=object)
