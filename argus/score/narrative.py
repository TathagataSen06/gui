"""Analyst-style alert narratives assembled from rule hits and model drivers."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .features import FEATURE_DESCRIPTIONS


def _fmt_value(feature: str, v: float) -> str:
    if any(k in feature for k in ("total", "volume", "amount", "spend", "inflow")):
        return f"${v:,.0f}"
    if "ratio" in feature or "share" in feature:
        return f"{v:.0%}"
    if feature in ("betweenness", "pagerank", "community_score", "strength"):
        return f"{v:.3g}"
    return f"{v:,.0f}"


def top_drivers(contrib: pd.Series, raw: pd.Series, k: int = 5) -> list[dict[str, Any]]:
    """Largest positive log-odds contributions with readable labels."""
    pos = contrib[contrib > 0.05].sort_values(ascending=False).head(k)
    return [
        {"feature": f, "label": FEATURE_DESCRIPTIONS.get(f, f), "value": float(raw[f]), "display_value": _fmt_value(f, float(raw[f])), "contribution": round(float(c), 3)}
        for f, c in pos.items()
    ]


def recommendation(risk: float, n_rules: int) -> str:
    if risk >= 70 or n_rules >= 4:
        return "Escalate to investigation"
    if risk >= 45:
        return "Analyst review"
    return "Monitor"


def compose(entity: pd.Series, hits: list[dict[str, Any]], drivers: list[dict[str, Any]], counterparties: list[tuple[str, float]]) -> str:
    """One-paragraph alert narrative (the kind written at the top of a case file)."""
    src = entity["sources"]
    sources = ", ".join(str(s) for s in src) if hasattr(src, "__iter__") and not isinstance(src, str) else str(src)
    kind = "organisation" if entity["entity_type"] == "org" else "individual"
    parts = [
        f"{entity['display_name']} ({kind}; resolved from {int(entity['n_records'])} record(s) across {sources}) "
        f"scores {entity['risk_score']:.0f}/100 (rules {entity['rule_score']:.0f}, model {entity['ml_score']:.0%}"
        + ("; rule floor applied - the model has little history for this pattern" if entity.get("risk_driver") == "rule floor" else "")
        + ")."
    ]
    if hits:
        ind = "; ".join(f"({i + 1}) {h['title']}: {h['reason']}" for i, h in enumerate(hits))
        parts.append(f"Indicators: {ind}.")
    else:
        parts.append("No rule-based indicators fired; the score is driven by the model.")
    if drivers:
        parts.append("Strongest model drivers: " + ", ".join(f"{d['label'].lower()} ({d['display_value']})" for d in drivers[:3]) + ".")
    if counterparties:
        parts.append("Largest money counterparties: " + ", ".join(f"{n} (${v:,.0f})" for n, v in counterparties[:3]) + ".")
    parts.append(f"Recommended action: {recommendation(float(entity['risk_score']), len(hits))}.")
    return " ".join(parts)
