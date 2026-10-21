"""Scoring stage: features -> rules (2-pass) -> ML -> blended, explained risk scores."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import joblib
import pandas as pd

from ..db import connect
from ..paths import RunPaths
from .features import FeatureConfig, build_features, calls_with_flagged, labels, transfers
from .model import ModelConfig, fit_and_score, model_bundle, score_with_pretrained
from .narrative import compose, top_drivers
from .rules import apply_rules


@dataclass
class ScoreConfig:
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    rule_weight: float = 0.4  # blend = 100 * (rule_weight * rules/100 + (1 - rule_weight) * model probability)
    rule_floor: float = 0.8  # risk = max(blend, rule_floor * rules): strong scenario evidence is never diluted
    # by a model that has seen few (or no) examples of that typology
    first_pass_flag: float = 40.0  # rule score that marks an entity as "high-risk" for the 2-pass contact feature
    alert_threshold: float = 45.0


def top_counterparties(con: duckdb.DuckDBPyConnection, names: dict[str, str], k: int = 3) -> dict[str, list[tuple[str, float]]]:
    tx = transfers(con)
    tx = tx[tx["src"] != tx["dst"]]
    names = dict(names)
    for i, n in zip(tx["src"], tx["payer_name"]):
        if i.startswith("ext:") and n:
            names.setdefault(i, n.title())
    for i, n in zip(tx["dst"], tx["payee_name"]):
        if i.startswith("ext:") and n:
            names.setdefault(i, n.title())
    both = pd.concat([tx[["src", "dst", "amount"]].rename(columns={"src": "e", "dst": "o"}), tx[["dst", "src", "amount"]].rename(columns={"dst": "e", "src": "o"})])
    agg = both.groupby(["e", "o"])["amount"].sum().reset_index().sort_values(["e", "amount"], ascending=[True, False])
    out: dict[str, list[tuple[str, float]]] = {}
    for e, g in agg.groupby("e"):
        out[e] = [(names.get(o, o.split(":", 1)[-1]), float(a)) for o, a in zip(g["o"].head(k), g["amount"].head(k))]
    return out


def run_scoring(con: duckdb.DuckDBPyConnection, cfg: ScoreConfig | None = None, model_path: Path | None = None, pretrained: Path | None = None) -> dict[str, Any]:
    """Score every entity. With ``pretrained`` the ML model is loaded instead of trained
    (needed for a world without confirmed cases, e.g. the clean control)."""
    cfg = cfg or ScoreConfig()
    t0 = time.perf_counter()
    X = build_features(con, cfg.features)
    rule1, _ = apply_rules(X)
    flagged = set(rule1[rule1 >= cfg.first_pass_flag].index) | set(X.index[X["watchlist_hit"] > 0])
    X["calls_with_flagged"] = calls_with_flagged(con, flagged).reindex(X.index).fillna(0.0)
    rule, hits = apply_rules(X)
    y, labelable = labels(con, X.index)
    groups = pd.Series(dict(con.execute("SELECT node_id, community_id FROM graph_nodes").fetchall())).reindex(X.index)
    if pretrained is not None:
        res = score_with_pretrained(X, joblib.load(pretrained))
    else:
        if int(y[labelable].sum()) < 2:
            raise ValueError("fewer than 2 confirmed cases: train on another world and pass pretrained=<model.joblib>")
        res = fit_and_score(X, y, labelable, groups, cfg.model, rule_score=rule)
        if model_path is not None:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model_bundle(res, X, labelable), model_path)

    ents = con.execute("SELECT entity_id, entity_type, display_name, n_records, sources FROM entities").df().set_index("entity_id").reindex(X.index)
    blend = 100 * (cfg.rule_weight * rule / 100 + (1 - cfg.rule_weight) * res.prob)
    floor = cfg.rule_floor * rule
    risk = blend.where(blend >= floor, floor).round(2)
    driver = pd.Series(["model+rules"] * len(risk), index=risk.index).where(blend >= floor, "rule floor")
    names = ents["display_name"].to_dict()
    cps = top_counterparties(con, names)
    rows = []
    for eid in X.index:
        drivers = top_drivers(res.contributions.loc[eid], X.loc[eid])
        row = ents.loc[eid].to_dict()
        row["sources"] = [str(s) for s in row["sources"]]
        row.update(
            entity_id=eid,
            rule_score=float(rule[eid]),
            ml_score=round(float(res.prob[eid]), 4),
            risk_score=float(risk[eid]),
            risk_driver=driver[eid],
            labelled=bool(labelable[eid]),
            rule_hits=hits[eid],
            ml_drivers=drivers,
        )
        row["narrative"] = compose(pd.Series(row), hits[eid], drivers, cps.get(eid, []))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["risk_score", "entity_id"], ascending=[False, True]).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    out["alert"] = out["risk_score"] >= cfg.alert_threshold
    top_reason = []
    for h, d in zip(out["rule_hits"], out["ml_drivers"]):
        if h:
            top_reason.append(max(h, key=lambda x: x["points"])["title"])
        elif d:
            top_reason.append(d[0]["label"])
        else:
            top_reason.append("")
    out["top_reason"] = top_reason
    persist = out.copy()
    persist["rule_hits"] = persist["rule_hits"].map(json.dumps)
    persist["ml_drivers"] = persist["ml_drivers"].map(json.dumps)
    persist["sources"] = persist["sources"].map(lambda s: [str(x) for x in s])

    def put(name: str, df: pd.DataFrame) -> None:
        con.register("_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _df")
        con.unregister("_df")

    cols = ["rank", "entity_id", "entity_type", "display_name", "risk_score", "rule_score", "ml_score", "risk_driver", "alert", "labelled", "top_reason", "n_records", "sources", "rule_hits", "ml_drivers", "narrative"]
    put("risk_scores", persist[cols])
    feat = X.reset_index().rename(columns={"index": "entity_id"})
    put("entity_features", feat)
    if res.importances is not None:
        put("model_importances", res.importances)
    contrib = res.contributions.reset_index().rename(columns={"index": "entity_id"})
    put("model_contributions", contrib)
    stats = {
        "entities": int(len(out)),
        "alerts": int(out["alert"].sum()),
        "metrics": res.metrics,
        "config": asdict(cfg),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    con.execute("CREATE TABLE IF NOT EXISTS run_meta (stage VARCHAR, payload JSON)")
    con.execute("DELETE FROM run_meta WHERE stage = 'score'")
    con.execute("INSERT INTO run_meta VALUES ('score', ?)", [json.dumps(stats, default=str)])
    return stats


def run(root: str | Path, cfg: ScoreConfig | None = None, pretrained: str | Path | None = None) -> dict[str, Any]:
    paths = RunPaths.at(root)
    con = connect(paths.db)
    try:
        return run_scoring(con, cfg, model_path=paths.model, pretrained=Path(pretrained) if pretrained else None)
    finally:
        con.close()
