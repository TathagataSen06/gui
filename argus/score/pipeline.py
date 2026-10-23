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
from .features import FeatureConfig, labels, transfers
from .model import ModelConfig, fit_and_score, model_bundle, score_with_pretrained
from .narrative import compose, top_drivers
from .rules import apply_rules
from .snapshot import features_with_contacts, scoring_window, snapshot_features, training_plan


@dataclass
class ScoreConfig:
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    rule_weight: float = 0.4  # blend = 100 * (rule_weight * rules/100 + (1 - rule_weight) * model probability)
    rule_floor: float = 0.8  # risk = max(blend, rule_floor * rules): strong scenario evidence is never diluted
    # by a model that has seen few (or no) examples of that typology
    first_pass_flag: float = 40.0  # rule score that marks an entity as "high-risk" for the 2-pass contact feature
    alert_threshold: float = 45.0
    # Point-in-time: train on features from the events before the first case was opened, score on the
    # latest window of the same length (see argus/score/snapshot.py). Off: every feature uses the whole window.
    point_in_time: bool = False


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


def _db_path(con: duckdb.DuckDBPyConnection) -> Path:
    """File behind ``con``; point-in-time snapshots copy it."""
    row = con.execute("SELECT path FROM duckdb_databases() WHERE database_name = current_database()").fetchone()
    if not row or not row[0]:
        raise ValueError("point-in-time scoring needs a file-backed warehouse")
    return Path(row[0])


def run_scoring(con: duckdb.DuckDBPyConnection, cfg: ScoreConfig | None = None, model_path: Path | None = None, pretrained: Path | None = None) -> dict[str, Any]:
    """Score every entity. With ``pretrained`` the ML model is loaded instead of trained
    (needed for a world without confirmed cases, e.g. the clean control)."""
    cfg = cfg or ScoreConfig()
    t0 = time.perf_counter()
    bundle = joblib.load(pretrained) if pretrained is not None else None
    pit = (bundle or {}).get("point_in_time") if bundle is not None else ({} if cfg.point_in_time else None)
    windows: dict[str, Any] = {}
    labels_as_of = None
    if pit is None:
        X = features_with_contacts(con, cfg.features, cfg.first_pass_flag)
    elif bundle is not None:  # a point-in-time model scores the latest window of the length it was trained on
        w_now = scoring_window(con, int(pit["window_days"]))
        X, _ = snapshot_features(con, _db_path(con), w_now, cfg.features, cfg.first_pass_flag)
        windows = {"score": w_now.to_dict()}
    else:
        w_train, w_now = training_plan(con)
        X_train, groups_train = snapshot_features(con, _db_path(con), w_train, cfg.features, cfg.first_pass_flag)
        X, _ = snapshot_features(con, _db_path(con), w_now, cfg.features, cfg.first_pass_flag)
        labels_as_of = w_now.end  # only outcomes decided by the scoring date
        windows = {"train": w_train.to_dict(), "score": w_now.to_dict(), "labels_as_of": labels_as_of.isoformat()}
    rule, hits = apply_rules(X)
    y, labelable = labels(con, X.index, as_of=labels_as_of)
    if bundle is not None:
        res = score_with_pretrained(X, bundle)
    else:
        if int(y[labelable].sum()) < 2:
            raise ValueError("fewer than 2 confirmed cases known: train on another world and pass pretrained=<model.joblib>")
        if labels_as_of is not None:
            windows["labelled_positives"] = int(y[labelable].sum())
        if pit is None:
            groups = pd.Series(dict(con.execute("SELECT node_id, community_id FROM graph_nodes").fetchall())).reindex(X.index)
            res = fit_and_score(X, y, labelable, groups, cfg.model, rule_score=rule)
            X_ref = X
        else:
            res = fit_and_score(X_train, y, labelable, groups_train, cfg.model, rule_score=rule, X_eval=X)
            X_ref = X_train
        if model_path is not None:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            pit_info = {"window_days": windows["train"]["days"], **windows} if pit is not None else None
            joblib.dump(model_bundle(res, X_ref, labelable, point_in_time=pit_info), model_path)

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
        "point_in_time": windows or None,
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
