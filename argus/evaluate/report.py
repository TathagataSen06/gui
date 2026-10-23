"""Run the full benchmark evaluation for a world and write a report."""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any

from ..db import connect
from ..generate.config import GeneratorConfig
from ..paths import RunPaths
from .metrics import clean_control_fpr, evaluate_detection, evaluate_er
from .truth import Truth


def _alert_threshold(con) -> float:  # type: ignore[no-untyped-def]
    row = con.execute("SELECT payload FROM run_meta WHERE stage = 'score'").fetchone()
    return float(json.loads(row[0])["config"]["alert_threshold"]) if row else 45.0


def build_clean_control(root: Path, model: Path) -> Path:
    """Generate a world with the same size but no planted rings, run the pipeline
    with the model trained on the main world, and return its directory."""
    from ..graph import run as run_graph
    from ..ingest import ingest
    from ..resolve import run as run_resolve
    from ..score import run as run_score

    manifest = json.loads((root / "manifest.json").read_text())
    c = manifest["config"]
    clean = root.parent / f"{root.name}_clean_control"
    if clean.exists():
        shutil.rmtree(clean)
    from ..generate.world import generate_world

    cfg = GeneratorConfig(n_people=c["n_people"], seed=int(c["seed"]) + 1000, start=date.fromisoformat(c["start"]), days=c["days"], scenario_mix={})
    generate_world(cfg, clean)
    ingest(clean)
    run_resolve(clean)
    run_graph(clean)
    run_score(clean, pretrained=model)
    return clean


def evaluate_run(root: str | Path, clean_control: bool = False) -> dict[str, Any]:
    paths = RunPaths.at(root)
    truth = Truth.load(paths.root)
    con = connect(paths.db, read_only=True)
    try:
        thr = _alert_threshold(con)
        meta = {r[0]: json.loads(r[1]) for r in con.execute("SELECT stage, payload FROM run_meta").fetchall()}
        result: dict[str, Any] = {
            "world": json.loads(paths.manifest.read_text())["config"],
            "entity_resolution": evaluate_er(con, truth),
            "detection": evaluate_detection(con, truth, thr),
            "model_cv": meta.get("score", {}).get("metrics", {}),
        }
        result["model_importances"] = con.execute("SELECT * FROM model_importances ORDER BY importance DESC LIMIT 12").df().to_dict(orient="records")
    finally:
        con.close()
    if clean_control:
        clean = build_clean_control(paths.root, paths.model)
        ccon = connect(RunPaths.at(clean).db, read_only=True)
        try:
            result["clean_control"] = clean_control_fpr(ccon, Truth.load(clean), thr)
            result["clean_control"]["world_seed"] = int(json.loads((clean / "manifest.json").read_text())["seed"])
        finally:
            ccon.close()
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "evaluation.json").write_text(json.dumps(result, indent=2, default=str))
    (paths.reports / "evaluation.md").write_text(render_markdown(result))
    return result


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{x:.1%}"


def render_markdown(r: dict[str, Any]) -> str:
    er = r["entity_resolution"]
    d = r["detection"]
    w = r["world"]
    lines = [
        "# Argus benchmark evaluation",
        "",
        f"World: {w['n_people']} synthetic people, seed {w['seed']}, {w['days']} days from {w['start']}.",
        "",
        "## Entity resolution",
        "",
        "| Slice | Records | True pairs | Pairwise P | Pairwise R | F1 | B-cubed F1 | Over-merged | Split entities |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in ("all_records", "person_records", "person_records_excl_burners", "org_records"):
        m = er[k]
        lines.append(f"| {k} | {m['records']} | {m['true_pairs']} | {m['pairwise_precision']:.3f} | {m['pairwise_recall']:.3f} | {m['pairwise_f1']:.3f} | {m['bcubed_f1']:.3f} | {m['overmerged_clusters']} | {m['split_entities']} |")
    ea = er["error_analysis"]
    lines += ["", "Missed links by likely cause: " + ", ".join(f"{k} = {v}" for k, v in ea["split_records_by_cause"].items()), ""]
    lines += ["## Detection", "", "| Score | ROC-AUC | Average precision |", "|---|---:|---:|"]
    for k in ("risk_score", "rule_score", "ml_score"):
        lines.append(f"| {k} | {d[k]['roc_auc']:.3f} | {d[k]['average_precision']:.3f} |")
    a = d["alerts"]
    lines += [
        "",
        f"Alerts at risk >= {a['threshold']}: {a['alerts']} ({a['true_positives']} true, {a['false_positives']} false); precision {_pct(a['precision'])}, recall {_pct(a['recall'])}, FPR {a['false_positive_rate']:.4f}.",
        "",
        "Precision@K: " + ", ".join(f"P@{k} = {v:.2f}" for k, v in d["precision_at_k"].items()),
        "",
        f"Rings detected (>= 50% of members alerted): **{d['rings_detected']} / {d['rings_total']}**; member recall {_pct(d['ring_member_recall'])} (excluding watch-listed seeds: {_pct(d['non_watchlisted_member_recall'])}).",
    ]
    pit = "scoring_window_days" in d
    if pit:
        w = d["scoring_window_days"]
        lines += [
            "",
            f"Point-in-time run: scores use features from days {w[0] + 1}-{w[1]} of the world only (day 1 = the first day); "
            "the community columns below come from the whole-window graph. "
            f"{d['rings_active_in_scoring_window']} of {d['rings_total']} rings were active for at least {d['min_active_days']} days in that window; "
            f"among them {d['rings_detected_among_active']} were detected and member recall is {_pct(d['ring_member_recall_among_active'])}.",
        ]
    lines += [
        "",
        "| Ring | Typology | Members | Alerted | Orgs alerted | Largest community share | Community flagged |" + (" Days in scoring window |" if pit else ""),
        "|---|---|---:|---:|---:|---:|---|" + ("---:|" if pit else ""),
    ]
    for ring in d["rings"]:
        extra = f" {ring['days_in_scoring_window']} |" if pit else ""
        lines.append(f"| {ring['ring_id']} | {ring['typology']} | {ring['members']} | {ring['members_alerted']} | {ring['orgs_alerted']} | {ring['largest_community_share']:.2f} | {'yes' if ring['community_flagged'] else 'no'} |" + extra)
    lines += ["", "False positives among benign look-alikes (decoys):", "", "| Decoy | Participants | Alerted | Rate |", "|---|---:|---:|---:|"]
    for k, v in d["decoys"].items():
        lines.append(f"| {k} | {v['participants']} | {v['alerted']} | {_pct(v['rate'])} |")
    lines.append(f"| scam victims | {d['victims']['victims']} | {d['victims']['alerted']} | {_pct(d['victims']['alerted'] / max(1, d['victims']['victims']))} |")
    cq = d["communities"]
    lines += ["", f"Flagged communities: {cq['flagged_communities']} ({cq['flagged_with_ring_member']} contain a ring member; {cq['rings_touching_flagged_community']} rings touch a flagged community)."]
    if "clean_control" in r:
        c = r["clean_control"]
        lines += [
            "",
            "## Clean control (no planted networks, model trained on the main world)",
            "",
            f"{c['alerts']} alerts among {c['entities']} entities: entity FPR {c['entity_false_positive_rate']:.4f}; {c['persons_alerted']} of {c['persons']} people alerted (person FPR {c['person_false_positive_rate']:.4f}); 99th percentile risk score {c['risk_score_p99']}.",
        ]
    cv = r.get("model_cv", {})
    if cv:
        lines += ["", "## Model cross-validation (grouped by community, labelled entities)", "", "| Model | ROC-AUC | Average precision | Precision@k |", "|---|---:|---:|---:|"]
        for k in ("logreg", "gbm", "blend", "rules"):
            if k in cv:
                lines.append(f"| {k} | {cv[k]['roc_auc']:.3f} | {cv[k]['average_precision']:.3f} | {cv[k]['precision_at_k']:.3f} |")
        lines.append(f"\nChosen model: **{cv.get('chosen_model')}**.")
    if r.get("model_importances"):
        lines += ["", "Top features:", "", "| Feature | Importance | Held-out permutation importance |", "|---|---:|---:|"]
        for row in r["model_importances"][:10]:
            lines.append(f"| {row['feature']} | {row['importance']:.3f} | {row.get('heldout_permutation_importance', float('nan')):.4f} |")
    return "\n".join(lines) + "\n"
