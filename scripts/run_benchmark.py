"""Multi-seed benchmark: generate -> pipeline -> evaluate (with clean control) per seed.

Usage::

    python scripts/run_benchmark.py --people 1500 --seeds 42 7 2024 --out data/bench
    # writes reports/benchmark.md and reports/benchmark.json (repository root)

Results are reported as mean +/- standard deviation across seeds so a single
lucky world cannot make the system look better than it is.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from argus.evaluate import evaluate_run  # noqa: E402
from argus.generate.config import GeneratorConfig  # noqa: E402
from argus.generate.world import generate_world  # noqa: E402
from argus.graph import run as run_graph  # noqa: E402
from argus.ingest import ingest  # noqa: E402
from argus.resolve import run as run_resolve  # noqa: E402
from argus.score import run as run_score  # noqa: E402

METRICS: dict[str, tuple[str, ...]] = {
    "ER pairwise precision (persons, excl. anonymous lines)": ("entity_resolution", "person_records_excl_burners", "pairwise_precision"),
    "ER pairwise recall (persons, excl. anonymous lines)": ("entity_resolution", "person_records_excl_burners", "pairwise_recall"),
    "ER pairwise F1 (persons, excl. anonymous lines)": ("entity_resolution", "person_records_excl_burners", "pairwise_f1"),
    "ER pairwise recall (all person records)": ("entity_resolution", "person_records", "pairwise_recall"),
    "Model CV ROC-AUC (labelled entities)": ("model_cv", "{chosen}", "roc_auc"),
    "Model CV average precision": ("model_cv", "{chosen}", "average_precision"),
    "Rules-only CV ROC-AUC": ("model_cv", "rules", "roc_auc"),
    "Rules-only CV average precision": ("model_cv", "rules", "average_precision"),
    "Logistic regression CV ROC-AUC": ("model_cv", "logreg", "roc_auc"),
    "Gradient boosting CV ROC-AUC": ("model_cv", "gbm", "roc_auc"),
    "Alert precision (entities)": ("detection", "alerts", "precision"),
    "Alert recall (entities incl. fragments)": ("detection", "alerts", "recall"),
    "Ring-member recall (person level)": ("detection", "ring_member_recall",),
    "Ring-member recall excl. watch-listed seeds": ("detection", "non_watchlisted_member_recall",),
    "Rings detected (of 7)": ("detection", "rings_detected",),
    "Precision@25": ("detection", "precision_at_k", "25"),
    "Clean-world person FPR": ("clean_control", "person_false_positive_rate",),
    "Clean-world alerts": ("clean_control", "alerts",),
}


def get(d: dict, path: tuple[str, ...]):  # type: ignore[no-untyped-def]
    cur = d
    for p in path:
        if p == "{chosen}":
            p = d["model_cv"]["chosen_model"]
        cur = cur[p]
    return cur


def run_seed(out: Path, people: int, seed: int) -> dict:
    root = out / f"seed_{seed}"
    t = time.perf_counter()
    generate_world(GeneratorConfig(n_people=people, seed=seed), root)
    ingest(root)
    run_resolve(root)
    run_graph(root)
    run_score(root)
    res = evaluate_run(root, clean_control=True)
    res["seconds"] = round(time.perf_counter() - t, 1)
    print(f"seed {seed}: done in {res['seconds']}s", flush=True)
    return res


def _run_seed_isolated(args: tuple[str, int, int]) -> dict:
    out, people, seed = args
    return run_seed(Path(out), people, seed)


def run_seeds(out: Path, people: int, seeds: list[int]) -> dict[int, dict]:
    """Run every seed in a fresh process, so memory held by one world's stages (DuckDB buffers,
    the Python heap) cannot pile up into the next; six worlds in one process were once OOM-killed."""
    ctx = multiprocessing.get_context("spawn")
    results = {}
    for s in seeds:
        with ctx.Pool(1) as pool:
            results[s] = pool.apply(_run_seed_isolated, ((str(out), people, s),))
    return results


def _fmt(v: float) -> str:
    return f"{v:.4f}" if 0 < abs(v) < 0.01 else f"{v:.3f}"


def role_recall(out: Path, seeds: list[int], threshold: float = 45.0) -> dict[tuple[str, str], list[int]]:
    """Missed / total ring members per (typology, role), person level."""
    from argus.db import connect
    from argus.evaluate.truth import Truth, person_entities

    acc: dict[tuple[str, str], list[int]] = {}
    for s in seeds:
        root = out / f"seed_{s}"
        con = connect(root / "argus.duckdb", read_only=True)
        try:
            t = Truth.load(root)
            pe = person_entities(con, t)
            risk = dict(con.execute("SELECT entity_id, risk_score FROM risk_scores").fetchall())
        finally:
            con.close()
        for ring in t.rings:
            for m in ring["members"]:
                r = max((risk.get(e, 0.0) for e in pe.get(m["person_id"], [])), default=0.0)
                k = (ring["typology"], m["role"])
                acc.setdefault(k, [0, 0])
                acc[k][1] += 1
                acc[k][0] += int(r >= threshold)
    return acc


def render(people: int, seeds: list[int], results: dict, roles: dict | None) -> str:
    rows = []
    for label, path in METRICS.items():
        vals = [float(get(r, path)) for r in results.values()]
        rows.append({"metric": label, "values": vals, "mean": statistics.mean(vals), "std": statistics.pstdev(vals)})
    typ: dict[str, list[float]] = {}
    for r in results.values():
        for ring in r["detection"]["rings"]:
            typ.setdefault(ring["typology"], []).append(ring["member_recall"])
    decoys: dict[str, list[float]] = {}
    for r in results.values():
        for k, v in r["detection"]["decoys"].items():
            decoys.setdefault(k, []).append(v["rate"])
        decoys.setdefault("scam victims", []).append(r["detection"]["victims"]["alerted"] / max(1, r["detection"]["victims"]["victims"]))
    md = [
        "# Argus benchmark (multi-seed)",
        "",
        f"{len(seeds)} independent synthetic worlds of {people:,} people (seeds {', '.join(map(str, seeds))}), "
        "182-day window, 7 planted networks each (2 shell-layering, 2 structuring, 2 mule-funnel, 1 insider), "
        "21 benign look-alike decoys each, plus a clean control world per seed with no planted networks "
        "(scored with the model trained on the main world). Alert threshold: risk >= 45.",
        "",
        "| Metric | Mean | Std | Per seed |",
        "|---|---:|---:|---|",
    ]
    for r in rows:
        md.append(f"| {r['metric']} | {_fmt(r['mean'])} | {_fmt(r['std'])} | {', '.join(_fmt(v) for v in r['values'])} |")
    md += ["", "## Member recall by typology", "", "| Typology | Mean member recall | Rings |", "|---|---:|---:|"]
    for k, v in sorted(typ.items()):
        md.append(f"| {k} | {statistics.mean(v):.2f} | {len(v)} |")
    if roles:
        md += ["", "## Member recall by role (all seeds)", "", "| Typology | Role | Alerted / members |", "|---|---|---:|"]
        for (t, role), (hit, tot) in sorted(roles.items()):
            md.append(f"| {t} | {role} | {hit} / {tot} |")
    md += ["", "## Alert rate among benign look-alikes (main worlds)", "", "| Decoy | Mean alert rate |", "|---|---:|"]
    for k, v in sorted(decoys.items()):
        md.append(f"| {k} | {statistics.mean(v):.3f} |")
    secs = sum(float(r.get("seconds", 0)) for r in results.values())
    md += ["", f"Total runtime: {secs:.0f}s (generation, pipeline, evaluation and clean control for all seeds; one seed at a time, each in a fresh process).", ""]
    return "\n".join(md)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", type=int, default=1500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2024])
    ap.add_argument("--out", default="data/bench")
    ap.add_argument("--report-dir", default=str(ROOT / "reports"))
    ap.add_argument("--render-only", action="store_true", help="re-render reports from an existing benchmark.json")
    a = ap.parse_args()
    out = Path(a.out)
    rep = Path(a.report_dir)
    rep.mkdir(parents=True, exist_ok=True)
    if a.render_only:
        saved = json.loads((rep / "benchmark.json").read_text())
        people, seeds, results = saved["people"], saved["seeds"], {int(k): v for k, v in saved["per_seed"].items()}
    else:
        people, seeds = a.people, a.seeds
        results = run_seeds(out, people, seeds)
        (rep / "benchmark.json").write_text(json.dumps({"people": people, "seeds": seeds, "per_seed": results}, indent=2, default=str))
    roles = role_recall(out, seeds) if all((out / f"seed_{s}" / "argus.duckdb").exists() for s in seeds) else None
    md = render(people, seeds, results, roles)
    (rep / "benchmark.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
