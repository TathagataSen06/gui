"""Run the external benchmarks and write ``reports/realworld.md`` / ``.json``.

A dataset whose files cannot be obtained is reported as *not run*, with the
hosts that would need to be reachable; no number is ever filled in for it.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any, Callable

import sklearn

from . import creditcard, dblp_acm, elliptic
from .baselines import Baseline, for_protocol
from .sources import CATALOG, DEFAULT_ROOT, DatasetUnavailable, dataset_dir, fetch

RUNNERS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "elliptic": lambda d: elliptic.run(d),
    "creditcard": lambda d: creditcard.run(d / "creditcard.csv"),
    "dblp_acm": lambda d: dblp_acm.run(d),
}
ORDER = ("elliptic", "creditcard", "dblp_acm")


def run_all(datasets: list[str] | None = None, root: str | Path = DEFAULT_ROOT, report_dir: str | Path = "reports", download: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "machine": platform.machine()},
        "datasets": {},
    }
    prev = Path(report_dir) / "realworld.json"
    if prev.exists():  # keep earlier results for datasets not re-run now
        out["datasets"] = json.loads(prev.read_text()).get("datasets", {})
    for key in datasets or list(ORDER):
        d = dataset_dir(key, root)
        entry: dict[str, Any] = {"status": "not_run"}
        try:
            prov = fetch(key, root) if download else json.loads((d / "PROVENANCE.json").read_text())
            t = time.perf_counter()
            entry = {"status": "ok", "provenance": prov, "result": RUNNERS[key](d), "seconds": round(time.perf_counter() - t, 1)}
        except (DatasetUnavailable, FileNotFoundError) as e:
            entry = {"status": "not_run", "reason": str(e).splitlines()[0], "hosts": CATALOG[key].hosts(), "manual": CATALOG[key].manual}
        out["datasets"][key] = entry
        print(f"{key}: {entry['status']}" + (f" in {entry.get('seconds')}s" if entry["status"] == "ok" else f" ({entry['reason']})"), flush=True)
    rep = Path(report_dir)
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "realworld.json").write_text(json.dumps(out, indent=2, default=str))
    (rep / "realworld.md").write_text(render(out))
    return out


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _f(v: Any, nd: int = 3) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def _ci(ci: list[float] | tuple[float, float]) -> str:
    return f"{ci[0]:.3f}-{ci[1]:.3f}"


def _baseline_rows(bs: list[Baseline], cols: list[str]) -> list[str]:
    rows = []
    for b in bs:
        cells = " | ".join(_f(b.metrics.get(c)) for c in cols)
        note = f" ({b.note})" if b.note else ""
        rows.append(f"| {b.method} | {cells} | [{b.source}]({b.url}){note} |")
    return rows


def _provenance(entry: dict[str, Any]) -> list[str]:
    prov = entry["provenance"]
    lines = [f"Data: {prov['dataset']}. License: {prov['license']}.", ""]
    for name, f in prov["files"].items():
        src = f["source"] if f["source"] == "local" else f"<{f['source']}> ({f.get('source_kind', '?')})"
        lines.append(f"- `{name}`: {f['bytes']:,} bytes, sha256 `{f['sha256'][:16]}...`, from {src}")
    return lines + [""]


def _not_run(key: str, entry: dict[str, Any]) -> list[str]:
    return [
        f"**Not run in this environment.** {entry.get('reason', '')[:1].upper()}{entry.get('reason', '')[1:]}",
        "",
        f"To run it, allow one of these hosts in the environment's network policy: {', '.join(f'`{h}`' for h in entry.get('hosts', []))}; "
        f"or place the files by hand: {entry.get('manual', '')} Then run `argus realworld run --dataset {key}`.",
        "",
        "No result is reported for this dataset.",
        "",
    ]


def _render_elliptic(r: dict[str, Any]) -> list[str]:
    md = [
        f"{r['nodes']:,} transactions, {r['edges']:,} edges. Train: time steps 1-34 ({r['train']['rows']:,} labelled, {r['train']['illicit']:,} illicit). "
        f"Test: steps 35-49 ({r['test']['rows']:,} labelled, {r['test']['illicit']:,} illicit). Illicit class, threshold 0.5.",
        "",
        "| Model | Precision | Recall | F1 | Micro F1 | Source |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for res in sorted(r["results"].values(), key=lambda x: (not x["model"].startswith("argus"), x["model"], x["features"])):
        label = {"argus_logreg": "**Argus** logistic regression", "argus_gbm": "**Argus** gradient boosting",
                 "weber_rf": "Weber RF settings, re-run here", "weber_logreg": "Weber LR settings, re-run here"}[res["model"]]
        md.append(f"| {label}, {res['features']} | {_f(res['precision'])} | {_f(res['recall'])} | {_f(res['f1'])} | {_f(res['micro_f1'])} | measured here |")
    md += _baseline_rows(for_protocol("elliptic", "temporal_1_34"), ["precision", "recall", "f1", "micro_f1"])
    best = max(r["results"].values(), key=lambda x: x["f1"])
    steps = sorted(best["per_step"].items(), key=lambda kv: int(kv[0]))
    md += [
        "",
        f"Illicit F1 per test time step for the best run here ({best['model']}, {best['features']}). The dark market shut down at step 43:",
        "",
        "| Step | " + " | ".join(str(s) for s, _ in steps) + " |",
        "|---|" + "---:|" * len(steps),
        "| Illicit | " + " | ".join(str(v["illicit"]) for _, v in steps) + " |",
        "| F1 | " + " | ".join(_f(v["f1"], 2) for _, v in steps) + " |",
        "",
        f"F1 before step 43: {_f(best['f1_before_step_43'])}; from step 43 on: {_f(best['f1_from_step_43'])}.",
        "",
    ]
    return md


def _model_name(key: str) -> str:
    return {"logreg": "logistic regression", "gbm": "gradient boosting"}.get(key, key)


def _render_creditcard(r: dict[str, Any]) -> list[str]:
    oot, rnd = r["out_of_time"], r["random_60_40"]
    md = [
        f"{r['rows']:,} transactions, {r['frauds']} frauds ({r['prevalence']:.3%}). Features: V1-V28 and log(1 + Amount); `Time` only orders rows.",
        "",
        f"**Out-of-time** ({oot['protocol']}): {oot['test_rows']:,} test rows, {oot['test_frauds']} frauds. 95% bootstrap intervals.",
        "",
        "| Model | ROC-AUC | AUPRC | Source |",
        "|---|---:|---:|---|",
    ]
    for name, m in oot["models"].items():
        md.append(f"| **Argus** {_model_name(name)} | {_f(m['roc_auc'])} ({_ci(m['roc_auc_ci95'])}) | {_f(m['auprc'])} ({_ci(m['auprc_ci95'])}) | measured here |")
    md += _baseline_rows(for_protocol("creditcard", "out_of_time"), ["roc_auc", "auprc"])
    md += ["", "Alert budgets on the out-of-time test set (top-scored transactions alerted):", "", "| Model | Budget | Alerts | Frauds caught | Precision | Recall |", "|---|---:|---:|---:|---:|---:|"]
    for name, m in oot["models"].items():
        for b, v in m["budgets"].items():
            md.append(f"| Argus {_model_name(name)} | {b} | {v['alerts']} | {v['frauds_caught']} | {_f(v['precision'])} | {_f(v['recall'])} |")
    md += [
        "",
        f"**Random split** ({rnd['protocol']}), mean and standard deviation over seeds:",
        "",
        "| Model | AUPRC | ROC-AUC | Source |",
        "|---|---:|---:|---|",
    ]
    for name, m in rnd["models"].items():
        md.append(f"| **Argus** {_model_name(name)} | {_f(m['auprc']['mean'])} ± {_f(m['auprc']['std'])} | {_f(m['roc_auc']['mean'])} ± {_f(m['roc_auc']['std'])} | measured here |")
    md += _baseline_rows(for_protocol("creditcard", "random_60_40"), ["auprc", "roc_auc"])
    md.append("")
    for name, m in rnd["models"].items():
        aucs = m["roc_auc"]["values"]
        if max(aucs) - min(aucs) > 0.05:
            md += [f"Argus {_model_name(name)} is unstable on this data: its ROC-AUC ranged from {min(aucs):.3f} to {max(aucs):.3f} across the seeds above "
                   "(per seed: " + ", ".join(f"{a:.3f}" for a in aucs) + ").", ""]
    return md


def _render_dblp(r: dict[str, Any]) -> list[str]:
    ft = r["full_table"]
    bl = ft["blocking"]
    md = [
        f"{r['dblp']:,} DBLP and {r['acm']:,} ACM records, {r['gold_pairs']:,} true matches.",
        "",
        f"**Whole problem.** Blocking kept {bl['candidate_pairs']:,} of {bl['cartesian_pairs']:,} pairs (reduction {bl['reduction_ratio']:.3f}) "
        f"and {bl['pair_completeness']:.1%} of the true matches. Precision and recall are over all {r['gold_pairs']:,} true matches, so blocking misses count.",
        "",
        "| Matcher | Precision | Recall | F1 | Labels | Source |",
        "|---|---:|---:|---:|---|---|",
    ]
    names = {"fellegi_sunter": "**Argus** Fellegi-Sunter (u from random pairs, m by EM)", "argus_logreg": "**Argus** logistic regression", "argus_gbm": "**Argus** gradient boosting"}
    for k, m in ft["matchers"].items():
        labels = "none" if m["labels_used"] == 0 else m["labels_used"]
        md.append(f"| {names[k]} | {_f(m['precision'])} | {_f(m['recall'])} | {_f(m['f1'])} | {labels} | measured here |")
    md += [
        "",
        "Published results on the whole problem (not on the pre-blocked split). Their protocols differ from this one and from each other; the notes say how:",
        "",
        "| Method | Precision | Recall | F1 | Source |",
        "|---|---:|---:|---:|---|",
    ]
    md += _baseline_rows(for_protocol("dblp_acm", "full_table"), ["precision", "recall", "f1"])
    if "pre_blocked" in r:
        pb = r["pre_blocked"]
        md += [
            "",
            f"**Pre-blocked split** ({pb['protocol']}): {pb['pairs']['train']['pairs']:,} / {pb['pairs']['valid']['pairs']:,} / {pb['pairs']['test']['pairs']:,} pairs.",
            "",
            "| Model | Precision | Recall | F1 | Source |",
            "|---|---:|---:|---:|---|",
        ]
        for k, m in pb["models"].items():
            label = names[k].replace(" (u from random pairs, m by EM)", "")
            md.append(f"| {label} (threshold {m['threshold']:.2f}; F1 at 0.5: {_f(m['f1_at_0.5'])}) | {_f(m['precision'])} | {_f(m['recall'])} | {_f(m['f1'])} | measured here |")
        md += _baseline_rows(for_protocol("dblp_acm", "pre_blocked"), ["precision", "recall", "f1"])
    md.append("")
    return md


SECTIONS = {
    "elliptic": ("Elliptic Bitcoin transactions: graph features and models", "graph features, supervised models", _render_elliptic),
    "creditcard": ("ULB credit-card fraud: supervised models at 0.17% prevalence", "supervised models", _render_creditcard),
    "dblp_acm": ("DBLP-ACM: entity resolution", "blocking, comparators, embedding, Fellegi-Sunter, clustering", _render_dblp),
}

CAVEATS = {
    "elliptic": [
        "Elliptic validates the scoring and graph-feature code, not entity resolution: it has no identity attributes.",
        "Weber's numbers are one unseeded run; random-forest results on this split move by a few points between runs.",
        "Argus's graph features are hand-built centralities, not the GCN embeddings of the paper's +NE rows.",
    ],
    "creditcard": [
        "Only the transaction scorer is tested: the data has no cardholder, merchant or device key, so no entity resolution or graph features.",
        "The data spans two days, so 'out-of-time' means a roughly ten-hour hold-out, not drift over months.",
        "The out-of-time test set has about 75 frauds, hence the wide intervals. Random-split numbers are not comparable with out-of-time ones.",
    ],
    "dblp_acm": [
        "The entity is a publication, not a person. Author names are public bibliographic metadata, used only as a string field and never written to any report.",
        "Field normalization (venue aliases, HTML entities) was written for this dataset; the Argus resolver's person-specific comparators do not apply to bibliographic records.",
        "DBLP-ACM is an easy benchmark; most published methods score above 0.96 F1. It shows the components work on real records, not that they are state of the art.",
        "One change was made after the first run on this data: levels never seen among random pairs had been given an arbitrary u of 1e-6 instead of the same additive smoothing as the other levels. "
        "Fixing that moved the unsupervised Fellegi-Sunter F1 from 0.974 to the value above; the supervised and pre-blocked results did not change.",
    ],
}


def render(out: dict[str, Any]) -> str:
    ds = out["datasets"]
    env = out["environment"]
    md = [
        "# Argus on public benchmark data",
        "",
        "Argus's models (unchanged configuration) and entity-resolution building blocks, run on three public research datasets and compared with "
        "published results, under the same protocol where one exists; each table says how the published protocols differ. "
        "None of the datasets identifies a person. The person-level pipeline (`argus ingest` onwards) still accepts only synthetic data.",
        "",
        "Where Argus falls short on this data is analyzed in [`docs/FAILURE_MODES.md`](../docs/FAILURE_MODES.md#6-on-public-benchmark-data).",
        "",
        f"Environment: Python {env['python']}, scikit-learn {env['scikit_learn']}, {env['machine']}. Regenerate with `argus realworld run`.",
        "",
        "| Dataset | Tests | Status |",
        "|---|---|---|",
    ]
    for key in ORDER:
        if key in ds:
            title, tests, _ = SECTIONS[key]
            status = "measured" if ds[key]["status"] == "ok" else "**not run** (data host not reachable)"
            md.append(f"| {CATALOG[key].title} | {tests} | {status} |")
    for key in ORDER:
        if key not in ds:
            continue
        title, _, fn = SECTIONS[key]
        md += ["", f"## {title}", ""]
        entry = ds[key]
        if entry["status"] != "ok":
            md += _not_run(key, entry)
            continue
        md += _provenance(entry)
        md += fn(entry["result"])
        md += ["Caveats:", ""] + [f"- {c}" for c in CAVEATS[key]] + [""]
        md += [f"Citation: {CATALOG[key].citation}", ""]
    return "\n".join(md).rstrip() + "\n"
