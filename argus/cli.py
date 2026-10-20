"""Command-line interface: ``python -m argus <command>``.

Commands::

    generate   fabricate a synthetic world (raw feeds + ground truth)
    ingest     load raw feeds into the DuckDB warehouse (common event schema)
    resolve    entity resolution
    graph      relationship graph analytics
    score      risk scoring
    run        all of the above in sequence
    evaluate   benchmark metrics against ground truth (optionally with a clean control world)
    app        launch the Streamlit operator console
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

from .generate.config import GeneratorConfig


def _print(title: str, payload: dict) -> None:
    print(f"\n== {title} ==")
    print(json.dumps(payload, indent=2, default=str))


def cmd_generate(a: argparse.Namespace) -> dict:
    from .generate.world import generate_world

    mix = None
    if a.clean:
        mix = {}
    cfg = GeneratorConfig(n_people=a.people, seed=a.seed, start=date.fromisoformat(a.start), days=a.days, **({"scenario_mix": mix} if mix is not None else {}))
    res = generate_world(cfg, a.data)
    return res.stats


def cmd_ingest(a: argparse.Namespace) -> dict:
    from .ingest import ingest

    r = ingest(a.data)
    return {"events": r.total_events, "records": r.total_records, "sources": r.sources}


def cmd_resolve(a: argparse.Namespace) -> dict:
    from .resolve import run

    st = run(a.data).stats
    st.pop("scorer", None)
    return st


def cmd_graph(a: argparse.Namespace) -> dict:
    from .graph import run

    st = run(a.data)
    st.pop("config", None)
    return st


def cmd_score(a: argparse.Namespace) -> dict:
    from .score import run

    st = run(a.data)
    st.pop("config", None)
    return st


def cmd_run(a: argparse.Namespace) -> dict:
    out = {}
    if not a.skip_generate:
        t = time.perf_counter()
        out["generate"] = cmd_generate(a)
        _print(f"generate ({time.perf_counter() - t:.0f}s)", {k: v for k, v in out["generate"].items() if not k.startswith("bank_")})
    for name, fn in (("ingest", cmd_ingest), ("resolve", cmd_resolve), ("graph", cmd_graph), ("score", cmd_score)):
        t = time.perf_counter()
        out[name] = fn(a)
        _print(f"{name} ({time.perf_counter() - t:.0f}s)", out[name])
    return out


def cmd_evaluate(a: argparse.Namespace) -> dict:
    from .evaluate.report import evaluate_run

    return evaluate_run(a.data, clean_control=a.clean_control)


def cmd_app(a: argparse.Namespace) -> dict:
    app = Path(__file__).parent / "app" / "dashboard.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app), "--", "--data", str(a.data)], check=False)
    return {}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="argus", description="Synthetic-data investigative analytics benchmark (research use only).")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name: str, fn, gen: bool = False) -> argparse.ArgumentParser:  # type: ignore[no-untyped-def]
        sp = sub.add_parser(name)
        sp.add_argument("--data", default="data/demo", help="world / run directory")
        if gen:
            sp.add_argument("--people", type=int, default=1500)
            sp.add_argument("--seed", type=int, default=42)
            sp.add_argument("--start", default="2026-01-01")
            sp.add_argument("--days", type=int, default=182)
            sp.add_argument("--clean", action="store_true", help="generate without planted networks (false-positive control)")
        sp.set_defaults(fn=fn)
        return sp

    add("generate", cmd_generate, gen=True)
    add("ingest", cmd_ingest)
    add("resolve", cmd_resolve)
    add("graph", cmd_graph)
    add("score", cmd_score)
    r = add("run", cmd_run, gen=True)
    r.add_argument("--skip-generate", action="store_true")
    e = add("evaluate", cmd_evaluate)
    e.add_argument("--clean-control", action="store_true", help="also build a clean world and measure false positives")
    add("app", cmd_app)
    a = p.parse_args(argv)
    res = a.fn(a)
    if a.command not in ("run", "app"):
        _print(a.command, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
