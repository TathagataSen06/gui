"""Graph stage: build, analyse, persist."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import duckdb
import networkx as nx
import pandas as pd

from ..db import connect
from ..paths import RunPaths
from .analytics import centrality, communities, community_profiles, controller_map, find_money_cycles, investigative_graph, load_transfers
from .build import GraphConfig, build_projection


def run_graph(con: duckdb.DuckDBPyConnection, cfg: GraphConfig | None = None) -> dict[str, Any]:
    cfg = cfg or GraphConfig()
    tz = con.execute("SELECT value FROM manifest WHERE key = 'timezone'").fetchone()
    if tz:
        cfg.timezone = tz[0]
    t0 = time.perf_counter()
    G, idf, chan_edges, hubs = build_projection(con, cfg)
    cent = centrality(G, cfg)
    comm = communities(investigative_graph(G, cfg), cfg)
    cent["community_id"] = cent["node_id"].map(comm)
    ctrl = controller_map(con)
    cycles = find_money_cycles(load_transfers(con, cfg.min_cycle_amount), ctrl, cfg)
    prof = community_profiles(con, G, comm, cycles)

    def put(name: str, df: pd.DataFrame) -> None:
        con.register("_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _df")
        con.unregister("_df")

    put("graph_identifier_edges", idf)
    put("graph_edges", chan_edges.rename(columns={"a": "src", "b": "dst"}))
    put("graph_nodes", cent)
    put("communities", prof)
    cyc_df = pd.DataFrame(cycles, columns=["cycle_id", "length", "parties", "entities_path", "event_ids", "amounts", "start_ts", "end_ts", "duration_hours", "value_retention", "instances"])
    put("money_cycles", cyc_df)
    put("graph_hubs", pd.DataFrame({"identifier": sorted(hubs)}))
    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "communities": int(prof.shape[0]),
        "flagged_communities": int(prof["flagged"].sum()) if not prof.empty else 0,
        "money_cycles": len(cycles),
        "hubs_dropped": len(hubs),
        "components": nx.number_connected_components(G),
        "config": asdict(cfg),
        "seconds": round(time.perf_counter() - t0, 2),
    }
    con.execute("CREATE TABLE IF NOT EXISTS run_meta (stage VARCHAR, payload JSON)")
    con.execute("DELETE FROM run_meta WHERE stage = 'graph'")
    con.execute("INSERT INTO run_meta VALUES ('graph', ?)", [json.dumps(stats, default=str)])
    return stats


def load_graph(con: duckdb.DuckDBPyConnection) -> nx.Graph:
    """Rebuild the entity projection from persisted tables (used by scoring and the app)."""
    nodes = con.execute("SELECT node_id, kind, label, community_id FROM graph_nodes").fetchall()
    edges = con.execute("SELECT src, dst, channel, count, amount, weight FROM graph_edges ORDER BY src, dst, channel").fetchall()
    G = nx.Graph()
    for n, kind, label, cid in nodes:
        G.add_node(n, kind=kind, label=label, community_id=cid)
    for s, d, ch, cnt, amt, wgt in edges:
        if G.has_edge(s, d):
            G[s][d]["channels"][ch] = {"count": int(cnt), "amount": float(amt or 0), "weight": float(wgt)}
            G[s][d]["weight"] += float(wgt)
        else:
            G.add_edge(s, d, weight=float(wgt), channels={ch: {"count": int(cnt), "amount": float(amt or 0), "weight": float(wgt)}})
    return G


def run(root: str | Path, cfg: GraphConfig | None = None) -> dict[str, Any]:
    con = connect(RunPaths.at(root).db)
    try:
        return run_graph(con, cfg)
    finally:
        con.close()
