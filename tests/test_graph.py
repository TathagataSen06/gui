"""Relationship graph: cycles, communities, persisted analytics."""

from __future__ import annotations

import networkx as nx
import pandas as pd

from argus.graph.analytics import Transfer, communities, find_money_cycles
from argus.graph.build import GraphConfig
from argus.graph.pipeline import load_graph


def _t(i: int, day: float, amt: float, s: str, d: str) -> Transfer:
    return Transfer(f"e{i}", pd.Timestamp("2026-03-01") + pd.Timedelta(days=day), amt, s, d, s, d)


def test_detects_value_preserving_temporal_cycle() -> None:
    tx = [_t(1, 0, 50_000, "A", "B"), _t(2, 1, 49_250, "B", "C"), _t(3, 2.5, 48_500, "C", "A")]
    cyc = find_money_cycles(tx, {}, GraphConfig())
    assert len(cyc) == 1 and cyc[0]["length"] == 3 and cyc[0]["event_ids"] == ["e1", "e2", "e3"]


def test_ignores_loops_that_are_not_layering() -> None:
    cfg = GraphConfig()
    out_of_order = [_t(1, 5, 50_000, "A", "B"), _t(2, 1, 49_000, "B", "C"), _t(3, 2, 48_000, "C", "A")]
    amounts_differ = [_t(1, 0, 50_000, "A", "B"), _t(2, 1, 5_000, "B", "C"), _t(3, 2, 4_900, "C", "A")]
    too_slow = [_t(1, 0, 50_000, "A", "B"), _t(2, 30, 49_000, "B", "C"), _t(3, 60, 48_000, "C", "A")]
    for tx in (out_of_order, amounts_differ, too_slow):
        assert find_money_cycles(tx, {}, cfg) == []


def test_beneficial_owner_collapse_closes_round_trip() -> None:
    # money leaves person P's company, travels through two shells and returns to P personally
    tx = [_t(1, 0, 80_000, "FRONT", "S1"), _t(2, 1, 79_000, "S1", "S2"), _t(3, 2, 78_000, "S2", "P")]
    assert find_money_cycles(tx, {}, GraphConfig()) == []
    cyc = find_money_cycles(tx, {"FRONT": "P"}, GraphConfig())
    assert len(cyc) == 1


def test_louvain_is_deterministic() -> None:
    G = nx.connected_caveman_graph(6, 5)
    for _, _, d in G.edges(data=True):
        d["weight"] = 1.0
    cfg = GraphConfig(seed=3)
    assert communities(G, cfg) == communities(G, cfg)


def test_graph_tables(con) -> None:  # type: ignore[no-untyped-def]
    kinds = {r[0] for r in con.execute("SELECT DISTINCT kind FROM graph_nodes").fetchall()}
    assert {"person", "org"} <= kinds
    chans = {r[0] for r in con.execute("SELECT DISTINCT channel FROM graph_edges").fetchall()}
    assert {"money", "call", "email", "controls"} <= chans
    n = con.execute("SELECT count(*) FROM graph_nodes WHERE betweenness < 0 OR degree < 0").fetchone()[0]
    assert n == 0
    assert con.execute("SELECT count(*) FROM communities WHERE flagged").fetchone()[0] >= 1
    cyc = con.execute("SELECT count(*) FROM money_cycles").fetchone()[0]
    assert cyc >= 1  # the shell-layering rings round-trip money


def test_recency_weights_are_bounded(con) -> None:  # type: ignore[no-untyped-def]
    lo, hi = con.execute("SELECT min(weight), max(weight) FROM graph_edges WHERE channel <> 'controls'").fetchone()
    assert 0 <= lo and hi <= 1.0 + 1e-9


def test_load_graph_roundtrip(con) -> None:  # type: ignore[no-untyped-def]
    G = load_graph(con)
    assert G.number_of_nodes() == con.execute("SELECT count(*) FROM graph_nodes").fetchone()[0]
    u, v, d = next(iter(G.edges(data=True)))
    assert d["channels"] and d["weight"] > 0
