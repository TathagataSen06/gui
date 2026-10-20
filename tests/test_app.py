"""Operator console: data layer, rendering, and a headless Streamlit run."""

from __future__ import annotations

import sys
from pathlib import Path

from argus.app.data import Warehouse
from argus.app.graphview import render

APP = Path(__file__).resolve().parents[1] / "argus" / "app" / "dashboard.py"


def test_warehouse_queries(scored: Path) -> None:
    wh = Warehouse(scored)
    q = wh.alert_queue()
    assert q["rank"].tolist() == sorted(q["rank"]) and len(q) > 0
    top = q.iloc[0]["entity_id"]
    ent = wh.entity(top)
    assert ent["narrative"] and isinstance(ent["rule_hits"], list)
    assert not wh.records(top).empty
    ev = wh.events(top)
    assert {"ts_local", "channel", "description"} <= set(ev.columns) and len(ev) > 0
    assert ev["description"].str.len().min() > 5
    G = wh.subgraph(top, mode="ego", radius=1, max_nodes=30)
    assert top in G and G.number_of_nodes() <= 30
    html = render(G, focus=top)
    assert "vis-network" in html or "vis.Network" in html
    assert wh.kpis()["events"] > 0


def test_subgraph_handles_unscored_nodes(scored: Path) -> None:
    """Ego networks include accounts / external parties that have no risk score."""
    wh = Warehouse(scored)
    row = wh.q(
        """
        SELECT e.src FROM graph_edges e JOIN risk_scores r ON r.entity_id = e.src
        WHERE e.dst NOT IN (SELECT entity_id FROM risk_scores) LIMIT 1
        """
    )
    assert not row.empty
    G = wh.subgraph(row.iloc[0, 0], mode="ego", radius=1, max_nodes=80)
    unscored = [n for n, d in G.nodes(data=True) if d["risk"] is None]
    assert unscored and all(G.nodes[n]["alert"] is False for n in unscored)
    assert "vis" in render(G, focus=row.iloc[0, 0])
    for n in list(G)[:3]:
        G2 = wh.subgraph(n, mode="community", max_nodes=40)
        assert n in G2 or G2.number_of_nodes() <= 80


def test_dispositions_are_appended(scored: Path) -> None:
    wh = Warehouse(scored)
    eid = wh.alert_queue().iloc[0]["entity_id"]
    before = len(wh.actions(eid))
    wh.record_action(eid, "tester", "Close - false positive", "unit test", 50.0)
    assert len(wh.actions(eid)) == before + 1


def test_dashboard_renders_headless(scored: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("ARGUS_EAGER_TABS", "1")  # run every tab body, not just the selected one

    argv = sys.argv
    sys.argv = [str(APP), "--data", str(scored)]
    try:
        at = AppTest.from_file(str(APP), default_timeout=180)
        at.run()
    finally:
        sys.argv = argv
    assert not at.exception, [e.value for e in at.exception]
    assert at.title[0].value.startswith("Argus")
    assert len(at.tabs) == 5
    labels = [m.label for m in at.metric]
    assert "Open alerts" in labels and "Resolved entities" in labels
    # every tab body executes on each run: select an entity whose network has unscored nodes
    wh = Warehouse(scored)
    eid = wh.q("SELECT e.src FROM graph_edges e JOIN risk_scores r ON r.entity_id = e.src WHERE e.dst NOT IN (SELECT entity_id FROM risk_scores) LIMIT 1").iloc[0, 0]
    at.session_state["entity"] = eid
    sys.argv = [str(APP), "--data", str(scored)]
    try:
        at.sidebar.slider[0].set_value(0).run()
    finally:
        sys.argv = argv
    assert not at.exception, [e.value for e in at.exception]
