"""Argus operator console (Streamlit).

Run with::

    python -m argus app --data data/demo
    # or
    streamlit run argus/app/dashboard.py -- --data data/demo

Synthetic data only - research and fraud-detection benchmarking.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from argus.app.data import Warehouse  # noqa: E402
from argus.app.graphview import legend_html, render  # noqa: E402

TAB_NAMES = ["Alert queue", "Investigation", "Network", "Timeline", "Model & benchmark"]
CHANNEL_ORDER = ["money in", "money out", "cash", "card", "phone", "email", "access"]
CHANNEL_COLOURS = {"money in": "#7FB77E", "money out": "#E4572E", "cash": "#F28C28", "card": "#9AA0A6", "phone": "#F2B134", "email": "#5FC9C1", "access": "#B07CF0"}
PLOT_LAYOUT = dict(margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")


def data_dir() -> str:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--data", default=os.environ.get("ARGUS_DATA", "data/demo"))
    args, _ = p.parse_known_args(sys.argv[1:])
    return args.data


@st.cache_resource(show_spinner=False)
def warehouse(path: str) -> Warehouse:
    return Warehouse(path)


@st.cache_data(show_spinner=False)
def cached_queue(path: str) -> pd.DataFrame:
    return warehouse(path).alert_queue()


@st.cache_data(show_spinner="Loading entity activity...")
def cached_events(path: str, eid: str) -> pd.DataFrame:
    return warehouse(path).events(eid)


def md_safe(text: str) -> str:
    """Escape dollar signs so Streamlit Markdown does not render amounts as LaTeX."""
    return str(text).replace("$", "\\$")


def risk_badge(risk: float) -> str:
    colour = "#FF3B3B" if risk >= 70 else ("#F28C28" if risk >= 45 else "#7FB77E")
    return f'<span style="background:{colour};color:#111;padding:2px 10px;border-radius:12px;font-weight:700">{risk:.0f}/100</span>'


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


def tab_queue(view: pd.DataFrame) -> None:
    st.subheader(f"Ranked entities ({len(view):,} shown)")
    show = view[["rank", "risk_score", "rule_score", "ml_score", "display_name", "entity_type", "top_reason", "rules_fired", "n_records", "sources", "community_id", "entity_id"]]
    ev = st.dataframe(
        show,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="queue",
        height=560,
        column_config={
            "rank": st.column_config.NumberColumn("#", width="small"),
            "risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.0f"),
            "rule_score": st.column_config.NumberColumn("Rules", format="%.0f"),
            "ml_score": st.column_config.ProgressColumn("Model p", min_value=0.0, max_value=1.0, format="%.2f"),
            "display_name": "Entity",
            "entity_type": "Type",
            "top_reason": "Top reason",
            "rules_fired": "Rules",
            "n_records": "Records",
            "sources": "Sources",
            "community_id": "Community",
            "entity_id": "Entity ID",
        },
    )
    rows = ev.selection.rows if hasattr(ev, "selection") else []
    if rows:
        st.session_state["entity"] = show.iloc[rows[0]]["entity_id"]
        st.info(f"Selected **{show.iloc[rows[0]]['display_name']}** - open the Investigation tab.")


def tab_investigation(wh: Warehouse, ent: dict[str, Any], events: pd.DataFrame) -> None:
    eid = ent["entity_id"]
    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"### {ent['display_name']} &nbsp; {risk_badge(ent['risk_score'])}", unsafe_allow_html=True)
        st.caption(f"{eid} - {ent['entity_type']} - rank #{ent['rank']} - DOB {ent.get('dob') or 'n/a'} - {ent.get('address') or ''}")
        st.markdown(f"> {md_safe(ent['narrative'])}")
        st.markdown("#### Rule indicators")
        if ent["rule_hits"]:
            st.dataframe(pd.DataFrame(ent["rule_hits"])[["rule_id", "title", "points", "reason", "typology"]], hide_index=True)
        else:
            st.write("No rules fired.")
    with right:
        st.markdown("#### Score decomposition")
        if ent.get("risk_driver") == "rule floor":
            st.caption("Rule floor applied: risk = 0.8 x rule score, because the model has little history for this pattern.")
        fig = go.Figure(
            go.Bar(
                x=[ent["rule_score"] * 0.4, ent["ml_score"] * 100 * 0.6, ent["rule_score"] * 0.8],
                y=["Rules (40%)", "Model (60%)", "Rule floor (0.8x)"],
                orientation="h",
                marker_color=["#F28C28", "#4C9BE8", "#7A7A7A"],
            )
        )
        fig.update_layout(height=170, xaxis_title="points of the 0-100 risk score", **PLOT_LAYOUT)
        st.plotly_chart(fig)
        if ent["ml_drivers"]:
            d = pd.DataFrame(ent["ml_drivers"])
            fig = px.bar(d.iloc[::-1], x="contribution", y="label", orientation="h", hover_data=["display_value"], color_discrete_sequence=["#4C9BE8"])
            fig.update_layout(height=260, title="Model drivers (log-odds contribution)", yaxis_title="", **PLOT_LAYOUT)
            st.plotly_chart(fig)
        g1, g2, g3 = st.columns(3)
        g1.metric("Degree", int(ent.get("degree") or 0))
        g2.metric("Betweenness", f"{(ent.get('betweenness') or 0):.4f}")
        g3.metric("Community", ent.get("community_id") or "-")

    st.markdown("#### Resolved identity")
    recs = wh.records(eid)
    st.dataframe(recs, hide_index=True)
    links = wh.match_evidence(eid)
    if not links.empty:
        with st.expander(f"Why these {len(recs)} records were merged ({len(links)} links)"):
            st.dataframe(links[["record_a", "record_b", "match_probability", "match_weight", "rule", "agreements", "top_evidence", "name_jw"]], hide_index=True)

    a, b = st.columns(2)
    with a:
        st.markdown("#### Largest money movements")
        if not events.empty:
            mm = events[events["event_type"].isin(["transfer", "cash_deposit", "cash_withdrawal"])].sort_values("amount", ascending=False).head(15)
            st.dataframe(mm[["ts_local", "description", "amount"]].rename(columns={"ts_local": "time (local)"}), hide_index=True, column_config={"amount": st.column_config.NumberColumn(format="$%.2f")})
        cyc = wh.cycles(eid)
        if not cyc.empty:
            st.markdown("#### Circular payment paths")
            st.dataframe(cyc, hide_index=True)
    with b:
        st.markdown("#### Top contacts")
        if not events.empty:
            cc = events[events["event_type"].isin(["call", "sms", "email"])]
            if not cc.empty:
                top = cc.groupby(["counterparty", "event_type"]).size().rename("count").reset_index().sort_values("count", ascending=False).head(15)
                st.dataframe(top, hide_index=True)
        loc = wh.locations(eid)
        if not loc.empty:
            st.markdown("#### Places (badge readers)")
            st.dataframe(loc[["site_name", "reader_name", "zone", "visits", "after_hours", "denied"]].head(12), hide_index=True)
            st.map(loc.rename(columns={"lat": "latitude", "lon": "longitude"})[["latitude", "longitude"]], size=60)

    st.markdown("#### Disposition")
    d1, d2, d3 = st.columns([1, 1, 2])
    analyst = d1.text_input("Analyst", value="analyst-01")
    disp = d2.selectbox("Decision", ["Escalate to investigation", "Request more information", "Close - false positive", "Close - explained activity"])
    note = d3.text_input("Note")
    if st.button("Record disposition"):
        wh.record_action(eid, analyst, disp, note, float(ent["risk_score"]))
        st.success("Disposition recorded (case_actions.jsonl).")
    hist = wh.actions(eid)
    if not hist.empty:
        st.dataframe(hist, hide_index=True)


def tab_network(wh: Warehouse, ent: dict[str, Any]) -> None:
    eid = ent["entity_id"]
    n1, n2, n3 = st.columns([1, 1, 2])
    mode = n1.radio("View", ["ego", "community"], horizontal=True)
    radius = n2.slider("Ego radius", 1, 2, 1, disabled=mode == "community")
    max_nodes = n3.slider("Max nodes", 15, 120, 50)
    G = wh.subgraph(eid, mode=mode, radius=radius, max_nodes=max_nodes)
    st.markdown(legend_html(), unsafe_allow_html=True)
    page = render(G, focus=eid)  # labels/tooltips are HTML-escaped in graphview.render
    if hasattr(st, "iframe"):
        st.iframe(page, height=660)
    else:  # Streamlit < 1.50
        components.html(page, height=660, scrolling=False)
    cid = ent.get("community_id")
    if cid:
        prof = wh.community(cid)
        if prof:
            st.markdown(f"#### Community {cid} - {'**FLAGGED**' if prof.get('flagged') else 'not flagged'}")
            cols = ["size", "n_persons", "n_orgs", "density", "tightness", "internal_money", "internal_money_share", "internal_wires", "round_share_large", "sub_threshold_cash", "money_cycles", "crypto_outflow", "young_orgs", "shared_address_orgs", "money_flow_score", "reasons"]
            st.dataframe(pd.DataFrame([{c: prof.get(c) for c in cols}]), hide_index=True)


def tab_timeline(events: pd.DataFrame) -> None:
    if events.empty:
        st.write("No events.")
        return
    chans = st.multiselect("Channels", CHANNEL_ORDER, default=[c for c in CHANNEL_ORDER if c != "card"])
    tl = events[events["channel"].isin(chans)].copy()
    tl["size"] = tl["amount"].clip(lower=1).map(lambda v: min(30.0, 4 + v ** 0.33))
    fig = px.scatter(
        tl, x="ts_local", y="channel", color="channel", size="size", hover_name="description",
        hover_data={"size": False, "ts_local": True, "channel": False}, category_orders={"channel": CHANNEL_ORDER},
        color_discrete_map=CHANNEL_COLOURS, height=420,
    )
    fig.update_layout(xaxis_title="", yaxis_title="", showlegend=False, **PLOT_LAYOUT)
    st.plotly_chart(fig)
    money = events[events["channel"].isin(["money in", "money out", "cash"])].copy()
    if not money.empty:
        money["day"] = money["ts_local"].dt.tz_localize(None).dt.floor("D")
        money["signed"] = money.apply(lambda r: r["amount"] if r["direction"] == "in" else -r["amount"], axis=1)
        daily = money.groupby(["day", "direction"])["signed"].sum().reset_index()
        fig2 = px.bar(daily, x="day", y="signed", color="direction", color_discrete_map={"in": "#7FB77E", "out": "#E4572E"}, height=260, title="Daily money in / out (USD)")
        fig2.update_layout(xaxis_title="", yaxis_title="", **PLOT_LAYOUT)
        st.plotly_chart(fig2)
    st.dataframe(tl[["ts_local", "channel", "description"]].sort_values("ts_local", ascending=False).head(500), hide_index=True)


def tab_model(wh: Warehouse) -> None:
    meta = wh.meta()
    sc = meta.get("score", {}).get("metrics", {})
    if sc:
        rows = [{"model": m, **sc[m]} for m in ("logreg", "gbm", "blend", "rules") if m in sc]
        st.markdown(f"#### Cross-validated ranking quality (grouped by community; chosen: **{sc.get('chosen_model')}**)")
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    imp = wh.importances()
    if not imp.empty:
        fig = px.bar(imp.head(15).iloc[::-1], x="importance", y="feature", orientation="h", color_discrete_sequence=["#B07CF0"], height=420, title="Feature importance")
        fig.update_layout(yaxis_title="", **PLOT_LAYOUT)
        st.plotly_chart(fig)
    er = meta.get("resolve", {})
    if er:
        st.markdown("#### Entity resolution run")
        st.json({k: v for k, v in er.items() if k != "scorer"}, expanded=False)
    md = wh.evaluation_markdown()
    if md:
        st.markdown("---")
        st.markdown(md_safe(md))
    else:
        st.info("Run `python -m argus evaluate --data <dir> --clean-control` to add benchmark results here.")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Argus console", page_icon=":satellite:", layout="wide")
    st.markdown(
        '<div style="background:#3a1010;border:1px solid #FF3B3B;padding:6px 12px;border-radius:6px;margin-bottom:8px">'
        "<b>SYNTHETIC DATA - RESEARCH AND FRAUD-DETECTION BENCHMARKING ONLY.</b> Every person, account, phone number and event "
        "shown here was generated by <code>argus.generate</code>. This console is not designed for, and must not be used for, monitoring real individuals."
        "</div>",
        unsafe_allow_html=True,
    )
    path = st.sidebar.text_input("World directory", data_dir())
    try:
        wh = warehouse(path)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    queue = cached_queue(path)
    k = wh.kpis()

    st.title("Argus - investigative analytics console")
    c = st.columns(6)
    c[0].metric("Events ingested", f"{k['events']:,}")
    c[1].metric("Identity records", f"{k['records']:,}")
    c[2].metric("Resolved entities", f"{k['entities']:,}")
    c[3].metric("Open alerts", f"{k['alerts']:,}")
    c[4].metric("Flagged communities", k["flagged_communities"])
    c[5].metric("Circular money paths", k["money_cycles"])

    st.sidebar.header("Alert queue filters")
    min_risk = st.sidebar.slider("Minimum risk", 0, 100, 45)
    types = st.sidebar.multiselect("Entity type", ["person", "org"], default=["person", "org"])
    search = st.sidebar.text_input("Search name / entity id")
    alerts_only = st.sidebar.checkbox("Alerts only", value=False)
    view = queue[(queue["risk_score"] >= min_risk) & queue["entity_type"].isin(types)]
    if alerts_only:
        view = view[view["alert"]]
    if search:
        s = search.lower()
        view = view[view["display_name"].str.lower().str.contains(s, na=False) | view["entity_id"].str.lower().str.contains(s)]
    if "entity" not in st.session_state:
        st.session_state["entity"] = queue.iloc[0]["entity_id"]
    options = view["entity_id"].tolist() or queue["entity_id"].head(50).tolist()
    labels = {e: f"{n if isinstance(n, str) and n else e}  ({e})" for e, n in zip(queue["entity_id"], queue["display_name"])}
    current = st.session_state["entity"] if st.session_state["entity"] in options else options[0]
    st.session_state["entity"] = st.sidebar.selectbox("Selected entity", options, index=options.index(current), format_func=lambda e: labels.get(e, e))

    eager = os.environ.get("ARGUS_EAGER_TABS") == "1"  # tests render every tab in one run
    try:
        tabs = st.tabs(TAB_NAMES) if eager else st.tabs(TAB_NAMES, key="main_tabs", on_change="rerun")
    except TypeError:  # Streamlit without lazy tabs
        tabs = st.tabs(TAB_NAMES)

    def is_open(i: int) -> bool:
        # Lazy tabs: only the selected tab runs. Besides speed, this matters for the network
        # view - vis.js would otherwise lay the graph out inside a hidden, zero-sized frame.
        return eager or getattr(tabs[i], "open", True) is not False

    with tabs[0]:
        if is_open(0):
            tab_queue(view)
    eid = st.session_state["entity"]
    ent = wh.entity(eid)
    with tabs[1]:
        if is_open(1):
            tab_investigation(wh, ent, cached_events(path, eid))
    with tabs[2]:
        if is_open(2):
            tab_network(wh, ent)
    with tabs[3]:
        if is_open(3):
            tab_timeline(cached_events(path, eid))
    with tabs[4]:
        if is_open(4):
            tab_model(wh)


if __name__ == "__main__":
    main()
