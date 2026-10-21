"""Interactive network rendering (pyvis / vis.js) for the operator console."""

from __future__ import annotations

import html
import json

import networkx as nx
from pyvis.network import Network

KIND_STYLE: dict[str, tuple[str, str]] = {
    # kind -> (colour, shape)
    "person": ("#4C9BE8", "dot"),
    "org": ("#B07CF0", "square"),
    "account": ("#7FB77E", "diamond"),
    "external_account": ("#9AA0A6", "diamond"),
    "phone": ("#F2B134", "triangle"),
    "mailbox": ("#5FC9C1", "triangleDown"),
}
CHANNEL_COLOUR: dict[str, str] = {"money": "#E4572E", "call": "#F2B134", "email": "#5FC9C1", "coloc": "#B07CF0", "controls": "#FFFFFF"}
ALERT_BORDER = "#FF3B3B"

VIS_OPTIONS = {
    "physics": {
        "barnesHut": {"gravitationalConstant": -4500, "centralGravity": 0.45, "springLength": 120, "springConstant": 0.035, "damping": 0.55, "avoidOverlap": 0.25},
        "minVelocity": 1.0,
        "stabilization": {"enabled": True, "iterations": 600, "updateInterval": 50, "fit": True},
    },
    "interaction": {"hover": True, "navigationButtons": True, "tooltipDelay": 120, "keyboard": False},
    "edges": {"smooth": False},
}
# Freeze the layout once it has settled, and re-fit whenever the frame changes size: Streamlit
# renders inactive tabs at 0x0, so a fit computed at load time would be wrong once shown.
_FREEZE_JS = """
                  network.once("stabilizationIterationsDone", function () {
                      network.setOptions({physics: {enabled: false}});
                      network.fit({animation: false});
                  });
                  if (window.ResizeObserver) {
                      new ResizeObserver(function () { network.redraw(); network.fit({animation: false}); }).observe(container);
                  }
                  return network;"""


def _dominant(channels: dict) -> str:
    for c in ("money", "controls", "coloc", "call", "email"):
        if c in channels:
            return c
    return next(iter(channels), "call")


def render(G: nx.Graph, focus: str | None = None, height: int = 640, dark: bool = True) -> str:
    """Return a self-contained HTML document for ``G``."""
    net = Network(height=f"{height}px", width="100%", bgcolor="#0E1117" if dark else "#FFFFFF", font_color="#E6E6E6" if dark else "#222222", cdn_resources="in_line")
    net.set_options(json.dumps(VIS_OPTIONS))
    for n, d in G.nodes(data=True):
        colour, shape = KIND_STYLE.get(d.get("kind") or "", ("#CCCCCC", "dot"))
        risk = d.get("risk")
        size = 10 + (risk or 0) * 0.35
        if n == focus:
            size += 10
        label = (d.get("label") or n)[:28]
        tip = f"<b>{html.escape(str(d.get('label') or n))}</b><br>{html.escape(n)}<br>kind: {d.get('kind')}<br>community: {d.get('community_id')}"
        if risk is not None:
            tip += f"<br>risk: {risk:.0f}/100"
        if d.get("reason"):
            tip += f"<br>{html.escape(d['reason'])}"
        net.add_node(
            n,
            label=label,
            title=tip,
            color={"background": colour, "border": ALERT_BORDER if d.get("alert") else colour, "highlight": {"background": "#FFFFFF", "border": ALERT_BORDER}},
            shape=shape,
            size=size,
            borderWidth=4 if d.get("alert") else 1,
            font={"size": 14 if n == focus else 11},
        )
    for u, v, d in G.edges(data=True):
        ch = d.get("channels", {})
        dom = _dominant(ch)
        tip = "<br>".join(f"{c}: {i['count']}x" + (f", ${i['amount']:,.0f}" if c == "money" else "") for c, i in ch.items())
        net.add_edge(u, v, color=CHANNEL_COLOUR.get(dom, "#888888"), width=1 + 2.5 * min(d.get("weight", 1), 3), title=tip, dashes=dom in ("call", "email"))
    return net.generate_html(notebook=False).replace("return network;", _FREEZE_JS, 1)


def legend_html() -> str:
    items = "".join(f'<span style="margin-right:14px"><span style="color:{c}">&#9679;</span> {k}</span>' for k, (c, _) in KIND_STYLE.items())
    edges = "".join(f'<span style="margin-right:14px"><span style="color:{c}">&#9644;</span> {k}</span>' for k, c in CHANNEL_COLOUR.items())
    return f'<div style="font-size:0.85rem">Nodes: {items}<br>Edges: {edges} &nbsp; <span style="color:{ALERT_BORDER}">&#9711;</span> alerted entity</div>'
