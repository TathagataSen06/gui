"""Graph analytics: centrality, communities, flagged clusters and money cycles."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import duckdb
import networkx as nx
import numpy as np
import pandas as pd

from .build import GraphConfig

ROUND_STEP = 250.0
ROUND_MIN = 5_000.0
SUB_THRESHOLD = (8_000.0, 10_000.0)


def centrality(G: nx.Graph, cfg: GraphConfig) -> pd.DataFrame:
    """Degree, strength, betweenness (k-sampled, distance = 1/weight) and PageRank."""
    H = G.copy()
    for _, _, d in H.edges(data=True):
        d["distance"] = 1.0 / max(d["weight"], 1e-6)
    n = H.number_of_nodes()
    k = min(cfg.betweenness_k, n) if n > 2 else None
    btw = nx.betweenness_centrality(H, k=k, weight="distance", seed=cfg.seed, normalized=True) if n > 2 else {v: 0.0 for v in H}
    pr = nx.pagerank(H, weight="weight") if H.number_of_edges() else {v: 0.0 for v in H}
    clus = nx.clustering(H)
    rows = []
    for v in sorted(H.nodes):
        rows.append(
            {
                "node_id": v,
                "kind": H.nodes[v].get("kind"),
                "label": H.nodes[v].get("label"),
                "degree": H.degree(v),
                "strength": float(H.degree(v, weight="weight")),
                "betweenness": float(btw.get(v, 0.0)),
                "pagerank": float(pr.get(v, 0.0)),
                "clustering": float(clus.get(v, 0.0)),
            }
        )
    return pd.DataFrame(rows)


def investigative_graph(G: nx.Graph, cfg: GraphConfig) -> nx.Graph:
    """The subgraph that shapes communities.

    Louvain on the full relationship graph mostly recovers families, friend
    groups and teams - every ring member also has a normal social life. For
    *money-flow* communities we keep only ties that carry investigative
    signal: substantial money (aggregate >= ``community_min_money``),
    beneficial-ownership control and small-group co-presence. Money edges of
    payroll/biller-like organisations (fan-out above
    ``community_org_fanout_cap``) are excluded so employers do not glue their
    whole workforce into one cluster.
    """
    org_money_deg: dict[str, int] = defaultdict(int)
    for u, v, d in G.edges(data=True):
        if "money" in d["channels"]:
            for x in (u, v):
                if G.nodes[x].get("kind") == "org":
                    org_money_deg[x] += 1
    H = nx.Graph()
    H.add_nodes_from(sorted(G.nodes))
    for u, v, d in sorted(G.edges(data=True), key=lambda e: (e[0], e[1])):
        ch = d["channels"]
        w = 0.0
        m = ch.get("money")
        if m and m["amount"] >= cfg.community_min_money and not any(G.nodes[x].get("kind") == "org" and org_money_deg[x] > cfg.community_org_fanout_cap for x in (u, v)):
            w += m["weight"]
        for c in ("controls", "coloc"):
            if c in ch:
                w += ch[c]["weight"]
        if w > 0:
            H.add_edge(u, v, weight=w)
    return H


def communities(G: nx.Graph, cfg: GraphConfig) -> dict[str, str]:
    """Louvain communities (deterministic given the seed); node -> community id."""
    if G.number_of_edges() == 0:
        return {v: f"C-{i:04d}" for i, v in enumerate(sorted(G))}
    comms = nx.community.louvain_communities(G, weight="weight", resolution=cfg.louvain_resolution, seed=cfg.seed)
    comms = sorted((sorted(c) for c in comms), key=lambda c: (-len(c), c[0]))
    out = {}
    for i, c in enumerate(comms):
        cid = f"C-{i:04d}"
        for v in c:
            out[v] = cid
    return out


# --------------------------------------------------------------------------
# Money cycles
# --------------------------------------------------------------------------


@dataclass
class Transfer:
    event_id: str
    ts: pd.Timestamp
    amount: float
    src: str
    dst: str
    src_entity: str
    dst_entity: str


def load_transfers(con: duckdb.DuckDBPyConnection, min_amount: float) -> list[Transfer]:
    q = """
    SELECT e.event_id, e.ts, CAST(e.attributes->>'amount' AS DOUBLE) AS amount, e.actor_ids[1] AS src, e.actor_ids[2] AS dst,
           coalesce(o1.entity_id, e.actor_ids[1]) AS src_entity, coalesce(o2.entity_id, e.actor_ids[2]) AS dst_entity
    FROM events e
    LEFT JOIN (SELECT identifier, min(entity_id) AS entity_id FROM identifier_entity GROUP BY 1) o1 ON o1.identifier = e.actor_ids[1]
    LEFT JOIN (SELECT identifier, min(entity_id) AS entity_id FROM identifier_entity GROUP BY 1) o2 ON o2.identifier = e.actor_ids[2]
    WHERE e.event_type = 'transfer' AND CAST(e.attributes->>'amount' AS DOUBLE) >= ?
    ORDER BY e.ts, e.event_id
    """
    return [Transfer(*r) for r in con.execute(q, [min_amount]).fetchall()]


def controller_map(con: duckdb.DuckDBPyConnection) -> dict[str, str]:
    """org entity -> controlling person entity (first beneficial owner)."""
    rows = con.execute("SELECT dst_entity, min(src_entity) FROM entity_links WHERE relation = 'controls' GROUP BY 1").fetchall()
    return dict(rows)


def find_money_cycles(transfers: list[Transfer], ctrl: dict[str, str], cfg: GraphConfig) -> list[dict[str, Any]]:
    """Value-preserving temporal cycles on the beneficial-owner-collapsed money graph.

    A cycle ``n1 -> n2 -> ... -> nk -> n1`` counts only if there is a sequence
    of transfers along it with increasing timestamps, each hop within
    ``cycle_max_gap_days`` of the previous, and each amount between
    ``cycle_min_retention`` and ``cycle_max_growth`` times the previous one
    (money moving on, minus fees) - the signature of layering/round-tripping
    rather than of unrelated payments that happen to form a loop.
    """
    def party(e: str) -> str:
        return ctrl.get(e, e)

    by_edge: dict[tuple[str, str], list[Transfer]] = defaultdict(list)
    for t in transfers:
        a, b = party(t.src_entity), party(t.dst_entity)
        if a != b:
            by_edge[(a, b)].append(t)
    D = nx.DiGraph()
    D.add_edges_from(sorted(by_edge))
    max_gap = pd.Timedelta(days=cfg.cycle_max_gap_days)
    found: dict[tuple[str, ...], dict[str, Any]] = {}
    for cyc in nx.simple_cycles(D, length_bound=cfg.cycle_max_len):
        k = len(cyc)
        best = None
        for r in range(k):
            nodes = cyc[r:] + cyc[:r]
            edges = [(nodes[i], nodes[(i + 1) % k]) for i in range(k)]
            for start in by_edge[edges[0]]:
                chain = [start]
                ok = True
                for e in edges[1:]:
                    prev = chain[-1]
                    nxt = next(
                        (
                            t
                            for t in by_edge[e]
                            if prev.ts < t.ts <= prev.ts + max_gap and cfg.cycle_min_retention * prev.amount <= t.amount <= cfg.cycle_max_growth * prev.amount
                        ),
                        None,
                    )
                    if nxt is None:
                        ok = False
                        break
                    chain.append(nxt)
                if ok:
                    best = (nodes, chain)
                    break
            if best:
                break
        if best:
            nodes, chain = best
            key = tuple(sorted(nodes))
            if key in found:
                found[key]["instances"] += 1
                continue
            dur = (chain[-1].ts - chain[0].ts).total_seconds() / 3600
            found[key] = {
                "cycle_id": "CYC-" + hashlib.sha1("|".join(key).encode()).hexdigest()[:10].upper(),
                "length": k,
                "parties": list(nodes),
                "entities_path": [chain[0].src_entity] + [t.dst_entity for t in chain],
                "event_ids": [t.event_id for t in chain],
                "amounts": [round(t.amount, 2) for t in chain],
                "start_ts": chain[0].ts,
                "end_ts": chain[-1].ts,
                "duration_hours": round(dur, 1),
                "value_retention": round(chain[-1].amount / chain[0].amount, 4),
                "instances": 1,
            }
    return sorted(found.values(), key=lambda c: c["cycle_id"])


# --------------------------------------------------------------------------
# Community money-flow profiling
# --------------------------------------------------------------------------


def org_red_flags(con: duckdb.DuckDBPyConnection, young_days: int = 365, min_shared: int = 3) -> tuple[set[str], set[str]]:
    """Org entities incorporated recently, and orgs sharing a registered street address with others.

    Many companies at one street address (with different suites) is the
    classic footprint of a registered-agent / virtual-office shell factory.
    """
    rows = con.execute(
        """
        SELECT re.entity_id, ir.address, ir.attributes->>'incorporation_date' AS inc,
               (SELECT value FROM manifest WHERE key = 'window_start') AS ws
        FROM identity_records ir JOIN record_entity re USING (record_id)
        WHERE ir.entity_type = 'org' AND ir.source = 'bank'
        """
    ).fetchall()
    from ..resolve.normalize import parse_address

    young: set[str] = set()
    by_addr: dict[str, set[str]] = defaultdict(set)
    for eid, addr, inc, ws in rows:
        if inc and ws and (pd.Timestamp(ws) - pd.Timestamp(inc)).days <= young_days:
            young.add(eid)
        a = parse_address(addr)
        if a.key and a.zip:
            by_addr[f"{a.key}|{a.zip}"].add(eid)
    shared = {e for ents in by_addr.values() if len(ents) >= min_shared for e in ents}
    return young, shared


def community_profiles(con: duckdb.DuckDBPyConnection, G: nx.Graph, comm: dict[str, str], cycles: list[dict[str, Any]], z_threshold: float = 3.5) -> pd.DataFrame:
    """Per-community structure and money-flow statistics, with an anomaly flag.

    Flags *tightly connected clusters with unusual money flow*: communities
    whose internal money volume, round-amount share, sub-threshold cash,
    circular flows and multi-channel density are extreme relative to the
    other communities (robust z-scores).
    """
    tr = con.execute(
        """
        SELECT e.event_id, CAST(e.attributes->>'amount' AS DOUBLE) AS amount, e.attributes->>'channel' AS channel,
               coalesce(o1.entity_id, e.actor_ids[1]) AS s, coalesce(o2.entity_id, e.actor_ids[2]) AS d
        FROM events e
        LEFT JOIN (SELECT identifier, min(entity_id) AS entity_id FROM identifier_entity GROUP BY 1) o1 ON o1.identifier = e.actor_ids[1]
        LEFT JOIN (SELECT identifier, min(entity_id) AS entity_id FROM identifier_entity GROUP BY 1) o2 ON o2.identifier = e.actor_ids[2]
        WHERE e.event_type = 'transfer'
        """
    ).df()
    cash = con.execute(
        """
        SELECT ie.entity_id, CAST(e.attributes->>'amount' AS DOUBLE) AS amount
        FROM events e JOIN identifier_entity ie ON ie.identifier = e.actor_ids[1]
        WHERE e.event_type = 'cash_deposit'
        """
    ).df()
    crypto = con.execute(
        """
        SELECT ie.entity_id, sum(CAST(e.attributes->>'amount' AS DOUBLE)) AS amount
        FROM events e JOIN identifier_entity ie ON ie.identifier = e.actor_ids[1]
        WHERE e.event_type = 'card_purchase' AND (e.attributes->>'mcc') = '6051'
        GROUP BY 1
        """
    ).df()
    crypto["c"] = crypto["entity_id"].map(comm)
    crypto_by_c = crypto.groupby("c")["amount"].sum().to_dict()
    tr["cs"] = tr["s"].map(comm)
    tr["cd"] = tr["d"].map(comm)
    internal = tr[(tr["cs"] == tr["cd"]) & tr["cs"].notna() & (tr["s"] != tr["d"])]
    boundary = tr[(tr["cs"] != tr["cd"])]
    cash["c"] = cash["entity_id"].map(comm)
    sub = cash[(cash["amount"] >= SUB_THRESHOLD[0]) & (cash["amount"] < SUB_THRESHOLD[1])]
    cyc_count: dict[str, int] = defaultdict(int)
    for c in cycles:
        cs = {comm.get(p) for p in c["parties"]}
        for x in cs:
            if x:
                cyc_count[x] += 1
    members: dict[str, list[str]] = defaultdict(list)
    for v, c in comm.items():
        members[c].append(v)
    young, shared_addr = org_red_flags(con)
    rows = []
    for cid in sorted(members):
        mem = members[cid]
        sg = G.subgraph(mem)
        persons = [v for v in mem if G.nodes[v].get("kind") == "person"]
        it = internal[internal["cs"] == cid]
        bo = boundary[(boundary["cs"] == cid) | (boundary["cd"] == cid)]
        vol_in = float(it["amount"].sum())
        vol_b = float(bo["amount"].sum())
        big = it[it["amount"] >= ROUND_MIN]
        chans = defaultdict(int)
        for _, _, d in sg.edges(data=True):
            for ch in d["channels"]:
                chans[ch] += 1
        multi = sum(1 for _, _, d in sg.edges(data=True) if len(d["channels"]) >= 2)
        rows.append(
            {
                "community_id": cid,
                "size": len(mem),
                "n_persons": len(persons),
                "n_orgs": sum(1 for v in mem if G.nodes[v].get("kind") == "org"),
                "density": round(nx.density(sg), 4) if len(mem) > 1 else 0.0,
                "edges": sg.number_of_edges(),
                "multichannel_edges": multi,
                "internal_money": round(vol_in, 2),
                "boundary_money": round(vol_b, 2),
                "internal_money_share": round(vol_in / (vol_in + vol_b), 4) if vol_in + vol_b else 0.0,
                "internal_transfers": int(len(it)),
                "internal_wires": int((it["channel"] == "WIRE").sum()),
                "large_internal_transfers": int(len(big)),
                "round_share_large": round(float(((big["amount"] % ROUND_STEP) < 0.005).mean()), 4) if len(big) else 0.0,
                "young_orgs": sum(1 for v in mem if v in young),
                "shared_address_orgs": sum(1 for v in mem if v in shared_addr),
                "sub_threshold_cash": int(sub[sub["c"] == cid].shape[0]),
                "money_cycles": int(cyc_count.get(cid, 0)),
                "crypto_outflow": round(float(crypto_by_c.get(cid, 0.0)), 2),
                "channels": dict(sorted(chans.items())),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    elig = df["size"].between(3, 80) & (df["n_persons"] >= 2)
    # money-flow features (what makes a cluster *unusual*) with per-feature scale floors:
    # most communities have zero cycles / structured cash, so MAD is often 0.
    money_feats = {
        "large_internal_volume": (np.log1p(df["internal_money"].where(df["large_internal_transfers"] > 0, 0.0)), 1.0),
        "internal_wires": (np.log1p(df["internal_wires"]), 0.5),
        "sub_threshold_cash": (np.log1p(df["sub_threshold_cash"]), 0.5),
        "money_cycles": (df["money_cycles"].astype(float), 0.35),
        "round_large_transfers": (df["round_share_large"] * np.log1p(df["large_internal_transfers"]), 0.35),
        "young_or_shared_address_orgs": ((df["young_orgs"] + df["shared_address_orgs"]).astype(float), 0.5),
        "quasi_cash_outflow": (np.log1p(df["crypto_outflow"]), 3.0),
    }
    z = pd.DataFrame(index=df.index)
    for name, (series, floor) in money_feats.items():
        ref = series[elig] if elig.any() else series
        med = float(ref.median())
        mad = float((ref - med).abs().median())
        scale = max(1.4826 * mad, floor)
        z[name] = ((series - med) / scale).clip(lower=0, upper=25)
    df["money_flow_score"] = (z.pow(2).sum(axis=1) ** 0.5).round(3)
    # tightness: share of member pairs connected, and ties spanning several channels
    df["tightness"] = (df["density"] + df["multichannel_edges"] / df["size"].clip(lower=1)).round(4)
    tight_ref = float(df.loc[elig, "tightness"].quantile(0.25)) if elig.any() else 0.0
    df["anomaly_score"] = df["money_flow_score"]
    reasons = []
    for i in df.index:
        reasons.append([f"{k} z={z.loc[i, k]:.1f}" for k in z.columns if z.loc[i, k] >= z_threshold])
    df["reasons"] = [json.dumps(r) for r in reasons]
    df["flagged"] = elig & (df["money_flow_score"] >= z_threshold) & (df["tightness"] >= tight_ref) & pd.Series([len(r) > 0 for r in reasons], index=df.index)
    df["channels"] = df["channels"].map(json.dumps)
    return df.sort_values(["flagged", "anomaly_score"], ascending=[False, False]).reset_index(drop=True)
