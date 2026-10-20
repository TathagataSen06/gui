"""Build the relationship graph from resolved events.

Two views are produced:

* **identifier edges** (``graph_identifier_edges``): account->account transfers,
  phone->phone calls, mailbox->mailbox e-mail, card->reader visits - exactly
  what the feeds observed, aggregated with frequency, volume and recency.
* **entity projection** (a NetworkX ``Graph``): identifiers are collapsed onto
  the person/org entity that owns them (from entity resolution). Identifiers
  nobody owns (external accounts, unregistered numbers) stay as their own
  nodes unless they behave like public hubs (switchboards, utilities), which
  are dropped because they connect everyone to everyone.

Edge weights combine channels (money, call, e-mail, co-location, control) on a
common [0, 1] scale so a tie that spans several channels counts for more than
a heavy tie on one channel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import duckdb
import networkx as nx
import pandas as pd

CHANNELS = ("money", "call", "email", "coloc", "controls")


@dataclass
class GraphConfig:
    half_life_days: float = 45.0
    coloc_window_min: int = 15
    coloc_bucket_min: int = 30
    coloc_max_occupancy: int = 6
    coloc_excluded_zones: tuple[str, ...] = ("TRANSIT",)
    hub_degree_cap: int = 25
    min_cycle_amount: float = 1_000.0
    cycle_max_len: int = 6
    cycle_max_gap_days: float = 10.0
    cycle_min_retention: float = 0.80
    cycle_max_growth: float = 1.02
    betweenness_k: int = 64  # sampled sources for approximate betweenness (exact is O(VE))
    louvain_resolution: float = 2.0
    community_min_money: float = 2_000.0  # aggregate tie volume for a money edge to shape communities
    community_org_fanout_cap: int = 25  # payroll/biller-like orgs are excluded from community structure
    seed: int = 0
    timezone: str = "America/New_York"
    reference_epoch: float | None = None  # "now" for recency weighting; defaults to the last event


def _decay_sql(cfg: GraphConfig, ts: str = "e.ts") -> str:
    """Exponential recency decay relative to the end of the observation window.

    ``cfg.reference_epoch`` is resolved once per run (see :func:`resolve_reference`)
    and inlined as a literal: a scalar sub-query here would be re-evaluated per row.
    """
    lam = math.log(2) / (cfg.half_life_days * 86400.0)
    return f"exp(-{lam} * ({cfg.reference_epoch} - epoch({ts})))"


def resolve_reference(con: duckdb.DuckDBPyConnection, cfg: GraphConfig) -> None:
    if cfg.reference_epoch is None:
        cfg.reference_epoch = float(con.execute("SELECT epoch(max(ts)) FROM events").fetchone()[0] or 0.0)


def identifier_edges(con: duckdb.DuckDBPyConnection, cfg: GraphConfig) -> pd.DataFrame:
    """Aggregate observed interactions between identifiers (directed), one query per channel."""
    resolve_reference(con, cfg)
    decay = _decay_sql(cfg)
    amount = "CAST(e.attributes->>'amount' AS DOUBLE)"
    queries = {
        "money": f"""
            SELECT e.actor_ids[1] AS src, e.actor_ids[2] AS dst, count(*) AS n, sum({amount}) AS amount,
                   sum({decay}) AS recency, sum({amount} * {decay}) AS amount_recency, min(e.ts) AS first_ts, max(e.ts) AS last_ts
            FROM events e WHERE e.event_type = 'transfer' GROUP BY 1, 2""",
        "call": f"""
            SELECT e.actor_ids[1] AS src, e.actor_ids[2] AS dst, count(*) AS n, sum(CAST(e.attributes->>'duration_sec' AS DOUBLE)) AS amount,
                   sum({decay}) AS recency, NULL::DOUBLE AS amount_recency, min(e.ts) AS first_ts, max(e.ts) AS last_ts
            FROM events e WHERE e.event_type IN ('call', 'sms') GROUP BY 1, 2""",
        "visit": f"""
            SELECT e.actor_ids[1] AS src, e.actor_ids[2] AS dst, count(*) AS n,
                   sum(CASE WHEN e.event_type = 'access_denied' THEN 1 ELSE 0 END)::DOUBLE AS amount,
                   sum({decay}) AS recency, NULL::DOUBLE AS amount_recency, min(e.ts) AS first_ts, max(e.ts) AS last_ts
            FROM events e WHERE e.event_type IN ('access_granted', 'access_denied') GROUP BY 1, 2""",
    }
    frames = []
    for channel, q in queries.items():
        df = con.execute(q).df()
        df.insert(2, "channel", channel)
        frames.append(df)
    # e-mail fans out to every recipient; materialize the UNNEST before aggregating
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE _mail AS
        SELECT e.actor_ids[1] AS src, UNNEST(e.actor_ids[2:]) AS dst, e.ts,
               coalesce(CAST(e.attributes->>'attachment_bytes' AS DOUBLE), 0) AS bytes
        FROM events e
        WHERE e.event_type = 'email' AND coalesce(CAST(e.attributes->>'bulk' AS BOOLEAN), false) = false
        """
    )
    mail = con.execute(
        f"""
        SELECT src, dst, count(*) AS n, sum(bytes) AS amount, sum({_decay_sql(cfg, 'ts')}) AS recency,
               NULL::DOUBLE AS amount_recency, min(ts) AS first_ts, max(ts) AS last_ts
        FROM _mail GROUP BY 1, 2
        """
    ).df()
    mail.insert(2, "channel", "email")
    frames.append(mail)
    df = pd.concat(frames, ignore_index=True)
    df.columns = ["src", "dst", "channel", "count", "amount", "recency", "amount_recency", "first_ts", "last_ts"]
    return df.sort_values(["channel", "src", "dst"]).reset_index(drop=True)


def colocation_edges(con: duckdb.DuckDBPyConnection, cfg: GraphConfig) -> pd.DataFrame:
    """Small-group co-presence: two entities badging at the same low-traffic reader within minutes.

    High-traffic contexts (turnstiles, any reader-bucket with more than
    ``coloc_max_occupancy`` distinct people) are excluded; each co-presence is
    weighted by ``1/(occupancy-1)`` so a two-person meeting counts more than a
    six-person one.
    """
    resolve_reference(con, cfg)
    excluded = ", ".join(f"'{z}'" for z in cfg.coloc_excluded_zones) or "''"
    decay = _decay_sql(cfg, "l1.ts")
    q = f"""
    WITH sw AS (
        SELECT e.ts, ie.entity_id, e.attributes->>'reader_id' AS reader, e.attributes->>'zone' AS zone,
               e.attributes->>'site_type' AS site_type,
               hour(timezone('{cfg.timezone}', e.ts AT TIME ZONE 'UTC')) AS local_hour
        FROM events e JOIN identifier_entity ie ON ie.identifier = e.actor_ids[1]
        WHERE e.event_type = 'access_granted' AND (e.attributes->>'zone') NOT IN ({excluded})
    ),
    b AS (SELECT *, time_bucket(INTERVAL {cfg.coloc_bucket_min} MINUTE, ts) AS bkt FROM sw),
    occ AS (SELECT reader, bkt, count(DISTINCT entity_id) AS occupancy FROM b GROUP BY 1, 2),
    lo AS (SELECT b.*, occ.occupancy FROM b JOIN occ USING (reader, bkt) WHERE occ.occupancy BETWEEN 2 AND {cfg.coloc_max_occupancy})
    SELECT l1.entity_id AS a, l2.entity_id AS b, count(*) AS n,
           sum({decay} / (l1.occupancy - 1)) AS recency,
           sum(CASE WHEN l1.local_hour >= 20 OR l1.local_hour < 6 THEN 1 ELSE 0 END) AS after_hours,
           min(l1.ts) AS first_ts, max(l1.ts) AS last_ts,
           list(DISTINCT l1.reader ORDER BY l1.reader) AS readers
    FROM lo l1 JOIN lo l2
      ON l1.reader = l2.reader AND l1.bkt = l2.bkt AND l1.entity_id < l2.entity_id
     AND abs(epoch(l1.ts) - epoch(l2.ts)) <= {cfg.coloc_window_min * 60}
    GROUP BY 1, 2
    ORDER BY 1, 2
    """
    return con.execute(q).df()


def build_projection(con: duckdb.DuckDBPyConnection, cfg: GraphConfig) -> tuple[nx.Graph, pd.DataFrame, pd.DataFrame, set[str]]:
    """Return (entity graph, identifier edges, entity-level channel edges, hub identifiers)."""
    idf = identifier_edges(con, cfg)
    ents = con.execute("SELECT entity_id, entity_type, display_name, n_records, sources FROM entities").df().set_index("entity_id")

    # hubs: unowned identifiers with too many distinct counterparties (service lines, billers)
    con.register("_idf", idf)
    con.execute("CREATE OR REPLACE TEMP TABLE _own AS SELECT identifier, min(entity_id) AS entity_id FROM identifier_entity GROUP BY 1")
    hubs = {
        r[0]
        for r in con.execute(
            f"""
            WITH d AS (SELECT src AS x, dst AS y FROM _idf WHERE channel <> 'visit' UNION SELECT dst, src FROM _idf WHERE channel <> 'visit')
            SELECT x FROM d WHERE x NOT IN (SELECT identifier FROM _own) GROUP BY x HAVING count(DISTINCT y) > {int(cfg.hub_degree_cap)}
            """
        ).fetchall()
    }
    con.register("_hubs", pd.DataFrame({"identifier": sorted(hubs)}, dtype=object))
    agg = con.execute(
        """
        WITH m AS (
            SELECT coalesce(o1.entity_id, i.src) AS u, coalesce(o2.entity_id, i.dst) AS v, i.*
            FROM _idf i
            LEFT JOIN _own o1 ON o1.identifier = i.src
            LEFT JOIN _own o2 ON o2.identifier = i.dst
            WHERE i.channel IN ('money', 'call', 'email')
              AND i.src NOT IN (SELECT identifier FROM _hubs) AND i.dst NOT IN (SELECT identifier FROM _hubs)
              AND NOT starts_with(i.src, 'merchant:') AND NOT starts_with(i.dst, 'merchant:')
        )
        SELECT least(u, v) AS a, greatest(u, v) AS b, channel, sum(count) AS count, sum(amount) AS amount,
               sum(recency) AS recency, sum(amount_recency) AS amount_recency, min(first_ts) AS first_ts, max(last_ts) AS last_ts
        FROM m WHERE u <> v GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
        """
    ).df()
    con.unregister("_idf")
    con.unregister("_hubs")
    co = colocation_edges(con, cfg)
    if not co.empty:
        co_e = pd.DataFrame({"a": co["a"], "b": co["b"], "channel": "coloc", "count": co["n"], "amount": co["after_hours"].astype(float), "recency": co["recency"], "amount_recency": None, "first_ts": co["first_ts"], "last_ts": co["last_ts"]})
        agg = pd.concat([agg, co_e], ignore_index=True)
    links = con.execute("SELECT src_entity, dst_entity FROM entity_links WHERE relation = 'controls'").df()
    if not links.empty:
        a = links[["src_entity", "dst_entity"]].min(axis=1)
        b = links[["src_entity", "dst_entity"]].max(axis=1)
        agg = pd.concat([agg, pd.DataFrame({"a": a, "b": b, "channel": "controls", "count": 1, "amount": 0.0, "recency": 1.0, "amount_recency": None, "first_ts": None, "last_ts": None})], ignore_index=True)

    # channel weight on [0, 1]: log-scaled relative to the channel's 99th percentile
    def scale(df: pd.DataFrame) -> pd.Series:
        base = df["amount_recency"].fillna(df["recency"]) if df["channel"].iloc[0] == "money" else df["recency"]
        base = base.fillna(0).clip(lower=0)
        ref = max(float(base.quantile(0.99)), 1e-9)
        return (base.map(math.log1p) / math.log1p(ref)).clip(upper=1.0)

    agg["weight"] = 0.0
    for ch, idx in agg.groupby("channel").groups.items():
        agg.loc[idx, "weight"] = 1.0 if ch == "controls" else scale(agg.loc[idx]).values
    agg = agg.sort_values(["a", "b", "channel"]).reset_index(drop=True)

    G = nx.Graph()
    nodes = sorted(set(agg["a"]) | set(agg["b"]) | set(ents.index))
    for n in nodes:
        if n in ents.index:
            e = ents.loc[n]
            G.add_node(n, kind=e["entity_type"], label=e["display_name"], n_records=int(e["n_records"]))
        else:
            kind = n.split(":", 1)[0]
            G.add_node(n, kind={"acct": "account", "ext": "external_account", "msisdn": "phone", "email": "mailbox"}.get(kind, kind), label=n.split(":", 1)[1], n_records=0)
    edges: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for a, b, ch, cnt, amt, wgt in zip(agg["a"], agg["b"], agg["channel"], agg["count"], agg["amount"], agg["weight"]):
        edges.setdefault((a, b), {})[ch] = {"count": int(cnt), "amount": float(amt) if amt == amt and amt is not None else 0.0, "weight": float(wgt)}
    for (a, b), chans in edges.items():
        G.add_edge(a, b, weight=float(sum(c["weight"] for c in chans.values())), channels=chans)
    return G, idf, agg, hubs
