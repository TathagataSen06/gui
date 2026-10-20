"""Read-only data access for the operator console (no Streamlit dependency)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import networkx as nx
import pandas as pd

from ..db import connect
from ..paths import RunPaths


class Warehouse:
    """Thin query layer over a scored ``argus.duckdb``."""

    def __init__(self, root: str | Path) -> None:
        self.paths = RunPaths.at(root)
        if not self.paths.db.exists():
            raise FileNotFoundError(f"{self.paths.db} not found - run `python -m argus run --data {root}` first")
        self.con: duckdb.DuckDBPyConnection = connect(self.paths.db, read_only=True)
        self.tz = (self.q("SELECT value FROM manifest WHERE key = 'timezone'").iloc[0, 0]) if self.has("manifest") else "UTC"

    # ---- helpers ---------------------------------------------------------
    def q(self, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
        return self.con.execute(sql, params or []).df()

    def has(self, table: str) -> bool:
        return bool(self.con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [table]).fetchone()[0])

    def meta(self) -> dict[str, Any]:
        if not self.has("run_meta"):
            return {}
        return {s: json.loads(p) for s, p in self.con.execute("SELECT stage, payload FROM run_meta").fetchall()}

    # ---- overview --------------------------------------------------------
    def kpis(self) -> dict[str, Any]:
        one = lambda sql: self.con.execute(sql).fetchone()[0]  # noqa: E731
        return {
            "events": one("SELECT count(*) FROM events"),
            "records": one("SELECT count(*) FROM identity_records"),
            "entities": one("SELECT count(*) FROM entities"),
            "alerts": one("SELECT count(*) FROM risk_scores WHERE alert"),
            "flagged_communities": one("SELECT count(*) FROM communities WHERE flagged"),
            "money_cycles": one("SELECT count(*) FROM money_cycles"),
            "sources": self.q("SELECT source, count(*) AS events FROM events GROUP BY 1 ORDER BY 2 DESC"),
            "window": self.q("SELECT min(ts) AS first, max(ts) AS last FROM events").iloc[0].to_dict(),
        }

    def alert_queue(self) -> pd.DataFrame:
        df = self.q(
            """
            SELECT r.rank, r.entity_id, r.display_name, r.entity_type, r.risk_score, r.rule_score, r.ml_score, r.alert,
                   r.top_reason, r.n_records, r.sources, n.community_id, json_array_length(r.rule_hits) AS rules_fired
            FROM risk_scores r LEFT JOIN graph_nodes n ON n.node_id = r.entity_id
            ORDER BY r.rank
            """
        )
        df["sources"] = df["sources"].map(lambda s: ", ".join(s) if s is not None else "")
        return df

    # ---- entity ----------------------------------------------------------
    def entity(self, eid: str) -> dict[str, Any]:
        r = self.q("SELECT * FROM risk_scores WHERE entity_id = ?", [eid])
        if r.empty:
            raise KeyError(eid)
        row = r.iloc[0].to_dict()
        row["rule_hits"] = json.loads(row["rule_hits"])
        row["ml_drivers"] = json.loads(row["ml_drivers"])
        e = self.q("SELECT dob, address FROM entities WHERE entity_id = ?", [eid]).iloc[0].to_dict()
        row.update(e)
        g = self.q("SELECT degree, strength, betweenness, pagerank, community_id FROM graph_nodes WHERE node_id = ?", [eid])
        row.update(g.iloc[0].to_dict() if not g.empty else {})
        return row

    def records(self, eid: str) -> pd.DataFrame:
        return self.q(
            """
            SELECT ir.source, ir.record_id, ir.name, ir.dob, ir.address, ir.phone, ir.email, ir.owned_ids
            FROM record_entity re JOIN identity_records ir USING (record_id)
            WHERE re.entity_id = ? ORDER BY ir.source, ir.record_id
            """,
            [eid],
        )

    def match_evidence(self, eid: str) -> pd.DataFrame:
        """Why the entity's records were linked (pairwise Fellegi-Sunter evidence)."""
        df = self.q(
            """
            SELECT p.record_a, p.record_b, p.match_weight, p.match_probability, p.linked, p.evidence
            FROM er_pairs p
            JOIN record_entity a ON a.record_id = p.record_a
            JOIN record_entity b ON b.record_id = p.record_b
            WHERE a.entity_id = ? AND b.entity_id = ? AND p.linked
            ORDER BY p.match_probability DESC, p.match_weight DESC
            """,
            [eid, eid],
        )
        if df.empty:
            return df
        ev = df["evidence"].map(json.loads)
        df["rule"] = ev.map(lambda e: e.get("rule") or "")
        df["agreements"] = ev.map(lambda e: ", ".join(f"{k}={v}" for k, v in e["levels"].items() if v not in ("null", "ok")))
        df["top_evidence"] = ev.map(lambda e: ", ".join(f"{k} {w:+.1f}" for k, w in sorted(e["weights"].items(), key=lambda kv: -abs(kv[1]))[:4]))
        df["name_jw"] = ev.map(lambda e: e.get("name_jw"))
        return df.drop(columns=["evidence"])

    def identifiers(self, eid: str) -> list[str]:
        return [r[0] for r in self.con.execute("SELECT identifier FROM identifier_entity WHERE entity_id = ? ORDER BY 1", [eid]).fetchall()]

    def names_for(self, identifiers: list[str]) -> dict[str, str]:
        if not identifiers:
            return {}
        df = self.q(
            "SELECT ie.identifier, e.display_name, e.entity_id FROM identifier_entity ie JOIN entities e USING (entity_id) WHERE ie.identifier IN (SELECT UNNEST(?))",
            [identifiers],
        )
        return dict(zip(df["identifier"], df["display_name"]))

    def events(self, eid: str, limit: int = 20_000) -> pd.DataFrame:
        """All events touching the entity's identifiers, with readable descriptions."""
        ids = self.identifiers(eid)
        if not ids:
            return pd.DataFrame(columns=["ts", "source", "event_type", "channel", "direction", "amount", "counterparty", "description"])
        df = self.q(
            f"""
            SELECT DISTINCT e.event_id, e.ts, e.source, e.event_type, e.actor_ids, e.actor_roles, e.attributes
            FROM event_actors a JOIN events e USING (event_id)
            WHERE a.actor_id IN (SELECT UNNEST(?))
            ORDER BY e.ts LIMIT {int(limit)}
            """,
            [ids],
        )
        if df.empty:
            return df
        mine = set(ids)
        others = sorted({x for lst in df["actor_ids"] for x in lst if x not in mine})
        names = self.names_for(others)
        rows = [describe(r, mine, names) for r in df.itertuples(index=False)]
        out = pd.DataFrame(rows)
        out["ts_local"] = pd.to_datetime(out["ts"]).dt.tz_localize("UTC").dt.tz_convert(self.tz)
        return out

    def transfers(self, eid: str, min_amount: float = 0.0) -> pd.DataFrame:
        ev = self.events(eid)
        if ev.empty:
            return ev
        t = ev[ev["event_type"].isin(["transfer", "cash_deposit", "cash_withdrawal"]) & (ev["amount"] >= min_amount)]
        return t.sort_values("amount", ascending=False)

    def contacts(self, eid: str) -> pd.DataFrame:
        ev = self.events(eid)
        if ev.empty:
            return ev
        c = ev[ev["event_type"].isin(["call", "sms", "email"])]
        if c.empty:
            return c
        return (
            c.groupby(["counterparty", "event_type"]).agg(count=("ts", "size"), first=("ts_local", "min"), last=("ts_local", "max")).reset_index().sort_values("count", ascending=False)
        )

    def locations(self, eid: str) -> pd.DataFrame:
        ev = self.events(eid)
        if ev.empty or not self.has("ref_readers"):
            return pd.DataFrame()
        a = ev[ev["event_type"].isin(["access_granted", "access_denied"])]
        if a.empty:
            return pd.DataFrame()
        a = a.assign(reader_id=a["counterparty_id"].str.replace("loc:", "", regex=False))
        rd = self.q("SELECT reader_id, reader_name, site_name, site_type, zone, lat, lon FROM ref_readers")
        m = a.merge(rd, on="reader_id", how="left")
        m["after_hours"] = m["ts_local"].dt.hour.map(lambda h: h >= 20 or h < 6)
        return m.groupby(["site_name", "reader_name", "zone", "lat", "lon"]).agg(visits=("ts", "size"), denied=("event_type", lambda s: int((s == "access_denied").sum())), after_hours=("after_hours", "sum")).reset_index().sort_values("visits", ascending=False)

    def cycles(self, eid: str) -> pd.DataFrame:
        df = self.q("SELECT * FROM money_cycles WHERE list_contains(parties, ?) OR list_contains(entities_path, ?)", [eid, eid])
        if df.empty:
            return df
        names = dict(self.con.execute("SELECT entity_id, display_name FROM entities").fetchall())
        df["path"] = df["entities_path"].map(lambda p: "  ->  ".join(names.get(x, x) for x in p))
        df["amounts"] = df["amounts"].map(lambda a: " / ".join(f"${x:,.0f}" for x in a))
        return df[["cycle_id", "length", "path", "amounts", "start_ts", "end_ts", "duration_hours", "value_retention"]]

    # ---- graph -----------------------------------------------------------
    def community(self, cid: str) -> dict[str, Any]:
        df = self.q("SELECT * FROM communities WHERE community_id = ?", [cid])
        return df.iloc[0].to_dict() if not df.empty else {}

    def subgraph(self, eid: str, mode: str = "ego", radius: int = 1, max_nodes: int = 60) -> nx.Graph:
        """Ego network (strongest ties first) or the entity's community."""
        if mode == "community":
            cid = self.con.execute("SELECT community_id FROM graph_nodes WHERE node_id = ?", [eid]).fetchone()
            nodes = [r[0] for r in self.con.execute("SELECT node_id FROM graph_nodes WHERE community_id = ?", [cid[0] if cid else None]).fetchall()]
            nodes = nodes[: max_nodes * 2]
        else:
            nodes = [eid]
            frontier = [eid]
            for _ in range(radius):
                nxt: list[str] = []
                for n in frontier:
                    nb = self.q("SELECT CASE WHEN src = ? THEN dst ELSE src END AS nb, sum(weight) AS w FROM graph_edges WHERE src = ? OR dst = ? GROUP BY 1 ORDER BY 2 DESC LIMIT 25", [n, n, n])
                    for x in nb["nb"]:
                        if x not in nodes:
                            nodes.append(x)
                            nxt.append(x)
                frontier = nxt
        nodes = nodes[:max_nodes]
        G = nx.Graph()
        if not nodes:
            return G
        info = self.q(
            """
            SELECT n.node_id, n.kind, n.label, n.community_id, n.degree, r.risk_score, r.alert, r.top_reason
            FROM graph_nodes n LEFT JOIN risk_scores r ON r.entity_id = n.node_id
            WHERE n.node_id IN (SELECT UNNEST(?))
            """,
            [nodes],
        )
        for row in info.itertuples(index=False):
            # accounts / external parties have no risk score: nullable columns hold NA
            G.add_node(
                row.node_id,
                kind=row.kind,
                label=row.label,
                community_id=row.community_id,
                degree=row.degree,
                risk=None if pd.isna(row.risk_score) else float(row.risk_score),
                alert=False if pd.isna(row.alert) else bool(row.alert),
                reason="" if pd.isna(row.top_reason) else str(row.top_reason),
            )
        edges = self.q("SELECT src, dst, channel, count, amount, weight FROM graph_edges WHERE src IN (SELECT UNNEST(?)) AND dst IN (SELECT UNNEST(?))", [nodes, nodes])
        for r in edges.itertuples(index=False):
            if G.has_edge(r.src, r.dst):
                G[r.src][r.dst]["channels"][r.channel] = {"count": int(r.count), "amount": float(r.amount or 0)}
                G[r.src][r.dst]["weight"] += float(r.weight)
            else:
                G.add_edge(r.src, r.dst, weight=float(r.weight), channels={r.channel: {"count": int(r.count), "amount": float(r.amount or 0)}})
        return G

    # ---- model / reports -------------------------------------------------
    def importances(self) -> pd.DataFrame:
        return self.q("SELECT * FROM model_importances ORDER BY importance DESC") if self.has("model_importances") else pd.DataFrame()

    def evaluation_markdown(self) -> str | None:
        p = self.paths.reports / "evaluation.md"
        return p.read_text() if p.exists() else None

    # ---- dispositions ------------------------------------------------------
    def actions(self, eid: str | None = None) -> pd.DataFrame:
        p = self.paths.case_actions
        if not p.exists():
            return pd.DataFrame(columns=["at", "entity_id", "analyst", "disposition", "note", "risk_score"])
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        df = pd.DataFrame(rows)
        return df[df["entity_id"] == eid] if eid else df

    def record_action(self, eid: str, analyst: str, disposition: str, note: str, risk: float) -> None:
        rec = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "entity_id": eid, "analyst": analyst or "analyst", "disposition": disposition, "note": note, "risk_score": risk}
        with self.paths.case_actions.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")


def _money(x: float) -> str:
    return f"${x:,.2f}"


def describe(r: Any, mine: set[str], names: dict[str, str]) -> dict[str, Any]:
    """Turn a raw event row into a timeline-friendly description relative to the subject."""
    attrs = json.loads(r.attributes) if r.attributes else {}
    actors = list(r.actor_ids)
    others = [a for a in actors if a not in mine]
    cp_id = others[0] if others else (actors[1] if len(actors) > 1 else "")
    cp = names.get(cp_id) or attrs.get("payee_name") or attrs.get("payer_name") or attrs.get("merchant_name") or cp_id.split(":", 1)[-1]
    amount = float(attrs.get("amount", 0) or 0)
    et = r.event_type
    direction = ""
    if et == "transfer":
        outgoing = actors[0] in mine
        direction = "out" if outgoing else "in"
        ch = attrs.get("channel", "")
        desc = f"{ch} {_money(amount)} {'to' if outgoing else 'from'} {cp}"
        if attrs.get("narrative"):
            desc += f" - {attrs['narrative']}"
        channel = f"money {direction}"
    elif et == "card_purchase":
        direction = "out"
        desc = f"Card {_money(amount)} at {attrs.get('merchant_name', cp)} (MCC {attrs.get('mcc', '')})"
        channel = "card"
    elif et == "cash_deposit":
        direction = "in"
        desc = f"Cash deposit {_money(amount)} at {attrs.get('branch_id', '')}"
        channel = "cash"
    elif et == "cash_withdrawal":
        direction = "out"
        desc = f"Cash withdrawal {_money(amount)} at {attrs.get('branch_id', '')}"
        channel = "cash"
    elif et in ("call", "sms"):
        outgoing = actors[0] in mine
        direction = "out" if outgoing else "in"
        dur = int(attrs.get("duration_sec", 0) or 0)
        what = "SMS" if et == "sms" else f"Call {dur // 60}m{dur % 60:02d}s ({attrs.get('call_result', '').lower()})"
        desc = f"{what} {'to' if outgoing else 'from'} {cp} [{cp_id.split(':', 1)[-1]}]"
        channel = "phone"
    elif et == "email":
        outgoing = actors[0] in mine
        direction = "out" if outgoing else "in"
        n_att = int(attrs.get("n_attachments", 0) or 0)
        mb = float(attrs.get("attachment_bytes", 0) or 0) / 1e6
        desc = f"E-mail {'to' if outgoing else 'from'} {cp}: \"{attrs.get('subject') or '(no subject)'}\"" + (f" + {n_att} attachment(s), {mb:.1f} MB" if n_att else "")
        channel = "email"
        amount = mb
    else:
        granted = et == "access_granted"
        desc = f"Badge {'GRANTED' if granted else 'DENIED'} {attrs.get('site_name', '')} / {attrs.get('reader_id', '').split('-', 2)[-1]} ({attrs.get('zone', '')})" + ("" if granted else f" - {attrs.get('deny_reason', '')}")
        channel = "access"
        cp_id = actors[1] if len(actors) > 1 else ""
    return {"event_id": r.event_id, "ts": r.ts, "source": r.source, "event_type": et, "channel": channel, "direction": direction, "amount": amount, "counterparty": cp, "counterparty_id": cp_id, "description": desc}
