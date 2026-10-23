"""Entity-level risk features computed from the resolved warehouse.

Every feature is defined on *resolved entities*, so behaviour spread across
several accounts, phone lines and mailboxes of one person is added up - the
main payoff of entity resolution. Features are grouped as:

money          velocity, fan-in/out, pass-through, round amounts, structuring,
               wires, quasi-cash, circular flows, young-account inflows
communications contact breadth, contact with watch-listed / burner lines,
               calls shortly before large money movements, night activity
location       after-hours restricted-zone access, denials, after-hours meetings
digital        large attachments sent to external / own personal mailboxes
graph          centrality, community money-flow flags, shell-company control
"""

from __future__ import annotations

import bisect
from collections import defaultdict, deque
from dataclasses import dataclass

import duckdb
import pandas as pd

from ..graph.analytics import org_red_flags

REGULAR_INCOME_MARKERS = ("PAYROLL", "SOC SEC", "UI BENEFIT", "CARD SETTLEMENT")


@dataclass
class FeatureConfig:
    sub_threshold: tuple[float, float] = (8_000.0, 10_000.0)
    large_amount: float = 5_000.0
    round_step: float = 250.0
    pass_through_hours: float = 72.0
    coordination_hours: float = 4.0
    exfil_bytes: float = 5_000_000.0
    young_account_days: int = 90


FEATURE_DESCRIPTIONS: dict[str, str] = {
    "inflow_total": "Total incoming transfers (USD)",
    "outflow_total": "Total outgoing transfers (USD)",
    "fan_in": "Distinct non-payroll payers",
    "fan_out": "Distinct payees",
    "txn_velocity": "Outgoing non-card transfers per active day",
    "pass_through_ratio": "Share of non-payroll inflow sent onward within 72h",
    "pass_through_amount": "Inflow sent onward within 72h (USD)",
    "large_round_transfers": "Transfers >= $5k that are multiples of $250",
    "round_amount_share": "Share of transfers >= $1k that are multiples of $250",
    "sub_threshold_cash": "Cash deposits between $8k and $10k",
    "sub_threshold_branches": "Distinct branches used for $8k-$10k cash deposits",
    "cash_deposit_total": "Total cash deposited (USD)",
    "wire_count": "Wire transfers sent or received",
    "wire_volume": "Wire volume sent or received (USD)",
    "quasi_cash_spend": "Crypto / quasi-cash card purchases (USD)",
    "young_account_inflow": "Inflow into accounts opened within 90 days of the window (USD)",
    "money_cycles": "Value-preserving circular payment paths involved in",
    "cycle_volume": "Amount entering those cycles (USD)",
    "controls_red_flag_orgs": "Controlled orgs that are newly incorporated or share a registered address",
    "call_partners": "Distinct call/SMS counterparties",
    "calls_with_watchlisted": "Calls/SMS with watch-listed entities",
    "calls_with_flagged": "Calls/SMS with entities scoring high on rules (2-pass)",
    "burner_contacts": "Calls/SMS with unregistered prepaid lines",
    "pre_txn_coordination": "Large transfers preceded (<4h) by a call between the parties",
    "pre_cash_contacts": "Large cash deposits preceded (<4h) by a call with a prepaid line",
    "night_comm_share": "Share of calls/SMS between 22:00 and 06:00",
    "after_hours_restricted": "After-hours accesses to restricted zones",
    "access_denied": "Denied badge swipes",
    "after_hours_coloc": "After-hours small-group co-locations",
    "exfil_emails": "E-mails with >= 5 MB attachments to other domains",
    "degree": "Graph degree",
    "strength": "Weighted graph degree",
    "betweenness": "Betweenness centrality",
    "pagerank": "PageRank",
    "community_flagged": "Member of a flagged money-flow community",
    "community_score": "Money-flow anomaly score of the entity's community",
    "watchlist_hit": "Resolved to an internal watch-list entry",
    "is_org": "Entity is an organisation",
}


def _own(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE OR REPLACE TEMP TABLE _own AS SELECT identifier, min(entity_id) AS entity_id FROM identifier_entity GROUP BY 1")


def _tz(con: duckdb.DuckDBPyConnection) -> str:
    r = con.execute("SELECT value FROM manifest WHERE key = 'timezone'").fetchone()
    return r[0] if r else "America/New_York"


def transfers(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    _own(con)
    return con.execute(
        """
        SELECT e.event_id, e.ts, CAST(e.attributes->>'amount' AS DOUBLE) AS amount, e.attributes->>'channel' AS channel,
               e.actor_ids[1] AS src_id, e.actor_ids[2] AS dst_id,
               coalesce(o1.entity_id, e.actor_ids[1]) AS src, coalesce(o2.entity_id, e.actor_ids[2]) AS dst,
               upper(coalesce(e.attributes->>'narrative', '') || ' ' || coalesce(e.attributes->>'narrative_in', '')) AS narrative,
               coalesce(e.attributes->>'payer_name', '') AS payer_name,
               coalesce(e.attributes->>'payee_name', '') AS payee_name
        FROM events e
        LEFT JOIN _own o1 ON o1.identifier = e.actor_ids[1]
        LEFT JOIN _own o2 ON o2.identifier = e.actor_ids[2]
        WHERE e.event_type = 'transfer'
        ORDER BY e.ts, e.event_id
        """
    ).df()


def _regular(narr: pd.Series, payer: pd.Series) -> pd.Series:
    text = narr.fillna("") + " " + payer.fillna("")
    mask = pd.Series(False, index=narr.index)
    for m in REGULAR_INCOME_MARKERS:
        mask |= text.str.contains(m, regex=False)
    return mask | text.str.contains("TREAS", regex=False)


def money_features(con: duckdb.DuckDBPyConnection, cfg: FeatureConfig) -> pd.DataFrame:
    tx = transfers(con)
    tx = tx[tx["src"] != tx["dst"]].copy()
    tx["regular"] = _regular(tx["narrative"], tx["payer_name"])
    tx["is_round"] = (tx["amount"] % cfg.round_step).abs() < 0.005
    out = pd.DataFrame()
    inn = tx.groupby("dst").agg(inflow_total=("amount", "sum"), n_in=("amount", "size"))
    fan_in = tx[~tx["regular"]].groupby("dst")["src"].nunique().rename("fan_in")
    outg = tx.groupby("src").agg(outflow_total=("amount", "sum"), n_out=("amount", "size"), fan_out=("dst", "nunique"))
    days_active = pd.concat([tx[["src", "ts"]].rename(columns={"src": "e"}), tx[["dst", "ts"]].rename(columns={"dst": "e"})])
    days_active["d"] = days_active["ts"].dt.date
    active = days_active.groupby("e")["d"].nunique().rename("active_days")
    both = pd.concat([tx[["src", "amount", "is_round", "channel"]].rename(columns={"src": "e"}), tx[["dst", "amount", "is_round", "channel"]].rename(columns={"dst": "e"})])
    large = both[both["amount"] >= cfg.large_amount]
    k1 = both[both["amount"] >= 1_000]
    lr = large[large["is_round"]].groupby("e").size().rename("large_round_transfers")
    rs = k1.groupby("e")["is_round"].mean().rename("round_amount_share")
    wires = both[both["channel"] == "WIRE"].groupby("e").agg(wire_count=("amount", "size"), wire_volume=("amount", "sum"))
    out = pd.concat([inn, fan_in, outg, active, lr, rs, wires], axis=1)
    out["txn_velocity"] = out["n_out"].fillna(0) / out["active_days"].clip(lower=1)
    # pass-through: non-regular inflow forwarded onward within the horizon (FIFO matching)
    pt_amt: dict[str, float] = {}
    pt_ratio: dict[str, float] = {}
    horizon = pd.Timedelta(hours=cfg.pass_through_hours)
    card = con.execute(
        """
        SELECT o.entity_id AS src, e.ts, CAST(e.attributes->>'amount' AS DOUBLE) AS amount
        FROM events e JOIN _own o ON o.identifier = e.actor_ids[1]
        WHERE (e.event_type = 'card_purchase' AND (e.attributes->>'mcc') = '6051') OR e.event_type = 'cash_withdrawal'
        """
    ).df()
    outs = pd.concat([tx[["src", "ts", "amount"]], card])
    out_groups = {k: list(zip(g["ts"], g["amount"])) for k, g in outs.groupby("src")}
    for ent, g_in in tx[~tx["regular"]].groupby("dst"):
        ins = list(zip(g_in["ts"], g_in["amount"]))
        total_in = float(sum(a for _, a in ins))
        stream = sorted([(t, 0, a) for t, a in ins] + [(t, 1, a) for t, a in out_groups.get(ent, [])], key=lambda x: (x[0], x[1]))
        open_in: deque[list] = deque()
        matched = 0.0
        for t, kind, a in stream:
            while open_in and t - open_in[0][0] > horizon:
                open_in.popleft()
            if kind == 0:
                open_in.append([t, a])
                continue
            need = a
            while open_in and need > 1e-9:
                take = min(open_in[0][1], need)
                open_in[0][1] -= take
                need -= take
                matched += take
                if open_in[0][1] <= 1e-9:
                    open_in.popleft()
        pt_amt[ent] = matched
        pt_ratio[ent] = matched / total_in if total_in else 0.0
    out["pass_through_amount"] = pd.Series(pt_amt)
    out["pass_through_ratio"] = pd.Series(pt_ratio)
    # cash
    cash = con.execute(
        """
        SELECT o.entity_id AS e, e.ts, CAST(e.attributes->>'amount' AS DOUBLE) AS amount, e.attributes->>'branch_id' AS branch
        FROM events e JOIN _own o ON o.identifier = e.actor_ids[1] WHERE e.event_type = 'cash_deposit'
        """
    ).df()
    lo, hi = cfg.sub_threshold
    st = cash[(cash["amount"] >= lo) & (cash["amount"] < hi)]
    out["sub_threshold_cash"] = st.groupby("e").size()
    out["sub_threshold_branches"] = st.groupby("e")["branch"].nunique()
    out["cash_deposit_total"] = cash.groupby("e")["amount"].sum()
    out["quasi_cash_spend"] = con.execute(
        """
        SELECT o.entity_id AS e, sum(CAST(e.attributes->>'amount' AS DOUBLE)) AS v
        FROM events e JOIN _own o ON o.identifier = e.actor_ids[1]
        WHERE e.event_type = 'card_purchase' AND (e.attributes->>'mcc') = '6051' GROUP BY 1
        """
    ).df().set_index("e")["v"]
    # young accounts
    ws = con.execute("SELECT value FROM manifest WHERE key = 'window_start'").fetchone()[0]
    if con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'ref_accounts'").fetchone()[0]:
        young = {r[0] for r in con.execute(f"SELECT account_id FROM ref_accounts WHERE open_date >= DATE '{ws}' - INTERVAL {cfg.young_account_days} DAY").fetchall()}
        yt = tx[tx["dst_id"].isin(young)]
        out["young_account_inflow"] = yt.groupby("dst")["amount"].sum()
    return out


def cycle_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows = con.execute("SELECT parties, entities_path, amounts FROM money_cycles").fetchall()
    cnt: dict[str, int] = defaultdict(int)
    vol: dict[str, float] = defaultdict(float)
    for parties, path, amounts in rows:
        for e in set(parties) | set(path):
            cnt[e] += 1
            vol[e] += float(amounts[0])
    return pd.DataFrame({"money_cycles": pd.Series(cnt, dtype=float), "cycle_volume": pd.Series(vol, dtype=float)})


def comm_features(con: duckdb.DuckDBPyConnection, cfg: FeatureConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (features, entity-level call table used by the 2-pass feature)."""
    tz = _tz(con)
    calls = con.execute(
        f"""
        SELECT o1.entity_id AS a, o2.entity_id AS b, e.ts, e.event_type,
               hour(timezone('{tz}', e.ts AT TIME ZONE 'UTC')) AS h
        FROM events e
        JOIN _own o1 ON o1.identifier = e.actor_ids[1]
        JOIN _own o2 ON o2.identifier = e.actor_ids[2]
        WHERE e.event_type IN ('call', 'sms')
        """
    ).df()
    # self-calls are dropped here, not in SQL: as a join predicate, `o1.entity_id <> o2.entity_id` can make
    # DuckDB plan a nested-loop join of the two ownership tables (~60M rows, ~10 GB on a clean 1,500-person world)
    calls = calls[calls["a"] != calls["b"]]
    both = pd.concat([calls.rename(columns={"a": "e", "b": "o"}), calls.rename(columns={"b": "e", "a": "o"})], ignore_index=True)
    out = pd.DataFrame()
    out["call_partners"] = both.groupby("e")["o"].nunique()
    out["night_comm_share"] = both.assign(n=(both["h"] >= 22) | (both["h"] < 6)).groupby("e")["n"].mean()
    wl = {r[0] for r in con.execute("SELECT DISTINCT re.entity_id FROM record_entity re WHERE re.record_id LIKE 'watchlist:%'").fetchall()}
    out["calls_with_watchlisted"] = both[both["o"].isin(wl)].groupby("e").size()
    burners = burner_entities(con)
    out["burner_contacts"] = both[both["o"].isin(burners)].groupby("e").size()
    # coordination: large transfers preceded by a call between the (beneficial-owner) parties
    ctrl = dict(con.execute("SELECT dst_entity, min(src_entity) FROM entity_links WHERE relation = 'controls' GROUP BY 1").fetchall())
    party = lambda x: ctrl.get(x, x)  # noqa: E731
    pair_calls: dict[tuple[str, str], list[pd.Timestamp]] = defaultdict(list)
    for a, b, ts in zip(calls["a"], calls["b"], calls["ts"]):
        k = tuple(sorted((a, b)))
        pair_calls[k].append(ts)
    for k in pair_calls:
        pair_calls[k].sort()
    win = pd.Timedelta(hours=cfg.coordination_hours)
    tx = transfers(con)
    tx = tx[(tx["amount"] >= cfg.large_amount) & (tx["src"] != tx["dst"])]
    coord: dict[str, int] = defaultdict(int)
    for s, d, ts in zip(tx["src"], tx["dst"], tx["ts"]):
        ps, pd_ = party(s), party(d)
        if ps == pd_:
            continue
        lst = pair_calls.get(tuple(sorted((ps, pd_))), [])
        i = bisect.bisect_left(lst, ts - win)
        if i < len(lst) and lst[i] < ts:
            for e in {s, d, ps, pd_}:
                coord[e] += 1
    out["pre_txn_coordination"] = pd.Series(coord, dtype=float)
    # large cash deposits preceded by contact with a prepaid/unregistered line
    cash = con.execute(
        """
        SELECT o.entity_id AS e, e.ts FROM events e JOIN _own o ON o.identifier = e.actor_ids[1]
        WHERE e.event_type = 'cash_deposit' AND CAST(e.attributes->>'amount' AS DOUBLE) >= ?
        """,
        [cfg.large_amount],
    ).df()
    burner_calls: dict[str, list[pd.Timestamp]] = defaultdict(list)
    for e, o, ts in zip(both["e"], both["o"], both["ts"]):
        if o in burners:
            burner_calls[e].append(ts)
    for k in burner_calls:
        burner_calls[k].sort()
    pc: dict[str, int] = defaultdict(int)
    for e, ts in zip(cash["e"], cash["ts"]):
        lst = burner_calls.get(e, [])
        i = bisect.bisect_left(lst, ts - win)
        if i < len(lst) and lst[i] < ts:
            pc[e] += 1
    out["pre_cash_contacts"] = pd.Series(pc, dtype=float)
    return out, both


def burner_entities(con: duckdb.DuckDBPyConnection) -> set[str]:
    """Entities known only through a prepaid line (no bank, badge or mailbox identity)."""
    rows = con.execute(
        """
        SELECT re.entity_id
        FROM record_entity re JOIN identity_records ir USING (record_id)
        GROUP BY 1
        HAVING bool_and(ir.source = 'telco') AND bool_and(ir.attributes->>'account_type' = 'PREPAID')
        """
    ).fetchall()
    return {r[0] for r in rows}


def location_digital_features(con: duckdb.DuckDBPyConnection, cfg: FeatureConfig) -> pd.DataFrame:
    tz = _tz(con)
    acc = con.execute(
        f"""
        SELECT o.entity_id AS e,
               sum(CASE WHEN e.event_type = 'access_granted' AND (e.attributes->>'zone') = 'RESTRICTED'
                         AND (hour(timezone('{tz}', e.ts AT TIME ZONE 'UTC')) >= 20 OR hour(timezone('{tz}', e.ts AT TIME ZONE 'UTC')) < 6) THEN 1 ELSE 0 END) AS after_hours_restricted,
               sum(CASE WHEN e.event_type = 'access_denied' THEN 1 ELSE 0 END) AS access_denied
        FROM events e JOIN _own o ON o.identifier = e.actor_ids[1]
        WHERE e.event_type IN ('access_granted', 'access_denied')
        GROUP BY 1
        """
    ).df().set_index("e")
    coloc = con.execute("SELECT src AS a, dst AS b, amount FROM graph_edges WHERE channel = 'coloc'").df()
    ah = pd.concat([coloc[["a", "amount"]].rename(columns={"a": "e"}), coloc[["b", "amount"]].rename(columns={"b": "e"})]).groupby("e")["amount"].sum()
    mail = con.execute(
        """
        SELECT o.entity_id AS e, e.actor_ids AS ids, CAST(e.attributes->>'attachment_bytes' AS DOUBLE) AS bytes
        FROM events e JOIN _own o ON o.identifier = e.actor_ids[1]
        WHERE e.event_type = 'email' AND CAST(e.attributes->>'attachment_bytes' AS DOUBLE) >= ?
        """,
        [cfg.exfil_bytes],
    ).df()
    ex: dict[str, int] = defaultdict(int)
    for e, ids in zip(mail["e"], mail["ids"]):
        dom = ids[0].split("@", 1)[-1]
        if any(r.split("@", 1)[-1] != dom for r in ids[1:]):
            ex[e] += 1
    out = acc.copy()
    out["after_hours_coloc"] = ah
    out["exfil_emails"] = pd.Series(ex, dtype=float)
    return out


def graph_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    g = con.execute(
        """
        SELECT n.node_id AS e, n.degree, n.strength, n.betweenness, n.pagerank,
               coalesce(c.flagged, false)::INT AS community_flagged, coalesce(c.anomaly_score, 0) AS community_score
        FROM graph_nodes n LEFT JOIN communities c USING (community_id)
        """
    ).df().set_index("e")
    young, shared = org_red_flags(con)
    links = con.execute("SELECT src_entity, dst_entity FROM entity_links WHERE relation = 'controls'").fetchall()
    rf: dict[str, int] = defaultdict(int)
    for s, d in links:
        if d in young or d in shared:
            rf[s] += 1
    for d in young | shared:
        rf[d] += 1
    g["controls_red_flag_orgs"] = pd.Series(rf, dtype=float)
    return g


def build_features(con: duckdb.DuckDBPyConnection, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """Feature matrix indexed by entity_id (all resolved entities)."""
    cfg = cfg or FeatureConfig()
    ents = con.execute("SELECT entity_id, entity_type, display_name, n_records, sources FROM entities ORDER BY entity_id").df().set_index("entity_id")
    wl = {r[0] for r in con.execute("SELECT DISTINCT entity_id FROM record_entity WHERE record_id LIKE 'watchlist:%'").fetchall()}
    parts = [money_features(con, cfg), cycle_features(con)]
    comm, _ = comm_features(con, cfg)
    parts += [comm, location_digital_features(con, cfg), graph_features(con)]
    X = pd.DataFrame(index=ents.index)
    for p in parts:
        X = X.join(p[[c for c in p.columns if c not in ("n_in", "n_out", "active_days")]], how="left")
    X["watchlist_hit"] = X.index.isin(wl).astype(int)
    X["is_org"] = (ents["entity_type"] == "org").astype(int)
    for c in FEATURE_DESCRIPTIONS:
        if c not in X.columns:
            X[c] = 0.0
    X = X[[c for c in FEATURE_DESCRIPTIONS]].astype(float).fillna(0.0)
    X["calls_with_flagged"] = 0.0  # filled by the 2-pass scorer
    return X


def calls_with_flagged(con: duckdb.DuckDBPyConnection, flagged: set[str]) -> pd.Series:
    """Second-pass feature: contact volume with entities the first pass rated high."""
    _own(con)
    df = con.execute(
        """
        SELECT o1.entity_id AS a, o2.entity_id AS b
        FROM events e
        JOIN _own o1 ON o1.identifier = e.actor_ids[1]
        JOIN _own o2 ON o2.identifier = e.actor_ids[2]
        WHERE e.event_type IN ('call', 'sms')
        """
    ).df()
    df = df[df["a"] != df["b"]]  # not in SQL, see comm_features
    both = pd.concat([df.rename(columns={"a": "e", "b": "o"}), df.rename(columns={"b": "e", "a": "o"})])
    return both[both["o"].isin(flagged)].groupby("e").size().astype(float)


def labels(con: duckdb.DuckDBPyConnection, index: pd.Index) -> tuple[pd.Series, pd.Series]:
    """(label, labelable) per entity from historical case outcomes.

    An entity is *labelable* if it holds a bank customer record (case outcomes
    are keyed by bank customers); it is positive if any of its records was in
    a CONFIRMED_SUSPICIOUS case.
    """
    pos = {r[0] for r in con.execute("SELECT DISTINCT re.entity_id FROM case_labels c JOIN record_entity re ON re.record_id = c.subject_record_id WHERE c.disposition = 'CONFIRMED_SUSPICIOUS'").fetchall()}
    bank = {r[0] for r in con.execute("SELECT DISTINCT entity_id FROM record_entity WHERE record_id LIKE 'bank:%' AND record_id NOT LIKE '%:bo'").fetchall()}
    y = pd.Series(index.isin(pos).astype(int), index=index)
    m = pd.Series(index.isin(bank), index=index)
    return y, m


__all__ = ["FeatureConfig", "FEATURE_DESCRIPTIONS", "build_features", "calls_with_flagged", "labels", "burner_entities", "transfers"]
