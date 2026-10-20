"""Benchmark metrics: entity resolution, ring detection, false positives."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ..resolve.evaluate import er_metrics, recall_by_source_pair
from ..resolve.normalize import dob_relation, parse_iso
from .truth import Truth, entity_truth, person_entities


# --------------------------------------------------------------------------
# Entity resolution
# --------------------------------------------------------------------------


def evaluate_er(con: duckdb.DuckDBPyConnection, truth: Truth) -> dict[str, Any]:
    pred = dict(con.execute("SELECT record_id, entity_id FROM record_entity").fetchall())
    recs = sorted(pred)
    person_recs = [r for r in recs if truth.record_kind.get(r) in ("person", "burner", "anonymous_line") or (r not in truth.record_truth and not r.startswith("email:"))]
    no_burner = [r for r in recs if truth.record_kind.get(r) == "person"]
    org_recs = [r for r in recs if truth.record_kind.get(r) == "org"]
    out = {
        "all_records": er_metrics(pred, truth.record_truth, recs).to_dict(),
        "person_records": er_metrics(pred, truth.record_truth, person_recs).to_dict(),
        "person_records_excl_burners": er_metrics(pred, truth.record_truth, no_burner).to_dict(),
        "org_records": er_metrics(pred, truth.record_truth, org_recs).to_dict(),
        "recall_by_source_pair": recall_by_source_pair(pred, {r: truth.record_truth[r] for r in no_burner}),
    }
    out["error_analysis"] = er_error_analysis(con, truth, pred)
    return out


def er_error_analysis(con: duckdb.DuckDBPyConnection, truth: Truth, pred: dict[str, str], examples: int = 5) -> dict[str, Any]:
    """Categorize missed links (splits) and wrong links (merges) by likely cause."""
    rows = con.execute("SELECT record_id, source, name, dob, address, phone, email FROM identity_records").df().set_index("record_id")
    by_true: dict[str, list[str]] = defaultdict(list)
    for r in pred:
        t = truth.record_truth.get(r)
        if t and truth.record_kind.get(r) == "person":
            by_true[t].append(r)
    split_causes: Counter[str] = Counter()
    split_examples: list[dict[str, Any]] = []
    for t, recs in sorted(by_true.items()):
        ents = Counter(pred[r] for r in recs)
        if len(ents) <= 1:
            continue
        main = ents.most_common(1)[0][0]
        for r in recs:
            if pred[r] == main:
                continue
            row = rows.loc[r]
            mains = [rows.loc[x] for x in recs if pred[x] == main]
            cause = _split_cause(row, mains)
            split_causes[cause] += 1
            if len(split_examples) < examples:
                split_examples.append({"true_id": t, "record": r, "cause": cause, "name": row["name"], "dob": row["dob"], "matched_names": [m["name"] for m in mains[:3]]})
    merge_examples = []
    merges = 0
    by_pred: dict[str, list[str]] = defaultdict(list)
    for r, e in pred.items():
        by_pred[e].append(r)
    for e, recs in sorted(by_pred.items()):
        ids = {truth.record_truth.get(r) for r in recs if truth.record_truth.get(r)}
        if len(ids) > 1:
            merges += 1
            if len(merge_examples) < examples:
                merge_examples.append({"entity_id": e, "true_ids": sorted(ids), "names": [rows.loc[r]["name"] for r in recs][:6]})
    return {"split_records_by_cause": dict(split_causes.most_common()), "split_examples": split_examples, "overmerged_entities": merges, "merge_examples": merge_examples}


def _split_cause(row: pd.Series, mains: list[pd.Series]) -> str:
    if row.name.startswith("telco:") and (not row["name"] or " " not in str(row["name"]).strip()):
        return "anonymous_prepaid_line"
    if row.name.startswith("email:") and not row["name"]:
        return "mailbox_without_display_name"
    d = parse_iso(row["dob"])
    if d and any(dob_relation(d, parse_iso(m["dob"])) == "diff" for m in mains if m["dob"]):
        return "conflicting_dob"
    if not row["phone"] and not row["email"]:
        return "no_shared_identifier"
    return "weak_name_evidence"


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def _rank_metrics(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    if y.sum() == 0 or y.sum() == len(y):
        return {"roc_auc": float("nan"), "average_precision": float("nan")}
    return {"roc_auc": round(float(roc_auc_score(y, s)), 4), "average_precision": round(float(average_precision_score(y, s)), 4)}


def evaluate_detection(con: duckdb.DuckDBPyConnection, truth: Truth, alert_threshold: float) -> dict[str, Any]:
    scores = con.execute("SELECT entity_id, entity_type, risk_score, rule_score, ml_score, alert, labelled FROM risk_scores").df().set_index("entity_id")
    et = entity_truth(con, truth).reindex(scores.index)
    y = et["suspicious"].fillna(False).astype(int).values
    out: dict[str, Any] = {"entities": int(len(scores)), "suspicious_entities": int(y.sum())}
    for col in ("risk_score", "rule_score", "ml_score"):
        out[col] = _rank_metrics(y, scores[col].values)
    order = np.argsort(-scores["risk_score"].values, kind="mergesort")
    out["precision_at_k"] = {str(k): round(float(y[order[:k]].mean()), 4) for k in (10, 25, 50, 100) if k <= len(y)}
    alerts = scores["risk_score"].values >= alert_threshold
    tp = int((alerts & (y == 1)).sum())
    fp = int((alerts & (y == 0)).sum())
    out["alerts"] = {
        "threshold": alert_threshold,
        "alerts": int(alerts.sum()),
        "true_positives": tp,
        "false_positives": fp,
        "precision": round(tp / max(1, tp + fp), 4),
        "recall": round(tp / max(1, int(y.sum())), 4),
        "false_positive_rate": round(fp / max(1, int((y == 0).sum())), 5),
    }
    # person-level view: a planted member counts as found if any entity holding their records alerts
    pe = person_entities(con, truth)
    risk = scores["risk_score"].to_dict()
    rings = []
    flagged_comm = {r[0] for r in con.execute("SELECT community_id FROM communities WHERE flagged").fetchall()}
    comm = dict(con.execute("SELECT node_id, community_id FROM graph_nodes").fetchall())
    for ring in truth.rings:
        members = [m["person_id"] for m in ring["members"]]
        found = []
        best_rank = []
        for m in members:
            ents = pe.get(m, [])
            r = max((risk.get(e, 0.0) for e in ents), default=0.0)
            found.append(r >= alert_threshold)
            best_rank.append(r)
        m_comms = Counter(comm.get(e) for m in members for e in pe.get(m, [])[:1])
        top_c, top_n = m_comms.most_common(1)[0] if m_comms else (None, 0)
        rings.append(
            {
                "ring_id": ring["ring_id"],
                "typology": ring["typology"],
                "members": len(members),
                "members_alerted": int(sum(found)),
                "member_recall": round(sum(found) / len(members), 3),
                "detected": sum(found) / len(members) >= 0.5,
                "orgs_alerted": int(sum(1 for o in ring["orgs"] for e in pe.get(o["org_id"], []) if risk.get(e, 0) >= alert_threshold)),
                "largest_community_share": round(top_n / len(members), 3),
                "community_flagged": bool(top_c in flagged_comm),
            }
        )
    out["rings"] = rings
    out["rings_detected"] = int(sum(r["detected"] for r in rings))
    out["rings_total"] = len(rings)
    out["ring_member_recall"] = round(sum(r["members_alerted"] for r in rings) / max(1, sum(r["members"] for r in rings)), 4)
    # non-seed recall: exclude members that sit on the watch-list (the "tip" an analyst already had)
    wl_true = set()
    for r, e in con.execute("SELECT record_id, entity_id FROM record_entity WHERE record_id LIKE 'watchlist:%'").fetchall():
        t = truth.record_truth.get(r)
        if t:
            wl_true.add(t)
    non_seed = [(m["person_id"]) for ring in truth.rings for m in ring["members"] if m["person_id"] not in wl_true]
    hit = sum(1 for m in non_seed if max((risk.get(e, 0.0) for e in pe.get(m, [])), default=0) >= alert_threshold)
    out["non_watchlisted_member_recall"] = round(hit / max(1, len(non_seed)), 4)
    out["decoys"] = decoy_false_positives(truth, pe, risk, alert_threshold)
    out["victims"] = {
        "victims": len(truth.victims),
        "alerted": int(sum(1 for v in truth.victims if max((risk.get(e, 0.0) for e in pe.get(v, [])), default=0) >= alert_threshold)),
    }
    out["communities"] = community_quality(con, truth, pe)
    return out


def decoy_false_positives(truth: Truth, pe: dict[str, list[str]], risk: dict[str, float], thr: float) -> dict[str, Any]:
    by_kind: dict[str, list[bool]] = defaultdict(list)
    for d in truth.decoys:
        for m in d["members"]:
            r = max((risk.get(e, 0.0) for e in pe.get(m["person_id"], [])), default=0.0)
            by_kind[d["kind"]].append(r >= thr)
    return {k: {"participants": len(v), "alerted": int(sum(v)), "rate": round(sum(v) / len(v), 3)} for k, v in sorted(by_kind.items())}


def community_quality(con: duckdb.DuckDBPyConnection, truth: Truth, pe: dict[str, list[str]]) -> dict[str, Any]:
    comm = dict(con.execute("SELECT node_id, community_id FROM graph_nodes").fetchall())
    flagged = {r[0] for r in con.execute("SELECT community_id FROM communities WHERE flagged").fetchall()}
    ring_comms = set()
    for ring in truth.rings:
        for m in ring["members"]:
            for e in pe.get(m["person_id"], []):
                if comm.get(e) in flagged:
                    ring_comms.add(comm[e])
    covered = 0
    for ring in truth.rings:
        if any(comm.get(e) in flagged for m in ring["members"] for e in pe.get(m["person_id"], [])):
            covered += 1
    return {
        "flagged_communities": len(flagged),
        "flagged_with_ring_member": len(ring_comms),
        "flag_precision": round(len(ring_comms) / max(1, len(flagged)), 3),
        "rings_touching_flagged_community": covered,
    }


def clean_control_fpr(con: duckdb.DuckDBPyConnection, truth: Truth, alert_threshold: float) -> dict[str, Any]:
    """False-positive profile on a world with no planted networks (every alert is false)."""
    scores = con.execute("SELECT entity_id, risk_score FROM risk_scores").df().set_index("entity_id")
    pe = person_entities(con, truth)
    n = len(scores)
    alerts = int((scores["risk_score"] >= alert_threshold).sum())
    persons = truth.person_labels["person_id"].tolist()
    alerted_persons = sum(1 for p in persons if max((scores["risk_score"].get(e, 0.0) for e in pe.get(p, [])), default=0.0) >= alert_threshold)
    return {
        "entities": n,
        "alerts": alerts,
        "entity_false_positive_rate": round(alerts / max(1, n), 5),
        "persons": len(persons),
        "persons_alerted": alerted_persons,
        "person_false_positive_rate": round(alerted_persons / max(1, len(persons)), 5),
        "decoys": decoy_false_positives(truth, pe, scores["risk_score"].to_dict(), alert_threshold),
        "risk_score_p99": round(float(np.percentile(scores["risk_score"], 99)), 2),
    }
