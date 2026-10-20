"""Argus's graph features and models on the Elliptic Bitcoin transaction graph.

Protocol (Weber et al., 2019, Table 1): temporal split, train on time steps
1-34 and test on 35-49, labelled nodes only; illicit is the positive class and
the headline metric is illicit F1 at the default 0.5 threshold.

Feature sets:

* ``LF``: time step + 93 local transaction features (the paper's first 94).
* ``AF``: LF + 72 one-hop aggregated features (the paper's 166).
* ``AF+GF``: AF + Argus graph features computed on the payment-flow graph
  (degree, PageRank, sampled betweenness, Louvain community size, two-hop
  reach). Elliptic has no edges across time steps, so these use only the
  structure of a node's own time step, and no labels.

Models: Argus's logistic regression and gradient boosting exactly as configured
for synthetic entities (:func:`argus.score.model.make_models`), plus the
paper's own random forest (50 trees, 50 max features) and default logistic
regression as reproduction checks, so a gap to the published numbers can be
attributed to the model rather than to data handling.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from ..score.model import make_models

N_NODES = 203_769
N_EDGES = 234_355
N_ILLICIT = 4_545
N_LICIT = 42_019
N_LOCAL = 93
N_AGG = 72
TRAIN_LAST_STEP = 34
DARK_MARKET_STEP = 43
GRAPH_FEATURES = ["g_in_degree", "g_out_degree", "g_pagerank", "g_betweenness", "g_community_size", "g_in_2hop", "g_out_2hop"]


@dataclass
class EllipticConfig:
    seed: int = 0
    betweenness_k: int = 64
    louvain_resolution: float = 1.0
    feature_sets: tuple[str, ...] = ("LF", "AF", "AF+GF")
    models: tuple[str, ...] = ("argus_logreg", "argus_gbm", "weber_rf", "weber_logreg")


@dataclass
class EllipticData:
    nodes: pd.DataFrame  # txId, time_step, local_*, agg_*, label (1 illicit, 0 licit, -1 unknown)
    edges: pd.DataFrame  # src, dst

    @property
    def local_cols(self) -> list[str]:
        return [c for c in self.nodes.columns if c.startswith("local_")]

    @property
    def agg_cols(self) -> list[str]:
        return [c for c in self.nodes.columns if c.startswith("agg_")]


def load(root: str | Path, expect: bool = True) -> EllipticData:
    """Read the three official CSVs and check the documented counts."""
    root = Path(root)
    feat = pd.read_csv(root / "elliptic_txs_features.csv", header=None)
    n_feat = feat.shape[1] - 2
    n_local = min(N_LOCAL, n_feat)
    feat.columns = ["txId", "time_step"] + [f"local_{i}" for i in range(n_local)] + [f"agg_{i}" for i in range(n_feat - n_local)]
    cls = pd.read_csv(root / "elliptic_txs_classes.csv")
    cls.columns = ["txId", "class"]
    edges = pd.read_csv(root / "elliptic_txs_edgelist.csv")
    edges.columns = ["src", "dst"]
    lab = cls["class"].astype(str).map({"1": 1, "2": 0}).fillna(-1).astype(int)
    nodes = feat.merge(cls.assign(label=lab)[["txId", "label"]], on="txId", how="left", validate="one_to_one")
    nodes["label"] = nodes["label"].fillna(-1).astype(int)
    if expect:
        got = (len(nodes), len(edges), int((nodes.label == 1).sum()), int((nodes.label == 0).sum()), n_feat)
        want = (N_NODES, N_EDGES, N_ILLICIT, N_LICIT, N_LOCAL + N_AGG)
        if got != want:
            raise ValueError(f"Elliptic counts (nodes, edges, illicit, licit, features) = {got}, expected {want}")
    return EllipticData(nodes, edges)


def graph_features(data: EllipticData, cfg: EllipticConfig) -> pd.DataFrame:
    """Argus-style structural features of every transaction (no labels used)."""
    G = nx.DiGraph()
    G.add_nodes_from(data.nodes["txId"].tolist())
    G.add_edges_from(data.edges[["src", "dst"]].itertuples(index=False, name=None))
    U = G.to_undirected(as_view=True)
    pr = nx.pagerank(G, alpha=0.85)
    bc = nx.betweenness_centrality(U, k=min(cfg.betweenness_k, U.number_of_nodes()), seed=cfg.seed, normalized=True)
    comms = nx.community.louvain_communities(U, resolution=cfg.louvain_resolution, seed=cfg.seed)
    csize = {n: len(c) for c in comms for n in c}
    in2 = {n: len({p for q in G.predecessors(n) for p in G.predecessors(q)}) for n in G}
    out2 = {n: len({s for q in G.successors(n) for s in G.successors(q)}) for n in G}
    ids = data.nodes["txId"]
    return pd.DataFrame({
        "txId": ids,
        "g_in_degree": ids.map(dict(G.in_degree())).to_numpy(),
        "g_out_degree": ids.map(dict(G.out_degree())).to_numpy(),
        "g_pagerank": np.log1p(ids.map(pr).to_numpy() * 1e4),
        "g_betweenness": np.log1p(ids.map(bc).to_numpy() * 1e4),
        "g_community_size": np.log1p(ids.map(csize).to_numpy()),
        "g_in_2hop": np.log1p(ids.map(in2).to_numpy()),
        "g_out_2hop": np.log1p(ids.map(out2).to_numpy()),
    })


def feature_columns(data: EllipticData, name: str) -> list[str]:
    lf = ["time_step", *data.local_cols]
    if name == "LF":
        return lf
    if name == "AF":
        return lf + data.agg_cols
    if name == "AF+GF":
        return lf + data.agg_cols + GRAPH_FEATURES
    raise ValueError(f"unknown feature set {name}")


def make_model(name: str, seed: int) -> Any:
    if name.startswith("argus_"):
        return clone(make_models(seed)[name.removeprefix("argus_")])
    if name == "weber_rf":
        return RandomForestClassifier(n_estimators=50, max_features=50, random_state=seed, n_jobs=-1)
    if name == "weber_logreg":
        return LogisticRegression()  # scikit-learn defaults, as in the paper
    raise ValueError(f"unknown model {name}")


def illicit_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = (p >= 0.5).astype(int)
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro")),
        "auprc": float(average_precision_score(y, p)) if y.sum() else float("nan"),
    }


def run(root: str | Path, cfg: EllipticConfig | None = None, expect: bool = True) -> dict[str, Any]:
    cfg = cfg or EllipticConfig()
    data = load(root, expect)
    t = time.perf_counter()
    if any("GF" in f for f in cfg.feature_sets):
        data.nodes = data.nodes.merge(graph_features(data, cfg), on="txId", how="left")
    gf_seconds = round(time.perf_counter() - t, 1)
    lab = data.nodes[data.nodes["label"] >= 0]
    tr = lab[lab["time_step"] <= TRAIN_LAST_STEP]
    te = lab[lab["time_step"] > TRAIN_LAST_STEP]
    y_te = te["label"].to_numpy()
    results: dict[str, Any] = {}
    for fs in cfg.feature_sets:
        cols = feature_columns(data, fs)
        for mname in cfg.models:
            t = time.perf_counter()
            m = make_model(mname, cfg.seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                m.fit(tr[cols], tr["label"])
            p = m.predict_proba(te[cols])[:, 1]
            per_step = {}
            for step, idx in te.groupby("time_step").indices.items():
                ys, ps = y_te[idx], p[idx]
                per_step[int(step)] = {"illicit": int(ys.sum()), "f1": float(f1_score(ys, (ps >= 0.5).astype(int), zero_division=0))}
            before = te["time_step"].to_numpy() < DARK_MARKET_STEP
            f1_of = lambda mask: float(f1_score(y_te[mask], (p[mask] >= 0.5).astype(int), zero_division=0)) if mask.any() else None  # noqa: E731
            results[f"{mname}|{fs}"] = {
                "model": mname, "features": fs, "n_features": len(cols),
                **illicit_metrics(y_te, p),
                "f1_before_step_43": f1_of(before),
                "f1_from_step_43": f1_of(~before),
                "per_step": per_step,
                "seconds": round(time.perf_counter() - t, 1),
            }
    return {
        "dataset": "elliptic",
        "nodes": len(data.nodes), "edges": len(data.edges),
        "train": {"rows": len(tr), "illicit": int(tr["label"].sum())},
        "test": {"rows": len(te), "illicit": int(y_te.sum())},
        "graph_feature_seconds": gf_seconds,
        "results": results,
    }
