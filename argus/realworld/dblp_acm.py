"""Argus's comparators, embedding and models on the DBLP-ACM linkage benchmark.

From Argus this uses the Jaro-Winkler comparator, the character n-gram LSA
embedding, ``er_metrics`` and the supervised models. Blocking, the
Fellegi-Sunter matcher and the one-to-one linking are written here for
bibliographic records; the person resolver's ``block``, ``FellegiSunterScorer``
and ``ConstrainedUnionFind`` do not apply to them.

DBLP-ACM links 2,616 DBLP records to 2,294 ACM records describing the same
database papers (2,224 true matches). The entity is a *publication*; author
names are only used as one more string field and are never printed, stored in
reports or resolved as people.

Two protocols, whose numbers must not be compared with each other:

* ``full_table`` (Kopcke et al., 2010; ZeroER, 2020): block the whole
  2,616 x 2,294 problem, score candidates, cluster, and measure precision and
  recall against all 2,224 gold pairs, so matches lost in blocking count as
  misses. Two matchers:

  - ``fellegi_sunter``: unsupervised. u-probabilities come from random record
    pairs, as the person resolver does (with add-0.5 smoothing over every
    level); m-probabilities and the
    match prior are fitted by EM on the candidate pairs (no labels). Links need
    a posterior of at least 0.90, Argus's default threshold.
  - ``argus_logreg`` / ``argus_gbm``: supervised, Argus's models on continuous
    comparison features, scored out-of-fold with 5 folds grouped by DBLP record.

  Clustering is strongest-first with a one-to-one constraint (each source is
  duplicate-free), the clean-clean analogue of Argus's cannot-link rule.

* ``pre_blocked`` (DeepMatcher, SIGMOD 2018; Ditto, PVLDB 2021): the fixed
  12,363 candidate pairs split 3:1:1. Train on ``train``, pick the threshold
  on ``valid``, report F1 on ``test``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold

from ..resolve.embedding import CharNgramEmbedder
from ..resolve.evaluate import er_metrics
from ..resolve.matching import jw
from ..resolve.normalize import ascii_fold
from ..score.model import make_models

N_DBLP = 2_616
N_ACM = 2_294
N_MATCH = 2_224
STOP = {"a", "an", "and", "the", "of", "for", "in", "on", "to", "with", "by", "from", "using", "data", "database", "databases", "system", "systems"}
PAIR_FEATURES = [
    "title_jw", "title_token_jaccard", "title_token_sort", "title_cosine", "title_exact",
    "authors_token_set", "authors_last_jaccard", "authors_missing",
    "venue_same", "venue_token_jaccard", "year_diff", "year_missing",
]


@dataclass
class DblpAcmConfig:
    seed: int = 0
    knn: int = 10  # embedding neighbours per record in blocking
    max_token_df: int = 30  # title tokens shared by more records than this do not block
    fs_threshold: float = 0.90
    u_sample_pairs: int = 20_000
    em_iterations: int = 100
    folds: int = 5


# --------------------------------------------------------------------------
# Loading and normalization
# --------------------------------------------------------------------------


def _read_csv(path: Path, **kw: Any) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, keep_default_na=False, **kw)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"cannot decode {path}")


def norm_text(s: str) -> str:
    s = ascii_fold(str(s or "")).lower().replace("&amp;", " ").replace("&mdash;", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", s)).strip()


def canonical_venue(s: str) -> str:
    v = norm_text(s)
    if "record" in v:
        return "sigmod_record"
    if "journal" in v and ("vldb" in v or "large data" in v):
        return "vldb_journal"
    if "transactions on database" in v or v == "tods":
        return "tods"
    if "vldb" in v or "very large data" in v:
        return "vldb"
    if "sigmod" in v or "management of data" in v:
        return "sigmod"
    return v


def _prep(df: pd.DataFrame, source: str) -> pd.DataFrame:
    out = pd.DataFrame({"id": df["id"].astype(str).str.strip(), "source": source})
    out["title"] = df["title"].map(norm_text)
    out["authors"] = df["authors"].map(norm_text)
    out["venue_raw"] = df["venue"].map(norm_text)
    out["venue"] = df["venue"].map(canonical_venue)
    out["year"] = pd.to_numeric(df["year"], errors="coerce")
    out["title_tokens"] = out["title"].map(lambda t: frozenset(t.split()))
    out["author_last"] = df["authors"].map(lambda a: frozenset(norm_text(x).split()[-1] for x in str(a).split(",") if norm_text(x)))
    return out


@dataclass
class DblpAcmData:
    dblp: pd.DataFrame
    acm: pd.DataFrame
    gold: set[tuple[str, str]]  # (dblp id, acm id)


def load(root: str | Path, expect: bool = True) -> DblpAcmData:
    root = Path(root)
    dblp = _prep(_read_csv(root / "DBLP2.csv"), "dblp")
    acm = _prep(_read_csv(root / "ACM.csv"), "acm")
    m = _read_csv(root / "DBLP-ACM_perfectMapping.csv", header=None)
    if not m.empty and str(m.iloc[0, 1]).strip().lower().startswith("id"):  # header row present in some copies
        m = m.iloc[1:]
    gold = {(str(a).strip(), str(b).strip()) for a, b in m.iloc[:, :2].itertuples(index=False, name=None)}
    if expect and (len(dblp), len(acm), len(gold)) != (N_DBLP, N_ACM, N_MATCH):
        raise ValueError(f"DBLP-ACM counts {(len(dblp), len(acm), len(gold))}, expected {(N_DBLP, N_ACM, N_MATCH)}")
    return DblpAcmData(dblp.reset_index(drop=True), acm.reset_index(drop=True), gold)


# --------------------------------------------------------------------------
# Comparison features
# --------------------------------------------------------------------------


def _jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def compare(left: pd.DataFrame, right: pd.DataFrame, cos: np.ndarray) -> pd.DataFrame:
    """Continuous comparison features for aligned rows of ``left`` and ``right``."""
    rows = []
    for (_, a), (_, b), c in zip(left.iterrows(), right.iterrows(), cos):
        ya, yb = a["year"], b["year"]
        no_year = pd.isna(ya) or pd.isna(yb)
        no_auth = not a["authors"] or not b["authors"]
        rows.append({
            "title_jw": jw(a["title"], b["title"]),
            "title_token_jaccard": _jaccard(a["title_tokens"], b["title_tokens"]),
            "title_token_sort": fuzz.token_sort_ratio(a["title"], b["title"]) / 100,
            "title_cosine": float(c),
            "title_exact": float(a["title"] == b["title"] and bool(a["title"])),
            "authors_token_set": 0.0 if no_auth else fuzz.token_set_ratio(a["authors"], b["authors"]) / 100,
            "authors_last_jaccard": _jaccard(a["author_last"], b["author_last"]),
            "authors_missing": float(no_auth),
            "venue_same": float(a["venue"] == b["venue"] and bool(a["venue"])),
            "venue_token_jaccard": _jaccard(frozenset(a["venue_raw"].split()), frozenset(b["venue_raw"].split())),
            "year_diff": -1.0 if no_year else float(min(5, abs(ya - yb))),
            "year_missing": float(no_year),
        })
    return pd.DataFrame(rows, columns=PAIR_FEATURES)


LEVELS = {
    "title": ("exact", "jw95", "cos90", "tok50", "diff"),
    "authors": ("high", "mid", "diff", "null"),
    "venue": ("same", "diff"),
    "year": ("same", "near", "diff", "null"),
}


def levels(f: pd.DataFrame) -> pd.DataFrame:
    """Discrete agreement levels for Fellegi-Sunter (Argus-style comparison vector)."""
    title = np.select([f.title_exact == 1, f.title_jw >= 0.95, f.title_cosine >= 0.90, f.title_token_jaccard >= 0.5], ["exact", "jw95", "cos90", "tok50"], "diff")
    auth = np.select([f.authors_missing == 1, f.authors_token_set >= 0.9, f.authors_token_set >= 0.6], ["null", "high", "mid"], "diff")
    venue = np.where(f.venue_same == 1, "same", "diff")
    year = np.select([f.year_missing == 1, f.year_diff == 0, f.year_diff == 1], ["null", "same", "near"], "diff")
    return pd.DataFrame({"title": title, "authors": auth, "venue": venue, "year": year}, index=f.index)


# --------------------------------------------------------------------------
# Blocking
# --------------------------------------------------------------------------


def embed_titles(data: DblpAcmData, cfg: DblpAcmConfig) -> tuple[np.ndarray, np.ndarray]:
    """Argus's character n-gram LSA embedding, fitted once on every title of both sources."""
    emb = CharNgramEmbedder(dim=48, seed=cfg.seed).embed(data.dblp["title"].tolist() + data.acm["title"].tolist())
    return emb[: len(data.dblp)], emb[len(data.dblp):]


def block(data: DblpAcmData, ed: np.ndarray, ea: np.ndarray, cfg: DblpAcmConfig) -> tuple[np.ndarray, np.ndarray]:
    """Candidate (dblp_idx, acm_idx) pairs from embedding kNN and rare title tokens."""
    sim = ed @ ea.T
    k = min(cfg.knn, sim.shape[1])
    cand = {(i, int(j)) for i, row in enumerate(np.argpartition(-sim, k - 1, axis=1)[:, :k]) for j in row}
    k = min(cfg.knn, sim.shape[0])
    cand |= {(int(i), j) for j, col in enumerate(np.argpartition(-sim.T, k - 1, axis=1)[:, :k]) for i in col}
    inv: dict[str, list[int]] = {}
    for j, toks in enumerate(data.acm["title_tokens"]):
        for t in toks:
            if t not in STOP and len(t) >= 4:
                inv.setdefault(t, []).append(j)
    for i, toks in enumerate(data.dblp["title_tokens"]):
        for t in toks:
            js = inv.get(t, [])
            if 0 < len(js) <= cfg.max_token_df:
                cand.update((i, j) for j in js)
    pairs = np.array(sorted(cand), dtype=int)
    return pairs, sim[pairs[:, 0], pairs[:, 1]]


# --------------------------------------------------------------------------
# Matchers
# --------------------------------------------------------------------------


def _u_probs(data: DblpAcmData, ed: np.ndarray, ea: np.ndarray, cfg: DblpAcmConfig) -> dict[str, dict[str, float]]:
    """u-probabilities from random DBLP x ACM pairs (the Argus resolver's estimator)."""
    rng = np.random.default_rng(cfg.seed)
    n = min(cfg.u_sample_pairs, len(data.dblp) * len(data.acm))
    i = rng.integers(0, len(data.dblp), n)
    j = rng.integers(0, len(data.acm), n)
    cos = np.einsum("ij,ij->i", ed[i], ea[j])
    lv = levels(compare(data.dblp.iloc[i].reset_index(drop=True), data.acm.iloc[j].reset_index(drop=True), cos))
    # additive smoothing over every level, so a level never seen among random pairs gets a small, not arbitrary, u
    return {c: {k: (int((lv[c] == k).sum()) + 0.5) / (len(lv) + 0.5 * len(ks)) for k in ks} for c, ks in LEVELS.items()}


def fellegi_sunter_em(lv: pd.DataFrame, u: dict[str, dict[str, float]], iters: int) -> tuple[np.ndarray, dict[str, Any]]:
    """EM for m-probabilities and the match share with u held fixed; returns posteriors."""
    floor = 1e-6
    cats = {c: sorted(LEVELS[c]) for c in lv}
    codes = {c: pd.Categorical(lv[c], categories=cats[c]).codes for c in lv}
    agree = {"exact", "high", "same"}
    m = {c: np.array([0.9 if k in agree else 0.1 / max(1, len(cats[c]) - 1) for k in cats[c]]) for c in lv}
    log_u = sum(np.log(np.array([u[c][k] for k in cats[c]]))[codes[c]] for c in lv)
    lam = 0.1
    post = np.zeros(len(lv))
    for _ in range(iters):
        log_m = sum(np.log(np.maximum(m[c], floor))[codes[c]] for c in lv)
        new = 1 / (1 + np.exp(np.clip(np.log(1 - lam) + log_u - np.log(lam) - log_m, -50, 50)))
        lam = float(np.clip(new.mean(), 1e-4, 0.9))
        m = {c: np.bincount(codes[c], weights=new, minlength=len(cats[c])) / new.sum() for c in lv}
        done = np.abs(new - post).max() < 1e-6
        post = new
        if done:
            break
    params = {
        "match_share": round(lam, 4),
        "m": {c: {k: round(float(v), 4) for k, v in zip(cats[c], m[c])} for c in lv},
        "u": {c: {k: round(v, 4) for k, v in u[c].items()} for c in u},
    }
    return post, params


def one_to_one(pairs: np.ndarray, score: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Strongest-first linking where each record is matched at most once."""
    used_l: set[int] = set()
    used_r: set[int] = set()
    out = []
    for k in np.lexsort((pairs[:, 1], pairs[:, 0], -score)):
        if score[k] < threshold:
            break
        i, j = int(pairs[k, 0]), int(pairs[k, 1])
        if i in used_l or j in used_r:
            continue
        used_l.add(i)
        used_r.add(j)
        out.append((i, j))
    return out


def evaluate_links(data: DblpAcmData, links: list[tuple[int, int]]) -> dict[str, Any]:
    pred: dict[str, str] = {}
    for i, j in links:
        c = f"c{i}"
        pred[f"d:{data.dblp.id[i]}"] = c
        pred[f"a:{data.acm.id[j]}"] = c
    truth: dict[str, str] = {}
    for a, b in data.gold:
        truth[f"d:{a}"] = truth[f"a:{b}"] = f"t:{a}"
    recs = [f"d:{x}" for x in data.dblp.id] + [f"a:{x}" for x in data.acm.id]
    m = er_metrics(pred, truth, recs)
    # recompute from counts: er_metrics rounds to 4 decimals, and rounding again for the report can be off by one
    p = m.true_positive_pairs / m.predicted_pairs if m.predicted_pairs else 1.0
    r = m.true_positive_pairs / m.true_pairs if m.true_pairs else 1.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "predicted_pairs": m.predicted_pairs, "true_positive_pairs": m.true_positive_pairs}


def full_table(data: DblpAcmData, cfg: DblpAcmConfig) -> dict[str, Any]:
    ed, ea = embed_titles(data, cfg)
    pairs, cos = block(data, ed, ea, cfg)
    ids = list(zip(data.dblp.id.to_numpy()[pairs[:, 0]], data.acm.id.to_numpy()[pairs[:, 1]]))
    y = np.array([p in data.gold for p in ids], dtype=int)
    feats = compare(data.dblp.iloc[pairs[:, 0]].reset_index(drop=True), data.acm.iloc[pairs[:, 1]].reset_index(drop=True), cos)
    out: dict[str, Any] = {
        "protocol": "full table: block all 2,616 x 2,294 records, score, cluster one-to-one; blocking misses count against recall",
        "blocking": {
            "candidate_pairs": len(pairs),
            "cartesian_pairs": len(data.dblp) * len(data.acm),
            "reduction_ratio": 1 - len(pairs) / (len(data.dblp) * len(data.acm)),
            "pair_completeness": float(y.sum() / len(data.gold)),
        },
        "matchers": {},
    }
    u = _u_probs(data, ed, ea, cfg)
    post, params = fellegi_sunter_em(levels(feats), u, cfg.em_iterations)
    out["matchers"]["fellegi_sunter"] = {**evaluate_links(data, one_to_one(pairs, post, cfg.fs_threshold)), "labels_used": 0, "em": params}
    groups = pairs[:, 0]
    for name in ("logreg", "gbm"):
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=cfg.folds).split(feats, y, groups):
            m = clone(make_models(cfg.seed)[name]).fit(feats.iloc[tr], y[tr])
            oof[te] = m.predict_proba(feats.iloc[te])[:, 1]
        out["matchers"][f"argus_{name}"] = {**evaluate_links(data, one_to_one(pairs, oof, 0.5)), "labels_used": f"{cfg.folds}-fold out-of-fold, grouped by DBLP record"}
    return out


# --------------------------------------------------------------------------
# Pre-blocked (DeepMatcher / Ditto) protocol
# --------------------------------------------------------------------------

_COL = re.compile(r"COL (\w+) VAL (.*?)(?= COL \w+ VAL |$)")


def parse_ditto_side(s: str) -> dict[str, str]:
    return {k: v.strip() for k, v in _COL.findall(s.strip())}


def load_ditto_split(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    left, right, y = [], [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        left.append(parse_ditto_side(parts[0]))
        right.append(parse_ditto_side(parts[1]))
        y.append(int(parts[2]))
    to_df = lambda rows: _prep(pd.DataFrame(rows).reindex(columns=["title", "authors", "venue", "year"]).fillna("").assign(id=range(len(rows))), "x")  # noqa: E731
    return to_df(left), to_df(right), np.array(y)


def pre_blocked(root: str | Path, cfg: DblpAcmConfig) -> dict[str, Any]:
    root = Path(root)
    splits = {s: load_ditto_split(root / f"{s}.txt") for s in ("train", "valid", "test")}
    titles = [t for l, r, _ in splits.values() for t in (*l["title"], *r["title"])]
    emb = CharNgramEmbedder(dim=48, seed=cfg.seed).embed(titles)
    feats, off = {}, 0
    for s, (l, r, _) in splits.items():
        n = len(l)
        cos = np.einsum("ij,ij->i", emb[off: off + n], emb[off + n: off + 2 * n])
        off += 2 * n
        feats[s] = compare(l, r, cos)
    out: dict[str, Any] = {
        "protocol": "pre-blocked candidate pairs (DeepMatcher/Ditto split 3:1:1); threshold chosen on valid, F1 on test",
        "pairs": {s: {"pairs": int(len(v[2])), "matches": int(v[2].sum())} for s, v in splits.items()},
        "models": {},
    }
    for name in ("logreg", "gbm"):
        m = clone(make_models(cfg.seed)[name]).fit(feats["train"], splits["train"][2])
        pv = m.predict_proba(feats["valid"])[:, 1]
        grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
        thr = float(max(grid, key=lambda t: (f1_score(splits["valid"][2], pv >= t), -abs(t - 0.5))))
        pt = m.predict_proba(feats["test"])[:, 1]
        yt = splits["test"][2]
        out["models"][f"argus_{name}"] = {
            "threshold": thr,
            "precision": float(precision_score(yt, pt >= thr)),
            "recall": float(recall_score(yt, pt >= thr)),
            "f1": float(f1_score(yt, pt >= thr)),
            "f1_at_0.5": float(f1_score(yt, pt >= 0.5)),
        }
    return out


def run(root: str | Path, cfg: DblpAcmConfig | None = None, expect: bool = True, pre_blocked_root: str | Path | None = None) -> dict[str, Any]:
    cfg = cfg or DblpAcmConfig()
    data = load(root, expect)
    res: dict[str, Any] = {"dataset": "dblp_acm", "dblp": len(data.dblp), "acm": len(data.acm), "gold_pairs": len(data.gold), "full_table": full_table(data, cfg)}
    pb = Path(pre_blocked_root) if pre_blocked_root else Path(root) / "pre_blocked"
    if (pb / "test.txt").exists():
        res["pre_blocked"] = pre_blocked(pb, cfg)
    return res
