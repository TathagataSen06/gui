"""Entity-resolution metrics against ground truth."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Iterable, Mapping


def _pairs(n: int) -> int:
    return n * (n - 1) // 2


@dataclass
class ERMetrics:
    records: int
    true_entities: int
    predicted_entities: int
    true_pairs: int
    predicted_pairs: int
    true_positive_pairs: int
    pairwise_precision: float
    pairwise_recall: float
    pairwise_f1: float
    bcubed_precision: float
    bcubed_recall: float
    bcubed_f1: float
    overmerged_clusters: int
    split_entities: int

    def to_dict(self) -> dict:
        return asdict(self)


def er_metrics(predicted: Mapping[str, str], truth: Mapping[str, str], records: Iterable[str] | None = None) -> ERMetrics:
    """Pairwise and B-cubed precision/recall.

    ``predicted`` and ``truth`` map record_id -> cluster id. Records missing
    from ``truth`` are treated as singletons (they describe nobody else).
    """
    recs = sorted(records if records is not None else predicted)
    t = {r: truth.get(r, f"__singleton__{r}") for r in recs}
    p = {r: predicted.get(r, f"__pred_singleton__{r}") for r in recs}
    joint = Counter((p[r], t[r]) for r in recs)
    p_sizes = Counter(p.values())
    t_sizes = Counter(t.values())
    tp = sum(_pairs(c) for c in joint.values())
    pp = sum(_pairs(c) for c in p_sizes.values())
    tpairs = sum(_pairs(c) for c in t_sizes.values())
    prec = tp / pp if pp else 1.0
    rec = tp / tpairs if tpairs else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    bp = sum(joint[(p[r], t[r])] / p_sizes[p[r]] for r in recs) / len(recs) if recs else 1.0
    br = sum(joint[(p[r], t[r])] / t_sizes[t[r]] for r in recs) / len(recs) if recs else 1.0
    bf = 2 * bp * br / (bp + br) if bp + br else 0.0
    truths_per_pred: dict[str, set[str]] = defaultdict(set)
    preds_per_truth: dict[str, set[str]] = defaultdict(set)
    for r in recs:
        truths_per_pred[p[r]].add(t[r])
        preds_per_truth[t[r]].add(p[r])
    return ERMetrics(
        records=len(recs),
        true_entities=len(t_sizes),
        predicted_entities=len(p_sizes),
        true_pairs=tpairs,
        predicted_pairs=pp,
        true_positive_pairs=tp,
        pairwise_precision=round(prec, 4),
        pairwise_recall=round(rec, 4),
        pairwise_f1=round(f1, 4),
        bcubed_precision=round(bp, 4),
        bcubed_recall=round(br, 4),
        bcubed_f1=round(bf, 4),
        overmerged_clusters=sum(1 for s in truths_per_pred.values() if len(s) > 1),
        split_entities=sum(1 for s in preds_per_truth.values() if len(s) > 1),
    )


def blocking_recall(candidate_pairs: set[tuple[str, str]], truth: Mapping[str, str]) -> dict[str, float]:
    """Pair completeness (share of true matching pairs that blocking proposed)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for r, e in truth.items():
        groups[e].append(r)
    true_pairs = {tuple(sorted(pq)) for g in groups.values() for pq in combinations(sorted(g), 2)}
    found = sum(1 for pq in true_pairs if pq in candidate_pairs)
    return {"true_pairs": len(true_pairs), "proposed_true_pairs": found, "pair_completeness": round(found / len(true_pairs), 4) if true_pairs else 1.0, "candidate_pairs": len(candidate_pairs)}


def recall_by_source_pair(predicted: Mapping[str, str], truth: Mapping[str, str]) -> dict[str, dict[str, float]]:
    """Pairwise recall broken down by the two records' source systems."""
    groups: dict[str, list[str]] = defaultdict(list)
    for r, e in truth.items():
        groups[e].append(r)
    hit: Counter[str] = Counter()
    tot: Counter[str] = Counter()
    for g in groups.values():
        for a, b in combinations(sorted(g), 2):
            key = "-".join(sorted((a.split(":", 1)[0], b.split(":", 1)[0])))
            tot[key] += 1
            if predicted.get(a) is not None and predicted.get(a) == predicted.get(b):
                hit[key] += 1
    return {k: {"pairs": tot[k], "recall": round(hit[k] / tot[k], 4)} for k in sorted(tot)}
