"""Pairwise record comparison with Fellegi-Sunter match weights.

Every comparison maps a record pair to a discrete *agreement level*
(e.g. last name: exact / fuzzy / diff / null). The evidence contributed by a
level is the log-likelihood ratio ``log2(m / u)``:

* ``m`` = P(level | records are the same entity) - set from domain knowledge
  of the defect rates (documented below, deliberately not fitted to labels);
* ``u`` = P(level | different entities) - **estimated from the data** by
  comparing random record pairs. For exact agreement on a value, ``u`` is
  term-frequency adjusted: agreeing on "smith" is weaker evidence than
  agreeing on "kowalczyk".

The total weight is converted to a posterior match probability with a prior
derived from the expected number of records per entity. Each scored pair keeps
its per-comparison weights as evidence, so every merge is explainable.

Two guards correct known weaknesses of the conditional-independence
assumption behind Fellegi-Sunter:

* **household dependence** - relatives share surname *and* address, so when
  first names disagree, address (and the embedding, which encodes the
  address) is treated as uninformative instead of as independent evidence;
* **corroboration** - unless first *and* last name agree exactly, a link
  needs at least one non-name agreement (date of birth, phone, e-mail, street
  address or employer). Near-namesakes ("Tina Gonzalez" / "Tina Gonzales")
  otherwise link on name similarity alone, because u for a fuzzy surname is
  estimated on random pairs, not on the phonetically-blocked candidates where
  fuzzy agreement among non-matches is common.

The embedding comparison is skipped when either record lacks an address: on
a name-only text it would count the name evidence a second time.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from .normalize import NameParts, dob_relation
from .records import Rec

M_PROBS: dict[str, dict[str, float]] = {
    "first": {"exact": 0.84, "fuzzy": 0.06, "initial": 0.04, "diff": 0.06},
    "last": {"exact": 0.85, "fuzzy": 0.07, "diff": 0.08},
    "dob": {"exact": 0.90, "swap": 0.03, "near": 0.05, "diff": 0.02},
    "phone": {"exact": 0.90, "diff": 0.10},
    "email": {"exact": 0.90, "diff": 0.10},
    "address": {"exact": 0.60, "street": 0.10, "zip": 0.12, "diff": 0.18},
    "employer": {"match": 0.85, "diff": 0.15},
    "embedding": {"high": 0.55, "mid": 0.30, "low": 0.15},
    "suffix": {"conflict": 0.02, "ok": 0.98},
    "org_name": {"exact": 0.85, "fuzzy": 0.10, "diff": 0.05},
    "reg_no": {"exact": 0.95, "diff": 0.05},
}
PERSON_COMPARISONS = ("first", "last", "dob", "phone", "email", "address", "employer", "embedding", "suffix")
ORG_COMPARISONS = ("org_name", "reg_no", "address", "phone", "email")
TF_LEVELS = {("first", "exact"), ("last", "exact"), ("phone", "exact"), ("email", "exact"), ("address", "exact"), ("org_name", "exact")}


@dataclass
class MatchConfig:
    threshold: float = 0.90  # posterior probability required to link
    records_per_entity: float = 4.0  # prior: expected records describing one entity
    use_embeddings: bool = True
    u_sample_pairs: int = 60_000
    seed: int = 0
    deterministic_identifier_rule: bool = True
    household_guard: bool = True
    require_corroboration: bool = True


@dataclass
class PairScore:
    i: int
    j: int
    levels: dict[str, str]
    weights: dict[str, float]
    weight: float
    prob: float
    rule: str | None = None
    name_jw: float = 0.0
    emb_cos: float | None = None

    @property
    def is_match(self) -> bool:  # set by the scorer
        return self._match

    _match: bool = field(default=False, repr=False)

    def evidence(self) -> dict[str, Any]:
        return {
            "levels": self.levels,
            "weights": {k: round(v, 2) for k, v in self.weights.items() if v},
            "name_jw": round(self.name_jw, 3),
            "embedding_cosine": None if self.emb_cos is None else round(float(self.emb_cos), 3),
            "rule": self.rule,
        }


# --------------------------------------------------------------------------
# Comparison functions
# --------------------------------------------------------------------------


def jw(a: str | None, b: str | None) -> float:
    if not a or not b:
        return 0.0
    return float(JaroWinkler.similarity(a, b))


def cmp_first(a: NameParts, b: NameParts) -> str:
    if not a.first or not b.first:
        return "null"
    if a.first_forms & b.first_forms:
        return "exact"
    if len(a.first) == 1 or len(b.first) == 1:
        ia = {f[0] for f in a.first_forms} | {a.first[0]}
        ib = {f[0] for f in b.first_forms} | {b.first[0]}
        return "initial" if ia & ib else "diff"
    if max(jw(x, y) for x in a.first_forms for y in b.first_forms) >= 0.88:
        return "fuzzy"
    return "diff"


def cmp_last(a: NameParts, b: NameParts) -> str:
    if not a.last or not b.last:
        return "null"
    if a.last == b.last:
        return "exact"
    if jw(a.last, b.last) >= 0.90:
        return "fuzzy"
    return "diff"


def cmp_suffix(a: NameParts, b: NameParts) -> str:
    if a.suffix and b.suffix and a.suffix != b.suffix:
        return "conflict"
    return "ok"


def cmp_set(a: frozenset[str], b: frozenset[str]) -> str:
    if not a or not b:
        return "null"
    return "exact" if a & b else "diff"


def cmp_email(a: frozenset[str], b: frozenset[str], corporate: frozenset[str]) -> str:
    """Like :func:`cmp_set`, but a work address vs. a personal address is not a disagreement.

    People hold several mailboxes; disagreement is only evidence when both
    records describe the same *kind* of mailbox.
    """
    if not a or not b:
        return "null"
    if a & b:
        return "exact"
    kind = lambda s: {e.split("@", 1)[-1] in corporate for e in s}  # noqa: E731
    ka, kb = kind(a), kind(b)
    if len(ka) == 1 and len(kb) == 1 and ka != kb:
        return "null"
    return "diff"


def cmp_address(r: Rec, s: Rec) -> str:
    A, B = r.addr, s.addr
    if (not A.key and not A.zip) or (not B.key and not B.zip):
        return "null"
    if A.key and B.key and A.key == B.key and (A.zip == B.zip or not A.zip or not B.zip):
        return "exact"
    if A.number and A.number == B.number and A.street and B.street and jw(A.street, B.street) >= 0.9:
        return "street"
    if A.zip and A.zip == B.zip:
        return "zip"
    return "diff"


def cmp_employer(r: Rec, s: Rec) -> str:
    if not r.employer or not s.employer:
        return "null"
    return "match" if fuzz.token_set_ratio(r.employer, s.employer) >= 90 else "diff"


def cmp_embedding(cos: float | None) -> str:
    if cos is None:
        return "null"
    if cos >= 0.90:
        return "high"
    if cos >= 0.75:
        return "mid"
    return "low"


def cmp_org_name(r: Rec, s: Rec) -> str:
    if not r.org_name or not s.org_name:
        return "null"
    if r.org_name == s.org_name:
        return "exact"
    return "fuzzy" if jw(r.org_name, s.org_name) >= 0.93 else "diff"


def cmp_reg(r: Rec, s: Rec) -> str:
    if not r.reg_no or not s.reg_no:
        return "null"
    return "exact" if r.reg_no == s.reg_no else "diff"


def best_name_pair(r: Rec, s: Rec) -> tuple[NameParts, NameParts]:
    """Pick the (name, alias) combination that agrees best (aliases, display-name variants)."""
    if not r.aliases and not s.aliases:
        return r.name, s.name
    score = {"exact": 3, "fuzzy": 2, "initial": 1, "diff": 0, "null": 0}
    best = (r.name, s.name)
    best_s = -1
    for a in [r.name] + r.aliases:
        for b in [s.name] + s.aliases:
            sc = score[cmp_first(a, b)] + score[cmp_last(a, b)]
            if sc > best_s:
                best, best_s = (a, b), sc
    return best


def compare(r: Rec, s: Rec, cos: float | None, corporate: frozenset[str] = frozenset()) -> tuple[dict[str, str], float]:
    """Agreement levels for a pair, plus a name Jaro-Winkler for reporting."""
    if r.entity_type == "org":
        return {
            "org_name": cmp_org_name(r, s),
            "reg_no": cmp_reg(r, s),
            "address": cmp_address(r, s),
            "phone": cmp_set(r.phones, s.phones),
            "email": cmp_set(r.emails, s.emails),
        }, jw(r.org_name, s.org_name)
    a, b = best_name_pair(r, s)
    name_jw = max(jw(a.canonical_full, b.canonical_full), jw(a.full, b.full))
    if not r.addr.key or not s.addr.key:
        cos = None
    return {
        "first": cmp_first(a, b),
        "last": cmp_last(a, b),
        "dob": dob_relation(r.dob, s.dob),
        "phone": cmp_set(r.phones, s.phones),
        "email": cmp_email(r.emails, s.emails, corporate),
        "address": cmp_address(r, s),
        "employer": cmp_employer(r, s),
        "embedding": cmp_embedding(cos),
        "suffix": cmp_suffix(a, b),
    }, name_jw


# --------------------------------------------------------------------------
# Scorer
# --------------------------------------------------------------------------


class FellegiSunterScorer:
    """Estimates u-probabilities from the data and scores candidate pairs."""

    def __init__(self, records: list[Rec], cfg: MatchConfig, emb: np.ndarray | None = None) -> None:
        self.records = records
        self.cfg = cfg
        self.emb = emb
        self.u: dict[str, dict[str, float]] = {}
        self._tf: dict[str, Counter[str]] = {}
        self._tf_n: dict[str, int] = {}
        # domains seen on employee badge records are corporate mail domains
        self.corporate_domains = frozenset(
            e.split("@", 1)[-1] for r in records if r.source == "access" and r.attrs.get("card_type") == "EMPLOYEE" for e in r.emails
        )
        self._build_tf()
        self._estimate_u()
        by_type = Counter(r.entity_type for r in records)
        n = max(2, by_type.get("person", 2))
        lam = min(0.5, (cfg.records_per_entity - 1) / (n - 1))
        self.prior_log2_odds = math.log2(lam / (1 - lam))

    def _build_tf(self) -> None:
        tf: dict[str, Counter[str]] = {k: Counter() for k in ("first", "last", "phone", "email", "address", "org_name")}
        for r in self.records:
            if r.entity_type == "person":
                for f in r.name.first_forms:
                    tf["first"][f] += 1
                if r.name.last:
                    tf["last"][r.name.last] += 1
                if r.addr.key:
                    tf["address"][r.addr.key] += 1
            elif r.org_name:
                tf["org_name"][r.org_name] += 1
            for p in r.phones:
                tf["phone"][p] += 1
            for e in r.emails:
                tf["email"][e] += 1
        self._tf = tf
        self._tf_n = {k: max(1, sum(1 for r in self.records if self._has(k, r))) for k in tf}

    @staticmethod
    def _has(kind: str, r: Rec) -> bool:
        return bool(
            {
                "first": r.name.first,
                "last": r.name.last,
                "phone": r.phones,
                "email": r.emails,
                "address": r.addr.key,
                "org_name": r.org_name,
            }[kind]
        )

    def _cos(self, i: int, j: int) -> float | None:
        if self.emb is None:
            return None
        return float(self.emb[i] @ self.emb[j])

    def _estimate_u(self) -> None:
        rng = np.random.default_rng(self.cfg.seed)
        counts: dict[str, Counter[str]] = {k: Counter() for k in M_PROBS}
        totals: Counter[str] = Counter()
        for etype in ("person", "org"):
            idx = np.array([r.idx for r in self.records if r.entity_type == etype])
            if len(idx) < 2:
                continue
            k = self.cfg.u_sample_pairs if etype == "person" else min(5_000, self.cfg.u_sample_pairs)
            a = rng.choice(idx, size=k)
            b = rng.choice(idx, size=k)
            keep = a != b
            for i, j in zip(a[keep], b[keep]):
                levels, _ = compare(self.records[int(i)], self.records[int(j)], self._cos(int(i), int(j)), self.corporate_domains)
                for c, lv in levels.items():
                    if lv != "null":
                        counts[c][lv] += 1
                        totals[c] += 1
        for c, levels in M_PROBS.items():
            n = totals[c]
            self.u[c] = {lv: (counts[c][lv] + 1) / (n + len(levels)) for lv in levels}

    def _u_tf(self, comp: str, r: Rec, s: Rec) -> float:
        kind = comp
        tf, n = self._tf[kind], self._tf_n[kind]
        if comp == "first":
            vals = [tf[f] for f in (r.name.first_forms & s.name.first_forms)] or [1]
        elif comp == "last":
            vals = [tf[r.name.last]] if r.name.last else [1]  # type: ignore[index]
        elif comp == "phone":
            vals = [tf[p] for p in (r.phones & s.phones)] or [1]
        elif comp == "email":
            vals = [tf[e] for e in (r.emails & s.emails)] or [1]
        elif comp == "address":
            vals = [tf[r.addr.key]] if r.addr.key else [1]  # type: ignore[index]
        else:
            vals = [tf[r.org_name]] if r.org_name else [1]  # type: ignore[index]
        return max(max(vals) / n, 1.0 / n)

    def score(self, i: int, j: int) -> PairScore:
        r, s = self.records[i], self.records[j]
        cos = self._cos(i, j) if r.entity_type == "person" else None
        levels, name_jw = compare(r, s, cos, self.corporate_domains)
        weights: dict[str, float] = {}
        for comp, lv in levels.items():
            if lv == "null":
                weights[comp] = 0.0
                continue
            m = M_PROBS[comp][lv]
            u = self._u_tf(comp, r, s) if (comp, lv) in TF_LEVELS else self.u[comp][lv]
            weights[comp] = math.log2(m / max(u, 1e-9))
        if not self.cfg.use_embeddings:
            weights["embedding"] = 0.0
        if r.entity_type == "person" and self.cfg.household_guard and levels["first"] == "diff":
            weights["address"] = min(weights.get("address", 0.0), 0.0)
            weights["embedding"] = min(weights.get("embedding", 0.0), 0.0)
        total = sum(weights.values())
        logit = total + self.prior_log2_odds
        prob = 1.0 / (1.0 + 2.0 ** (-max(min(logit, 60), -60)))
        ps = PairScore(i, j, levels, weights, total, prob, name_jw=name_jw, emb_cos=cos)
        rule = None
        if self.cfg.deterministic_identifier_rule and r.entity_type == "person":
            ident = levels["phone"] == "exact" or levels["email"] == "exact"
            no_conflict = levels["first"] != "diff" and levels["last"] != "diff" and levels["dob"] != "diff" and levels["suffix"] != "conflict"
            if ident and no_conflict:
                rule = "shared_identifier"
        if r.entity_type == "org" and levels.get("reg_no") == "exact":
            rule = "registration_number"
        ps.rule = rule
        corroborated = True
        if r.entity_type == "person" and self.cfg.require_corroboration:
            corroborated = (
                (levels["first"] == "exact" and levels["last"] == "exact")
                or levels["dob"] in ("exact", "swap", "near")
                or levels["phone"] == "exact"
                or levels["email"] == "exact"
                or levels["address"] in ("exact", "street")
                or levels["employer"] == "match"
            )
        ps._match = (prob >= self.cfg.threshold and corroborated) or rule is not None
        return ps

    def describe(self) -> dict[str, Any]:
        return {
            "m": M_PROBS,
            "u": {c: {k: round(v, 6) for k, v in lv.items()} for c, lv in self.u.items()},
            "prior_log2_odds": round(self.prior_log2_odds, 3),
            "threshold": self.cfg.threshold,
        }
