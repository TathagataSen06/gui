"""Constrained union-find clustering of matched record pairs.

Plain transitive closure over-merges: a DOB-less e-mail record that looks
like both "Robert Kowalski Sr." and "Robert Kowalski Jr." would chain father
and son into one entity. We therefore process links in descending match
probability and refuse a union when the two clusters carry incompatible
dates of birth (a *cannot-link* constraint).
"""

from __future__ import annotations

from datetime import date
from typing import Iterable

from .normalize import dobs_compatible


class ConstrainedUnionFind:
    def __init__(self, n: int, dobs: list[date | None]) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n
        self.dobs: list[set[date]] = [{d} if d else set() for d in dobs]
        self.refused = 0
        self.refused_pairs: list[tuple[int, int]] = []

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def compatible(self, a: int, b: int) -> bool:
        da, db = self.dobs[a], self.dobs[b]
        if not da or not db:
            return True
        return any(dobs_compatible(x, y) for x in da for y in db)

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        if not self.compatible(ra, rb):
            self.refused += 1
            self.refused_pairs.append((a, b))
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.dobs[ra] |= self.dobs[rb]
        return True

    def clusters(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            out.setdefault(self.find(i), []).append(i)
        return out


def cluster_pairs(n: int, dobs: list[date | None], links: Iterable[tuple[int, int, float]]) -> ConstrainedUnionFind:
    """Cluster ``n`` records given ``(i, j, prob)`` links (processed strongest first)."""
    uf = ConstrainedUnionFind(n, dobs)
    for i, j, _ in sorted(links, key=lambda t: (-t[2], t[0], t[1])):
        uf.union(i, j)
    return uf
