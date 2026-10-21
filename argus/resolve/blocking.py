"""Candidate generation (blocking).

Comparing every pair of records is quadratic, so candidate pairs are drawn
only from records that share at least one *blocking key*. Keys are designed
so that each realistic defect leaves at least one key intact:

==========  =============================================  ============================
key         definition                                      survives
==========  =============================================  ============================
``ph``      exact E.164 phone                               any name noise
``em``      exact normalized e-mail                         any name noise
``nm``      metaphone(last) + initial of canonical first    typos, nicknames, initials
``fd``      metaphone(canonical first) + date of birth      surname change (marriage)
``ad``      house number + metaphone(street) + ZIP          name noise, missing DOB
``ln``      metaphone(last) + birth year                    first-name noise
``org``     normalized org name / tax registration number   corporate suffix variation
==========  =============================================  ============================

Blocks larger than ``max_block`` are dropped (and reported): a phone number
shared by hundreds of records is a switchboard, not an identity signal.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

from .normalize import NameParts, metaphone
from .records import Rec


@dataclass
class BlockingResult:
    pairs: set[tuple[int, int]]
    key_counts: dict[str, int] = field(default_factory=dict)
    dropped_blocks: dict[str, int] = field(default_factory=dict)
    pair_sources: dict[tuple[int, int], set[str]] = field(default_factory=dict)


def _name_keys(n: NameParts) -> set[str]:
    keys: set[str] = set()
    if n.last and n.first:
        lm = metaphone(n.last)
        initials = {f[0] for f in (n.first_forms or {n.first})}
        for i in initials:
            keys.add(f"nm:{lm}:{i}")
    return keys


def record_keys(r: Rec) -> set[str]:
    keys: set[str] = set()
    for p in r.phones:
        keys.add(f"ph:{p}")
    for e in r.emails:
        keys.add(f"em:{e}")
    if r.entity_type == "org":
        if r.org_name:
            keys.add(f"org:{r.org_name}")
        if r.reg_no:
            keys.add(f"reg:{r.reg_no}")
        return keys
    names = [r.name] + r.aliases
    for n in names:
        keys |= _name_keys(n)
        if n.first and r.dob:
            for f in n.first_forms or {n.first}:
                if len(f) > 1:
                    keys.add(f"fd:{metaphone(f)}:{r.dob.isoformat()}")
        if n.last and r.dob:
            keys.add(f"ln:{metaphone(n.last)}:{r.dob.year}")
    if r.addr.number and r.addr.street and r.addr.zip:
        keys.add(f"ad:{r.addr.number}:{metaphone(r.addr.street.split()[0])}:{r.addr.zip}")
    return keys


def block(records: list[Rec], max_block: int = 40, max_identifier_block: int = 12) -> BlockingResult:
    """Return candidate index pairs ``(i, j)`` with ``i < j``."""
    index: dict[str, list[int]] = defaultdict(list)
    for r in records:
        for k in sorted(record_keys(r)):
            index[k].append(r.idx)
    pairs: set[tuple[int, int]] = set()
    pair_sources: dict[tuple[int, int], set[str]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    dropped: dict[str, int] = defaultdict(int)
    for key in sorted(index):
        members = index[key]
        kind = key.split(":", 1)[0]
        cap = max_identifier_block if kind in ("ph", "em") else max_block
        if len(members) < 2:
            continue
        if len(members) > cap:
            dropped[kind] += 1
            continue
        for i, j in combinations(sorted(set(members)), 2):
            if records[i].entity_type != records[j].entity_type:
                continue
            pairs.add((i, j))
            pair_sources[(i, j)].add(kind)
            counts[kind] += 1
    return BlockingResult(pairs, dict(counts), dict(dropped), dict(pair_sources))
