"""Entity resolution: normalization, comparisons, constrained clustering, quality vs. truth."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from argus.resolve.clustering import ConstrainedUnionFind
from argus.resolve.evaluate import blocking_recall, er_metrics
from argus.resolve.matching import cmp_first, cmp_last
from argus.resolve.normalize import dob_relation, normalize_org_name, parse_address, parse_name


def test_parse_name_layouts() -> None:
    a = parse_name("SMITH JR, ROBERT JAMES")
    assert (a.first, a.last, a.suffix, a.middle) == ("robert", "smith", "jr", ("james",))
    b = parse_name("Bob J. Smith")
    assert b.first == "bob" and "robert" in b.first_forms and b.last == "smith"
    c = parse_name('"Smith, Bob"')
    assert (c.first, c.last) == ("bob", "smith")
    assert parse_name("Mr. O'Brien-Kelly, Liz").last == "obrienkelly"
    assert parse_name(None).first is None
    assert parse_name("bob").last is None


def test_name_comparisons() -> None:
    assert cmp_first(parse_name("Bill Jones"), parse_name("William Jones")) == "exact"
    assert cmp_first(parse_name("W. Jones"), parse_name("William Jones")) == "initial"
    assert cmp_last(parse_name("Ann Kowalczyk"), parse_name("Ann Kowalcyzk")) == "fuzzy"
    assert cmp_last(parse_name("Ann Smith"), parse_name("Ann Brown")) == "diff"


def test_address_standardization() -> None:
    a = parse_address("12 North Maple Street Apt. 4B, Lake Paul, NJ 08812")
    b = parse_address("12 N MAPLE ST #4B, LAKE PAUL, NJ 08812")
    assert a.key == b.key == "12 n maple st" and a.zip == b.zip == "08812" and a.unit == b.unit == "4b"
    assert normalize_org_name("Bluefin Holdings, LLC") == normalize_org_name("BLUEFIN HOLDINGS LLC") == "bluefin holdings"


def test_dob_relations() -> None:
    d = date(1980, 3, 7)
    assert dob_relation(d, date(1980, 7, 3)) == "swap"
    assert dob_relation(d, date(1981, 3, 7)) == "near"
    assert dob_relation(d, date(1962, 11, 2)) == "diff"
    assert dob_relation(d, None) == "null"


def test_cannot_link_blocks_father_son_chain() -> None:
    # 0 = father (1962), 1 = son (1991), 2 = DOB-less record that looks like both
    uf = ConstrainedUnionFind(3, [date(1962, 4, 1), date(1991, 9, 9), None])
    assert uf.union(0, 2)
    assert not uf.union(1, 2)
    assert uf.find(0) == uf.find(2) != uf.find(1) and uf.refused == 1


def test_er_metrics_on_toy_clusters() -> None:
    truth = {"a": "1", "b": "1", "c": "1", "d": "2", "e": "2"}
    pred = {"a": "X", "b": "X", "c": "Y", "d": "Z", "e": "Z"}
    m = er_metrics(pred, truth)
    assert m.true_pairs == 4 and m.predicted_pairs == 2 and m.true_positive_pairs == 2
    assert m.pairwise_precision == 1.0 and m.pairwise_recall == 0.5
    br = blocking_recall({("a", "b"), ("d", "e")}, truth)
    assert br["pair_completeness"] == 0.5


def test_resolution_quality_against_truth(con, world_dir: Path) -> None:  # type: ignore[no-untyped-def]
    truth = pd.read_csv(world_dir / "ground_truth/record_truth.csv", dtype=str)
    person = truth[truth["entity_kind"] == "person"]
    pred = dict(con.execute("SELECT record_id, entity_id FROM record_entity").fetchall())
    m = er_metrics(pred, dict(zip(person["record_id"], person["true_id"])), list(person["record_id"]))
    assert m.pairwise_precision >= 0.97
    assert m.pairwise_recall >= 0.85
    assert m.predicted_entities < m.records  # it actually merged things


def test_merge_evidence_is_explainable(con) -> None:  # type: ignore[no-untyped-def]
    row = con.execute("SELECT evidence FROM er_pairs WHERE linked LIMIT 1").fetchone()
    import json

    ev = json.loads(row[0])
    assert set(ev) >= {"levels", "weights", "name_jw", "rule"}
    assert con.execute("SELECT count(*) FROM entity_links WHERE relation = 'controls'").fetchone()[0] > 0
