"""External-benchmark harness: loaders, protocols, provenance and reporting.

The fixtures below are tiny generated files in each dataset's real format; no
network access and no real data are needed.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from argus.realworld import creditcard, dblp_acm, elliptic, sources
from argus.realworld.report import render, run_all


# --------------------------------------------------------------------------
# Fixtures in the real file formats
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def creditcard_csv(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(0)
    n = 3000
    y = (rng.random(n) < 0.03).astype(int)
    df = pd.DataFrame({"Time": np.sort(rng.uniform(0, 172_800, n))})
    for i in range(1, 29):
        df[f"V{i}"] = rng.normal(size=n) + (2.5 * y if i in (4, 11) else 0)
    df["Amount"] = rng.exponential(80, n)
    df["Class"] = [f"'{v}'" for v in y]  # quoted labels, as in one public mirror
    p = tmp_path_factory.mktemp("cc") / "creditcard.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture(scope="module")
def elliptic_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(1)
    d = tmp_path_factory.mktemp("ell")
    rows, edges, classes = [], [], []
    tx = 1000
    for step in range(30, 41):
        ids = list(range(tx, tx + 60))
        tx += 60
        for t in ids:
            lab = rng.choice(["1", "2", "unknown"], p=[0.15, 0.45, 0.40])
            shift = 2.0 if lab == "1" else 0.0
            rows.append([t, step, *(rng.normal(size=6) + shift)])
            classes.append((t, lab))
        for _ in range(80):
            a, b = rng.choice(ids, 2, replace=False)
            edges.append((a, b))
    pd.DataFrame(rows).to_csv(d / "elliptic_txs_features.csv", header=False, index=False)
    pd.DataFrame(edges, columns=["txId1", "txId2"]).drop_duplicates().to_csv(d / "elliptic_txs_edgelist.csv", index=False)
    pd.DataFrame(classes, columns=["txId", "class"]).to_csv(d / "elliptic_txs_classes.csv", index=False)
    return d


WORDS = "query optimization index transaction stream xml mining join cache spatial temporal view olap cube schema peer ranking skyline sampling recovery".split()
NAMES = "alice chen bob kumar carol smith dan lee erin park frank zhou gina rossi hank novak".split()


@pytest.fixture(scope="module")
def dblp_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(2)
    d = tmp_path_factory.mktemp("dblp")
    dblp, acm, gold = [], [], []
    for i in range(60):
        title = " ".join(rng.choice(WORDS, 5, replace=False))
        authors = ", ".join(f"{rng.choice(NAMES)} {rng.choice(NAMES)}" for _ in range(2))
        year = int(rng.integers(1995, 2004))
        dblp.append({"id": f"conf/x/{i}", "title": title.title(), "authors": authors, "venue": "SIGMOD Conference", "year": year})
        if i < 45:  # matched in ACM with formatting noise
            acm.append({"id": 9000 + i, "title": title + " .", "authors": authors.upper(), "venue": "International Conference on Management of Data", "year": year})
            gold.append((f"conf/x/{i}", 9000 + i))
    for j in range(15):
        acm.append({"id": 9500 + j, "title": " ".join(rng.choice(WORDS, 5, replace=False)), "authors": "x y", "venue": "VLDB", "year": 2000})
    pd.DataFrame(dblp).to_csv(d / "DBLP2.csv", index=False)
    pd.DataFrame(acm).to_csv(d / "ACM.csv", index=False)
    pd.DataFrame(gold).to_csv(d / "DBLP-ACM_perfectMapping.csv", index=False, header=False)
    pb = d / "pre_blocked"
    pb.mkdir()
    side = lambda r: f"COL title VAL {r['title'].lower()} COL authors VAL {r['authors'].lower()} COL venue VAL {r['venue'].lower()} COL year VAL {r['year']}"  # noqa: E731
    for split, rng_seed in (("train", 3), ("valid", 4), ("test", 5)):
        r2 = np.random.default_rng(rng_seed)
        lines = []
        for _ in range(80):
            i = int(r2.integers(0, 45))
            if r2.random() < 0.5:
                lines.append(f"{side(dblp[i])}\t{side(acm[i])}\t1")
            else:
                j = int(r2.integers(0, len(acm)))
                lines.append(f"{side(dblp[i])}\t{side(acm[j])}\t{int(j == i)}")
        (pb / f"{split}.txt").write_text("\n".join(lines) + "\n")
    return d


# --------------------------------------------------------------------------
# Credit card
# --------------------------------------------------------------------------


def test_creditcard_load_checks_counts(creditcard_csv: Path) -> None:
    df = creditcard.load(creditcard_csv, expect=None)
    assert df["Class"].isin([0, 1]).all() and "log_amount" in df
    with pytest.raises(ValueError, match="expected"):
        creditcard.load(creditcard_csv)


def test_creditcard_protocols(creditcard_csv: Path) -> None:
    cfg = creditcard.CreditCardConfig(bootstrap=50, random_repeats=2)
    r = creditcard.run(creditcard_csv, cfg, expect=None)
    oot = r["out_of_time"]
    assert oot["train_rows"] == int(0.8 * r["rows"]) and oot["train_rows"] + oot["test_rows"] == r["rows"]
    for m in oot["models"].values():
        assert m["roc_auc"] > 0.9 and m["roc_auc_ci95"][0] <= m["roc_auc"] <= m["roc_auc_ci95"][1]
        assert set(m["budgets"]) == {"0.1%", "0.5%"}
    assert len(r["random_60_40"]["models"]["gbm"]["auprc"]["values"]) == 2
    assert oot["auto"]["chosen"] in cfg.models and set(oot["auto"]["validation_auprc"]) == set(cfg.models)
    assert "Time" not in r["features"]


# --------------------------------------------------------------------------
# Elliptic
# --------------------------------------------------------------------------


def test_elliptic_temporal_split_and_metrics(elliptic_dir: Path) -> None:
    cfg = elliptic.EllipticConfig(betweenness_k=8, models=("argus_logreg", "weber_rf"))
    r = elliptic.run(elliptic_dir, cfg, expect=False)
    assert r["train"]["rows"] > 0 and r["test"]["rows"] > 0
    res = r["results"]["argus_logreg|AF+GF"]
    assert res["n_features"] == 1 + 6 + len(elliptic.GRAPH_FEATURES)
    assert set(res["per_step"]) == set(range(35, 41)) and res["f1"] > 0.5
    assert r["results"]["weber_rf|LF"]["n_features"] == 7


def test_elliptic_graph_features_cover_every_time_step(elliptic_dir: Path) -> None:
    """Sampled betweenness is computed per time step, so no step is left all-zero by the sampling."""
    data = elliptic.load(elliptic_dir, expect=False)
    gf = elliptic.graph_features(data, elliptic.EllipticConfig(betweenness_k=4))
    by_step = gf.assign(step=data.nodes["time_step"].to_numpy()).groupby("step")["g_betweenness"].max()
    assert (by_step > 0).all()


def test_elliptic_rejects_wrong_counts(elliptic_dir: Path) -> None:
    with pytest.raises(ValueError, match="expected"):
        elliptic.load(elliptic_dir)


# --------------------------------------------------------------------------
# DBLP-ACM
# --------------------------------------------------------------------------


def test_dblp_venue_and_ditto_parsing() -> None:
    assert dblp_acm.canonical_venue("International Conference on Management of Data") == dblp_acm.canonical_venue("SIGMOD Conference")
    assert dblp_acm.canonical_venue("The VLDB Journal &mdash; The International Journal on Very Large Data Bases") == "vldb_journal"
    side = dblp_acm.parse_ditto_side("COL title VAL a b c COL authors VAL x , y COL venue VAL vldb COL year VAL 2001")
    assert side == {"title": "a b c", "authors": "x , y", "venue": "vldb", "year": "2001"}


def test_dblp_full_table_and_pre_blocked(dblp_dir: Path) -> None:
    cfg = dblp_acm.DblpAcmConfig(u_sample_pairs=2000, folds=3, knn=3)
    r = dblp_acm.run(dblp_dir, cfg, expect=False)
    ft = r["full_table"]
    assert ft["blocking"]["pair_completeness"] == 1.0
    fs = ft["matchers"]["fellegi_sunter"]
    assert fs["labels_used"] == 0 and fs["f1"] > 0.9
    assert fs["recall"] == fs["true_positive_pairs"] / 45  # unrounded, from counts
    assert ft["matchers"]["argus_gbm"]["predicted_pairs"] <= 45  # one-to-one
    assert r["pre_blocked"]["models"]["argus_logreg"]["f1"] > 0.9


def test_one_to_one_links_each_record_once() -> None:
    pairs = np.array([[0, 0], [0, 1], [1, 1], [2, 2]])
    links = dblp_acm.one_to_one(pairs, np.array([0.99, 0.98, 0.97, 0.5]), threshold=0.9)
    assert links == [(0, 0), (1, 1)]


# --------------------------------------------------------------------------
# Provenance and reporting
# --------------------------------------------------------------------------


def test_fetch_records_provenance_and_rejects_bad_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src = tmp_path / "src"
    src.mkdir()
    with zipfile.ZipFile(src / "pack.zip", "w") as z:
        z.writestr("inner/a.csv", "x\n1\n")
    (src / "b.csv").write_text("y\n2\n")
    spec = sources.DatasetSpec(
        key="toy", title="Toy", homepage="", license="CC0", citation="",
        files=(
            sources.FileSpec("a.csv", (sources.Source((src / "pack.zip").as_uri(), member="a.csv"),)),
            sources.FileSpec("b.csv", (sources.Source((src / "missing.csv").as_uri()), sources.Source((src / "b.csv").as_uri(), "mirror")), size=4),
        ),
    )
    monkeypatch.setitem(sources.CATALOG, "toy", spec)
    prov = sources.fetch("toy", tmp_path / "ext")
    assert (tmp_path / "ext" / "toy" / "a.csv").read_text() == "x\n1\n"
    assert prov["files"]["b.csv"]["source_kind"] == "mirror" and len(prov["files"]["b.csv"]["sha256"]) == 64
    assert not list((tmp_path / "ext" / "toy").glob("*.zip"))
    bad = sources.DatasetSpec(key="bad", title="Bad", homepage="", license="", citation="",
                              files=(sources.FileSpec("b.csv", (sources.Source((src / "b.csv").as_uri()),), size=999),))
    monkeypatch.setitem(sources.CATALOG, "bad", bad)
    with pytest.raises(sources.DatasetUnavailable, match="documented"):
        sources.fetch("bad", tmp_path / "ext")


def test_report_marks_unavailable_data_as_not_run(tmp_path: Path, dblp_dir: Path) -> None:
    out = run_all(["elliptic"], root=tmp_path / "empty", report_dir=tmp_path / "rep", download=False)
    assert out["datasets"]["elliptic"]["status"] == "not_run"
    md = (tmp_path / "rep" / "realworld.md").read_text()
    assert "Not run in this environment" in md and "data.pyg.org" in md and "| Model |" not in md
    r = dblp_acm.run(dblp_dir, dblp_acm.DblpAcmConfig(u_sample_pairs=2000, folds=3, knn=3), expect=False)
    prov = {"dataset": "DBLP-ACM", "license": "research", "files": {"ACM.csv": {"source": "local", "bytes": 1, "sha256": "0" * 64}}}
    md = render({"environment": {"python": "3", "scikit_learn": "1", "machine": "x"}, "datasets": {"dblp_acm": {"status": "ok", "provenance": prov, "result": r}}})
    assert "**Argus** Fellegi-Sunter" in md and "Ditto" in md and "measured here" in md
    json.dumps(r)  # results must be JSON-serializable
