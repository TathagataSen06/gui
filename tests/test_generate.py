"""Synthetic data generator: determinism, safety of identifiers, noise, ground truth."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from argus.generate.config import GeneratorConfig, Rng
from argus.generate.identifiers import IdentifierFactory, is_fictional_phone, is_synthetic_email
from argus.generate.noise import corrupt_dob, format_phone, nickname, phone_digit_typo, typo
from argus.generate.state import Clock, us_holidays
from argus.generate.world import generate_world


def _digest(root: Path) -> dict[str, str]:
    files = sorted(p for p in root.rglob("*") if p.is_file() and ("raw" in p.parts or "ground_truth" in p.parts))
    return {str(p.relative_to(root)): hashlib.sha1(p.read_bytes()).hexdigest() for p in files}


def test_same_seed_is_byte_identical(tmp_path: Path) -> None:
    cfg = GeneratorConfig(n_people=140, seed=5, days=35)
    generate_world(cfg, tmp_path / "a")
    generate_world(cfg, tmp_path / "b")
    assert _digest(tmp_path / "a") == _digest(tmp_path / "b")


def test_different_seed_differs(tmp_path: Path) -> None:
    generate_world(GeneratorConfig(n_people=140, seed=5, days=35), tmp_path / "a")
    generate_world(GeneratorConfig(n_people=140, seed=6, days=35), tmp_path / "b")
    assert _digest(tmp_path / "a") != _digest(tmp_path / "b")


def test_layout_and_manifest(world_dir: Path) -> None:
    m = json.loads((world_dir / "manifest.json").read_text())
    assert m["synthetic"] is True and m["marker"].startswith("ARGUS-SYNTHETIC")
    for rel in ("raw/bank/FMB/customers.csv", "raw/telco/subscribers.csv", "raw/access_control/cardholders.csv", "raw/watchlist/watchlist.csv", "raw/labels/case_outcomes.csv"):
        assert (world_dir / rel).exists(), rel
    assert list((world_dir / "raw/telco/cdr").glob("cdr_*.csv"))
    assert list((world_dir / "raw/email").glob("headers_*.jsonl"))
    assert list((world_dir / "raw/access_control/events").glob("access_*.jsonl"))


def test_all_phone_numbers_and_emails_are_fictional(world_dir: Path) -> None:
    subs = pd.read_csv(world_dir / "raw/telco/subscribers.csv", dtype=str)
    assert all(is_fictional_phone("+" + m) for m in subs["msisdn"])
    for f in sorted((world_dir / "raw/bank").glob("*/customers.csv")):
        c = pd.read_csv(f, dtype=str, keep_default_na=False)
        for e in c["email"]:
            if e:
                assert is_synthetic_email(e), e
    cdr = pd.concat(pd.read_csv(f, dtype=str) for f in sorted((world_dir / "raw/telco/cdr").glob("*.csv")))
    called = cdr["called_number"].str.replace("+", "", regex=False).str[-10:]
    assert called.str.match(r"^[2-9]\d{2}55501\d{2}$").all()


def test_ground_truth_is_consistent(world_dir: Path) -> None:
    gt = world_dir / "ground_truth"
    rings = json.loads((gt / "scenarios.json").read_text())
    persons = set(pd.read_csv(gt / "persons.csv", dtype=str)["person_id"])
    labels = pd.read_csv(gt / "person_labels.csv", dtype=str, keep_default_na=False)
    assert {r["typology"] for r in rings} == {"shell_layering", "structuring", "mule_funnel", "insider_collusion"}
    members = {m["person_id"] for r in rings for m in r["members"]}
    assert members <= persons
    assert set(labels.loc[labels["label"] == "1", "person_id"]) == members
    planted = pd.read_csv(gt / "planted_events.csv", dtype=str)
    assert set(planted["ring_id"]) == {r["ring_id"] for r in rings}
    assert set(planted["source"]) >= {"bank", "telco", "email", "access"}
    truth = pd.read_csv(gt / "record_truth.csv", dtype=str)
    assert truth["record_id"].is_unique


def test_structuring_deposits_sit_under_threshold(world_dir: Path) -> None:
    gt = world_dir / "ground_truth"
    planted = pd.read_csv(gt / "planted_events.csv", dtype=str)
    rings = {r["ring_id"]: r["typology"] for r in json.loads((gt / "scenarios.json").read_text())}
    ids = set(planted.loc[planted["ring_id"].map(rings) == "structuring", "native_id"])
    tx = pd.concat(pd.read_csv(f, dtype=str) for f in sorted((world_dir / "raw/bank").glob("*/transactions_*.csv")))
    cash = tx[tx["txn_id"].isin(ids) & (tx["channel"] == "CASH")]["amount"].astype(float)
    assert len(cash) > 0 and cash.between(8000, 10000, inclusive="left").all()


def test_noise_functions() -> None:
    rng = np.random.default_rng(0)
    for word in ("Jonathan", "Kowalczyk", "Margaret"):
        t = typo(word, rng)
        assert t != word or len(word) < 3
        assert abs(len(t) - len(word)) <= 1
    assert nickname("Robert", np.random.default_rng(1)) in {"Bob", "Rob", "Bobby", "Robbie"}
    assert nickname("Xavier", rng) == "Xavier"
    assert format_phone("+12015550142", "paren") == "(201) 555-0142"
    assert format_phone("+12015550142", "one_dash") == "1-201-555-0142"
    for _ in range(50):
        assert is_fictional_phone(phone_digit_typo("+12015550142", rng))
    d = date(1985, 3, 7)
    assert corrupt_dob(d, rng, swap_p=1.0, typo_p=0.0) == date(1985, 7, 3)


def test_identifier_factory_unique_and_fictional() -> None:
    f = IdentifierFactory(np.random.default_rng(3))
    nums = {f.msisdn() for _ in range(3000)}
    assert len(nums) == 3000 and all(is_fictional_phone(n) for n in nums)
    assert f.reg_no().startswith("00-")


def test_calendar_and_rng_streams() -> None:
    assert date(2026, 7, 3) in us_holidays(2026)  # Independence Day observed (July 4 is a Saturday)
    clock = Clock(date(2026, 1, 1), 31, "America/New_York")
    assert not clock.business[0]  # New Year's Day
    a, b = Rng(1).stream("bank").random(), Rng(1).stream("bank").random()
    assert a == b and Rng(1).stream("bank").random() != Rng(1).stream("telco").random()


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        GeneratorConfig(n_people=50)
