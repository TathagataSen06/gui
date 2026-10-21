"""Shared fixtures: one small synthetic world, generated and fully processed once per session."""

from __future__ import annotations

from pathlib import Path

import pytest

from argus.generate.config import GeneratorConfig
from argus.generate.world import generate_world

SMALL = dict(n_people=320, seed=11, days=120)


@pytest.fixture(scope="session")
def world_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("world")
    generate_world(GeneratorConfig(**SMALL), root)
    return root


@pytest.fixture(scope="session")
def ingested(world_dir: Path) -> Path:
    from argus.ingest import ingest

    ingest(world_dir)
    return world_dir


@pytest.fixture(scope="session")
def resolved(ingested: Path) -> Path:
    from argus.resolve import run

    run(ingested)
    return ingested


@pytest.fixture(scope="session")
def graphed(resolved: Path) -> Path:
    from argus.graph import run

    run(resolved)
    return resolved


@pytest.fixture(scope="session")
def scored(graphed: Path) -> Path:
    from argus.score import run

    run(graphed)
    return graphed


@pytest.fixture()
def con(scored: Path):  # type: ignore[no-untyped-def]
    from argus.db import connect

    c = connect(scored / "argus.duckdb", read_only=True)
    yield c
    c.close()
