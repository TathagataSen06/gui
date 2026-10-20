"""DuckDB connection helper used by every stage."""

from __future__ import annotations

from pathlib import Path

import duckdb


def connect(db_path: str | Path, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the warehouse with interactive niceties (progress bar) disabled."""
    con = duckdb.connect(str(db_path), read_only=read_only)
    con.execute("SET enable_progress_bar = false")
    return con
