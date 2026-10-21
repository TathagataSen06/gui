"""Filesystem layout of a generated world / pipeline run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    """All paths used by one synthetic world and the pipeline run on top of it.

    Layout::

        <root>/
          manifest.json          synthetic-data manifest (required by ingest)
          raw/<source>/...       raw feeds, one directory per source system
          ground_truth/...       hidden truth for evaluation only
          argus.duckdb           warehouse produced by the pipeline
          reports/               evaluation reports
          case_actions.jsonl     operator dispositions recorded by the dashboard
    """

    root: Path

    @classmethod
    def at(cls, root: str | Path) -> "RunPaths":
        return cls(Path(root).expanduser().resolve())

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def ground_truth(self) -> Path:
        return self.root / "ground_truth"

    @property
    def db(self) -> Path:
        return self.root / "argus.duckdb"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def model(self) -> Path:
        return self.root / "models" / "risk_model.joblib"

    @property
    def case_actions(self) -> Path:
        return self.root / "case_actions.jsonl"

    def ensure(self) -> "RunPaths":
        for p in (self.root, self.raw, self.ground_truth, self.reports):
            p.mkdir(parents=True, exist_ok=True)
        return self
