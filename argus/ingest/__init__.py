"""Ingestion: raw feeds -> common event schema in DuckDB."""

from .base import SourceAdapter, register_adapter, registered_adapters
from .guard import NonSyntheticDataError
from .pipeline import IngestReport, connect, ingest

__all__ = ["SourceAdapter", "register_adapter", "registered_adapters", "ingest", "connect", "IngestReport", "NonSyntheticDataError"]
