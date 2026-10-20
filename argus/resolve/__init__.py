"""Entity resolution: blocking, Fellegi-Sunter matching, constrained clustering."""

from .matching import MatchConfig
from .pipeline import ResolutionResult, resolve, run

__all__ = ["MatchConfig", "ResolutionResult", "resolve", "run"]
