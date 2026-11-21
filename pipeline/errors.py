"""Exceptions with stable codes, so a backend can store them and show a useful message."""
from __future__ import annotations


class AnalysisError(Exception):
    """A video could not be analysed. `code` is stable and machine-readable."""

    def __init__(self, code: str, message: str, detail: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail or {}

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, **({"detail": self.detail} if self.detail else {})}


class AnalysisCancelled(Exception):
    """The caller asked to stop. Completed work and checkpoints are kept; resume by re-running."""
