"""Abstract base class that every adapter must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AdapterBase(ABC):
    """Protocol for all medical AI adapters.

    Every adapter wraps a vendor project (or built-in tool) and exposes a
    uniform ``run()`` interface so the orchestrator can invoke them
    interchangeably.
    """

    name: str
    modality: str  # genomics | ehr | imaging | drug | wearable | reasoning
    description: str

    # Set by the adapter if vendor deps are missing or setup failed.
    available: bool = True
    unavailable_reason: str = ""

    @abstractmethod
    async def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute the adapter's pipeline and return a standardised result.

        Returns
        -------
        dict with keys:
            source      – adapter name
            answer      – human-readable answer string
            evidence    – list of supporting evidence strings
            confidence  – float 0-1
            raw         – vendor-specific raw output (for debugging)
        """

    def capabilities(self) -> list[str]:
        """Return capability tags used for routing (override in subclasses)."""
        return []

    def result(
        self,
        answer: str,
        evidence: list[str] | None = None,
        confidence: float = 0.5,
        raw: Any = None,
    ) -> dict[str, Any]:
        """Helper to build a standardised result dict."""
        return {
            "source": self.name,
            "answer": answer,
            "evidence": evidence or [],
            "confidence": confidence,
            "raw": raw,
        }

    def mark_unavailable(self, reason: str) -> None:
        self.available = False
        self.unavailable_reason = reason

    def __repr__(self) -> str:
        status = "available" if self.available else f"unavailable ({self.unavailable_reason})"
        return f"<{self.__class__.__name__} [{self.modality}] {status}>"
