"""Adapter for ChatMED/Prompt-to-Pill — end-to-end drug discovery + clinical trial simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.adapter_base import AdapterBase
from harness.adapters._vendor_mixin import try_vendor


class Prompt2PillAdapter(AdapterBase):
    name = "prompt2pill"
    modality = "drug"
    description = "End-to-end drug discovery pipeline with clinical trial simulation."

    def __init__(self, config: dict | None = None, **kwargs: Any):
        self._config = config or {}
        self._llm = kwargs.get("llm")
        self._vendor_path = Path(self._config.get("vendor_path", "vendors/Prompt-to-Pill"))

    def capabilities(self) -> list[str]:
        return ["drug_discovery", "molecule_generation", "clinical_trial_simulation", "lead_optimization"]

    async def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        vendor_result = await try_vendor("prompt2pill", query, context, timeout=20)
        if vendor_result is not None:
            return self.result(
                answer=vendor_result["answer"],
                evidence=vendor_result.get("evidence", []),
                confidence=vendor_result.get("confidence", 0.5),
                raw=vendor_result.get("raw"),
            )
        return self.result(answer=f"Prompt-to-Pill vendor unavailable. Query: {query}", confidence=0.2)
