"""Adapter for mims-harvard/TxAgent — therapeutic reasoning with tool universe."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.adapter_base import AdapterBase
from harness.adapters._vendor_mixin import try_vendor


class TxAgentAdapter(AdapterBase):
    name = "txagent"
    modality = "reasoning"
    description = "Therapeutic reasoning agent with drug/treatment tool universe (Harvard)."

    def __init__(self, config: dict | None = None, **kwargs: Any):
        self._config = config or {}
        self._llm = kwargs.get("llm")
        self._vendor_path = Path(self._config.get("vendor_path", "vendors/TxAgent"))

    def capabilities(self) -> list[str]:
        return ["therapeutic_reasoning", "drug_interaction", "treatment_recommendation", "pharmacology", "clinical_guidelines"]

    async def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        vendor_result = await try_vendor("txagent", query, context, timeout=20)
        if vendor_result is not None:
            return self.result(
                answer=vendor_result["answer"],
                evidence=vendor_result.get("evidence", []),
                confidence=vendor_result.get("confidence", 0.5),
                raw=vendor_result.get("raw"),
            )
        return self.result(answer=f"TxAgent vendor unavailable. Query: {query}", confidence=0.2)
