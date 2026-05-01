"""Adapter for wshi83/EhrAgent — code-generation agent for EHR reasoning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.adapter_base import AdapterBase
from harness.adapters._vendor_mixin import try_vendor


class EHRAgentAdapter(AdapterBase):
    name = "ehragent"
    modality = "ehr"
    description = "Code-generation agent for EHR reasoning over structured patient data (EMNLP 2024)."

    def __init__(self, config: dict | None = None, **kwargs: Any):
        self._config = config or {}
        self._llm = kwargs.get("llm")
        self._vendor_path = Path(self._config.get("vendor_path", "vendors/EhrAgent"))

    def capabilities(self) -> list[str]:
        return ["ehr_reasoning", "patient_data_query", "code_generation", "temporal_reasoning", "lab_interpretation"]

    async def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        vendor_result = await try_vendor("ehragent", query, context, timeout=20)
        if vendor_result is not None:
            return self.result(
                answer=vendor_result["answer"],
                evidence=vendor_result.get("evidence", []),
                confidence=vendor_result.get("confidence", 0.5),
                raw=vendor_result.get("raw"),
            )
        return self.result(answer=f"EhrAgent vendor unavailable. Query: {query}", confidence=0.2)
