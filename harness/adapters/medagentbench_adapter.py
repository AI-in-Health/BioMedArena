"""Adapter for clinical EHR benchmark — FHIR environment + benchmark tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.adapter_base import AdapterBase
from harness.adapters._vendor_mixin import try_vendor


class MedAgentBenchAdapter(AdapterBase):
    name = "medagentbench"
    modality = "ehr"
    description = "FHIR-based EHR agent environment with 300 benchmark tasks over 100 patients (NEJM AI)."

    def __init__(self, config: dict | None = None, **kwargs: Any):
        self._config = config or {}
        self._llm = kwargs.get("llm")
        self._vendor_path = Path(self._config.get("vendor_path", "vendors/MedAgentBench"))
        self._fhir_url = self._config.get("fhir_url", "http://localhost:8080/fhir")

    def capabilities(self) -> list[str]:
        return ["fhir_query", "patient_lookup", "ehr_tasks", "clinical_workflow", "order_entry"]

    async def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        vendor_result = await try_vendor("medagentbench", query, context, timeout=20)
        if vendor_result is not None:
            return self.result(
                answer=vendor_result["answer"],
                evidence=vendor_result.get("evidence", []),
                confidence=vendor_result.get("confidence", 0.5),
                raw=vendor_result.get("raw"),
            )
        return self.result(
            answer=f"MedAgentBench vendor unavailable. FHIR URL: {self._fhir_url}. Query: {query}",
            confidence=0.2,
        )
