"""Adapter for clinical simulation benchmark — multimodal clinical scenarios."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.adapter_base import AdapterBase
from harness.adapters._vendor_mixin import try_vendor


class AgentClinicAdapter(AdapterBase):
    name = "agentclinic"
    modality = "reasoning"
    description = "Multimodal clinical simulation and diagnostic reasoning benchmark."

    def __init__(self, config: dict | None = None, **kwargs: Any):
        self._config = config or {}
        self._llm = kwargs.get("llm")
        self._vendor_path = Path(self._config.get("vendor_path", "vendors/AgentClinic"))

    def capabilities(self) -> list[str]:
        return ["clinical_simulation", "diagnostic_reasoning", "patient_interaction", "multimodal_diagnosis"]

    async def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        vendor_result = await try_vendor("agentclinic", query, context, timeout=20)
        if vendor_result is not None:
            return self.result(
                answer=vendor_result["answer"],
                evidence=vendor_result.get("evidence", []),
                confidence=vendor_result.get("confidence", 0.5),
                raw=vendor_result.get("raw"),
            )
        return self.result(answer=f"AgentClinic vendor unavailable. Query: {query}", confidence=0.2)
