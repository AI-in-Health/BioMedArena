"""Adapter for PKU-AICare/ColaCare — multi-agent EHR collaboration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.adapter_base import AdapterBase
from harness.adapters._vendor_mixin import try_vendor


class ColaCareAdapter(AdapterBase):
    name = "colacare"
    modality = "ehr"
    description = "Multi-agent collaborative EHR analysis for clinical prediction (WWW 2025)."

    def __init__(self, config: dict | None = None, **kwargs: Any):
        self._config = config or {}
        self._llm = kwargs.get("llm")
        self._vendor_path = Path(self._config.get("vendor_path", "vendors/ColaCare"))

    def capabilities(self) -> list[str]:
        return ["ehr_prediction", "mortality_prediction", "readmission_prediction", "multi_agent_collaboration"]

    async def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        vendor_result = await try_vendor("colacare", query, context, timeout=20)
        if vendor_result is not None:
            return self.result(
                answer=vendor_result["answer"],
                evidence=vendor_result.get("evidence", []),
                confidence=vendor_result.get("confidence", 0.5),
                raw=vendor_result.get("raw"),
            )
        return self.result(answer=f"ColaCare vendor unavailable. Query: {query}", confidence=0.2)
