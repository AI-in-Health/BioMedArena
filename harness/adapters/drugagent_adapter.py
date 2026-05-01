"""Adapter for FermiQ/drugagent — multi-agent drug discovery ML programming."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.adapter_base import AdapterBase
from harness.adapters._vendor_mixin import try_vendor


class DrugAgentAdapter(AdapterBase):
    name = "drugagent"
    modality = "drug"
    description = "Multi-agent drug discovery via ML programming (AAAI 2025 Workshop)."

    def __init__(self, config: dict | None = None, **kwargs: Any):
        self._config = config or {}
        self._llm = kwargs.get("llm")
        self._vendor_path = Path(self._config.get("vendor_path", "vendors/drugagent"))

    def capabilities(self) -> list[str]:
        return ["drug_discovery", "molecular_property_prediction", "drug_repurposing", "admet_prediction", "compound_screening"]

    async def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        vendor_result = await try_vendor("drugagent", query, context, timeout=20)
        if vendor_result is not None:
            return self.result(
                answer=vendor_result["answer"],
                evidence=vendor_result.get("evidence", []),
                confidence=vendor_result.get("confidence", 0.5),
                raw=vendor_result.get("raw"),
            )
        return self.result(answer=f"DrugAgent vendor unavailable. Query: {query}", confidence=0.2)
