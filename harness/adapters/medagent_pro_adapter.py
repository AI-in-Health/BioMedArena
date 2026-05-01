"""Adapter for jinlab-imvr/MedAgent-Pro — multi-modal diagnosis via agentic workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.adapter_base import AdapterBase
from harness.adapters._vendor_mixin import try_vendor


class MedAgentProAdapter(AdapterBase):
    name = "medagent_pro"
    modality = "imaging"
    description = "Multi-modal medical image diagnosis via agentic workflow (MedAgent-Pro)."

    def __init__(self, config: dict | None = None, **kwargs: Any):
        self._config = config or {}
        self._llm = kwargs.get("llm")
        self._vendor_path = Path(self._config.get("vendor_path", "vendors/MedAgent-Pro"))

    def capabilities(self) -> list[str]:
        return ["medical_imaging", "multi_modal_diagnosis", "radiology", "pathology_imaging", "agentic_diagnosis"]

    async def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        vendor_result = await try_vendor("medagent_pro", query, context, timeout=25)
        if vendor_result is not None:
            return self.result(
                answer=vendor_result["answer"],
                evidence=vendor_result.get("evidence", []),
                confidence=vendor_result.get("confidence", 0.5),
                raw=vendor_result.get("raw"),
            )
        return self.result(answer=f"MedAgent-Pro vendor unavailable. Query: {query}", confidence=0.2)
