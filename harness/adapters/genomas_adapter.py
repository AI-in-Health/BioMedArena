"""Adapter for Liu-Hy/GenoMAS — multi-agent gene expression analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.adapter_base import AdapterBase
from harness.adapters._vendor_mixin import try_vendor


class GenoMASAdapter(AdapterBase):
    name = "genomas"
    modality = "genomics"
    description = "Multi-agent gene expression analysis (Data Engineer + Code Reviewer + Domain Expert)."

    def __init__(self, config: dict | None = None, **kwargs: Any):
        self._config = config or {}
        self._llm = kwargs.get("llm")
        self._vendor_path = Path(self._config.get("vendor_path", "vendors/GenoMAS"))

    def capabilities(self) -> list[str]:
        return ["gene_expression", "multi_agent_analysis", "code_review", "domain_expertise", "statistical_analysis"]

    async def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        vendor_result = await try_vendor("genomas", query, context, timeout=20)
        if vendor_result is not None:
            return self.result(
                answer=vendor_result["answer"],
                evidence=vendor_result.get("evidence", []),
                confidence=vendor_result.get("confidence", 0.5),
                raw=vendor_result.get("raw"),
            )
        return self.result(answer=f"GenoMAS vendor unavailable. Query: {query}", confidence=0.2)
