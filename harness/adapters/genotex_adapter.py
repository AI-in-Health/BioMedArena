"""Adapter for gene expression analysis benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.adapter_base import AdapterBase
from harness.adapters._vendor_mixin import try_vendor


class GenoTEXAdapter(AdapterBase):
    name = "genotex"
    modality = "genomics"
    description = "Gene expression analysis: GEO/TCGA preprocessing, statistical analysis, DEG identification."

    def __init__(self, config: dict | None = None, **kwargs: Any):
        self._config = config or {}
        self._llm = kwargs.get("llm")
        self._vendor_path = Path(self._config.get("vendor_path", "vendors/GenoTEX"))

    def capabilities(self) -> list[str]:
        return [
            "gene_expression",
            "differential_expression",
            "geo_analysis",
            "tcga_analysis",
            "statistical_analysis",
            "trait_gene_association",
        ]

    async def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        vendor_result = await try_vendor("genotex", query, context, timeout=20)
        if vendor_result is not None:
            return self.result(
                answer=vendor_result["answer"],
                evidence=vendor_result.get("evidence", []),
                confidence=vendor_result.get("confidence", 0.5),
                raw=vendor_result.get("raw"),
            )

        ctx = context or {}
        trait = ctx.get("trait") or ctx.get("disease") or query
        return self.result(
            answer=f"GenoTEX vendor unavailable. Would analyze gene expression for trait '{trait}'.",
            confidence=0.2,
            raw={"mode": "fallback"},
        )
