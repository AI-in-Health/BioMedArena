"""Adapter for ncbi/GeneGPT — tool-augmented LLM for genomic QA via NCBI Web APIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.adapter_base import AdapterBase
from harness.adapters._vendor_mixin import try_vendor


class GeneGPTAdapter(AdapterBase):
    name = "genegpt"
    modality = "genomics"
    description = "Tool-augmented LLM for genomic question answering via NCBI E-utilities (GeneGPT)."

    def __init__(self, config: dict | None = None, **kwargs: Any):
        self._config = config or {}
        self._llm = kwargs.get("llm")
        self._vendor_path = Path(self._config.get("vendor_path", "vendors/GeneGPT"))

    def capabilities(self) -> list[str]:
        return [
            "genomic_qa",
            "gene_lookup",
            "snp_lookup",
            "gene_disease_association",
            "ncbi_api_tools",
        ]

    async def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        # 1. Try real vendor subprocess
        vendor_result = await try_vendor("genegpt", query, context, timeout=20)
        if vendor_result is not None:
            return self.result(
                answer=vendor_result["answer"],
                evidence=vendor_result.get("evidence", []),
                confidence=vendor_result.get("confidence", 0.5),
                raw=vendor_result.get("raw"),
            )

        # 2. Fallback: explain what GeneGPT would do
        return self.result(
            answer=(
                f"GeneGPT vendor unavailable. Original used Codex (deprecated); "
                f"native fallback: searches NCBI E-utilities for genomic QA. Query: {query}"
            ),
            confidence=0.2,
            raw={"mode": "fallback"},
        )
