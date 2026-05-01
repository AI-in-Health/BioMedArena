"""Adapter for ncbi-nlp/GeneAgent — gene-set analysis with self-verification via NCBI APIs."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from harness.adapter_base import AdapterBase


class GeneAgentAdapter(AdapterBase):
    name = "geneagent"
    modality = "genomics"
    description = "Gene-set analysis and self-verification pipeline using NCBI APIs (GeneAgent, Nature Methods)."

    def __init__(self, config: dict | None = None, **kwargs: Any):
        self._config = config or {}
        self._llm = kwargs.get("llm")
        self._vendor_path = Path(self._config.get("vendor_path", "vendors/GeneAgent"))
        # We use native implementation, so vendor clone not required
        # (but we still check to be informative)

    def capabilities(self) -> list[str]:
        return [
            "gene_set_analysis",
            "gene_function",
            "pathway_analysis",
            "ncbi_verification",
            "gene_enrichment",
        ]

    def _run_sync(self, query: str, genes: list[str]) -> dict[str, Any]:
        # Placeholder — real async native impl called from run() below
        return self.result(
            answer="Use async path",
            confidence=0.0,
        )

    async def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Use native GeneAgent implementation (vendor prompts + our LLM)."""
        if not self.available:
            return self.result(answer=self.unavailable_reason, confidence=0.0)

        # Import here to avoid circular deps
        try:
            from harness.adapters.geneagent_native import GeneAgentNative
        except ImportError as exc:
            return self.result(answer=f"GeneAgent native import failed: {exc}", confidence=0.0)

        ctx = context or {}
        genes = ctx.get("genes", [])
        if not genes:
            # Try extracting from query
            import re as _re
            genes = [g for g in _re.findall(r"\b([A-Z][A-Z0-9]{1,5})\b", query)
                     if g not in {"DNA", "RNA", "PCR", "BMI"}][:5]

        if not genes:
            return self.result(
                answer="GeneAgent requires a gene list (context['genes'] or detectable gene symbols in query).",
                confidence=0.1,
            )

        try:
            llm = self._llm  # set in __init__ via kwargs
            agent = GeneAgentNative(llm=llm)
            result = await agent.analyze(genes)
            return self.result(
                answer=result.get("summary", ""),
                evidence=result.get("evidence", []),
                confidence=0.8,
                raw=result,
            )
        except Exception as exc:
            return self.result(answer=f"GeneAgent error: {exc}", confidence=0.0)
