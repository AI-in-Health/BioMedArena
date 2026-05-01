"""Adapter for mitmedialab/MDAgents — adaptive multi-agent medical decision-making.

Uses native reimplementation (harness.adapters.mdagents_native) that reuses
the vendor's multi-agent workflow but plugs into our modern LLMClient.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.adapter_base import AdapterBase


class MDAgentsAdapter(AdapterBase):
    name = "mdagents"
    modality = "reasoning"
    description = "Adaptive multi-agent medical decision-making (native reimpl. of MDAgents, NeurIPS 2024)."

    def __init__(self, config: dict | None = None, **kwargs: Any):
        self._config = config or {}
        self._llm = kwargs.get("llm")
        self._vendor_path = Path(self._config.get("vendor_path", "vendors/MDAgents"))

    def capabilities(self) -> list[str]:
        return [
            "clinical_reasoning",
            "differential_diagnosis",
            "treatment_planning",
            "multi_agent_consultation",
            "medical_decision_making",
        ]

    async def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._llm is None:
            return self.result(answer="MDAgents requires an LLM client.", confidence=0.0)

        try:
            from harness.adapters.mdagents_native import MDAgentsNative
        except ImportError as exc:
            return self.result(answer=f"MDAgents native import failed: {exc}", confidence=0.0)

        try:
            agent = MDAgentsNative(llm=self._llm)
            answer, difficulty = await agent.answer(query)
            return self.result(
                answer=answer,
                evidence=[f"Difficulty: {difficulty}", "Multi-agent panel consultation"],
                confidence=0.8 if difficulty == "advanced" else 0.75,
                raw={"difficulty": difficulty, "answer": answer},
            )
        except Exception as exc:
            return self.result(answer=f"MDAgents error: {exc}", confidence=0.0)
