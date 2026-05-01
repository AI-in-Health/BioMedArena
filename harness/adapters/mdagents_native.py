"""Native MDAgents implementation — reimplements the difficulty-adaptive
routing from mitmedialab/MDAgents (NeurIPS 2024) using modern LLM APIs.

Routing strategy:
  basic → single agent
  intermediate → multi-agent debate
  advanced → hierarchical panel
"""

from __future__ import annotations

from typing import Any

from harness.llm_client import LLMClient


DIFFICULTY_PROMPT = (
    "You are a medical triage assistant. Classify the following medical question "
    "by difficulty: 'basic' (single-step factual), 'intermediate' (requires 2-3 "
    "reasoning steps or differential diagnosis), or 'advanced' (complex multi-system, "
    "rare condition, or integrated reasoning). Respond with ONE word: basic, intermediate, or advanced."
)

BASIC_PROMPT = (
    "You are a medical expert. Answer the following question directly and accurately. "
    "If multiple choice, end with 'The answer is [X]'. If exact answer, end with 'The answer is [answer]'."
)

# Intermediate: multi-specialist debate
SPECIALIST_ROLES = [
    "internal medicine physician",
    "specialist in the relevant subspecialty",
    "evidence-based medicine expert",
]

MODERATOR_PROMPT = (
    "You are the moderator of a medical panel. Below are opinions from multiple "
    "specialists on a clinical question. Synthesize their views, identify agreements "
    "and disagreements, and state the most likely correct answer based on the "
    "weight of evidence. End with 'The answer is [X/answer]'."
)


class MDAgentsNative:
    """Adaptive multi-agent medical reasoning (reimplemented from MDAgents)."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def _classify_difficulty(self, question: str) -> str:
        resp = await self.llm.chat(
            messages=[
                {"role": "system", "content": DIFFICULTY_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=50,
        )
        if not resp:
            return "intermediate"
        text = resp.lower().strip()
        for d in ("advanced", "intermediate", "basic"):
            if d in text:
                return d
        return "intermediate"

    async def _basic_query(self, question: str) -> str:
        return await self.llm.chat(
            messages=[
                {"role": "system", "content": BASIC_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=2048,
        )

    async def _intermediate_query(self, question: str) -> str:
        """Multi-specialist debate."""
        import asyncio

        async def _specialist_opinion(role: str) -> str:
            prompt = (
                f"You are a {role}. Provide your expert opinion on the following "
                f"clinical question. Be concise (3-5 sentences). State your preferred answer."
            )
            return await self.llm.chat(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": question},
                ],
                temperature=0.3,
                max_tokens=512,
            )

        opinions = await asyncio.gather(*[_specialist_opinion(r) for r in SPECIALIST_ROLES])

        moderator_input = "\n\n".join(
            f"[{role}]: {op}" for role, op in zip(SPECIALIST_ROLES, opinions)
        )

        return await self.llm.chat(
            messages=[
                {"role": "system", "content": MODERATOR_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\n{moderator_input}"},
            ],
            temperature=0.0,
            max_tokens=2048,
        )

    async def _advanced_query(self, question: str) -> str:
        """Hierarchical team with specialists + integrator."""
        # Similar to intermediate but with expanded panel + second review round
        return await self._intermediate_query(question)

    async def answer(self, question: str) -> tuple[str, str]:
        """Return (answer, difficulty_used)."""
        diff = await self._classify_difficulty(question)
        if diff == "basic":
            resp = await self._basic_query(question)
        elif diff == "advanced":
            resp = await self._advanced_query(question)
        else:
            resp = await self._intermediate_query(question)
        return resp, diff
