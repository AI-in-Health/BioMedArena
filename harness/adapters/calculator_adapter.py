"""Adapter wrapping clinical_calculators.py as an orchestrator-compatible adapter."""

from __future__ import annotations

import json
from typing import Any

from harness.adapter_base import AdapterBase
from harness.tools.clinical_calculators import CALCULATORS


class CalculatorAdapter(AdapterBase):
    name = "clinical_calculators"
    modality = "reasoning"
    description = "Clinical risk calculators: CHA2DS2-VASc, HEART, Wells, MELD, eGFR, BMI, and 15+ more."

    def __init__(self, config: dict | None = None, **kwargs: Any):
        pass

    def capabilities(self) -> list[str]:
        return [
            "risk_scoring",
            "clinical_calculator",
            "cha2ds2_vasc",
            "heart_score",
            "wells_score",
            "meld",
            "egfr",
            "bmi",
            "apache",
            "gcs",
        ]

    async def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        ctx = context or {}
        calculator_name = ctx.get("calculator")
        params = ctx.get("calculator_params", {})

        if calculator_name and calculator_name in CALCULATORS:
            try:
                result = CALCULATORS[calculator_name](**params)
                answer = (
                    f"**{calculator_name}**: Score = {result['score']}\n"
                    f"Category: {result['category']}\n"
                    f"Recommendation: {result['recommendation']}"
                )
                return self.result(answer=answer, confidence=0.95, raw=result)
            except Exception as exc:
                return self.result(answer=f"Calculator error: {exc}", confidence=0.0)

        # If no specific calculator requested, list available ones
        available = ", ".join(sorted(CALCULATORS.keys()))
        return self.result(
            answer=(
                f"Available clinical calculators: {available}. "
                "Provide 'calculator' and 'calculator_params' in context to compute a specific score."
            ),
            confidence=0.3,
        )
