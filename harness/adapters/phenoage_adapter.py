"""Adapter wrapping PhenoAge biological age calculation."""

from __future__ import annotations

from typing import Any

from harness.adapter_base import AdapterBase
from harness.tools.phenoage import compute_phenoage


class PhenoAgeAdapter(AdapterBase):
    name = "phenoage"
    modality = "wearable"
    description = "Levine PhenoAge biological age from 9 blood biomarkers + chronological age."

    def __init__(self, config: dict | None = None, **kwargs: Any):
        pass

    def capabilities(self) -> list[str]:
        return ["biological_age", "aging", "biomarker_analysis", "phenoage"]

    async def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        ctx = context or {}
        biomarkers = ctx.get("biomarkers") or ctx.get("labs") or {}
        age = ctx.get("age") or ctx.get("chronological_age")

        required = [
            "albumin", "creatinine", "glucose", "crp",
            "lymphocyte_pct", "mcv", "rdw", "alkaline_phosphatase", "wbc",
        ]
        missing = [k for k in required if k not in biomarkers]
        if missing or age is None:
            return self.result(
                answer=(
                    f"PhenoAge requires chronological age and 9 biomarkers. "
                    f"Missing: {', '.join(missing) if missing else 'none'}"
                    f"{'; chronological age' if age is None else ''}. "
                    "Provide in context as 'biomarkers' dict + 'age'."
                ),
                confidence=0.1,
            )

        result = compute_phenoage(
            albumin=float(biomarkers["albumin"]),
            creatinine=float(biomarkers["creatinine"]),
            glucose=float(biomarkers["glucose"]),
            crp=float(biomarkers["crp"]),
            lymphocyte_pct=float(biomarkers["lymphocyte_pct"]),
            mcv=float(biomarkers["mcv"]),
            rdw=float(biomarkers["rdw"]),
            alkaline_phosphatase=float(biomarkers["alkaline_phosphatase"]),
            wbc=float(biomarkers["wbc"]),
            chronological_age=float(age),
        )

        answer = (
            f"**PhenoAge: {result['phenoage']}** (chronological: {result['chronological_age']})\n"
            f"Age gap: {result['age_gap']:+.1f} years\n"
            f"Mortality score: {result['mortality_score']}\n"
            f"{result['interpretation']}"
        )
        return self.result(answer=answer, confidence=0.85, raw=result)
