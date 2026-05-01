"""Levine's PhenoAge biological age calculation.

Based on: Levine ME et al. "An epigenetic biomarker of aging for lifespan and
healthspan." Aging (2018).
"""

from __future__ import annotations

import math
from typing import Any


# Coefficients from the original PhenoAge paper (Cox PH model on NHANES)
_COEFFICIENTS = {
    "albumin": -0.0336,
    "creatinine": 0.0095,
    "glucose": 0.1953,
    "crp_log": 0.0954,     # log(CRP in mg/dL)
    "lymphocyte_pct": -0.0120,
    "mcv": 0.0268,
    "rdw": 0.3306,
    "alkaline_phosphatase": 0.0019,
    "wbc": 0.0554,
    "age": 0.0804,
}

_INTERCEPT = -19.9067


def compute_phenoage(
    albumin: float,          # g/dL
    creatinine: float,       # mg/dL
    glucose: float,          # mg/dL
    crp: float,              # mg/dL (NOT mg/L — divide mg/L by 10)
    lymphocyte_pct: float,   # percentage (e.g. 30 for 30%)
    mcv: float,              # fL
    rdw: float,              # percentage
    alkaline_phosphatase: float,  # U/L
    wbc: float,              # 1000 cells/μL
    chronological_age: float,
) -> dict[str, Any]:
    """Calculate PhenoAge biological age.

    Returns dict with phenoage, age_gap, mortality_score, and interpretation.
    """
    crp_log = math.log(max(crp, 0.001))  # avoid log(0)

    # Linear predictor (xb)
    xb = (
        _COEFFICIENTS["albumin"] * albumin
        + _COEFFICIENTS["creatinine"] * creatinine
        + _COEFFICIENTS["glucose"] * glucose
        + _COEFFICIENTS["crp_log"] * crp_log
        + _COEFFICIENTS["lymphocyte_pct"] * lymphocyte_pct
        + _COEFFICIENTS["mcv"] * mcv
        + _COEFFICIENTS["rdw"] * rdw
        + _COEFFICIENTS["alkaline_phosphatase"] * alkaline_phosphatase
        + _COEFFICIENTS["wbc"] * wbc
        + _COEFFICIENTS["age"] * chronological_age
        + _INTERCEPT
    )

    # Mortality score (Gompertz)
    # Parameters from Levine et al.
    gamma = 0.0076927
    lambda_val = 0.0022802

    mortality_score = 1 - math.exp(-lambda_val * math.exp(xb) * (math.exp(120 * gamma) - 1) / gamma)

    # Convert mortality score back to PhenoAge
    # PhenoAge = 141.50225 + ln(-0.00553 * ln(1 - mortality_score)) / 0.090165
    if mortality_score < 1:
        inner = -0.00553 * math.log(1 - mortality_score)
        if inner > 0:
            phenoage = 141.50225 + math.log(inner) / 0.090165
        else:
            phenoage = chronological_age
    else:
        phenoage = chronological_age + 20  # cap

    phenoage = round(phenoage, 1)
    age_gap = round(phenoage - chronological_age, 1)

    # Interpretation
    if age_gap <= -5:
        interpretation = "Significantly younger biological age — excellent health trajectory."
    elif age_gap <= -1:
        interpretation = "Slightly younger biological age — good health indicators."
    elif age_gap <= 1:
        interpretation = "Biological age approximately matches chronological age."
    elif age_gap <= 5:
        interpretation = "Slightly older biological age — some risk factors may be elevated."
    else:
        interpretation = "Significantly older biological age — consider lifestyle and medical interventions."

    return {
        "phenoage": phenoage,
        "chronological_age": chronological_age,
        "age_gap": age_gap,
        "mortality_score": round(mortality_score, 4),
        "interpretation": interpretation,
        "biomarkers": {
            "albumin": albumin,
            "creatinine": creatinine,
            "glucose": glucose,
            "crp": crp,
            "lymphocyte_pct": lymphocyte_pct,
            "mcv": mcv,
            "rdw": rdw,
            "alkaline_phosphatase": alkaline_phosphatase,
            "wbc": wbc,
        },
    }
