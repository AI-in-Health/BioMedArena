"""Composite health scores from wearable/biomarker data.

Produces 0-100 scores for overall health, sleep, activity, and stress/resilience.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def overall_health_score(
    sleep: dict[str, Any] | None = None,
    activity: dict[str, Any] | None = None,
    hrv: dict[str, Any] | None = None,
    vitals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute composite health score from sub-domains."""
    components: dict[str, float] = {}
    weights: dict[str, float] = {}

    if sleep:
        s = sleep_score(sleep)
        components["sleep"] = s["score"]
        weights["sleep"] = 0.30

    if activity:
        a = activity_score(activity)
        components["activity"] = a["score"]
        weights["activity"] = 0.30

    if hrv:
        r = resilience_score(hrv)
        components["resilience"] = r["score"]
        weights["resilience"] = 0.20

    if vitals:
        v = vitals_score(vitals)
        components["vitals"] = v["score"]
        weights["vitals"] = 0.20

    if not components:
        return {"score": 0, "components": {}, "interpretation": "Insufficient data."}

    # Normalise weights
    total_w = sum(weights.values())
    composite = sum(components[k] * weights[k] / total_w for k in components)
    composite = round(composite, 1)

    if composite >= 80:
        interp = "Excellent overall health indicators."
    elif composite >= 60:
        interp = "Good overall health; some areas for improvement."
    elif composite >= 40:
        interp = "Fair health; multiple areas need attention."
    else:
        interp = "Poor health indicators; recommend medical consultation."

    return {"score": composite, "components": components, "interpretation": interp}


def sleep_score(data: dict[str, Any]) -> dict[str, Any]:
    """Score 0-100 from sleep data.

    Expected keys: duration_hours, efficiency (0-1), deep_pct, rem_pct,
                   awakenings, latency_min.
    """
    score = 50.0  # baseline

    duration = data.get("duration_hours", 7)
    if 7 <= duration <= 9:
        score += 20
    elif 6 <= duration < 7 or 9 < duration <= 10:
        score += 10
    else:
        score -= 10

    efficiency = data.get("efficiency", 0.85)
    score += (efficiency - 0.75) * 100  # +25 for 100% eff, 0 for 75%

    deep_pct = data.get("deep_pct", 0.15)
    score += (deep_pct - 0.10) * 100  # bonus for deep sleep

    rem_pct = data.get("rem_pct", 0.20)
    score += (rem_pct - 0.15) * 50

    awakenings = data.get("awakenings", 2)
    score -= max(0, awakenings - 1) * 3

    score = round(max(0, min(100, score)), 1)

    if score >= 80:
        interp = "Excellent sleep quality."
    elif score >= 60:
        interp = "Good sleep quality."
    elif score >= 40:
        interp = "Fair sleep quality; consider sleep hygiene improvements."
    else:
        interp = "Poor sleep quality; sleep study may be warranted."

    return {"score": score, "interpretation": interp}


def activity_score(data: dict[str, Any]) -> dict[str, Any]:
    """Score 0-100 from activity data.

    Expected keys: steps, active_minutes, distance_km, calories_active,
                   exercise_sessions_week.
    """
    score = 0.0

    steps = data.get("steps", 0)
    score += min(steps / 10000, 1.0) * 35  # 35 pts for 10k steps

    active_min = data.get("active_minutes", 0)
    score += min(active_min / 30, 1.0) * 25  # 25 pts for 30 active min

    sessions = data.get("exercise_sessions_week", 0)
    score += min(sessions / 5, 1.0) * 25  # 25 pts for 5 sessions/week

    distance = data.get("distance_km", 0)
    score += min(distance / 8, 1.0) * 15

    score = round(max(0, min(100, score)), 1)

    if score >= 80:
        interp = "Very active lifestyle."
    elif score >= 60:
        interp = "Moderately active; meeting basic guidelines."
    elif score >= 40:
        interp = "Below recommended activity levels."
    else:
        interp = "Sedentary; significant increase in activity recommended."

    return {"score": score, "interpretation": interp}


def resilience_score(data: dict[str, Any]) -> dict[str, Any]:
    """Stress/resilience score 0-100 from HRV and related data.

    Expected keys: rmssd_ms, sdnn_ms, resting_hr, respiratory_rate.
    """
    score = 50.0

    rmssd = data.get("rmssd_ms", 30)
    if rmssd >= 50:
        score += 25
    elif rmssd >= 30:
        score += 15
    elif rmssd >= 20:
        score += 5
    else:
        score -= 10

    resting_hr = data.get("resting_hr", 70)
    if resting_hr <= 55:
        score += 20
    elif resting_hr <= 65:
        score += 10
    elif resting_hr <= 75:
        score += 0
    else:
        score -= 10

    score = round(max(0, min(100, score)), 1)

    if score >= 80:
        interp = "High resilience; excellent autonomic balance."
    elif score >= 60:
        interp = "Good resilience."
    elif score >= 40:
        interp = "Moderate stress; consider stress management."
    else:
        interp = "High stress indicators; rest and recovery recommended."

    return {"score": score, "interpretation": interp}


def vitals_score(data: dict[str, Any]) -> dict[str, Any]:
    """Score 0-100 from vital signs.

    Expected keys: sbp, dbp, spo2, temperature, resting_hr.
    """
    score = 100.0

    sbp = data.get("sbp", 120)
    dbp = data.get("dbp", 80)
    if sbp > 140 or dbp > 90:
        score -= 25
    elif sbp > 130 or dbp > 85:
        score -= 10

    spo2 = data.get("spo2", 98)
    if spo2 < 90:
        score -= 30
    elif spo2 < 95:
        score -= 15

    temp = data.get("temperature", 36.8)
    if temp > 38.0 or temp < 35.5:
        score -= 20
    elif temp > 37.5:
        score -= 5

    score = round(max(0, min(100, score)), 1)

    if score >= 80:
        interp = "Vitals within normal range."
    elif score >= 60:
        interp = "Some vitals slightly outside optimal range."
    else:
        interp = "Vitals concerning; medical review recommended."

    return {"score": score, "interpretation": interp}
