"""Adapter for wearable/biomarker data — Terra API format loader + analysis."""

from __future__ import annotations

from typing import Any

from harness.adapter_base import AdapterBase
from harness.tools.health_scores import overall_health_score, sleep_score, activity_score, resilience_score


class WearableAdapter(AdapterBase):
    name = "wearable"
    modality = "wearable"
    description = "Wearable device data analysis: sleep, activity, HRV, vitals scoring."

    def __init__(self, config: dict | None = None, **kwargs: Any):
        pass

    def capabilities(self) -> list[str]:
        return [
            "wearable_analysis",
            "sleep_analysis",
            "activity_analysis",
            "hrv_analysis",
            "health_score",
            "trend_analysis",
        ]

    async def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        ctx = context or {}

        sleep_data = ctx.get("sleep")
        activity_data = ctx.get("activity") or ctx.get("wearable_data")
        hrv_data = ctx.get("hrv")
        vitals_data = ctx.get("vitals")

        if not any([sleep_data, activity_data, hrv_data, vitals_data]):
            # Try to extract from flat context
            if "avg_steps_30d" in ctx or "steps" in ctx:
                activity_data = {
                    "steps": ctx.get("avg_steps_30d") or ctx.get("steps", 0),
                    "active_minutes": ctx.get("active_minutes", 0),
                    "exercise_sessions_week": ctx.get("exercise_sessions_week", 0),
                }
            if "rmssd_ms" in ctx or "resting_hr" in ctx:
                hrv_data = {
                    "rmssd_ms": ctx.get("rmssd_ms", 30),
                    "resting_hr": ctx.get("resting_hr", 70),
                }

        if not any([sleep_data, activity_data, hrv_data, vitals_data]):
            return self.result(
                answer="No wearable data provided. Include sleep, activity, hrv, or vitals in context.",
                confidence=0.1,
            )

        health = overall_health_score(
            sleep=sleep_data,
            activity=activity_data,
            hrv=hrv_data,
            vitals=vitals_data,
        )

        parts = [f"**Overall Health Score: {health['score']}/100**"]
        for comp_name, comp_score in health.get("components", {}).items():
            parts.append(f"- {comp_name.title()}: {comp_score}/100")
        parts.append(f"\n{health['interpretation']}")

        # Add trend info if available
        trend = (ctx.get("wearable_data") or {}).get("trend") or ctx.get("trend")
        if trend:
            parts.append(f"\nTrend: {trend}")

        return self.result(
            answer="\n".join(parts),
            evidence=[f"{k}: {v}/100" for k, v in health.get("components", {}).items()],
            confidence=0.7,
            raw=health,
        )
