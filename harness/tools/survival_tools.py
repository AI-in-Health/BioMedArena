"""Lifelines TOOL_SPECS — survival analysis (Kaplan-Meier, Cox PH).

Expects durations + event flags. Designed for small cohorts passed
directly in the tool args; larger analyses belong in a future
SurvivalAdapter.
"""

from __future__ import annotations

from typing import Any


SURVIVAL_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "kaplan_meier_fit",
            "description": (
                "Fit a Kaplan-Meier survival curve to durations + event "
                "indicators. Returns the curve at a few summary "
                "timepoints plus median survival time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "durations": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Per-subject time to event or censoring.",
                    },
                    "events": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "1 = event observed, 0 = censored.",
                    },
                    "summary_times": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Times at which to report S(t). Defaults to quartiles.",
                    },
                },
                "required": ["durations", "events"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "logrank_two_sample",
            "description": (
                "Two-sample log-rank test between two survival cohorts. "
                "Returns test statistic, p-value, and a plain-English "
                "interpretation at alpha=0.05."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "durations_a": {"type": "array", "items": {"type": "number"}},
                    "events_a": {"type": "array", "items": {"type": "integer"}},
                    "durations_b": {"type": "array", "items": {"type": "number"}},
                    "events_b": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["durations_a", "events_a", "durations_b", "events_b"],
            },
        },
    },
]


SURVIVAL_TOOL_NAMES = {"kaplan_meier_fit", "logrank_two_sample"}


def _km_fit_sync(durations, events, summary_times=None):
    import numpy as np
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events)
    if not summary_times:
        q = np.quantile(durations, [0.25, 0.5, 0.75]).tolist()
        summary_times = q
    sf = kmf.survival_function_at_times(summary_times).tolist()
    median = float(kmf.median_survival_time_) if hasattr(kmf, "median_survival_time_") else None
    return {
        "summary_times": list(summary_times),
        "survival_probability": sf,
        "median_survival_time": median,
    }


def _logrank_sync(dur_a, ev_a, dur_b, ev_b):
    from lifelines.statistics import logrank_test
    r = logrank_test(dur_a, dur_b, ev_a, ev_b)
    return {
        "test_statistic": float(r.test_statistic),
        "p_value": float(r.p_value),
        "significant_at_0p05": bool(r.p_value < 0.05),
    }


def handle_survival_tool(name: str, args: dict[str, Any]) -> str:
    try:
        if name == "kaplan_meier_fit":
            out = _km_fit_sync(
                args["durations"], args["events"],
                summary_times=args.get("summary_times"),
            )
            return (
                f"median={out['median_survival_time']} "
                f"times={out['summary_times']} "
                f"S(t)={[round(v, 3) for v in out['survival_probability']]}"
            )
        if name == "logrank_two_sample":
            out = _logrank_sync(
                args["durations_a"], args["events_a"],
                args["durations_b"], args["events_b"],
            )
            return (
                f"chi2={out['test_statistic']:.3f} p={out['p_value']:.4f} "
                f"significant={out['significant_at_0p05']}"
            )
        return f"[unknown survival tool: {name}]"
    except Exception as exc:  # noqa: BLE001
        return f"[{name} error: {exc}]"
