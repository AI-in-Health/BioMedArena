"""Unified metrics interface for all benchmarks.

Every benchmark produces a BenchmarkMetrics object with standardised fields,
enabling cross-benchmark comparison and leaderboard generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuestionMetric:
    """Per-question result across all benchmarks."""
    question_id: str
    benchmark: str
    mode: str            # simple_llm | deep_think | light | heavy
    category: str        # domain/subject
    question_text: str   # truncated
    expected: str
    predicted: str
    predicted_raw: str   # full LLM response

    # Core metrics
    task_success: bool              # did the task complete correctly?
    tool_calls_made: list[str] = field(default_factory=list)  # which tools/adapters were called
    tool_call_accuracy: float = 0.0  # fraction of tool calls that were useful/correct
    reasoning_faithfulness: float = 0.0  # how well reasoning aligns with evidence (0-1)

    latency_s: float = 0.0
    error: str | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics for one benchmark + one mode."""
    benchmark: str
    mode: str
    total: int
    task_success_rate: float        # fraction of tasks completed correctly
    tool_call_accuracy: float       # avg fraction of useful tool calls
    reasoning_faithfulness: float   # avg reasoning faithfulness score
    avg_latency_s: float

    # Breakdowns
    success_by_category: dict[str, float] = field(default_factory=dict)
    success_by_type: dict[str, float] = field(default_factory=dict)
    official_metrics: dict[str, Any] = field(default_factory=dict)

    per_question: list[QuestionMetric] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "mode": self.mode,
            "total": self.total,
            "task_success_rate": round(self.task_success_rate, 4),
            "tool_call_accuracy": round(self.tool_call_accuracy, 4),
            "reasoning_faithfulness": round(self.reasoning_faithfulness, 4),
            "avg_latency_s": round(self.avg_latency_s, 2),
            "success_by_category": {k: round(v, 4) for k, v in self.success_by_category.items()},
            "success_by_type": {k: round(v, 4) for k, v in self.success_by_type.items()},
            "official_metrics": self.official_metrics,
        }


def build_metrics(
    benchmark: str,
    mode: str,
    questions: list[QuestionMetric],
) -> BenchmarkMetrics:
    """Aggregate per-question metrics into a BenchmarkMetrics."""
    total = len(questions)
    if total == 0:
        return BenchmarkMetrics(
            benchmark=benchmark, mode=mode, total=0,
            task_success_rate=0, tool_call_accuracy=0,
            reasoning_faithfulness=0, avg_latency_s=0,
            official_metrics={"status": "unavailable", "reason": "no_records"},
            per_question=questions,
        )

    success_rate = sum(1 for q in questions if q.task_success) / total
    avg_tool_acc = sum(q.tool_call_accuracy for q in questions) / total
    avg_faith = sum(q.reasoning_faithfulness for q in questions) / total
    avg_lat = sum(q.latency_s for q in questions) / total

    # Per-category
    cat_success: dict[str, list[bool]] = {}
    for q in questions:
        cat_success.setdefault(q.category, []).append(q.task_success)
    success_by_cat = {c: sum(v) / len(v) for c, v in cat_success.items()}

    try:
        from harness.eval.official_metrics import compute_official_metrics
        official_metrics = compute_official_metrics(benchmark, questions)
    except Exception as exc:
        official_metrics = {"status": "error", "reason": str(exc)}

    return BenchmarkMetrics(
        benchmark=benchmark,
        mode=mode,
        total=total,
        task_success_rate=success_rate,
        tool_call_accuracy=avg_tool_acc,
        reasoning_faithfulness=avg_faith,
        avg_latency_s=round(avg_lat, 2),
        success_by_category=success_by_cat,
        official_metrics=official_metrics,
        per_question=questions,
    )
