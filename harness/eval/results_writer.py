"""Aggregate per-task traces into a 22-column per-cell CSV row.

CSV header (exact 22-column order)::

    benchmark, backbone, mode, seed,
    n_tasks, n_correct, n_error, accuracy,
    total_cost_usd, avg_cost_per_task_usd,
    total_input_tokens, total_output_tokens,
    avg_input_tokens, avg_output_tokens,
    avg_tool_calls, avg_iterations, tool_call_success_rate,
    unique_tools_used, top_tool, top_tool_call_count,
    avg_latency_s, timestamp

Edge cases:
  - ``n_tasks == 0`` -> accuracy = 0.0 (not NaN)
  - No tool calls (simple_llm / deep_think) ->
      avg_tool_calls = 0,
      tool_call_success_rate = 1.0,
      unique_tools_used = 0,
      top_tool = "",
      top_tool_call_count = 0
  - Provider returns no token usage -> treated as 0.

Callers produce a ``list[TraceRecorder]`` (finalised) per cell, pass
it to :func:`aggregate_cell`, then append the returned row via
:func:`append_cell_row`.
"""
from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CSV_HEADER: list[str] = [
    "benchmark", "backbone", "mode", "seed",
    "n_tasks", "n_correct", "n_error", "accuracy",
    "total_cost_usd", "avg_cost_per_task_usd",
    "total_input_tokens", "total_output_tokens",
    "avg_input_tokens", "avg_output_tokens",
    "avg_tool_calls", "avg_iterations", "tool_call_success_rate",
    "unique_tools_used", "top_tool", "top_tool_call_count",
    "avg_latency_s", "timestamp",
]


def _iso_now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_row(benchmark: str, backbone: str, mode: str, seed: int) -> dict:
    """Canonical row for an empty cell (no tasks)."""
    return {
        "benchmark": benchmark,
        "backbone": backbone,
        "mode": mode,
        "seed": seed,
        "n_tasks": 0,
        "n_correct": 0,
        "n_error": 0,
        "accuracy": 0.0,
        "total_cost_usd": 0.0,
        "avg_cost_per_task_usd": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "avg_input_tokens": 0.0,
        "avg_output_tokens": 0.0,
        "avg_tool_calls": 0.0,
        "avg_iterations": 0.0,
        "tool_call_success_rate": 1.0,
        "unique_tools_used": 0,
        "top_tool": "",
        "top_tool_call_count": 0,
        "avg_latency_s": 0.0,
        "timestamp": _iso_now_utc(),
    }


def aggregate_cell(
    traces: list[Any],
    benchmark: str,
    backbone: str,
    mode: str,
    seed: int = 42,
    n_error_override: int | None = None,
) -> dict:
    """Compute one 22-field row from a list of finalised ``TraceRecorder`` objects."""
    n_tasks = len(traces)
    if n_tasks == 0:
        return _empty_row(benchmark, backbone, mode, seed)

    n_correct = sum(
        1 for t in traces
        if getattr(t, "scorer_result", None) and t.scorer_result.correct
    )
    if n_error_override is not None:
        n_error = int(n_error_override)
    else:
        n_error = sum(1 for t in traces if t.has_runtime_error())

    total_input = sum(t.total_input_tokens() for t in traces)
    total_output = sum(t.total_output_tokens() for t in traces)
    total_cost = sum(t.total_cost_usd() for t in traces)

    all_tool_calls = [c for t in traces for c in t.tool_calls]
    n_tc = len(all_tool_calls)
    n_tc_ok = sum(1 for c in all_tool_calls if c.success)
    tool_names = [c.name for c in all_tool_calls]
    counter = Counter(tool_names)
    top_name, top_count = (
        counter.most_common(1)[0] if counter else ("", 0)
    )

    total_iterations = sum(t.iterations for t in traces)
    total_latency = sum(t.total_latency_s for t in traces)

    return {
        "benchmark": benchmark,
        "backbone": backbone,
        "mode": mode,
        "seed": int(seed),
        "n_tasks": n_tasks,
        "n_correct": n_correct,
        "n_error": n_error,
        "accuracy": round(n_correct / n_tasks, 4),
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_per_task_usd": round(total_cost / n_tasks, 6),
        "total_input_tokens": int(total_input),
        "total_output_tokens": int(total_output),
        "avg_input_tokens": round(total_input / n_tasks, 2),
        "avg_output_tokens": round(total_output / n_tasks, 2),
        "avg_tool_calls": round(n_tc / n_tasks, 3),
        "avg_iterations": round(total_iterations / n_tasks, 3),
        "tool_call_success_rate": (
            round(n_tc_ok / n_tc, 4) if n_tc > 0 else 1.0
        ),
        "unique_tools_used": len(set(tool_names)),
        "top_tool": top_name,
        "top_tool_call_count": int(top_count),
        "avg_latency_s": round(total_latency / n_tasks, 3),
        "timestamp": _iso_now_utc(),
    }


def append_cell_row(csv_path: Path | str, row: dict) -> None:
    """Append ``row`` to ``csv_path``. Create the file with a header row
    if it does not exist yet."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        # Only emit the declared columns (extra keys ignored).
        writer.writerow({k: row.get(k, "") for k in CSV_HEADER})
