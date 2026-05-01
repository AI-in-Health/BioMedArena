"""Unit tests for ``harness.eval.results_writer``."""
from __future__ import annotations

import csv
from pathlib import Path

from harness.eval.results_writer import (
    CSV_HEADER,
    _empty_row,
    aggregate_cell,
    append_cell_row,
)
from harness.trace import TraceRecorder


def _make_trace(
    task_id: str = "t1",
    benchmark: str = "medcalc",
    backbone: str = "claude-sonnet-4-6",
    mode: str = "light",
    correct: bool = True,
    input_tokens: int = 100,
    output_tokens: int = 20,
    cost_usd: float = 0.001,
    iterations: int = 2,
    tool_calls: list | None = None,
    runtime_error: bool = False,
) -> TraceRecorder:
    tr = TraceRecorder(task_id, benchmark, backbone, mode)
    for _ in range(iterations):
        tr.increment_iteration()
    tr.record_llm_call(
        role="chat", system="s", messages=[{"role": "user", "content": "q"}],
        response_text="r", input_tokens=input_tokens,
        output_tokens=output_tokens, cost_usd=cost_usd,
        latency_ms=100,
        error=("boom" if runtime_error else None),
    )
    for name, success, err in (tool_calls or []):
        tr.record_tool_call(
            name=name, arguments={}, result="r",
            success=success, error=err, latency_ms=10,
        )
    tr.set_final_answer("ans")
    tr.set_scorer_result(correct=correct, method="primary:numeric")
    tr.finalize()
    return tr


def test_header_length_and_order():
    assert len(CSV_HEADER) == 22
    assert CSV_HEADER[0] == "benchmark"
    assert CSV_HEADER[-1] == "timestamp"


def test_empty_cell():
    row = aggregate_cell([], "b", "bb", "simple_llm", seed=7)
    assert set(row.keys()) == set(CSV_HEADER)
    assert row["n_tasks"] == 0
    assert row["accuracy"] == 0.0
    assert row["tool_call_success_rate"] == 1.0
    assert row["top_tool"] == ""
    assert row["top_tool_call_count"] == 0
    assert row["seed"] == 7


def test_aggregate_basic():
    tr1 = _make_trace(
        task_id="t1", correct=True,
        tool_calls=[("pubmed", True, None), ("calc", True, None)],
    )
    tr2 = _make_trace(
        task_id="t2", correct=False,
        tool_calls=[("pubmed", True, None), ("calc", False, "boom")],
    )
    row = aggregate_cell(
        [tr1, tr2], "medcalc", "claude-sonnet-4-6", "light",
    )
    assert row["n_tasks"] == 2
    assert row["n_correct"] == 1
    assert row["accuracy"] == 0.5
    assert row["total_input_tokens"] == 200
    assert row["total_output_tokens"] == 40
    assert row["avg_input_tokens"] == 100.0
    assert row["avg_output_tokens"] == 20.0
    assert row["avg_tool_calls"] == 2.0
    assert row["avg_iterations"] == 2.0
    # 3 successes out of 4 tool calls = 0.75
    assert row["tool_call_success_rate"] == 0.75
    assert row["unique_tools_used"] == 2
    assert row["top_tool"] == "pubmed"
    assert row["top_tool_call_count"] == 2


def test_no_tools_edge_case():
    tr = _make_trace(tool_calls=[], iterations=1)
    row = aggregate_cell([tr], "b", "bb", "simple_llm")
    assert row["avg_tool_calls"] == 0.0
    assert row["tool_call_success_rate"] == 1.0
    assert row["unique_tools_used"] == 0
    assert row["top_tool"] == ""
    assert row["top_tool_call_count"] == 0


def test_n_error_counts_runtime_errors():
    tr_ok = _make_trace(task_id="ok", correct=True)
    tr_err = _make_trace(task_id="err", correct=False, runtime_error=True)
    row = aggregate_cell([tr_ok, tr_err], "b", "bb", "simple_llm")
    assert row["n_error"] == 1


def test_n_error_override():
    tr = _make_trace(runtime_error=True)
    row = aggregate_cell([tr], "b", "bb", "m", n_error_override=99)
    assert row["n_error"] == 99


def test_append_cell_row_writes_header_once(tmp_path: Path):
    path = tmp_path / "results.csv"
    tr = _make_trace()
    row = aggregate_cell([tr], "medcalc", "claude-sonnet-4-6", "simple_llm")
    append_cell_row(path, row)
    append_cell_row(path, row)
    lines = path.read_text().splitlines()
    # 1 header + 2 data rows
    assert len(lines) == 3
    reader = csv.reader([lines[0]])
    header = next(reader)
    assert header == CSV_HEADER


def test_append_cell_row_ignores_extra_keys(tmp_path: Path):
    path = tmp_path / "results.csv"
    row = _empty_row("b", "bb", "m", 42)
    row["extra_junk"] = "should_not_appear"
    append_cell_row(path, row)
    text = path.read_text()
    assert "extra_junk" not in text
