"""Unit tests for ``harness.trace.TraceRecorder``."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.trace import (
    TraceRecorder,
    active_trace,
    get_active_trace,
)


def test_basic_recording():
    tr = TraceRecorder("t1", "medcalc", "claude-sonnet-4-5", "simple_llm")
    tr.increment_iteration()
    tr.record_llm_call(
        role="chat",
        system="sys",
        messages=[{"role": "user", "content": "q"}],
        response_text="42",
        input_tokens=100,
        output_tokens=3,
        cost_usd=0.001,
        latency_ms=123,
        finish_reason="stop",
    )
    tr.record_tool_call(
        name="calc", arguments={"x": 1},
        result="2", success=True, latency_ms=10,
    )
    tr.set_final_answer("42")
    tr.set_scorer_result(correct=True, method="primary:numeric")
    tr.finalize()

    assert tr.iterations == 1
    assert len(tr.llm_calls) == 1
    assert len(tr.tool_calls) == 1
    assert tr.total_input_tokens() == 100
    assert tr.total_output_tokens() == 3
    assert round(tr.total_cost_usd(), 4) == 0.001
    assert tr.n_tool_calls() == 1
    assert tr.n_tool_calls_success() == 1
    assert tr.tool_call_names() == ["calc"]
    assert tr.has_runtime_error() is False


def test_runtime_error_detection():
    tr = TraceRecorder("t1", "b", "bb", "simple_llm")
    tr.record_llm_call(
        role="chat", system="", messages=[], response_text="",
        error="boom",
    )
    tr.finalize()
    assert tr.has_runtime_error() is True


def test_tool_call_failure():
    tr = TraceRecorder("t1", "b", "bb", "light")
    tr.record_tool_call(
        name="x", arguments={}, result=None, success=False,
        error="HTTPError: 500",
    )
    tr.finalize()
    assert tr.n_tool_calls_success() == 0
    assert tr.n_tool_calls() == 1


def test_long_content_truncation():
    tr = TraceRecorder("t1", "b", "bb", "simple_llm")
    big = "x" * 5000
    tr.record_llm_call(
        role="chat", system=big,
        messages=[{"role": "user", "content": big}],
        response_text=big,
    )
    d = tr.to_dict()
    assert len(d["llm_calls"][0]["system"]) <= 4000
    assert len(d["llm_calls"][0]["response_text"]) <= 4000
    # Messages content is capped at 2000 chars + a short truncation tail.
    msg = d["llm_calls"][0]["messages"][0]["content"]
    assert len(msg) <= 2100  # "...[truncated]" tail


def test_dump_roundtrip(tmp_path: Path):
    tr = TraceRecorder("t1", "medcalc", "claude-sonnet-4-5", "deep_think")
    tr.increment_iteration()
    tr.record_llm_call(
        role="chat_think", system="s", messages=[{"role": "user", "content": "q"}],
        response_text="42", input_tokens=10, output_tokens=5, cost_usd=0.0001,
    )
    tr.set_final_answer("42")
    tr.set_scorer_result(
        correct=True, method="primary:numeric",
        details={"is_open_ended": False}, llm_judge_invoked=False,
    )
    tr.finalize()
    out = tmp_path / "traces" / "t1.json"
    tr.dump(out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["task_id"] == "t1"
    assert data["benchmark"] == "medcalc"
    assert data["mode"] == "deep_think"
    assert data["iterations"] == 1
    assert data["llm_calls"][0]["role"] == "chat_think"
    assert data["scorer_result"]["correct"] is True


def test_active_trace_context_manager():
    assert get_active_trace() is None
    tr = TraceRecorder("t1", "b", "bb", "simple_llm")
    with active_trace(tr):
        assert get_active_trace() is tr
    assert get_active_trace() is None


@pytest.mark.asyncio
async def test_active_trace_async_isolation():
    """Each async task sees its own bound recorder."""
    import asyncio

    async def task(tr):
        with active_trace(tr):
            await asyncio.sleep(0)
            return get_active_trace()

    tr_a = TraceRecorder("A", "b", "bb", "simple_llm")
    tr_b = TraceRecorder("B", "b", "bb", "simple_llm")
    a, b = await asyncio.gather(task(tr_a), task(tr_b))
    assert a is tr_a
    assert b is tr_b
