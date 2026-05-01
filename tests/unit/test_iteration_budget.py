"""Unit tests for iteration budget and per-tool timeout configuration.

Validates that:
- All benchmarks default to 50 iterations
- CLI/explicit overrides take precedence
- Per-tool timeouts are configured for key tools
- No blanket dispatch-level truncation (truncate_chars=0)
"""

from __future__ import annotations

import pytest

from harness.eval.function_calling_runner import (
    DEFAULT_MAX_ITERATIONS,
    TOOL_TIMEOUTS,
    FunctionCallingRunner,
    default_max_iterations_for,
)


class _DummyLLM:
    """Minimal stand-in so FunctionCallingRunner's __init__ doesn't
    need a real LLM. The runner only uses `self.llm` inside `run()`."""


def test_default_is_50():
    """All benchmarks should default to 50 iterations."""
    assert DEFAULT_MAX_ITERATIONS["_default"] == 50


def test_unknown_key_falls_back_to_default():
    assert default_max_iterations_for("not_a_real_benchmark") == 50
    assert default_max_iterations_for(None) == 50
    assert default_max_iterations_for("") == 50


@pytest.mark.parametrize("key", [
    "medcalc", "labbench_litqa2", "bixbench_closed_book",
    "hle_gold", "hle_bio_med_chem", "unknown_bench",
])
def test_constructor_resolves_to_50(key):
    """All benchmarks default to 50 iterations."""
    r = FunctionCallingRunner(_DummyLLM(), benchmark_key=key)
    assert r.max_iterations == 50


def test_explicit_max_iterations_wins_over_default():
    """Explicit override stays authoritative so callers can clamp
    budgets in cost-sensitive contexts."""
    r = FunctionCallingRunner(_DummyLLM(), max_iterations=3, benchmark_key="hle_gold")
    assert r.max_iterations == 3


def test_no_benchmark_key_defaults_to_50():
    r = FunctionCallingRunner(_DummyLLM())
    assert r.max_iterations == 50
    assert r.benchmark_key is None


def test_no_blanket_truncation_by_default():
    """Dispatch-level truncation should be disabled (0) by default,
    relying on context managers instead for long-context handling."""
    r = FunctionCallingRunner(_DummyLLM())
    assert r.truncate_chars == 0


def test_min_iterations_default_zero():
    """min_iterations defaults to 0 — model decides when to stop."""
    r = FunctionCallingRunner(_DummyLLM())
    assert r.min_iterations == 0


def test_per_tool_timeouts_exist():
    """All tools use a uniform 60s timeout via _default."""
    assert "_default" in TOOL_TIMEOUTS
    assert TOOL_TIMEOUTS["_default"] == 60


def test_token_truncation_default():
    """Token-based truncation defaults to 16000 tokens."""
    r = FunctionCallingRunner(_DummyLLM())
    assert r.truncate_tokens == 16000


def test_thinking_defaults():
    """Thinking is disabled by default; budget defaults to 8192."""
    r = FunctionCallingRunner(_DummyLLM())
    assert r.enable_thinking is False
    assert r.thinking_budget == 8192
