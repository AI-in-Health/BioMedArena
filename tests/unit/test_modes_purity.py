"""Purity tests for ``simple_llm`` and ``deep_think``.

Contract:
  - Both modes use the same system + user prompts (no mode-specific
    "think step by step" nudge at the prompt layer).
  - Both produce exactly 1 iteration per task.
  - Both invoke zero tool calls.
  - The only difference is that ``deep_think`` enables the provider's
    native reasoning budget (via ``chat_think``).

Tests mock the LLM client so no API call is made.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from harness.eval.benchmark_suite import BenchmarkSuite, SIMPLE_SYSTEM
from harness.trace import TraceRecorder, active_trace


@pytest.fixture
def mini_suite(tmp_path):
    """BenchmarkSuite wired against a minimal in-memory config file."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "llm:\n"
        "  provider: anthropic\n"
        "  model: claude-sonnet-4-5\n"
        "  api_key: fake-unused\n"
    )
    return BenchmarkSuite(config_path=str(cfg))


def _task() -> dict:
    return {
        "id": "t1",
        "question": "Compute eGFR for Cr=1.0 mg/dL",
        "answer": "90",
        "answer_type": "numeric",
        "category": "nephrology",
        "_benchmark_key": "medcalc",
    }


@pytest.mark.asyncio
async def test_simple_and_deep_share_system_prompt(mini_suite):
    """Same ``_build_pure_prompts`` output for both modes on the same task."""
    sys_s, user_s, imgs_s = mini_suite._build_pure_prompts(_task())
    sys_d, user_d, imgs_d = mini_suite._build_pure_prompts(_task())
    assert sys_s == sys_d
    assert user_s == user_d
    assert imgs_s == imgs_d
    # Must match the canonical simple_llm base prompt (up to benchmark hint).
    assert sys_s.startswith(SIMPLE_SYSTEM)


@pytest.mark.asyncio
async def test_simple_llm_invokes_exactly_one_iteration(mini_suite):
    tr = TraceRecorder("t1", "medcalc", "claude-sonnet-4-5", "simple_llm")
    with patch(
        "harness.llm_client.LLMClient.chat", new=AsyncMock(return_value="42"),
    ):
        with active_trace(tr):
            resp, tools = await mini_suite._run_simple(_task())
    assert resp == "42"
    assert tools == []
    assert tr.iterations == 1
    assert len(tr.tool_calls) == 0


@pytest.mark.asyncio
async def test_deep_think_invokes_exactly_one_iteration(mini_suite):
    tr = TraceRecorder("t1", "medcalc", "claude-sonnet-4-5", "deep_think")
    with patch(
        "harness.llm_client.LLMClient.chat_think",
        new=AsyncMock(return_value="42 (with reasoning)"),
    ):
        with active_trace(tr):
            resp, tools = await mini_suite._run_deep(_task())
    assert resp.startswith("42")
    assert tools == []
    assert tr.iterations == 1
    assert len(tr.tool_calls) == 0


@pytest.mark.asyncio
async def test_deep_think_calls_chat_think_not_chat(mini_suite):
    """deep_think must call ``chat_think`` (and NOT ``chat``)."""
    mock_chat_think = AsyncMock(return_value="42")
    mock_chat = AsyncMock(return_value="SHOULD_NOT_BE_CALLED")
    with patch("harness.llm_client.LLMClient.chat_think", new=mock_chat_think):
        with patch("harness.llm_client.LLMClient.chat", new=mock_chat):
            await mini_suite._run_deep(_task())
    mock_chat_think.assert_called_once()
    mock_chat.assert_not_called()


@pytest.mark.asyncio
async def test_simple_llm_calls_chat_not_chat_think(mini_suite):
    mock_chat = AsyncMock(return_value="42")
    mock_chat_think = AsyncMock(return_value="SHOULD_NOT_BE_CALLED")
    with patch("harness.llm_client.LLMClient.chat", new=mock_chat):
        with patch("harness.llm_client.LLMClient.chat_think", new=mock_chat_think):
            await mini_suite._run_simple(_task())
    mock_chat.assert_called_once()
    mock_chat_think.assert_not_called()


@pytest.mark.asyncio
async def test_deep_think_no_tool_schemas_attached(mini_suite):
    """The ``chat_think`` call carries no tool schemas (since the
    method signature doesn't accept any), which guarantees the model
    never sees the tool universe during deep_think."""
    captured = {}

    async def _fake(messages, **kwargs):
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return "42"

    with patch("harness.llm_client.LLMClient.chat_think", side_effect=_fake):
        await mini_suite._run_deep(_task())

    # ``chat_think`` accepts only (messages, thinking_budget, max_tokens).
    # Verify no "tools" kwarg leaked in.
    assert "tools" not in captured["kwargs"]
    assert any(m["role"] == "system" for m in captured["messages"])
    assert any(m["role"] == "user" for m in captured["messages"])
