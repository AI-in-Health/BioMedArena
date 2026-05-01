"""Regression tests for ``harness.llm_client`` edge cases."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _fake_usage(input_tokens: int = 10, output_tokens: int = 1):
    """Return an object that mimics Anthropic's ``usage`` attribute."""
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


@pytest.mark.asyncio
async def test_anthropic_refusal_empty_content_returns_empty_string():
    """``stop_reason='refusal'`` + ``content=[]`` must not crash."""
    from harness.llm_client import LLMClient

    fake_resp = SimpleNamespace(
        content=[],
        usage=_fake_usage(168, 1),
        stop_reason="refusal",
    )
    client = LLMClient(provider="anthropic", model="claude-sonnet-4-6",
                       api_key="fake")
    inner = MagicMock()
    inner.messages.create = AsyncMock(return_value=fake_resp)

    with patch.object(client, "_get_client", return_value=inner):
        result = await client.chat(
            messages=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "q"},
            ],
            temperature=0.0,
            max_tokens=1024,
        )

    assert result == ""
    inner.messages.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_anthropic_only_thinking_blocks_returns_empty_string():
    """If Anthropic returns only a ``thinking`` block (no ``text`` block),
    ``_chat_anthropic`` must still return a string, not crash."""
    from harness.llm_client import LLMClient

    thinking_block = SimpleNamespace(type="thinking", thinking="internal CoT")
    fake_resp = SimpleNamespace(
        content=[thinking_block],
        usage=_fake_usage(100, 20),
        stop_reason="end_turn",
    )
    client = LLMClient(provider="anthropic", model="claude-sonnet-4-6",
                       api_key="fake")
    inner = MagicMock()
    inner.messages.create = AsyncMock(return_value=fake_resp)

    with patch.object(client, "_get_client", return_value=inner):
        result = await client.chat(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0, max_tokens=256,
        )

    # A thinking-only response yields no visible text; we must get an
    # empty string, not raise.
    assert result == ""


@pytest.mark.asyncio
async def test_anthropic_mixed_text_and_other_blocks_concatenates_text():
    """When Anthropic returns a mix of text and non-text blocks, return
    only the concatenated text."""
    from harness.llm_client import LLMClient

    text_block_1 = SimpleNamespace(type="text", text="Hello ")
    tool_block = SimpleNamespace(type="tool_use", id="t1", name="x", input={})
    text_block_2 = SimpleNamespace(type="text", text="world")
    fake_resp = SimpleNamespace(
        content=[text_block_1, tool_block, text_block_2],
        usage=_fake_usage(50, 5),
        stop_reason="end_turn",
    )
    client = LLMClient(provider="anthropic", model="claude-sonnet-4-6",
                       api_key="fake")
    inner = MagicMock()
    inner.messages.create = AsyncMock(return_value=fake_resp)

    with patch.object(client, "_get_client", return_value=inner):
        result = await client.chat(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0, max_tokens=256,
        )

    assert result == "Hello world"


@pytest.mark.asyncio
async def test_anthropic_normal_single_text_block_still_works():
    """Baseline regression: the common case (single text block) is
    unchanged by the defensive extraction."""
    from harness.llm_client import LLMClient

    text_block = SimpleNamespace(type="text", text="42")
    fake_resp = SimpleNamespace(
        content=[text_block],
        usage=_fake_usage(10, 1),
        stop_reason="end_turn",
    )
    client = LLMClient(provider="anthropic", model="claude-sonnet-4-6",
                       api_key="fake")
    inner = MagicMock()
    inner.messages.create = AsyncMock(return_value=fake_resp)

    with patch.object(client, "_get_client", return_value=inner):
        result = await client.chat(
            messages=[{"role": "user", "content": "What is 6*7?"}],
            temperature=0.0, max_tokens=256,
        )

    assert result == "42"


@pytest.mark.asyncio
async def test_scorer_handles_empty_prediction():
    """Scorer dispatcher must handle empty prediction without crashing
    and without invoking the judge (empty candidate short-circuits)."""
    from harness.eval.llm_judge import score_with_fallback

    task = {
        "question": "What is 2+2?",
        "answer": "4",
        "answer_type": "numeric",
    }
    result = await score_with_fallback(task, prediction="")

    assert result["correct"] is False
    details = result.get("details") or {}
    assert details.get("judge_invoked") is False
