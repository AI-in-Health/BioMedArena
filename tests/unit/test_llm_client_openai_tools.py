"""OpenAI tool-calling compatibility tests."""
from __future__ import annotations

import pytest

from harness.llm_client import LLMClient


class _FakeToolCallFunction:
    name = "search"
    arguments = "{\"q\": \"test\"}"


class _FakeToolCall:
    id = "call_1"
    function = _FakeToolCallFunction()


class _FakeMessage:
    content = None
    tool_calls = [_FakeToolCall()]


class _FakeChoice:
    message = _FakeMessage()


class _FakeResponse:
    choices = [_FakeChoice()]


class _FakeCompletions:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return _FakeResponse()


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeOpenAIClient:
    def __init__(self):
        self.chat = _FakeChat()


@pytest.mark.asyncio
async def test_openai_reasoning_tool_calling_uses_max_completion_tokens():
    fake_client = _FakeOpenAIClient()
    llm = LLMClient(provider="openai", model="o4-mini", api_key="fake")
    llm._client = fake_client

    result = await llm.chat_with_tools(
        messages=[{"role": "user", "content": "Search for test"}],
        tools=[{
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            },
        }],
        temperature=0.2,
        max_tokens=123,
    )

    kwargs = fake_client.chat.completions.kwargs
    assert kwargs["max_completion_tokens"] == 123
    assert "max_tokens" not in kwargs
    assert "temperature" not in kwargs
    assert result["tool_calls"][0]["name"] == "search"


@pytest.mark.asyncio
async def test_non_reasoning_openai_tool_calling_keeps_standard_kwargs():
    fake_client = _FakeOpenAIClient()
    llm = LLMClient(provider="openai", model="gpt-4o", api_key="fake")
    llm._client = fake_client

    await llm.chat_with_tools(
        messages=[{"role": "user", "content": "Search for test"}],
        tools=[],
        temperature=0.2,
        max_tokens=123,
    )

    kwargs = fake_client.chat.completions.kwargs
    assert kwargs["max_tokens"] == 123
    assert kwargs["temperature"] == 0.2
    assert "max_completion_tokens" not in kwargs
