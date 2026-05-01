"""Unit tests for the OpenAI → Anthropic tool-message translator.

Anthropic expects tool results as user messages with `tool_result`
content blocks. These tests verify translation from OpenAI-style
`role="tool"` messages and assistant `tool_calls`.

Tests run offline — an AsyncMock stands in for the real
AsyncAnthropic client.
"""

from __future__ import annotations

import json
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from harness.llm_client import LLMClient, _openai_to_anthropic_messages


# ======================================================================
# Pure translator tests
# ======================================================================


def test_translator_extracts_system_prompt():
    sys, msgs = _openai_to_anthropic_messages([
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "hi"},
    ])
    assert sys == "Be concise."
    assert msgs == [{"role": "user", "content": "hi"}]


def test_translator_concatenates_multiple_system_messages():
    sys, _ = _openai_to_anthropic_messages([
        {"role": "system", "content": "Rule 1."},
        {"role": "system", "content": "Rule 2."},
        {"role": "user", "content": "q"},
    ])
    assert sys == "Rule 1.\n\nRule 2."


def test_translator_no_system():
    sys, msgs = _openai_to_anthropic_messages([
        {"role": "user", "content": "hi"},
    ])
    assert sys is None
    assert msgs == [{"role": "user", "content": "hi"}]


def test_translator_tool_call_pairs():
    """assistant-with-tool_calls + tool-result pair must translate to
    two Anthropic messages with the right content-block structure."""
    conv = [
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "calc", "arguments": '{"expr":"2+2"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "= 4"},
        {"role": "assistant", "content": "The answer is 4."},
    ]
    _, msgs = _openai_to_anthropic_messages(conv)
    assert len(msgs) == 4
    # 0: user (pass-through)
    assert msgs[0] == {"role": "user", "content": "What is 2+2?"}
    # 1: assistant with tool_use block
    assert msgs[1]["role"] == "assistant"
    assert isinstance(msgs[1]["content"], list)
    block = msgs[1]["content"][0]
    assert block["type"] == "tool_use"
    assert block["id"] == "call_1"
    assert block["name"] == "calc"
    assert block["input"] == {"expr": "2+2"}
    # 2: user with tool_result block (NOT role=tool)
    assert msgs[2]["role"] == "user"
    tr = msgs[2]["content"][0]
    assert tr["type"] == "tool_result"
    assert tr["tool_use_id"] == "call_1"
    assert tr["content"] == "= 4"
    # 3: pure-text assistant
    assert msgs[3] == {"role": "assistant", "content": "The answer is 4."}


def test_translator_assistant_text_with_tool_calls():
    """Assistant with BOTH text content and tool_calls → text block + tool_use."""
    conv = [
        {"role": "user", "content": "Look up X then answer."},
        {
            "role": "assistant",
            "content": "I'll look it up.",
            "tool_calls": [{
                "id": "t1", "type": "function",
                "function": {"name": "lookup", "arguments": '{"q":"X"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "X = 42"},
    ]
    _, msgs = _openai_to_anthropic_messages(conv)
    assert msgs[1]["role"] == "assistant"
    blocks = msgs[1]["content"]
    assert len(blocks) == 2
    assert blocks[0] == {"type": "text", "text": "I'll look it up."}
    assert blocks[1]["type"] == "tool_use"
    assert blocks[1]["name"] == "lookup"


def test_translator_coalesces_parallel_tool_results():
    """Two consecutive role=tool messages must merge into ONE Anthropic
    user message with two tool_result blocks."""
    conv = [
        {"role": "user", "content": "parallel"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "g", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": "result A"},
        {"role": "tool", "tool_call_id": "b", "content": "result B"},
    ]
    _, msgs = _openai_to_anthropic_messages(conv)
    # user, assistant(with 2 tool_use), user(with 2 tool_result)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    results = msgs[2]["content"]
    assert len(results) == 2
    assert results[0]["tool_use_id"] == "a"
    assert results[0]["content"] == "result A"
    assert results[1]["tool_use_id"] == "b"
    assert results[1]["content"] == "result B"


def test_translator_handles_json_decode_failure():
    """Malformed arguments string shouldn't crash the translator."""
    conv = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "t1", "type": "function",
                "function": {"name": "x", "arguments": "not-json{"},
            }],
        },
    ]
    _, msgs = _openai_to_anthropic_messages(conv)
    assert msgs[1]["content"][0]["input"] == {"__raw_arguments": "not-json{"}


def test_translator_accepts_dict_arguments():
    """Some OpenAI SDK versions supply arguments as dict, not str."""
    conv = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "t1", "type": "function",
                "function": {"name": "x", "arguments": {"k": 1}},
            }],
        },
    ]
    _, msgs = _openai_to_anthropic_messages(conv)
    assert msgs[1]["content"][0]["input"] == {"k": 1}


def test_translator_drops_empty_assistant():
    conv = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": ""},   # empty
        {"role": "user", "content": "q2"},
    ]
    _, msgs = _openai_to_anthropic_messages(conv)
    # Empty assistant dropped; remaining conversation stays valid
    assert len(msgs) == 2
    assert msgs[0]["content"] == "q"
    assert msgs[1]["content"] == "q2"


def test_translator_handles_list_content_on_tool():
    """tool content can occasionally arrive as a list of blocks."""
    conv = [
        {"role": "tool", "tool_call_id": "x",
         "content": [{"type": "text", "text": "first"},
                       {"type": "text", "text": "second"}]},
    ]
    _, msgs = _openai_to_anthropic_messages(conv)
    assert msgs[0]["content"][0]["type"] == "tool_result"
    assert "first" in msgs[0]["content"][0]["content"]
    assert "second" in msgs[0]["content"][0]["content"]


# ======================================================================
# End-to-end: _tools_anthropic with a mocked Anthropic client
# ======================================================================


def _make_fake_response(text: str = "", tool_calls=None):
    """Build a minimal stand-in for anthropic.types.Message."""
    blocks = []
    if text:
        blocks.append(types.SimpleNamespace(type="text", text=text))
    for tc in tool_calls or []:
        blocks.append(types.SimpleNamespace(
            type="tool_use",
            id=tc["id"],
            name=tc["name"],
            input=tc["input"],
        ))
    return types.SimpleNamespace(content=blocks)


@pytest.mark.asyncio
async def test_tools_anthropic_end_to_end_tool_call_roundtrip():
    """Simulate: send conversation with role=tool → client translates →
    assert the payload that HITS the Anthropic SDK matches the spec.
    """
    llm = LLMClient(provider="anthropic", model="claude-sonnet-4-5", api_key="fake")
    # Pre-populate the lazy client with a mock
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(
        return_value=_make_fake_response(text="42")
    )
    llm._client = fake_client

    tools = [{
        "type": "function",
        "function": {
            "name": "calc",
            "description": "evaluate",
            "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}},
        },
    }]
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": None,
         "tool_calls": [{
             "id": "tu_1", "type": "function",
             "function": {"name": "calc", "arguments": '{"expr":"2+2"}'},
         }]},
        {"role": "tool", "tool_call_id": "tu_1", "content": "= 4"},
    ]
    out = await llm.chat_with_tools(messages=conversation, tools=tools,
                                       temperature=0, max_tokens=100)

    # Caller-visible shape preserved
    assert out == {"content": "42", "tool_calls": []}

    # Inspect the actual call to the Anthropic SDK
    assert fake_client.messages.create.await_count == 1
    kwargs = fake_client.messages.create.await_args.kwargs
    # System is top-level, NOT in messages
    assert kwargs["system"] == "You are a helpful assistant."
    sent = kwargs["messages"]
    # No message should have role="tool" (the bug)
    assert all(m["role"] in ("user", "assistant") for m in sent), (
        f"got roles: {[m['role'] for m in sent]}"
    )
    # Find the tool-result carrier
    tr_msgs = [m for m in sent
                if m["role"] == "user"
                and isinstance(m.get("content"), list)
                and m["content"] and m["content"][0].get("type") == "tool_result"]
    assert len(tr_msgs) == 1
    tr = tr_msgs[0]["content"][0]
    assert tr["tool_use_id"] == "tu_1"
    assert tr["content"] == "= 4"
    # Tools correctly converted
    assert kwargs["tools"] == [{
        "name": "calc",
        "description": "evaluate",
        "input_schema": {"type": "object", "properties": {"expr": {"type": "string"}}},
    }]


@pytest.mark.asyncio
async def test_tools_anthropic_response_tool_use_parsed():
    """When Anthropic returns a tool_use block, the wrapper must surface
    it as {id, name, arguments-as-json-string} per our canonical shape."""
    llm = LLMClient(provider="anthropic", model="claude-sonnet-4-5", api_key="fake")
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(
        return_value=_make_fake_response(
            text="",
            tool_calls=[{"id": "use_1", "name": "calc", "input": {"expr": "1+1"}}],
        )
    )
    llm._client = fake_client

    out = await llm.chat_with_tools(
        messages=[{"role": "user", "content": "run calc"}],
        tools=[{"type": "function",
                  "function": {"name": "calc", "description": "",
                                 "parameters": {"type": "object"}}}],
        temperature=0, max_tokens=100,
    )
    assert out["content"] is None
    assert len(out["tool_calls"]) == 1
    tc = out["tool_calls"][0]
    assert tc["id"] == "use_1"
    assert tc["name"] == "calc"
    # arguments is a JSON string (OpenAI convention)
    assert json.loads(tc["arguments"]) == {"expr": "1+1"}
