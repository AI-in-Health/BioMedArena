"""Base classes and utilities for context management strategies.

Provides token estimation, message helpers, and the Strategy ABC.
Adapted for OpenAI-format messages used by BioMedArena.

OpenAI message format:
  {"role": "system"|"user"|"assistant"|"tool", "content": str|None,
   "tool_calls": [...], "tool_call_id": str}
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

try:
    import tiktoken
    _ENCODING = tiktoken.encoding_for_model("gpt-4")
except ImportError:
    _ENCODING = None

# Tool name sets for identifying compactable content
TOOL_NAMES_SEARCH = {"serper_search", "pubmed_search", "google_search"}
TOOL_NAMES_SCRAPE = {"jina_read_page", "visit", "scrape", "scrape_and_summarize"}
TOOL_NAMES_COMPACTABLE = TOOL_NAMES_SEARCH | TOOL_NAMES_SCRAPE | {
    "gene_lookup", "clinvar_lookup", "rxnav_drug", "openfda_adverse",
    "dailymed_label", "omim_lookup", "orphanet_lookup", "medlineplus_topic",
    "pubmed_search",
}


# ---------------------------------------------------------------------------
# Token estimation helpers
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Estimate token count for a string."""
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    # Fallback: rough estimate (1 token ≈ 4 chars)
    return len(text) // 4


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to a token budget, returning the truncated string."""
    if _ENCODING is not None:
        tokens = _ENCODING.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return _ENCODING.decode(tokens[:max_tokens])
    # Fallback
    max_chars = max_tokens * 4
    return text[:max_chars] if len(text) > max_chars else text


def estimate_message_tokens(msg: dict[str, Any]) -> int:
    """Estimate total tokens in a single OpenAI-format message."""
    total = 0
    content = msg.get("content")
    if isinstance(content, str) and content:
        total += estimate_tokens(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    total += estimate_tokens(block["text"])
                elif "image_url" in block:
                    total += 2000  # conservative estimate for images
                else:
                    total += 50
            elif isinstance(block, str):
                total += estimate_tokens(block)

    # Tool calls in assistant messages
    tool_calls = msg.get("tool_calls", [])
    for tc in tool_calls:
        fn = tc.get("function", {})
        total += estimate_tokens(fn.get("name", ""))
        args = fn.get("arguments", "")
        total += estimate_tokens(args if isinstance(args, str) else json.dumps(args))

    return total


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Total token estimate for a list of messages."""
    return sum(estimate_message_tokens(m) for m in messages)


# ---------------------------------------------------------------------------
# Message helpers (OpenAI format)
# ---------------------------------------------------------------------------


def find_tool_call_name(messages: list[dict[str, Any]], tool_call_id: str) -> tuple[str, dict]:
    """Find the tool call matching a given tool_call_id.

    Returns (tool_name, tool_args) or ("", {}) if not found.
    """
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []):
            if tc.get("id") == tool_call_id:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args_raw = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args = {}
                return name, args
    return "", {}


def find_safe_split(messages: list[dict[str, Any]], target_idx: int) -> int:
    """Find nearest safe split point at or after target_idx.

    Avoids breaking assistant+tool_calls / tool_result pairs:
    - If messages[idx] is a tool message, skip forward past all consecutive tool messages
    - If messages[idx] is an assistant with tool_calls, skip forward past its tool results
    """
    idx = target_idx
    while idx < len(messages):
        msg = messages[idx]
        role = msg.get("role")

        # Tool result message — its assistant is earlier, skip forward
        if role == "tool":
            idx += 1
            continue

        # Assistant with tool_calls — ensure tool results stay with it
        if role == "assistant" and msg.get("tool_calls"):
            # Find all subsequent tool messages
            end = idx + 1
            while end < len(messages) and messages[end].get("role") == "tool":
                end += 1
            # Don't split inside this group — advance past it
            idx = end
            continue

        break
    return min(idx, len(messages))


def is_cleared_content(content: str) -> bool:
    """Check if tool result content is already a cleared marker."""
    return content.startswith("[") and (
        "cleared" in content.lower()
        or "duplicate" in content.lower()
        or "omitted" in content.lower()
    )


# ---------------------------------------------------------------------------
# Strategy base class
# ---------------------------------------------------------------------------


class Strategy(ABC):
    """Base class for a single context management strategy.

    Each strategy has two hook points:
    - ``apply(messages)``: lightweight, called after every agent loop cycle
    - ``reduce(messages)``: heavyweight, called on context window overflow

    Strategies modify the messages list in-place.
    """

    name: str = "base"
    category: str = "base"

    @abstractmethod
    def apply(self, messages: list[dict[str, Any]]) -> None:
        """Lightweight operation called after every agent loop cycle.

        Should modify ``messages`` in-place.
        """

    def reduce(self, messages: list[dict[str, Any]]) -> int:
        """Heavy operation called on context window overflow.

        Should modify ``messages`` in-place.
        Returns estimated tokens freed. Default: 0 (no reduction).
        """
        return 0

    def get_state(self) -> dict[str, Any]:
        """Return strategy state for logging/analysis."""
        return {"name": self.name, "category": self.category}
