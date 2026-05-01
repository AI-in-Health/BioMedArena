"""Selective clearing strategies — surgically remove specific content types.

Strategies:
- ToolResultClearing: clear old tool results (search/scrape outputs)
- ThinkingClearing: remove reasoning/thinking from old assistant messages
- MediaClearing: remove images/documents from old messages
- Deduplication: replace duplicate search/scrape results with markers
"""

from __future__ import annotations

import logging
from typing import Any

from .base import (
    TOOL_NAMES_COMPACTABLE,
    TOOL_NAMES_SEARCH,
    TOOL_NAMES_SCRAPE,
    Strategy,
    estimate_message_tokens,
    estimate_messages_tokens,
    find_tool_call_name,
    is_cleared_content,
    truncate_to_tokens,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Tool Result Clearing
# ---------------------------------------------------------------------------


class ToolResultClearingStrategy(Strategy):
    """Replace old search/scrape tool results with a short marker.

    The most impactful clearing strategy for web research agents because
    tool results (search snippets, scraped pages) dominate context usage.
    """

    name = "tool_result_clearing"
    category = "selective_clearing"

    def __init__(
        self,
        keep_recent: int = 5,
        trigger_count: int = 8,
        max_result_tokens: int = 2048,
    ):
        self.keep_recent = keep_recent
        self.trigger_count = trigger_count
        self.max_result_tokens = max_result_tokens
        self._cleared_count = 0

    def apply(self, messages: list[dict[str, Any]]) -> None:
        # First: truncate oversized individual tool results
        self._truncate_oversized(messages)
        # Then: clear old tool results beyond keep_recent
        self._clear_old_results(messages, self.keep_recent)

    def reduce(self, messages: list[dict[str, Any]]) -> int:
        """Aggressive: keep only 2 most recent tool results."""
        before = estimate_messages_tokens(messages)
        self._clear_old_results(messages, keep_recent=2)
        after = estimate_messages_tokens(messages)
        freed = before - after
        if freed > 0:
            logger.info("[tool_result_clearing] aggressive reduce: freed %d tokens", freed)
        return freed

    def _truncate_oversized(self, messages: list[dict[str, Any]]) -> None:
        """Truncate individual tool results that exceed max_result_tokens."""
        for msg in messages:
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            from .base import estimate_tokens
            if estimate_tokens(content) > self.max_result_tokens:
                msg["content"] = (
                    truncate_to_tokens(content, self.max_result_tokens)
                    + f"\n\n[Truncated to {self.max_result_tokens} tokens]"
                )

    def _clear_old_results(
        self, messages: list[dict[str, Any]], keep_recent: int
    ) -> None:
        """Clear all but the N most recent compactable tool results."""
        # Collect indices of compactable tool results (newest last)
        compactable: list[int] = []
        for idx, msg in enumerate(messages):
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if is_cleared_content(content):
                continue
            tool_call_id = msg.get("tool_call_id", "")
            tool_name, _ = find_tool_call_name(messages, tool_call_id)
            if tool_name in TOOL_NAMES_COMPACTABLE:
                compactable.append(idx)

        if len(compactable) <= max(keep_recent, self.trigger_count):
            return

        to_clear = compactable[: len(compactable) - keep_recent]
        cleared = 0
        for idx in to_clear:
            messages[idx]["content"] = "[Old tool result cleared to save context]"
            cleared += 1

        if cleared:
            self._cleared_count += cleared
            logger.info(
                "[tool_result_clearing] cleared %d/%d results (keeping %d recent)",
                cleared, len(compactable), keep_recent,
            )

    def get_state(self) -> dict[str, Any]:
        return {**super().get_state(), "cleared_count": self._cleared_count}


# ---------------------------------------------------------------------------
# 2. Thinking Clearing
# ---------------------------------------------------------------------------


class ThinkingClearingStrategy(Strategy):
    """Remove verbose reasoning from old assistant messages.

    Once the model has acted on its reasoning, long chain-of-thought text
    is no longer needed. We truncate old assistant messages that are very long.
    """

    name = "thinking_clearing"
    category = "selective_clearing"

    def __init__(self, keep_recent_turns: int = 4):
        self.keep_recent_turns = keep_recent_turns
        self._cleared_count = 0

    def apply(self, messages: list[dict[str, Any]]) -> None:
        if len(messages) <= self.keep_recent_turns:
            return

        # Find assistant messages to potentially trim
        assistant_indices = [
            i for i, m in enumerate(messages) if m.get("role") == "assistant"
        ]

        if len(assistant_indices) <= self.keep_recent_turns:
            return

        # Trim all but the most recent keep_recent_turns assistant messages
        cutoff_indices = assistant_indices[:-self.keep_recent_turns]
        cleared = 0

        for idx in cutoff_indices:
            msg = messages[idx]
            content = msg.get("content")
            if not isinstance(content, str) or not content:
                continue
            # Only trim if content is very long (likely contains reasoning)
            from .base import estimate_tokens
            if estimate_tokens(content) > 500:
                # Keep first 200 tokens as a summary marker
                trimmed = truncate_to_tokens(content, 200)
                msg["content"] = trimmed + "\n[...reasoning truncated to save context]"
                cleared += 1

        if cleared:
            self._cleared_count += cleared
            logger.info(
                "[thinking_clearing] truncated %d old assistant messages (keeping last %d)",
                cleared, self.keep_recent_turns,
            )

    def reduce(self, messages: list[dict[str, Any]]) -> int:
        """Aggressive: clear ALL except the very last assistant."""
        before = estimate_messages_tokens(messages)
        original_keep = self.keep_recent_turns
        self.keep_recent_turns = 1
        self.apply(messages)
        self.keep_recent_turns = original_keep
        after = estimate_messages_tokens(messages)
        return before - after

    def get_state(self) -> dict[str, Any]:
        return {**super().get_state(), "cleared_count": self._cleared_count}


# ---------------------------------------------------------------------------
# 3. Media Clearing
# ---------------------------------------------------------------------------


class MediaClearingStrategy(Strategy):
    """Remove image/document content from old messages.

    In OpenAI format, images appear as content blocks with image_url.
    """

    name = "media_clearing"
    category = "selective_clearing"

    def __init__(self, keep_recent_turns: int = 4):
        self.keep_recent_turns = keep_recent_turns
        self._cleared_count = 0

    def apply(self, messages: list[dict[str, Any]]) -> None:
        if len(messages) <= self.keep_recent_turns:
            return

        cutoff = len(messages) - self.keep_recent_turns
        cleared = 0

        for idx in range(cutoff):
            msg = messages[idx]
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            new_content = []
            for block in content:
                if isinstance(block, dict):
                    if "image_url" in block:
                        new_content.append({"type": "text", "text": "[Image removed to save context]"})
                        cleared += 1
                    else:
                        new_content.append(block)
                else:
                    new_content.append(block)
            if cleared:
                msg["content"] = new_content

        if cleared:
            self._cleared_count += cleared
            logger.info("[media_clearing] removed %d media blocks", cleared)

    def reduce(self, messages: list[dict[str, Any]]) -> int:
        """Aggressive: clear ALL media except the very last turn."""
        before = estimate_messages_tokens(messages)
        original_keep = self.keep_recent_turns
        self.keep_recent_turns = 1
        self.apply(messages)
        self.keep_recent_turns = original_keep
        after = estimate_messages_tokens(messages)
        return before - after

    def get_state(self) -> dict[str, Any]:
        return {**super().get_state(), "cleared_count": self._cleared_count}


# ---------------------------------------------------------------------------
# 4. Deduplication
# ---------------------------------------------------------------------------


class DeduplicationStrategy(Strategy):
    """Replace duplicate search/scrape results with pointers to originals.

    Tracks queries and URLs seen so far; when the same query or URL
    appears again, replaces the tool result with a short reference marker.
    """

    name = "deduplication"
    category = "selective_clearing"

    def __init__(self):
        self._seen_queries: dict[str, int] = {}
        self._seen_urls: dict[str, int] = {}
        self._dedup_count = 0

    def apply(self, messages: list[dict[str, Any]]) -> None:
        for idx, msg in enumerate(messages):
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if is_cleared_content(content):
                continue

            tool_call_id = msg.get("tool_call_id", "")
            tool_name, tool_input = find_tool_call_name(messages, tool_call_id)

            if tool_name in TOOL_NAMES_SEARCH:
                query = tool_input.get("query", "").strip().lower()
                if not query:
                    continue
                if query in self._seen_queries and self._seen_queries[query] != idx:
                    msg["content"] = "[Duplicate search — see earlier result]"
                    self._dedup_count += 1
                else:
                    self._seen_queries[query] = idx

            elif tool_name in TOOL_NAMES_SCRAPE:
                url = tool_input.get("url", "").strip().rstrip("/").lower()
                if not url:
                    continue
                if url in self._seen_urls and self._seen_urls[url] != idx:
                    msg["content"] = "[Duplicate scrape — see earlier result]"
                    self._dedup_count += 1
                else:
                    self._seen_urls[url] = idx

    def get_state(self) -> dict[str, Any]:
        return {
            **super().get_state(),
            "dedup_count": self._dedup_count,
            "unique_queries": len(self._seen_queries),
            "unique_urls": len(self._seen_urls),
        }
