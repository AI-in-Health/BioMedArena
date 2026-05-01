"""Truncation strategies — drop old messages to stay within budget.

Strategies:
- SlidingWindow: keep only the last N messages
- FirstLast: keep first K (task context) + last N (recent work)
- TokenBudget: remove oldest messages until total tokens < budget
"""

from __future__ import annotations

import logging
from typing import Any

from .base import (
    Strategy,
    estimate_message_tokens,
    estimate_messages_tokens,
    find_safe_split,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Sliding Window
# ---------------------------------------------------------------------------


class SlidingWindowStrategy(Strategy):
    """Keep only the last ``window_size`` messages."""

    name = "sliding_window"
    category = "truncation"

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self._trimmed_count = 0

    def apply(self, messages: list[dict[str, Any]]) -> None:
        if len(messages) <= self.window_size:
            return

        # Find a safe split that doesn't break tool pairs
        target = len(messages) - self.window_size
        # Always keep the system message (index 0)
        target = max(target, 1) if messages and messages[0].get("role") == "system" else target
        split = find_safe_split(messages, target)
        if split <= 0 or split >= len(messages):
            return

        # Preserve system message
        preserved = []
        if messages[0].get("role") == "system":
            preserved = [messages[0]]
            if split <= 0:
                return

        removed = len(messages[:split]) - len(preserved)
        messages[:] = preserved + messages[split:]
        self._trimmed_count += removed
        logger.info(
            "[sliding_window] removed %d oldest messages, keeping %d",
            removed, len(messages),
        )

    def reduce(self, messages: list[dict[str, Any]]) -> int:
        """Aggressive: halve the window size temporarily."""
        before = estimate_messages_tokens(messages)
        emergency_window = max(6, self.window_size // 2)
        if len(messages) <= emergency_window:
            return 0
        target = len(messages) - emergency_window
        target = max(target, 1) if messages and messages[0].get("role") == "system" else target
        split = find_safe_split(messages, target)
        if split <= 0:
            return 0

        preserved = []
        if messages[0].get("role") == "system":
            preserved = [messages[0]]

        messages[:] = preserved + messages[split:]
        after = estimate_messages_tokens(messages)
        freed = before - after
        logger.info("[sliding_window] emergency reduce: freed %d tokens", freed)
        return freed

    def get_state(self) -> dict[str, Any]:
        return {**super().get_state(), "trimmed_count": self._trimmed_count}


# ---------------------------------------------------------------------------
# 2. First + Last
# ---------------------------------------------------------------------------


class FirstLastStrategy(Strategy):
    """Keep first ``keep_first`` messages + last ``keep_last`` messages."""

    name = "first_last"
    category = "truncation"

    def __init__(self, keep_first: int = 2, keep_last: int = 16):
        self.keep_first = keep_first
        self.keep_last = keep_last
        self._trimmed_count = 0

    def apply(self, messages: list[dict[str, Any]]) -> None:
        total_keep = self.keep_first + self.keep_last
        if len(messages) <= total_keep:
            return

        first_part = messages[: self.keep_first]
        last_start = len(messages) - self.keep_last
        last_start = find_safe_split(messages, last_start)
        last_part = messages[last_start:]

        removed = len(messages) - len(first_part) - len(last_part)
        if removed <= 0:
            return

        marker = {
            "role": "user",
            "content": f"[{removed} intermediate messages omitted to save context]",
        }
        messages[:] = first_part + [marker] + last_part
        self._trimmed_count += removed
        logger.info(
            "[first_last] kept first %d + last %d, dropped %d middle messages",
            len(first_part), len(last_part), removed,
        )

    def reduce(self, messages: list[dict[str, Any]]) -> int:
        """Aggressive: keep fewer messages."""
        before = estimate_messages_tokens(messages)
        emergency_last = max(4, self.keep_last // 2)
        if len(messages) <= self.keep_first + emergency_last:
            return 0

        first_part = messages[: self.keep_first]
        last_start = len(messages) - emergency_last
        last_start = find_safe_split(messages, last_start)
        last_part = messages[last_start:]

        marker = {
            "role": "user",
            "content": "[Messages aggressively trimmed due to context overflow]",
        }
        messages[:] = first_part + [marker] + last_part
        after = estimate_messages_tokens(messages)
        freed = before - after
        logger.info("[first_last] emergency reduce: freed %d tokens", freed)
        return freed

    def get_state(self) -> dict[str, Any]:
        return {**super().get_state(), "trimmed_count": self._trimmed_count}


# ---------------------------------------------------------------------------
# 3. Token Budget
# ---------------------------------------------------------------------------


class TokenBudgetStrategy(Strategy):
    """Remove oldest messages until total tokens < ``max_tokens``."""

    name = "token_budget"
    category = "truncation"

    def __init__(self, max_tokens: int = 150_000, min_keep: int = 6):
        self.max_tokens = max_tokens
        self.min_keep = min_keep
        self._tokens_freed_total = 0

    def apply(self, messages: list[dict[str, Any]]) -> None:
        total = estimate_messages_tokens(messages)
        if total <= self.max_tokens:
            return
        self._trim_to_budget(messages, self.max_tokens)

    def reduce(self, messages: list[dict[str, Any]]) -> int:
        """Aggressive: use 60% of normal budget."""
        emergency_budget = int(self.max_tokens * 0.6)
        before = estimate_messages_tokens(messages)
        self._trim_to_budget(messages, emergency_budget)
        after = estimate_messages_tokens(messages)
        freed = before - after
        logger.info("[token_budget] emergency reduce: freed %d tokens", freed)
        return freed

    def _trim_to_budget(self, messages: list[dict[str, Any]], budget: int) -> None:
        total = estimate_messages_tokens(messages)
        if total <= budget or len(messages) <= self.min_keep:
            return

        # Always preserve system message at index 0
        start_idx = 1 if messages and messages[0].get("role") == "system" else 0

        # Remove messages from after start_idx until under budget
        removed = 0
        idx = start_idx
        while total > budget and idx < len(messages) - self.min_keep:
            safe_idx = find_safe_split(messages, idx + 1)
            chunk = messages[idx:safe_idx]
            chunk_tokens = sum(estimate_message_tokens(m) for m in chunk)
            total -= chunk_tokens
            removed += len(chunk)
            idx = safe_idx

        if removed <= 0:
            return

        preserved = messages[:start_idx] if start_idx > 0 else []
        messages[:] = preserved + messages[idx:]
        self._tokens_freed_total += sum(
            estimate_message_tokens(m) for m in messages[:idx]
        )
        logger.info(
            "[token_budget] removed %d messages, now ~%d tokens (budget: %d)",
            removed, estimate_messages_tokens(messages), budget,
        )

    def get_state(self) -> dict[str, Any]:
        return {**super().get_state(), "tokens_freed_total": self._tokens_freed_total}
