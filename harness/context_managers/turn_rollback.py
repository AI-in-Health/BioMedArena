"""Turn rollback strategy — removes wasteful turns to save iterations and context.

Detects duplicate queries, tool errors, and useless results. When detected,
the offending assistant + tool-result turn is removed and guidance is appended.

Guidance modes:
- basic: Fixed template
- enriched: Programmatic — includes query/URL history
"""

from __future__ import annotations

import logging
from typing import Any

from .base import (
    TOOL_NAMES_SCRAPE,
    TOOL_NAMES_SEARCH,
    Strategy,
    estimate_message_tokens,
)

logger = logging.getLogger(__name__)

_VALID_GUIDANCE_MODES = ("basic", "enriched")


class TurnRollbackStrategy(Strategy):
    """Roll back wasteful turns (duplicates, tool errors).

    After each agent cycle, inspects the last assistant + tool-result turn.
    If a problem is detected, the turn is removed and guidance is appended.
    """

    name = "turn_rollback"
    category = "rollback"

    def __init__(
        self,
        max_consecutive: int = 5,
        guidance_mode: str = "basic",
    ):
        self.max_consecutive = max_consecutive
        if guidance_mode not in _VALID_GUIDANCE_MODES:
            guidance_mode = "basic"
        self.guidance_mode = guidance_mode
        self._consecutive = 0
        self._total_rollbacks = 0
        self._seen_queries: dict[str, int] = {}
        self._seen_urls: dict[str, int] = {}

    def apply(self, messages: list[dict[str, Any]]) -> None:
        if len(messages) < 2:
            self._consecutive = 0
            return

        reason = self._detect_issue(messages)
        if not reason:
            self._consecutive = 0
            self._track_last_turn(messages)
            return

        if self._consecutive >= self.max_consecutive:
            logger.warning(
                "[rollback] hit max consecutive rollbacks (%d), skipping: %s",
                self.max_consecutive, reason,
            )
            self._consecutive = 0
            self._track_last_turn(messages)
            return

        # Build guidance
        guidance = self._build_guidance(reason)

        # Pop the last turn (assistant + tool results)
        freed = self._pop_last_turn(messages)

        # Append guidance as a user message
        messages.append({"role": "user", "content": guidance})

        self._consecutive += 1
        self._total_rollbacks += 1
        logger.info(
            "[rollback] rolled back turn (%d tokens freed, %d consecutive): %s",
            freed, self._consecutive, reason,
        )

    def reduce(self, messages: list[dict[str, Any]]) -> int:
        return 0

    # ------------------------------------------------------------------
    # Guidance generation
    # ------------------------------------------------------------------

    def _build_guidance(self, reason: str) -> str:
        if self.guidance_mode == "enriched":
            return self._guidance_enriched(reason)
        return self._guidance_basic(reason)

    def _guidance_basic(self, reason: str) -> str:
        return (
            f"[System] Your last action was rolled back: {reason}. "
            "Please try a different approach — use different search keywords, "
            "try a different URL, or reason about what you already know."
        )

    def _guidance_enriched(self, reason: str) -> str:
        parts = [f"[System] Your last action was rolled back: {reason}."]

        if self._seen_queries:
            recent = list(self._seen_queries.keys())[-8:]
            q_list = "\n".join(f"  - {q!r}" for q in recent)
            parts.append(
                f"Queries already tried ({len(self._seen_queries)} total):\n{q_list}"
            )

        if self._seen_urls:
            recent = list(self._seen_urls.keys())[-5:]
            u_list = "\n".join(f"  - {u}" for u in recent)
            parts.append(
                f"URLs already visited ({len(self._seen_urls)} total):\n{u_list}"
            )

        parts.append(
            "You MUST use completely different search terms. Consider:\n"
            "- Synonyms or alternative phrasings\n"
            "- Different language (e.g., native language of the topic)\n"
            "- Related entities or events that might lead to the answer\n"
            "- A different type of source (news, Wikipedia, official databases)"
        )
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect_issue(self, messages: list[dict[str, Any]]) -> str | None:
        """Return a reason string if the last turn should be rolled back."""
        last_asst_idx = self._find_last_assistant(messages)
        if last_asst_idx is None:
            return None

        asst_msg = messages[last_asst_idx]
        tool_calls = asst_msg.get("tool_calls", [])

        # --- Duplicate search query or URL ---
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            try:
                import json
                args = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}

            if name in TOOL_NAMES_SEARCH:
                query = args.get("query", "").strip().lower()
                if query and query in self._seen_queries:
                    return f"duplicate search query: '{query}'"
            elif name in TOOL_NAMES_SCRAPE:
                url = args.get("url", "").strip().rstrip("/").lower()
                if url and url in self._seen_urls:
                    return f"duplicate URL scrape: '{url}'"

        # --- Tool error in results ---
        for i in range(last_asst_idx + 1, len(messages)):
            msg = messages[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if isinstance(content, str) and content.startswith("[") and "error" in content.lower():
                return f"tool error: {content[:100]}"

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_last_assistant(messages: list[dict[str, Any]]) -> int | None:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                return i
        return None

    def _track_last_turn(self, messages: list[dict[str, Any]]) -> None:
        """Record queries/URLs from the last turn (called when NO rollback)."""
        idx = self._find_last_assistant(messages)
        if idx is None:
            return
        for tc in messages[idx].get("tool_calls", []):
            fn = tc.get("function", {})
            name = fn.get("name", "")
            try:
                import json
                args = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}
            if name in TOOL_NAMES_SEARCH:
                query = args.get("query", "").strip().lower()
                if query:
                    self._seen_queries[query] = idx
            elif name in TOOL_NAMES_SCRAPE:
                url = args.get("url", "").strip().rstrip("/").lower()
                if url:
                    self._seen_urls[url] = idx

    @staticmethod
    def _pop_last_turn(messages: list[dict[str, Any]]) -> int:
        """Remove last assistant message and all subsequent tool messages."""
        last_asst_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_asst_idx = i
                break
        if last_asst_idx is None:
            return 0

        freed = 0
        while len(messages) > last_asst_idx:
            freed += estimate_message_tokens(messages.pop())
        return freed

    def get_state(self) -> dict[str, Any]:
        return {
            **super().get_state(),
            "guidance_mode": self.guidance_mode,
            "total_rollbacks": self._total_rollbacks,
            "unique_queries": len(self._seen_queries),
            "unique_urls": len(self._seen_urls),
        }
