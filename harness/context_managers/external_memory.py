"""External memory strategies — persist key information outside the context.

Strategies:
- SessionMemory: automatically extract key findings from tool results,
  persist them externally, re-inject after compaction
- Scratchpad: persistent <scratchpad> tags survive compaction events
"""

from __future__ import annotations

import logging
from typing import Any

from .base import (
    TOOL_NAMES_SCRAPE,
    TOOL_NAMES_SEARCH,
    Strategy,
    estimate_tokens,
    find_tool_call_name,
    is_cleared_content,
    truncate_to_tokens,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Session Memory
# ---------------------------------------------------------------------------


class SessionMemoryStrategy(Strategy):
    """Automatically extract key findings and re-inject after compaction.

    Maintains an external list of key findings (URLs, facts, data points)
    extracted from tool results. These survive compaction events and get
    re-injected as a user message so the model retains awareness of
    previously discovered information.
    """

    name = "session_memory"
    category = "external_memory"

    def __init__(
        self,
        max_findings: int = 10,
        max_tokens: int = 5000,
    ):
        self.max_findings = max_findings
        self.max_tokens = max_tokens
        self._findings: list[dict[str, str]] = []
        self._extraction_count = 0

    def apply(self, messages: list[dict[str, Any]]) -> None:
        """Extract key findings from tool results."""
        self._extract_findings(messages)

    def reduce(self, messages: list[dict[str, Any]]) -> int:
        """After compaction, re-inject preserved findings."""
        if not self._findings:
            return 0

        parts = ["## Previously Discovered Key Findings\n"]
        token_count = estimate_tokens(parts[0])

        for finding in self._findings:
            if finding["type"] == "scrape":
                entry = f"- **From** {finding['url']}:\n  {finding['content']}\n"
            elif finding["type"] == "search":
                entry = f"- **Search** \"{finding['query']}\":\n  {finding['content']}\n"
            else:
                entry = f"- {finding['content']}\n"

            entry_tokens = estimate_tokens(entry)
            if token_count + entry_tokens > self.max_tokens:
                break
            parts.append(entry)
            token_count += entry_tokens

        if len(parts) <= 1:
            return 0

        restoration_text = "\n".join(parts)
        restoration_msg = {"role": "user", "content": restoration_text}

        # Insert after position 0 (system prompt) or 1 if a summary exists
        insert_pos = 1
        # Skip system message
        if messages and messages[0].get("role") == "system":
            insert_pos = 1
        insert_pos = min(insert_pos, len(messages))
        messages.insert(insert_pos, restoration_msg)

        logger.info(
            "[session_memory] restored %d findings (%d tokens) after compaction",
            len(parts) - 1, token_count,
        )
        return 0

    def _extract_findings(self, messages: list[dict[str, Any]]) -> None:
        """Extract key findings from tool result messages."""
        new_findings: list[dict[str, str]] = []

        for msg in messages:
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if not content or len(content) < 50:
                continue
            if is_cleared_content(content):
                continue

            tool_call_id = msg.get("tool_call_id", "")
            tool_name, tool_input = find_tool_call_name(messages, tool_call_id)

            if tool_name in TOOL_NAMES_SCRAPE:
                url = tool_input.get("url", "")
                if url:
                    new_findings.append({
                        "type": "scrape",
                        "url": url,
                        "content": content[:500],
                    })
            elif tool_name in TOOL_NAMES_SEARCH:
                query = tool_input.get("query", "")
                if query:
                    new_findings.append({
                        "type": "search",
                        "query": query,
                        "content": content[:300],
                    })

        # Update findings: keep most recent, deduplicate by URL/query
        seen = set()
        merged = []
        for f in reversed(new_findings + self._findings):
            key = f.get("url", "") or f.get("query", "")
            if key and key not in seen:
                seen.add(key)
                merged.append(f)

        self._findings = list(reversed(merged[-self.max_findings:]))
        self._extraction_count += 1

    def get_state(self) -> dict[str, Any]:
        return {
            **super().get_state(),
            "findings_count": len(self._findings),
            "extraction_count": self._extraction_count,
        }


# ---------------------------------------------------------------------------
# 2. Scratchpad
# ---------------------------------------------------------------------------


class ScratchpadStrategy(Strategy):
    """Maintain a persistent scratchpad that survives compaction.

    The scratchpad is extracted from the model's responses (looking for
    <scratchpad>...</scratchpad> tags) and persisted across compaction events.

    The system prompt should instruct the model to maintain a scratchpad:
    "After each research step, update your scratchpad with key findings
    using <scratchpad>...</scratchpad> tags."
    """

    name = "scratchpad"
    category = "external_memory"

    # System prompt addition for scratchpad
    SYSTEM_PROMPT_ADDITION = """

## Scratchpad
After each research step, update your scratchpad with key findings
using <scratchpad>...</scratchpad> tags. Include:
- Key facts discovered so far
- URLs of useful pages
- What you still need to find
This scratchpad will be preserved if the conversation gets compressed."""

    def __init__(self, max_tokens: int = 3000):
        self.max_tokens = max_tokens
        self._scratchpad: str = ""
        self._update_count = 0

    def apply(self, messages: list[dict[str, Any]]) -> None:
        """Extract latest scratchpad from assistant messages."""
        # Search backwards for the most recent scratchpad
        for msg in reversed(messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if not isinstance(content, str) or not content:
                continue
            scratchpad = self._extract_scratchpad_text(content)
            if scratchpad:
                self._scratchpad = truncate_to_tokens(scratchpad, self.max_tokens)
                self._update_count += 1
                return

    def reduce(self, messages: list[dict[str, Any]]) -> int:
        """After compaction, re-inject the scratchpad."""
        if not self._scratchpad:
            return 0

        scratchpad_msg = {
            "role": "user",
            "content": (
                "## Your Research Scratchpad (preserved from before compaction)\n\n"
                f"{self._scratchpad}\n\n"
                "Continue your research. Update the scratchpad with any new findings "
                "using <scratchpad>...</scratchpad> tags."
            ),
        }

        # Insert after system prompt
        insert_pos = 1
        if messages and messages[0].get("role") == "system":
            insert_pos = 1
        insert_pos = min(insert_pos, len(messages))
        messages.insert(insert_pos, scratchpad_msg)

        logger.info(
            "[scratchpad] re-injected scratchpad (%d chars) after compaction",
            len(self._scratchpad),
        )
        return 0

    @staticmethod
    def _extract_scratchpad_text(text: str) -> str:
        """Extract content between <scratchpad>...</scratchpad> tags."""
        start_tag = "<scratchpad>"
        end_tag = "</scratchpad>"
        start = text.rfind(start_tag)
        if start == -1:
            return ""
        end = text.find(end_tag, start)
        if end == -1:
            return ""
        return text[start + len(start_tag): end].strip()

    def get_state(self) -> dict[str, Any]:
        return {
            **super().get_state(),
            "has_scratchpad": bool(self._scratchpad),
            "update_count": self._update_count,
        }
