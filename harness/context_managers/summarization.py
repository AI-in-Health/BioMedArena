"""Summarization strategies — compress old messages via LLM summaries.

Strategies:
- ConversationSummary: on overflow, summarize oldest chunk into one message
- ProgressiveSummary: proactively summarize every N cycles (rolling)
- IncrementalSummary: stay near context limit, compress minimally per trigger
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .base import (
    Strategy,
    estimate_messages_tokens,
    estimate_tokens,
    find_safe_split,
    is_cleared_content,
    truncate_to_tokens,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared summarization prompts
# ---------------------------------------------------------------------------

_RESEARCH_SUMMARY_PROMPT = """\
You are summarizing a web-research conversation. Produce a structured summary that \
preserves ALL factual findings, URLs, and data points discovered so far.

Format:
## Research Question
<the original user question>

## Key Findings
- Finding 1 (source: URL)
- Finding 2 (source: URL)
...

## Searches Performed
- query_1 -> brief result description
- query_2 -> brief result description
...

## Pages Read
- URL_1: key info extracted
- URL_2: key info extracted
...

## Open Questions
- What still needs to be investigated

Rules:
- Preserve exact data values, names, dates, and numbers
- Always include source URLs for each finding
- Be concise but do NOT drop any factual finding
- Do NOT address the user or add commentary
"""

_INCREMENTAL_SUMMARY_PROMPT = """\
You are updating a running research summary. You have the previous summary and \
new conversation turns. Merge the new information into the summary.

Previous summary:
{previous_summary}

Update the summary with any new findings, searches, or pages read from the \
conversation below. Keep the same format. Do not drop any previous findings.
"""

_CONDENSATION_PROMPT = """\
Condense the following research conversation into a brief summary. \
Keep all factual findings, URLs, and key data. Remove tool call details, \
verbose page content, and redundant information. Be concise.
"""


# ---------------------------------------------------------------------------
# Helper: condense tool results (no LLM needed)
# ---------------------------------------------------------------------------


def _condense_tool_result(text: str) -> str:
    """Condense a tool result while preserving the most valuable information."""
    # Visit results with evidence/summary structure
    if "Evidence" in text and "Summary" in text:
        # Extract Summary section
        summary_match = re.search(r"Summary:\s*\n?(.*)", text, re.DOTALL)
        if summary_match:
            summary = summary_match.group(1).strip()
            # Also grab any URL from the beginning
            url_match = re.match(r".*?(https?://\S+)", text)
            url = url_match.group(1) if url_match else ""
            if url:
                return f"[{url}] Summary: {summary[:500]}"
            return f"Summary: {summary[:500]}"

    # Search results (JSON-like format)
    if text.lstrip().startswith("{") and '"url"' in text:
        try:
            results = json.loads(text)
            condensed = []
            for title, info in results.items():
                url = info.get("url", "") if isinstance(info, dict) else ""
                condensed.append(f"- {title}: {url}")
            return "\n".join(condensed)
        except (json.JSONDecodeError, AttributeError):
            pass

    # General fallback: any tool result over 2000 tokens is truncated to 1000 tokens.
    # This handles biomedical tools (gene_lookup, clinvar_lookup, etc.) and any
    # other tool whose output doesn't match the patterns above.
    if estimate_tokens(text) > 2000:
        return truncate_to_tokens(text, 1000) + "\n[...truncated for context management]"

    # Short enough to keep as-is
    return text


# ---------------------------------------------------------------------------
# Helper: call LLM for summarization
# ---------------------------------------------------------------------------


def _call_llm_for_summary(
    llm: Any,
    messages_to_summarize: list[dict[str, Any]],
    system_prompt: str,
    user_instruction: str = "Please summarize this research conversation.",
) -> str:
    """Call the LLM synchronously (via asyncio) to generate a summary."""
    import asyncio

    # Build a condensed representation of the messages for the summarizer
    parts = []
    for msg in messages_to_summarize:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            continue
        if isinstance(content, str) and content:
            parts.append(f"[{role}] {content[:1000]}")
        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            fn = tc.get("function", {})
            parts.append(f"[tool_call] {fn.get('name', '')}({fn.get('arguments', '')[:200]})")

    conversation_text = "\n".join(parts)

    summary_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{user_instruction}\n\nConversation:\n{conversation_text}"},
    ]

    try:
        # Use the LLM client directly
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're already in an async context — create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(
                    asyncio.run, llm.chat(summary_messages, temperature=0.1, max_tokens=2048)
                ).result(timeout=30)
        else:
            result = asyncio.run(llm.chat(summary_messages, temperature=0.1, max_tokens=2048))
        return result
    except Exception as exc:
        logger.warning("[summarization] LLM call failed: %s", exc)
        # Fallback: just concatenate key content
        return conversation_text[:2000]


# ---------------------------------------------------------------------------
# 1. Conversation Summary
# ---------------------------------------------------------------------------


class ConversationSummaryStrategy(Strategy):
    """On context overflow, summarize the oldest chunk of messages."""

    name = "conversation_summary"
    category = "summarization"

    def __init__(
        self,
        summary_ratio: float = 0.4,
        preserve_recent: int = 10,
        system_prompt: str | None = None,
    ):
        self.summary_ratio = summary_ratio
        self.preserve_recent = preserve_recent
        self.system_prompt = system_prompt or _RESEARCH_SUMMARY_PROMPT
        self._compaction_count = 0
        self._llm = None  # Set externally by the runner

    def apply(self, messages: list[dict[str, Any]]) -> None:
        # Conversation summary is reactive only (triggered on overflow)
        pass

    def reduce(self, messages: list[dict[str, Any]]) -> int:
        """Summarize oldest messages to free context."""
        before = estimate_messages_tokens(messages)

        n_to_summarize = max(1, int(len(messages) * self.summary_ratio))
        n_to_summarize = min(n_to_summarize, len(messages) - self.preserve_recent)
        if n_to_summarize <= 0:
            return 0

        # Keep system message
        start_idx = 1 if messages and messages[0].get("role") == "system" else 0
        split = find_safe_split(messages, start_idx + n_to_summarize)
        if split <= start_idx or split >= len(messages):
            return 0

        msgs_to_summarize = messages[start_idx:split]
        remaining = messages[split:]

        if self._llm is not None:
            summary_text = _call_llm_for_summary(
                self._llm, msgs_to_summarize, self.system_prompt,
            )
        else:
            # Fallback: just concatenate key content
            parts = []
            for m in msgs_to_summarize:
                c = m.get("content", "")
                if isinstance(c, str) and c and m.get("role") != "tool":
                    parts.append(c[:200])
            summary_text = "## Conversation Summary\n" + "\n".join(parts[:10])

        summary_msg = {"role": "user", "content": summary_text}
        preserved = messages[:start_idx]
        messages[:] = preserved + [summary_msg] + remaining
        after = estimate_messages_tokens(messages)
        freed = before - after
        self._compaction_count += 1

        logger.info(
            "[conversation_summary] summarized %d messages into 1 (freed %d tokens)",
            len(msgs_to_summarize), freed,
        )
        return freed

    def get_state(self) -> dict[str, Any]:
        return {**super().get_state(), "compaction_count": self._compaction_count}


# ---------------------------------------------------------------------------
# 2. Progressive Summary
# ---------------------------------------------------------------------------


class ProgressiveSummaryStrategy(Strategy):
    """Proactively summarize every ``interval`` cycles to prevent overflow."""

    name = "progressive_summary"
    category = "summarization"

    def __init__(
        self,
        interval: int = 10,
        token_threshold: int = 80_000,
        preserve_recent: int = 8,
        system_prompt: str | None = None,
    ):
        self.interval = interval
        self.token_threshold = token_threshold
        self.preserve_recent = preserve_recent
        self.system_prompt = system_prompt or _RESEARCH_SUMMARY_PROMPT
        self._cycle_count = 0
        self._running_summary: str = ""
        self._summary_count = 0
        self._llm = None  # Set externally

    def apply(self, messages: list[dict[str, Any]]) -> None:
        self._cycle_count += 1

        if self._cycle_count % self.interval != 0:
            return

        total_tokens = estimate_messages_tokens(messages)
        if total_tokens < self.token_threshold:
            return

        if len(messages) <= self.preserve_recent + 2:
            return

        self._do_progressive_summary(messages)

    def reduce(self, messages: list[dict[str, Any]]) -> int:
        """On overflow, force an immediate progressive summary."""
        before = estimate_messages_tokens(messages)
        self._do_progressive_summary(messages)
        after = estimate_messages_tokens(messages)
        return before - after

    def _do_progressive_summary(self, messages: list[dict[str, Any]]) -> None:
        start_idx = 1 if messages and messages[0].get("role") == "system" else 0
        split = find_safe_split(messages, len(messages) - self.preserve_recent)
        if split <= start_idx or split >= len(messages):
            return

        msgs_to_summarize = messages[start_idx:split]
        remaining = messages[split:]

        if self._llm is not None:
            if self._running_summary:
                system_prompt = _INCREMENTAL_SUMMARY_PROMPT.format(
                    previous_summary=self._running_summary
                )
                instruction = "Update the summary with new findings from the conversation below."
            else:
                system_prompt = self.system_prompt
                instruction = "Summarize this research conversation."

            summary_text = _call_llm_for_summary(
                self._llm, msgs_to_summarize, system_prompt, instruction,
            )
            self._running_summary = summary_text
        else:
            parts = []
            for m in msgs_to_summarize:
                c = m.get("content", "")
                if isinstance(c, str) and c and m.get("role") != "tool":
                    parts.append(c[:200])
            summary_text = "## Progressive Summary\n" + "\n".join(parts[:10])

        summary_msg = {"role": "user", "content": summary_text}
        preserved = messages[:start_idx]
        messages[:] = preserved + [summary_msg] + remaining
        self._summary_count += 1

        logger.info(
            "[progressive_summary] cycle %d: summarized %d msgs (keeping %d recent)",
            self._cycle_count, len(msgs_to_summarize), len(remaining),
        )

    def get_state(self) -> dict[str, Any]:
        return {
            **super().get_state(),
            "cycle_count": self._cycle_count,
            "summary_count": self._summary_count,
            "has_running_summary": bool(self._running_summary),
        }


# ---------------------------------------------------------------------------
# 3. Incremental Summary
# ---------------------------------------------------------------------------


class IncrementalSummaryStrategy(Strategy):
    """Incremental condensation — stay near context limit, compress minimally.

    - Context grows freely until ``token_threshold`` is reached
    - On trigger: condense oldest uncondensed tool results (no LLM needed)
    - Fallback: when all are condensed, LLM-summarize a small batch
    """

    name = "incremental_summary"
    category = "summarization"

    def __init__(
        self,
        recent_count: int = 10,
        token_threshold: int = 100_000,
        batch_size: int = 5,
        llm_fallback_count: int = 15,
        system_prompt: str | None = None,
    ):
        self.recent_count = recent_count
        self.token_threshold = token_threshold
        self.batch_size = batch_size
        self.llm_fallback_count = llm_fallback_count
        self.system_prompt = system_prompt or _RESEARCH_SUMMARY_PROMPT
        self._compaction_count = 0
        self._condensed_up_to = 0
        self._llm = None  # Set externally

    def apply(self, messages: list[dict[str, Any]]) -> None:
        total_tokens = estimate_messages_tokens(messages)
        if total_tokens < self.token_threshold:
            return
        if len(messages) <= self.recent_count + 2:
            return
        self._incremental_condense(messages)

    def reduce(self, messages: list[dict[str, Any]]) -> int:
        """On overflow, force condensation (incremental first, LLM fallback)."""
        before = estimate_messages_tokens(messages)
        self._incremental_condense(messages)
        after = estimate_messages_tokens(messages)
        freed = before - after

        if freed < 3000:
            freed += self._llm_summary_fallback(messages)

        logger.info("[incremental_summary] reduce: freed %d tokens", freed)
        return freed

    def _incremental_condense(self, messages: list[dict[str, Any]]) -> None:
        """Condense the next batch of uncondensed tool result messages."""
        # Don't condense the most recent messages
        safe_end = max(0, len(messages) - self.recent_count)
        start = self._condensed_up_to
        if start >= safe_end:
            self._llm_summary_fallback(messages)
            return

        end = min(start + self.batch_size, safe_end)

        # Condense tool results in this batch
        condensed = 0
        for idx in range(start, end):
            msg = messages[idx]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str) or not content:
                continue
            if is_cleared_content(content):
                continue
            if len(content) > 500:
                msg["content"] = _condense_tool_result(content)
                condensed += 1

        self._condensed_up_to = end
        self._compaction_count += 1

        logger.info(
            "[incremental_summary] condensed messages[%d:%d] (%d tool results), "
            "condensed_up_to=%d",
            start, end, condensed, self._condensed_up_to,
        )

    def _llm_summary_fallback(self, messages: list[dict[str, Any]]) -> int:
        """When all messages are condensed, LLM-summarize the oldest batch."""
        before = estimate_messages_tokens(messages)

        start_idx = 1 if messages and messages[0].get("role") == "system" else 0
        n_to_summarize = min(self.llm_fallback_count, self._condensed_up_to)
        if n_to_summarize <= 1:
            return 0

        split = find_safe_split(messages, start_idx + n_to_summarize)
        if split <= start_idx or split >= len(messages):
            return 0

        old_msgs = messages[start_idx:split]
        remaining = messages[split:]

        if self._llm is not None:
            summary_text = _call_llm_for_summary(
                self._llm, old_msgs, self.system_prompt,
            )
        else:
            parts = []
            for m in old_msgs:
                c = m.get("content", "")
                if isinstance(c, str) and c and m.get("role") != "tool":
                    parts.append(c[:200])
            summary_text = "## Summary of Earlier Research\n" + "\n".join(parts[:10])

        summary_msg = {"role": "user", "content": summary_text}
        preserved = messages[:start_idx]
        messages[:] = preserved + [summary_msg] + remaining

        # Reset condensed pointer
        self._condensed_up_to = max(1, self._condensed_up_to - len(old_msgs) + 1)

        after = estimate_messages_tokens(messages)
        freed = before - after
        logger.info(
            "[incremental_summary] LLM fallback: summarized %d messages (freed %d tokens)",
            len(old_msgs), freed,
        )
        return freed

    def get_state(self) -> dict[str, Any]:
        return {
            **super().get_state(),
            "compaction_count": self._compaction_count,
            "condensed_up_to": self._condensed_up_to,
        }
