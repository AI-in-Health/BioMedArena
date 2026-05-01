"""Token-budget tracker.

Per-mode limits keep the LLM prompt under control and give us a
graceful degrade path (compress → truncate → force-answer) when a task
runs long. Used by FunctionCallingRunner (light and heavy modes).

Budgets (in prompt-tokens and completion-tokens):

  simple_llm        8k in / 2k out
  light             200k in / 16k out
  heavy             200k in / 16k out
  deep_think        16k in / 8k out
  self_consistency  16k in / 4k out (per sample)

Counts via tiktoken (`cl100k_base` is a solid approximation for the
models we use — GPT-4o, o4-mini, Claude Sonnet 4.5, Gemini 2.5; token
counts are estimates, not exact for non-OpenAI models).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeBudget:
    mode: str
    input_tokens: int
    output_tokens: int
    total_tokens: int | None = None  # if set, overrides input+output
    notes: str = ""


MODE_BUDGETS: dict[str, ModeBudget] = {
    "simple_llm": ModeBudget(
        mode="simple_llm", input_tokens=8_000, output_tokens=2_000,
        notes="One-shot LLM, no tools.",
    ),
    "light": ModeBudget(
        mode="light", input_tokens=200_000, output_tokens=16_000,
        notes="Single-turn tool calling — 200k input fits 15-25 full tool-result iterations; 16k output supports extended reasoning.",
    ),
    "heavy": ModeBudget(
        mode="heavy", input_tokens=200_000, output_tokens=16_000,
        notes="Multi-turn ReAct with tool retrieval — 200k input handles large evidence contexts; 16k output for detailed synthesis.",
    ),
    "deep_think": ModeBudget(
        mode="deep_think", input_tokens=16_000, output_tokens=8_000,
        notes="Extended thinking mode (reasoning tokens counted as output).",
    ),
    "self_consistency": ModeBudget(
        mode="self_consistency", input_tokens=16_000, output_tokens=4_000,
        notes="Per-sample budget; full-run multiplier = N samples.",
    ),
}


class BudgetExceededError(RuntimeError):
    """Raised when `TokenBudgetTracker.ensure_room()` is called with a
    hard=True constraint and no room remains.
    """


# ---------------------------------------------------------------------------
# Tokeniser — lazy import so that runs without tiktoken still function
# ---------------------------------------------------------------------------


_ENCODING = None  # tiktoken.Encoding or None
_ENCODING_TRIED = False


def _get_encoding():
    global _ENCODING, _ENCODING_TRIED
    if _ENCODING_TRIED:
        return _ENCODING
    _ENCODING_TRIED = True
    try:
        import tiktoken
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # noqa: BLE001
        logger.warning("tiktoken unavailable (%s) — token counts will be rough", exc)
        _ENCODING = None
    return _ENCODING


def count_tokens(text: str) -> int:
    """Return an estimated token count for `text`.

    tiktoken `cl100k_base` when available; falls back to
    `len(text) // 4` (a very rough approximation) otherwise.
    """
    if not text:
        return 0
    enc = _get_encoding()
    if enc is None:
        return max(1, len(text) // 4)
    return len(enc.encode(text))


def count_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Token count for a list of chat messages in OpenAI/Anthropic format.

    Sums per-message overhead (~4) + encoded content for each message.
    """
    total = 0
    for m in messages:
        total += 4  # fixed per-message overhead
        content = m.get("content")
        if isinstance(content, str):
            total += count_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text") or ""
                    total += count_tokens(text)
        # Tool calls
        tool_calls = m.get("tool_calls") or []
        for tc in tool_calls:
            fn = (tc.get("function") or {})
            total += count_tokens(fn.get("name") or "")
            args = fn.get("arguments")
            if isinstance(args, str):
                total += count_tokens(args)
    return total


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class TokenBudgetTracker:
    """Lightweight per-task token accountant.

    Typical use inside a harness mode:

        tracker = TokenBudgetTracker(mode="light")
        tracker.observe_input(messages=messages)
        ...
        # Before next LLM call:
        action = tracker.degrade_action()
        if action == "force_answer":
            return last_assistant_text
        elif action == "truncate":
            messages = tracker.truncate(messages)

    The tracker never mutates messages itself — callers apply the
    chosen strategy. This keeps the tracker pure and testable.
    """

    def __init__(self, mode: str = "light",
                   budget: ModeBudget | None = None):
        self.mode = mode
        self.budget = budget or MODE_BUDGETS.get(mode, MODE_BUDGETS["light"])
        self.input_used = 0
        self.output_used = 0
        self.iterations = 0

    # -------- observation ---------------------------------------------

    def observe_input(self, messages: list[dict[str, Any]] | None = None,
                        tokens: int | None = None) -> int:
        """Record an input-side token count. Supply either a messages
        list (which will be counted) or a precomputed `tokens` integer."""
        n = tokens if tokens is not None else count_messages_tokens(messages or [])
        self.input_used = max(self.input_used, n)  # high-water mark
        return n

    def observe_output(self, text: str | None = None,
                         tokens: int | None = None) -> int:
        n = tokens if tokens is not None else count_tokens(text or "")
        self.output_used += n
        return n

    def tick_iteration(self) -> None:
        self.iterations += 1

    # -------- budget accounting ---------------------------------------

    def input_remaining(self) -> int:
        if self.budget.total_tokens is not None:
            return max(0, self.budget.total_tokens - self.input_used - self.output_used)
        return max(0, self.budget.input_tokens - self.input_used)

    def output_remaining(self) -> int:
        if self.budget.total_tokens is not None:
            return max(0, self.budget.total_tokens - self.input_used - self.output_used)
        return max(0, self.budget.output_tokens - self.output_used)

    def ensure_room(self, required_input: int = 0, required_output: int = 0,
                       hard: bool = False) -> bool:
        """Return True if there's room for required_input/required_output.

        If `hard=True` and there isn't, raise BudgetExceededError.
        """
        ok_in = self.input_remaining() >= required_input
        ok_out = self.output_remaining() >= required_output
        if not (ok_in and ok_out):
            if hard:
                raise BudgetExceededError(
                    f"mode={self.mode} need in={required_input} out={required_output} "
                    f"but have in={self.input_remaining()} out={self.output_remaining()}"
                )
            return False
        return True

    # -------- degrade strategy ----------------------------------------

    def degrade_action(self, headroom_frac: float = 0.1) -> str:
        """Return one of: 'ok' | 'compress' | 'truncate' | 'force_answer'.

        Headroom fraction: when input is > (1 - headroom) of budget we
        suggest compression; > 1.0 we truncate; > 1.25 we force-answer.
        """
        if self.budget.total_tokens is not None:
            cap = self.budget.total_tokens
            used = self.input_used + self.output_used
        else:
            cap = self.budget.input_tokens
            used = self.input_used
        if cap <= 0:
            return "ok"
        frac = used / cap
        if frac > 1.25:
            return "force_answer"
        if frac > 1.0:
            return "truncate"
        if frac > 1.0 - headroom_frac:
            return "compress"
        return "ok"

    def truncate(self, messages: list[dict[str, Any]],
                    keep_system: bool = True, keep_last: int = 4) -> list[dict[str, Any]]:
        """Drop middle messages to fit. Simple, deterministic strategy:

        1. Always keep the system prompt (if `keep_system`).
        2. Always keep the user's original question (first user message).
        3. Keep the last `keep_last` messages.
        4. Drop everything in between.
        """
        if not messages:
            return []
        head: list[dict[str, Any]] = []
        if keep_system and messages[0].get("role") == "system":
            head.append(messages[0])
            rest = messages[1:]
        else:
            rest = messages
        # First user message
        first_user_idx = next(
            (i for i, m in enumerate(rest) if m.get("role") == "user"), None
        )
        if first_user_idx is not None:
            head.append(rest[first_user_idx])
        tail = rest[-keep_last:] if len(rest) > keep_last else rest
        # Avoid double-including first_user in tail
        if first_user_idx is not None and rest[first_user_idx] in tail:
            tail = [m for m in tail if m is not rest[first_user_idx]]
        return head + tail

    # -------- snapshot / report ---------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "input_used": self.input_used,
            "output_used": self.output_used,
            "iterations": self.iterations,
            "input_budget": self.budget.input_tokens,
            "output_budget": self.budget.output_tokens,
            "total_budget": self.budget.total_tokens,
            "degrade": self.degrade_action(),
        }
