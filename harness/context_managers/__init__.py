"""Context management strategies registry and composable manager.

All strategies are organized into 6 categories:
1. Truncation: SlidingWindow, FirstLast, TokenBudget
2. Summarization: ConversationSummary, ProgressiveSummary, IncrementalSummary
3. Selective Clearing: ToolResultClearing, ThinkingClearing, MediaClearing, Deduplication
4. External Memory: SessionMemory, Scratchpad
5. System Level: LargeContext, PromptCache
6. Rollback: TurnRollback

Each strategy can be independently toggled via CM_* environment variables.
With no CM_ variables set, behavior is identical to baseline (no CM).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .base import Strategy, estimate_messages_tokens
from .external_memory import ScratchpadStrategy, SessionMemoryStrategy
from .selective_clearing import (
    DeduplicationStrategy,
    MediaClearingStrategy,
    ThinkingClearingStrategy,
    ToolResultClearingStrategy,
)
from .summarization import (
    ConversationSummaryStrategy,
    IncrementalSummaryStrategy,
    ProgressiveSummaryStrategy,
)
from .system_level import LargeContextStrategy, PromptCacheStrategy
from .truncation import FirstLastStrategy, SlidingWindowStrategy, TokenBudgetStrategy
from .turn_rollback import TurnRollbackStrategy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "sliding_window": SlidingWindowStrategy,
    "first_last": FirstLastStrategy,
    "token_budget": TokenBudgetStrategy,
    "conversation_summary": ConversationSummaryStrategy,
    "progressive_summary": ProgressiveSummaryStrategy,
    "incremental_summary": IncrementalSummaryStrategy,
    "tool_result_clearing": ToolResultClearingStrategy,
    "thinking_clearing": ThinkingClearingStrategy,
    "media_clearing": MediaClearingStrategy,
    "deduplication": DeduplicationStrategy,
    "session_memory": SessionMemoryStrategy,
    "scratchpad": ScratchpadStrategy,
    "large_context": LargeContextStrategy,
    "prompt_cache": PromptCacheStrategy,
    "turn_rollback": TurnRollbackStrategy,
}


# ---------------------------------------------------------------------------
# ComposableContextManager
# ---------------------------------------------------------------------------


class ComposableContextManager:
    """Composes multiple strategies into a single context manager.

    Strategies run in insertion order. In ``apply_management()``, all
    lightweight strategies run. In ``reduce_context()``, heavyweight
    strategies run in order until enough context is freed.
    """

    def __init__(self, strategies: list[Strategy]):
        self.strategies = strategies

    def apply_management(self, messages: list[dict[str, Any]], **kwargs: Any) -> None:
        """Run all strategies' lightweight apply() hooks."""
        for strategy in self.strategies:
            try:
                strategy.apply(messages)
            except Exception as exc:
                logger.warning(
                    "[composable] strategy %s.apply() failed: %s",
                    strategy.name, exc,
                )

    def reduce_context(
        self, messages: list[dict[str, Any]], e: Exception | None = None, **kwargs: Any
    ) -> None:
        """Run strategies' reduce() hooks until enough context is freed.

        Tries each strategy in order. If any strategy frees > 5000 tokens,
        we stop and return (the agent loop will retry). If no strategy
        succeeds, re-raises the exception.
        """
        total_freed = 0

        for strategy in self.strategies:
            try:
                freed = strategy.reduce(messages)
                total_freed += freed
                if freed > 5000:
                    logger.info(
                        "[composable] %s freed %d tokens (total: %d), stopping",
                        strategy.name, freed, total_freed,
                    )
                    return
            except Exception as exc:
                logger.warning(
                    "[composable] strategy %s.reduce() failed: %s",
                    strategy.name, exc,
                )

        if total_freed > 2000:
            logger.info("[composable] total freed %d tokens across all strategies", total_freed)
            return

        # Nothing worked — re-raise
        if e:
            raise e

    def get_state(self) -> dict[str, Any]:
        return {"strategies": [s.get_state() for s in self.strategies]}


# ---------------------------------------------------------------------------
# Environment variable parsing
# ---------------------------------------------------------------------------

def _bool(key: str) -> bool:
    return os.environ.get(key, "").lower() in ("1", "true", "yes")


def _int(key: str, default: int) -> int:
    val = os.environ.get(key)
    return int(val) if val is not None else default


def _float(key: str, default: float) -> float:
    val = os.environ.get(key)
    return float(val) if val is not None else default


def parse_strategies_from_env() -> list[Strategy]:
    """Build a list of Strategy instances from CM_* environment variables.

    Returns an empty list if no CM_ variables are set (baseline behavior).
    """
    all_on = _bool("CM_ALL")

    strategies: list[Strategy] = []

    # --- Order matters: lightweight first, heavyweight last ---

    # 0. Turn rollback (must run before deduplication)
    if _bool("CM_TURN_ROLLBACK"):
        strategies.append(TurnRollbackStrategy(
            max_consecutive=_int("CM_ROLLBACK_MAX_CONSECUTIVE", 5),
            guidance_mode=os.environ.get("CM_ROLLBACK_GUIDANCE", "basic"),
        ))

    # 1. Selective clearing (lightweight, every cycle)
    if all_on or _bool("CM_DEDUPLICATION"):
        strategies.append(DeduplicationStrategy())

    if all_on or _bool("CM_TOOL_RESULT_CLEARING"):
        strategies.append(ToolResultClearingStrategy(
            keep_recent=_int("CM_CLEAR_KEEP_RECENT", 5),
            trigger_count=_int("CM_CLEAR_TRIGGER_COUNT", 8),
            max_result_tokens=_int("CM_CLEAR_MAX_RESULT_TOKENS", 2048),
        ))

    if all_on or _bool("CM_THINKING_CLEARING"):
        strategies.append(ThinkingClearingStrategy(
            keep_recent_turns=_int("CM_THINKING_KEEP_RECENT", 4),
        ))

    if all_on or _bool("CM_MEDIA_CLEARING"):
        strategies.append(MediaClearingStrategy(
            keep_recent_turns=_int("CM_MEDIA_KEEP_RECENT", 4),
        ))

    # 2. External memory (extraction in apply, restoration in reduce)
    if all_on or _bool("CM_SESSION_MEMORY"):
        strategies.append(SessionMemoryStrategy(
            max_findings=_int("CM_SESSION_MEMORY_MAX", 10),
            max_tokens=_int("CM_SESSION_MEMORY_TOKENS", 5000),
        ))

    if _bool("CM_SCRATCHPAD"):
        try:
            strategies.append(ScratchpadStrategy(
                max_tokens=_int("CM_SCRATCHPAD_MAX_TOKENS", 3000),
            ))
        except Exception as exc:
            logger.warning("[build_context_manager] CM_SCRATCHPAD is set but ScratchpadStrategy failed to initialise: %s", exc)

    # 3. Truncation (moderate weight)
    if _bool("CM_SLIDING_WINDOW"):
        strategies.append(SlidingWindowStrategy(
            window_size=_int("CM_SLIDING_WINDOW_SIZE", 20),
        ))

    if _bool("CM_FIRST_LAST"):
        strategies.append(FirstLastStrategy(
            keep_first=_int("CM_FIRST_LAST_FIRST", 2),
            keep_last=_int("CM_FIRST_LAST_LAST", 16),
        ))

    if _bool("CM_TOKEN_BUDGET"):
        strategies.append(TokenBudgetStrategy(
            max_tokens=_int("CM_TOKEN_BUDGET_MAX", 150_000),
            min_keep=_int("CM_TOKEN_BUDGET_MIN_KEEP", 6),
        ))

    # 4. Summarization (heavyweight, mainly in reduce)
    if all_on or _bool("CM_CONVERSATION_SUMMARY"):
        strategies.append(ConversationSummaryStrategy(
            summary_ratio=_float("CM_SUMMARY_RATIO", 0.4),
            preserve_recent=_int("CM_SUMMARY_PRESERVE_RECENT", 10),
        ))

    if _bool("CM_PROGRESSIVE_SUMMARY"):
        strategies.append(ProgressiveSummaryStrategy(
            interval=_int("CM_PROGRESSIVE_INTERVAL", 10),
            token_threshold=_int("CM_PROGRESSIVE_THRESHOLD", 80_000),
            preserve_recent=_int("CM_SUMMARY_PRESERVE_RECENT", 8),
        ))

    if _bool("CM_INCREMENTAL_SUMMARY"):
        try:
            strategies.append(IncrementalSummaryStrategy(
                recent_count=_int("CM_INCREMENTAL_RECENT", 10),
                token_threshold=_int("CM_INCREMENTAL_THRESHOLD", 100_000),
                batch_size=_int("CM_INCREMENTAL_BATCH", 5),
                llm_fallback_count=_int("CM_INCREMENTAL_LLM_BATCH", 15),
            ))
        except Exception as exc:
            logger.warning("[build_context_manager] CM_INCREMENTAL_SUMMARY is set but IncrementalSummaryStrategy failed to initialise: %s", exc)

    # 5. System level (configuration, no per-cycle action)
    if all_on or _bool("CM_LARGE_CONTEXT"):
        strategies.append(LargeContextStrategy(
            context_multiplier=_float("CM_LARGE_CONTEXT_MULTIPLIER", 1.5),
        ))

    if _bool("CM_PROMPT_CACHE"):
        strategies.append(PromptCacheStrategy())

    return strategies


def build_context_manager() -> ComposableContextManager | None:
    """Build a ComposableContextManager from env vars. Returns None if no CM enabled."""
    strategies = parse_strategies_from_env()
    if not strategies:
        # Warn when the caller explicitly opted in but nothing was registered.
        _CM_EXPLICIT_FLAGS = ("CM_SCRATCHPAD", "CM_INCREMENTAL_SUMMARY")
        active = [f for f in _CM_EXPLICIT_FLAGS if _bool(f)]
        if active:
            logger.warning(
                "[build_context_manager] %s env var(s) are set but no strategies were "
                "registered — context manager is disabled. Check for earlier warnings.",
                ", ".join(active),
            )
        return None
    return ComposableContextManager(strategies)
