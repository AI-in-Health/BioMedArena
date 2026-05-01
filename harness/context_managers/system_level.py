"""System-level strategies — configuration and architectural optimizations.

Strategies:
- LargeContext: increase token budgets for larger context windows
- PromptCache: optimize system prompt for better caching
"""

from __future__ import annotations

import logging
from typing import Any

from .base import Strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Large Context
# ---------------------------------------------------------------------------


class LargeContextStrategy(Strategy):
    """Configure for larger effective context by adjusting budgets.

    Meta-strategy that signals to other strategies and the environment
    to use larger token budgets.
    """

    name = "large_context"
    category = "system_level"

    def __init__(
        self,
        context_multiplier: float = 1.5,
        scraper_token_budget: int = 32768,
        max_result_tokens: int = 4096,
    ):
        self.context_multiplier = context_multiplier
        self.scraper_token_budget = scraper_token_budget
        self.max_result_tokens = max_result_tokens

    def apply(self, messages: list[dict[str, Any]]) -> None:
        pass

    def get_state(self) -> dict[str, Any]:
        return {
            **super().get_state(),
            "context_multiplier": self.context_multiplier,
            "scraper_token_budget": self.scraper_token_budget,
            "max_result_tokens": self.max_result_tokens,
        }


# ---------------------------------------------------------------------------
# 2. Prompt Cache
# ---------------------------------------------------------------------------


class PromptCacheStrategy(Strategy):
    """Optimize system prompt for caching efficiency.

    No-op at the ConversationManager level — caching happens at the API layer.
    Provides config that the environment can use to structure its system prompt.
    """

    name = "prompt_cache"
    category = "system_level"

    def __init__(self, cache_static_prefix: bool = True):
        self.cache_static_prefix = cache_static_prefix

    def apply(self, messages: list[dict[str, Any]]) -> None:
        pass

    def get_state(self) -> dict[str, Any]:
        return {
            **super().get_state(),
            "cache_static_prefix": self.cache_static_prefix,
        }
