"""Context-management subsystem for the harness.

Retrieval-based tool selection and token-budget tracking. Keeps the
LLM's prompt under control when scaling past 100 tools.
"""

from harness.context.tool_retrieval import (
    ToolRetriever,
    CORE_TOOL_NAMES,
    score_query_domain,
)
from harness.context.budget import (
    TokenBudgetTracker,
    MODE_BUDGETS,
    BudgetExceededError,
)

__all__ = [
    "ToolRetriever",
    "CORE_TOOL_NAMES",
    "score_query_domain",
    "TokenBudgetTracker",
    "MODE_BUDGETS",
    "BudgetExceededError",
]
