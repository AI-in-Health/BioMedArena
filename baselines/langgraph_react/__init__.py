"""LangGraph-ReAct baseline — head-to-head comparison target for the harness.

Self-contained package. Do NOT import from `harness.*` here — the
point of this baseline is a clean reference implementation using
stock langgraph primitives.
"""

from .agent import LangGraphReActBaseline
from .tool_adapter import specs_to_tools

__all__ = ["LangGraphReActBaseline", "specs_to_tools"]
