"""Tools ported from openai/plugins life-science-research skills.

Upstream: https://github.com/openai/plugins/tree/main/plugins/life-science-research
The upstream skills are Markdown prompt instructions for Codex; this package
re-implements each skill's underlying REST/GraphQL/SPARQL API as a native
TOOL_SPEC handler for our function-calling runner.

Registration is kept merge-safe: handlers live here, and the dispatch shim
is registered into harness.eval.function_calling_runner.MCP_ADAPTERS under
the ``olsp_`` prefix (OpenAI Life-Science Ported) — no edits to the core
runner file required.
"""

from __future__ import annotations

from harness.tools.openai_ported.specs import (
    OPENAI_PORTED_TOOL_SPECS,
    OPENAI_PORTED_TOOL_NAMES,
    OPENAI_PORTED_PREFIX,
)
from harness.tools.openai_ported.dispatcher import OpenAIPortedDispatcher

__all__ = [
    "OPENAI_PORTED_TOOL_SPECS",
    "OPENAI_PORTED_TOOL_NAMES",
    "OPENAI_PORTED_PREFIX",
    "OpenAIPortedDispatcher",
    "register_openai_ported",
]


_REGISTERED = False


def register_openai_ported() -> int:
    """Register ported TOOL_SPECs with the function-calling runner.

    Idempotent. Returns number of specs added (0 if already registered or
    if the runner module cannot be imported — e.g. during standalone unit
    tests of this package).
    """
    global _REGISTERED
    if _REGISTERED:
        return 0
    try:
        from harness.eval import function_calling_runner as fcr
    except Exception:
        return 0
    # Re-check after the import: harness.eval.__init__ may have called us
    # re-entrantly while we were loading, and already done the work.
    if _REGISTERED:
        return 0

    fcr.TOOL_SPECS.extend(OPENAI_PORTED_TOOL_SPECS)
    fcr.MCP_ADAPTERS[OPENAI_PORTED_PREFIX] = OpenAIPortedDispatcher()
    _REGISTERED = True
    return len(OPENAI_PORTED_TOOL_SPECS)
