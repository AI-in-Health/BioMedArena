"""Dispatch shim for the function-calling runner.

The runner's ``_execute_tool`` falls through to ``_mcp_adapter_for(name)``
for any tool name it doesn't recognise. We register this class into
``MCP_ADAPTERS`` under the ``olsp_`` prefix so the runner routes our
ported tool calls here without requiring edits to the core runner file.
"""

from __future__ import annotations

from typing import Any

from harness.tools.openai_ported.handlers import HANDLERS


class OpenAIPortedDispatcher:
    """Adapter-shaped dispatcher for the ``olsp_`` tool prefix."""

    async def call_tool(self, name: str, args: dict[str, Any]) -> str:
        fn = HANDLERS.get(name)
        if fn is None:
            return f"[unknown olsp tool: {name}]"
        try:
            return await fn(args or {})
        except Exception as exc:  # defensive: never break the runner loop
            return f"[{name} error: {exc}]"
