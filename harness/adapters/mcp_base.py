"""MCPAdapter base class — wrap an MCP server as both an AdapterBase and a
ToolSource (in the Inspect-AI sense).

One MCP server → N tools. The adapter connects lazily, enumerates the
server's tools on first use, converts each MCP `Tool` schema into an
OpenAI-style TOOL_SPECS entry with an `mcp_<server>_<tool>` prefix, and
exposes both:

1. `AdapterBase.run(query, context)` — for direct harness-mode dispatch
   (context['tool_name'] selects the MCP tool, context['tool_args'] is
   the argument dict). Useful for adapters that wrap a single logical
   capability.
2. `list_tool_specs()` / `call_tool(name, args)` — for the
   FunctionCallingRunner, which appends the specs to its TOOL_SPECS list
   and dispatches by name prefix.

Error semantics follow the Inspect-AI two-tier pattern:
- MCP tool returns `isError=True`  → model-visible string `[mcp_error: …]`
  so the LLM can retry with different args.
- Subprocess / transport failure → raises `MCPTransportError`; the
  caller decides whether to retry or abort.

Preset-driven startup via `MCP_SERVER_REGISTRY` (populated by concrete
adapters at import time) lets upstream code pick which servers to
launch via `harness/adapters/mcp_registry.yaml`.

Applies lessons from `.claude_workspace/HARNESS_ANALYSIS/inspect_ai.md`:
  L1 — model MCP server as ToolSource (N auto-registered specs)
  L2 — two-tier errors (ToolError-like string vs transport exception)
  L3 — one-shot `call_tool`, no internal tool-loop
  L4 — content-typed output ready for future multimodal
  L5 — registry-first presets
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, ClassVar

from harness.adapter_base import AdapterBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class MCPTransportError(RuntimeError):
    """Subprocess or protocol failure — not recoverable by retry from model."""


class MCPNotAvailableError(RuntimeError):
    """Raised when the `mcp` Python SDK is not installed."""


# ---------------------------------------------------------------------------
# Server spec (registry entries)
# ---------------------------------------------------------------------------


@dataclass
class MCPServerSpec:
    """Static description of how to launch an MCP stdio server.

    Attributes
    ----------
    name
        Short registry key (e.g. 'biomcp', 'dicom', 'tooluniverse').
    command
        Executable to spawn (e.g. 'uvx', '/path/to/python').
    args
        CLI args passed to the command.
    env
        Extra env vars merged onto `os.environ`. Secrets should be
        pulled from the process env, not hardcoded.
    description
        Human-readable summary (shown in adapter.description).
    modality
        Routing hint for the harness (genomics | drug | imaging | …).
    tool_prefix
        Prefix prepended to each MCP tool name to form the
        FunctionCallingRunner tool name. Default `mcp_<name>_`.
    allowed_tools
        If non-empty, only these MCP tool names are exposed. Useful for
        large servers (ToolUniverse has 200+ tools → whitelist 30–50).
    """

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    description: str = ""
    modality: str = "reasoning"
    tool_prefix: str | None = None
    allowed_tools: set[str] = field(default_factory=set)


# Populated at import time by concrete adapters. Ordered dict so presets
# in `mcp_registry.yaml` can reference insertion order.
MCP_SERVER_REGISTRY: dict[str, MCPServerSpec] = {}


def register_mcp_server(spec: MCPServerSpec) -> MCPServerSpec:
    """Register (or overwrite) an MCP server spec in the module registry."""
    MCP_SERVER_REGISTRY[spec.name] = spec
    return spec


# ---------------------------------------------------------------------------
# Persistent stdio session wrapper
# ---------------------------------------------------------------------------


class MCPServerSession:
    """Long-lived MCP stdio client running in a dedicated asyncio task.

    The `mcp` SDK uses context managers that must be entered and exited
    in the same task. We spawn a worker task, keep it alive, and dispatch
    `list_tools` / `call_tool` through asyncio queues.
    """

    def __init__(self, spec: MCPServerSpec, startup_timeout: int = 60):
        self.spec = spec
        self.startup_timeout = startup_timeout
        self._worker: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._stopped = asyncio.Event()
        self._request_q: asyncio.Queue | None = None
        self._start_error: BaseException | None = None
        self._tools_cache: list[Any] | None = None

    async def start(self) -> None:
        if self._worker is not None:
            return
        # Defer import so missing SDK is reported only when used
        try:
            from mcp import ClientSession, StdioServerParameters  # noqa: F401
            from mcp.client.stdio import stdio_client  # noqa: F401
        except ImportError as exc:
            raise MCPNotAvailableError(f"`mcp` SDK not installed: {exc}") from exc

        self._request_q = asyncio.Queue()
        self._worker = asyncio.create_task(self._run_worker(), name=f"mcp-{self.spec.name}")
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=self.startup_timeout)
        except asyncio.TimeoutError as exc:
            # Kill the worker before raising
            self._worker.cancel()
            raise MCPTransportError(
                f"MCP server '{self.spec.name}' did not initialise within "
                f"{self.startup_timeout}s"
            ) from exc
        if self._start_error is not None:
            raise MCPTransportError(
                f"MCP server '{self.spec.name}' failed to start: {self._start_error}"
            ) from self._start_error

    async def _run_worker(self) -> None:
        """Worker coroutine — owns the MCP client context managers."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        merged_env = {**os.environ, **self.spec.env}
        params = StdioServerParameters(
            command=self.spec.command,
            args=list(self.spec.args),
            env=merged_env,
        )
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._ready.set()
                    while True:
                        msg = await self._request_q.get()
                        if msg is None:
                            break  # shutdown sentinel
                        op, payload, fut = msg
                        try:
                            if op == "list_tools":
                                result = await session.list_tools()
                                fut.set_result(result)
                            elif op == "call_tool":
                                name, args, timeout = payload
                                import datetime as _dt
                                td = _dt.timedelta(seconds=timeout) if timeout else None
                                result = await session.call_tool(
                                    name, arguments=args or {},
                                    read_timeout_seconds=td,
                                )
                                fut.set_result(result)
                            else:
                                fut.set_exception(ValueError(f"unknown op: {op}"))
                        except BaseException as exc:  # noqa: BLE001
                            if not fut.done():
                                fut.set_exception(exc)
        except BaseException as exc:  # noqa: BLE001
            self._start_error = exc
            self._ready.set()  # unblock start()
        finally:
            self._stopped.set()

    async def list_tools(self) -> list[Any]:
        if self._tools_cache is not None:
            return self._tools_cache
        await self.start()
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        assert self._request_q is not None
        await self._request_q.put(("list_tools", None, fut))
        result = await fut
        tools = list(result.tools)
        if self.spec.allowed_tools:
            tools = [t for t in tools if t.name in self.spec.allowed_tools]
        self._tools_cache = tools
        return tools

    async def call_tool(self, name: str, args: dict[str, Any] | None = None,
                          timeout: float | None = 60.0) -> Any:
        await self.start()
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        assert self._request_q is not None
        await self._request_q.put(("call_tool", (name, args, timeout), fut))
        return await fut

    async def stop(self) -> None:
        if self._worker is None:
            return
        if self._request_q is not None:
            try:
                await self._request_q.put(None)
            except Exception:  # noqa: BLE001
                pass
        try:
            await asyncio.wait_for(self._stopped.wait(), timeout=5)
        except asyncio.TimeoutError:
            self._worker.cancel()
        self._worker = None
        self._request_q = None
        self._ready.clear()
        self._stopped.clear()
        self._tools_cache = None


# ---------------------------------------------------------------------------
# Content flattening (lesson 4: ready for multimodal)
# ---------------------------------------------------------------------------


def _content_to_text(content_list: list[Any]) -> str:
    """Flatten MCP content blocks to a single string.

    Text blocks are concatenated verbatim. Image / resource blocks are
    replaced with a [type] placeholder so an upstream LLM can see that
    non-text payload was returned. Upgrade later to return Content
    objects when we add multimodal handling.
    """
    parts: list[str] = []
    for item in content_list or []:
        t = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
        if t == "text":
            txt = getattr(item, "text", None) or (item.get("text") if isinstance(item, dict) else "")
            parts.append(str(txt))
        elif t == "image":
            mime = getattr(item, "mimeType", "") or "?"
            parts.append(f"[image {mime}]")
        elif t == "resource":
            parts.append("[resource]")
        else:
            # unknown content type — stringify best-effort
            parts.append(str(item)[:200])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tool-spec conversion (lesson 1: ToolSource → N TOOL_SPECS entries)
# ---------------------------------------------------------------------------


_SCHEMA_SAFE_KEYS = {
    "type", "properties", "required", "items", "enum", "description",
    "default", "anyOf", "oneOf", "allOf", "format",
}


def _sanitise_schema(schema: Any) -> dict[str, Any]:
    """Strip MCP-specific / non-standard fields from an inputSchema so the
    result is valid OpenAI/Gemini function-calling JSON schema."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    out: dict[str, Any] = {}
    for k, v in schema.items():
        if k in _SCHEMA_SAFE_KEYS:
            if k == "properties" and isinstance(v, dict):
                out[k] = {pk: _sanitise_schema(pv) for pk, pv in v.items()}
            elif k == "items":
                out[k] = _sanitise_schema(v)
            else:
                out[k] = v
    if "type" not in out:
        out["type"] = "object"
    if out["type"] == "object" and "properties" not in out:
        out["properties"] = {}
    return out


def mcp_tool_to_openai_spec(
    tool: Any,
    tool_prefix: str,
    description_suffix: str = "",
) -> dict[str, Any]:
    """Convert one MCP Tool to an OpenAI/Gemini function-calling spec dict."""
    raw_schema = getattr(tool, "inputSchema", None) or {}
    schema = _sanitise_schema(raw_schema)
    desc = (getattr(tool, "description", None) or "").strip()
    if description_suffix:
        desc = f"{desc} {description_suffix}".strip()
    return {
        "type": "function",
        "function": {
            "name": f"{tool_prefix}{tool.name}",
            "description": desc[:1024],
            "parameters": schema,
        },
    }


# ---------------------------------------------------------------------------
# MCPAdapter — AdapterBase subclass
# ---------------------------------------------------------------------------


class MCPAdapter(AdapterBase):
    """Single MCP server exposed as both an AdapterBase and a ToolSource.

    Subclass to bind to a specific server, or instantiate with a
    pre-registered `spec_name` lookup into `MCP_SERVER_REGISTRY`.
    """

    # Subclasses should set this, or pass `spec=` to __init__.
    default_spec: ClassVar[MCPServerSpec | None] = None

    def __init__(self,
                   spec: MCPServerSpec | None = None,
                   spec_name: str | None = None,
                   config: dict | None = None,
                   startup_timeout: int = 60,
                   per_tool_timeout: float = 60.0,
                   **kwargs: Any):
        chosen = spec
        if chosen is None and spec_name is not None:
            chosen = MCP_SERVER_REGISTRY.get(spec_name)
        if chosen is None:
            chosen = self.default_spec
        if chosen is None:
            raise ValueError(
                "MCPAdapter needs spec= or spec_name= or class-level "
                "default_spec (none provided)."
            )
        self.spec = chosen
        self._config = config or {}
        self._session: MCPServerSession | None = None
        self._per_tool_timeout = per_tool_timeout
        self._startup_timeout = startup_timeout

        # AdapterBase attributes
        self.name = f"mcp_{self.spec.name}"
        self.modality = self.spec.modality or "reasoning"
        self.description = (
            self.spec.description
            or f"MCP server '{self.spec.name}' exposing {len(self.spec.allowed_tools) or 'N'} tools."
        )

        # Availability probe: only the SDK being installed. Actual server
        # liveness is deferred until first call so __init__ stays cheap.
        try:
            import mcp  # noqa: F401
        except ImportError:
            self.mark_unavailable("`mcp` Python SDK not installed (pip install mcp)")

    # -------- Session lifecycle -------------------------------------

    def _get_session(self) -> MCPServerSession:
        if self._session is None:
            self._session = MCPServerSession(
                self.spec, startup_timeout=self._startup_timeout,
            )
        return self._session

    async def stop(self) -> None:
        if self._session is not None:
            await self._session.stop()
            self._session = None

    # -------- ToolSource interface (lesson 1) -----------------------

    def tool_prefix(self) -> str:
        return self.spec.tool_prefix or f"mcp_{self.spec.name}_"

    async def list_tool_specs(self) -> list[dict[str, Any]]:
        """Return OpenAI-style function specs for every MCP tool."""
        if not self.available:
            return []
        session = self._get_session()
        try:
            tools = await session.list_tools()
        except MCPNotAvailableError:
            self.mark_unavailable("`mcp` SDK missing")
            return []
        except MCPTransportError as exc:
            logger.warning("MCP list_tools(%s) failed: %s", self.spec.name, exc)
            self.mark_unavailable(f"transport: {exc}")
            return []
        prefix = self.tool_prefix()
        suffix = f"(via MCP server '{self.spec.name}')"
        return [mcp_tool_to_openai_spec(t, prefix, suffix) for t in tools]

    async def call_tool(self, name: str, args: dict[str, Any] | None = None) -> str:
        """One-shot tool call (lesson 3: no internal loop here).

        Strips the `tool_prefix` from `name` before forwarding.
        Returns a model-visible string. Two-tier errors (lesson 2):
          - `[mcp_error: …]` → LLM can retry
          - raises MCPTransportError → caller decides to abort / retry
        """
        if not self.available:
            return f"[mcp_error: adapter unavailable ({self.unavailable_reason})]"
        prefix = self.tool_prefix()
        mcp_name = name[len(prefix):] if name.startswith(prefix) else name
        session = self._get_session()
        try:
            result = await session.call_tool(
                mcp_name, args or {}, timeout=self._per_tool_timeout,
            )
        except MCPNotAvailableError as exc:
            self.mark_unavailable(str(exc))
            raise
        except asyncio.TimeoutError:
            return f"[mcp_error: {mcp_name} timed out after {self._per_tool_timeout}s]"
        except Exception as exc:  # noqa: BLE001
            # Unknown transport failure — surface as transport error so the
            # runner can decide whether to retry or abort.
            raise MCPTransportError(f"{self.spec.name}/{mcp_name}: {exc}") from exc

        is_error = bool(getattr(result, "isError", False))
        text = _content_to_text(getattr(result, "content", []) or [])
        if is_error:
            return f"[mcp_error: {mcp_name}: {text[:500]}]"
        # Prefer structured output JSON when present — more robust than text.
        structured = getattr(result, "structuredContent", None)
        if structured:
            try:
                return json.dumps(structured)[:20000]
            except (TypeError, ValueError):
                pass
        return text[:20000] if text else "[mcp: empty result]"

    # -------- AdapterBase.run (lesson 3: delegate to call_tool) ------

    async def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        ctx = context or {}
        tool_name = ctx.get("tool_name")
        tool_args = ctx.get("tool_args") or {}
        if not tool_name:
            # Without a concrete tool, list what's available
            specs = await self.list_tool_specs()
            names = [s["function"]["name"] for s in specs]
            return self.result(
                answer=(
                    f"MCPAdapter '{self.spec.name}' exposes {len(names)} tools. "
                    f"Pass context={{'tool_name': …, 'tool_args': {{…}}}} to invoke one."
                ),
                evidence=names[:20],
                confidence=0.3,
                raw={"tools": names},
            )
        try:
            text = await self.call_tool(tool_name, tool_args)
        except MCPTransportError as exc:
            return self.result(
                answer=f"MCP transport error: {exc}",
                evidence=[f"server={self.spec.name}", f"tool={tool_name}"],
                confidence=0.0,
            )
        confidence = 0.2 if text.startswith("[mcp_error:") else 0.8
        return self.result(
            answer=text,
            evidence=[f"mcp:{self.spec.name}.{tool_name}"],
            confidence=confidence,
            raw={"server": self.spec.name, "tool": tool_name, "args": tool_args},
        )
