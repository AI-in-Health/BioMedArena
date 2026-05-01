"""LangChain tool adapter — converts OpenAI-schema TOOL_SPECS to @tool decorators.

Self-contained: imports only from langchain_core / pydantic (third-party).
No imports from harness.* so this baseline can be benchmarked head-to-head
against the harness without sharing dispatch code.

Usage:
    from baselines.langgraph_react.tool_adapter import specs_to_tools

    tools = specs_to_tools(tool_specs, dispatch=lambda name, args: "result")
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model


_TYPE_MAP: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _schema_to_pydantic_model(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Convert a JSON-schema object into a pydantic model class."""
    props = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])
    fields: dict[str, Any] = {}
    for pname, pspec in props.items():
        ptype_str = (pspec or {}).get("type", "string")
        ptype = _TYPE_MAP.get(ptype_str, str)
        description = (pspec or {}).get("description", "")
        default_val = (pspec or {}).get("default")
        if pname in required:
            fields[pname] = (ptype, Field(..., description=description))
        elif default_val is not None:
            fields[pname] = (ptype, Field(default_val, description=description))
        else:
            fields[pname] = (ptype | None if False else ptype,
                              Field(default=None, description=description))
    # sanitised class name for pydantic
    cls_name = "".join(c for c in name.title() if c.isalnum()) or "Args"
    return create_model(f"{cls_name}Args", **fields)


def specs_to_tools(
    tool_specs: list[dict[str, Any]],
    dispatch: Callable[[str, dict[str, Any]], Any],
) -> list[StructuredTool]:
    """Build StructuredTool objects from OpenAI-schema specs.

    `dispatch(name, args) -> str|dict` is the callable the agent will
    invoke for each tool. Baseline keeps this dispatch minimal —
    concrete tool semantics are injected by the benchmark harness at
    call time (see agent.py for a default that just echoes).
    """
    tools: list[StructuredTool] = []
    for spec in tool_specs:
        if spec.get("type") != "function":
            continue
        fn_spec = spec.get("function") or {}
        name = fn_spec.get("name")
        if not name:
            continue
        description = fn_spec.get("description") or ""
        schema = fn_spec.get("parameters") or {"type": "object", "properties": {}}
        args_model = _schema_to_pydantic_model(name, schema)

        def _make_runner(tool_name: str):
            def _runner(**kwargs) -> str:
                result = dispatch(tool_name, kwargs)
                if asyncio.iscoroutine(result):
                    loop = asyncio.new_event_loop()
                    try:
                        result = loop.run_until_complete(result)
                    finally:
                        loop.close()
                if isinstance(result, (dict, list)):
                    return json.dumps(result)[:2000]
                return str(result)[:2000]
            _runner.__name__ = tool_name
            return _runner

        tools.append(StructuredTool.from_function(
            func=_make_runner(name),
            name=name,
            description=description,
            args_schema=args_model,
        ))
    return tools
