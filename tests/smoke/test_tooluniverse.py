"""Smoke tests for ToolUniverse MCP adapter.

ToolUniverse compact mode exposes 5 meta-tools that route to ~2214
underlying biomedical tools dynamically. First uvx invocation is slow
(~60-180s package install); subsequent sessions reuse the cache.
"""

from __future__ import annotations

import asyncio
import shutil

import pytest

pytest.importorskip("mcp")

_HAS_UVX = shutil.which("uvx") is not None


# ======================================================================
# Wiring
# ======================================================================


def test_tooluniverse_adapter_in_registry():
    from harness.adapters import ADAPTER_REGISTRY
    assert "ToolUniverseAdapter" in ADAPTER_REGISTRY


def test_mcp_server_registry_has_tooluniverse():
    from harness.adapters.mcp_base import MCP_SERVER_REGISTRY
    assert "tooluniverse" in MCP_SERVER_REGISTRY
    spec = MCP_SERVER_REGISTRY["tooluniverse"]
    assert spec.command == "uvx"
    assert "tooluniverse" in spec.args
    assert spec.tool_prefix == "mcp_tu_"


def test_tooluniverse_adapter_basic():
    from harness.adapters.tooluniverse_adapter import ToolUniverseAdapter
    a = ToolUniverseAdapter()
    assert a.available is True
    assert a.name == "mcp_tooluniverse"
    caps = a.capabilities()
    assert "tool_meta_dispatch" in caps
    assert "pubchem_query" in caps


def test_mcp_registry_yaml_includes_tooluniverse():
    import importlib.resources
    import yaml
    text = importlib.resources.files("harness.adapters").joinpath(
        "mcp_registry.yaml"
    ).read_text()
    data = yaml.safe_load(text)
    # Full preset must include tooluniverse
    assert "tooluniverse" in data["presets"]["full"]
    assert "tooluniverse" in data["presets"]["genomics"]


# ======================================================================
# Live MCP round-trip (slow)
# ======================================================================


@pytest.mark.skipif(not _HAS_UVX, reason="uvx not on PATH")
@pytest.mark.slow
async def test_tooluniverse_list_tools():
    """Compact mode should expose the 5 meta-tools."""
    from harness.adapters.tooluniverse_adapter import ToolUniverseAdapter
    a = ToolUniverseAdapter()
    try:
        specs = await asyncio.wait_for(a.list_tool_specs(), timeout=360)
    finally:
        await a.stop()
    names = [s["function"]["name"] for s in specs]
    # All should be prefixed
    assert all(n.startswith("mcp_tu_") for n in names)
    # The 5 meta-tools
    stripped = {n.removeprefix("mcp_tu_") for n in names}
    expected = {"list_tools", "grep_tools", "find_tools",
                 "get_tool_info", "execute_tool"}
    missing = expected - stripped
    assert not missing, f"expected meta-tools missing: {missing}"


@pytest.mark.skipif(not _HAS_UVX, reason="uvx not on PATH")
@pytest.mark.slow
async def test_tooluniverse_grep_tools():
    """grep_tools should return a non-empty result for 'drug'."""
    from harness.adapters.tooluniverse_adapter import ToolUniverseAdapter
    a = ToolUniverseAdapter()
    try:
        # First load to populate cache
        await asyncio.wait_for(a.list_tool_specs(), timeout=360)
        # Call grep_tools with a biomedical query likely to hit results
        text = await asyncio.wait_for(
            a.call_tool("mcp_tu_grep_tools", {"pattern": "drug"}),
            timeout=60,
        )
    finally:
        await a.stop()
    assert not text.startswith("[mcp_error:"), f"grep returned error: {text[:200]}"
    # Should mention at least one drug-related tool
    assert len(text) > 20
