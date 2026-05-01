"""Smoke tests for BioMCP adapter: MCPAdapter base + BioMCP server.

First run downloads ~80 packages into uv's tool cache (~60-120s). Tests
are marked slow and skip cleanly if the `mcp` SDK or `uvx` is absent.
"""

from __future__ import annotations

import asyncio
import shutil

import pytest

pytest.importorskip("mcp")

_HAS_UVX = shutil.which("uvx") is not None


# ======================================================================
# Registry / wiring
# ======================================================================


def test_biomcp_adapter_in_registry():
    from harness.adapters import ADAPTER_REGISTRY
    assert "BioMCPAdapter" in ADAPTER_REGISTRY


def test_mcp_server_registry_has_biomcp():
    from harness.adapters.mcp_base import MCP_SERVER_REGISTRY
    assert "biomcp" in MCP_SERVER_REGISTRY
    spec = MCP_SERVER_REGISTRY["biomcp"]
    assert spec.command == "uvx"
    assert "biomcp-python" in spec.args
    assert spec.tool_prefix == "mcp_biomcp_"


def test_biomcp_adapter_available_without_server():
    """Construction should not require the server to be up; availability
    only depends on the `mcp` SDK being importable."""
    from harness.adapters.biomcp_adapter import BioMCPAdapter
    a = BioMCPAdapter()
    assert a.available is True
    assert a.name == "mcp_biomcp"
    assert a.modality == "reasoning"
    # Capabilities should be non-empty and include a biomedical hint
    caps = a.capabilities()
    assert "pubmed_search" in caps
    assert "clinical_trial_search" in caps


def test_mcp_registry_yaml_loads():
    import importlib.resources
    import yaml
    text = importlib.resources.files("harness.adapters").joinpath(
        "mcp_registry.yaml"
    ).read_text()
    data = yaml.safe_load(text)
    assert data["version"] == 1
    assert "biomcp" in data["presets"]["minimal"]
    assert "biomcp" in data["presets"]["full"]


# ======================================================================
# Schema sanitisation
# ======================================================================


def test_sanitise_schema_strips_unknown_keys():
    from harness.adapters.mcp_base import _sanitise_schema
    raw = {
        "type": "object",
        "properties": {
            "q": {"type": "string", "vendor_hint": "ignored"},
            "n": {"type": "integer", "default": 5},
        },
        "required": ["q"],
        "x-vendor": "strip-me",
    }
    out = _sanitise_schema(raw)
    assert out["type"] == "object"
    assert "x-vendor" not in out
    assert out["required"] == ["q"]
    assert "vendor_hint" not in out["properties"]["q"]
    assert out["properties"]["n"]["default"] == 5


def test_sanitise_schema_recovers_non_object():
    from harness.adapters.mcp_base import _sanitise_schema
    out = _sanitise_schema("not a dict")  # type: ignore[arg-type]
    assert out == {"type": "object", "properties": {}}


def test_mcp_tool_to_openai_spec_shape():
    """Convert a mock MCP Tool to OpenAI function spec."""
    from harness.adapters.mcp_base import mcp_tool_to_openai_spec

    class _T:
        name = "article_searcher"
        description = "Search PubMed/PubTator3 for research articles."
        inputSchema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }

    spec = mcp_tool_to_openai_spec(_T(), tool_prefix="mcp_biomcp_")
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "mcp_biomcp_article_searcher"
    assert "query" in spec["function"]["parameters"]["properties"]
    assert spec["function"]["parameters"]["required"] == ["query"]


# ======================================================================
# Live MCP round-trip (slow)
# ======================================================================


@pytest.mark.skipif(not _HAS_UVX, reason="uvx not on PATH")
@pytest.mark.slow
async def test_biomcp_list_tools():
    """Spawn the BioMCP server via uvx and enumerate its tools."""
    from harness.adapters.biomcp_adapter import BioMCPAdapter
    a = BioMCPAdapter()
    try:
        specs = await asyncio.wait_for(a.list_tool_specs(), timeout=240)
    finally:
        await a.stop()
    assert len(specs) >= 10, f"expected >=10 tools, got {len(specs)}"
    names = [s["function"]["name"] for s in specs]
    # All specs should have the prefix
    assert all(n.startswith("mcp_biomcp_") for n in names)
    # BioMCP's `article_searcher` is a stable high-level tool
    assert any("article" in n for n in names), f"no article tool in {names[:5]}"


@pytest.mark.skipif(not _HAS_UVX, reason="uvx not on PATH")
@pytest.mark.slow
async def test_biomcp_call_tool_article_search():
    """Run one MCP tool round-trip via the adapter."""
    from harness.adapters.biomcp_adapter import BioMCPAdapter
    a = BioMCPAdapter()
    try:
        # Find the right article-search tool name (varies slightly across versions)
        specs = await asyncio.wait_for(a.list_tool_specs(), timeout=240)
        candidates = [s["function"]["name"] for s in specs
                        if "article" in s["function"]["name"] and "search" in s["function"]["name"]]
        assert candidates, "no article_search tool found"
        tool_name = candidates[0]
        text = await asyncio.wait_for(
            a.call_tool(tool_name, {"query": "CRISPR Cas9"}),
            timeout=120,
        )
    finally:
        await a.stop()
    # The tool should NOT return an MCP error marker, and should
    # return non-empty content.
    assert not text.startswith("[mcp_error:"), f"tool returned error: {text[:200]}"
    assert len(text) > 20


@pytest.mark.skipif(not _HAS_UVX, reason="uvx not on PATH")
@pytest.mark.slow
async def test_biomcp_through_adapter_run():
    """AdapterBase.run() path — no tool selection listing behaviour."""
    from harness.adapters.biomcp_adapter import BioMCPAdapter
    a = BioMCPAdapter()
    try:
        r = await asyncio.wait_for(a.run("", context={}), timeout=240)
    finally:
        await a.stop()
    # No tool_name -> listing mode, confidence = 0.3
    assert r["confidence"] == 0.3
    assert "tools" in r["raw"]
    assert any(n.startswith("mcp_biomcp_") for n in r["raw"]["tools"])


# ======================================================================
# FunctionCallingRunner integration (mocked LLM)
# ======================================================================


@pytest.mark.skipif(not _HAS_UVX, reason="uvx not on PATH")
@pytest.mark.slow
async def test_function_calling_runner_registers_mcp():
    """`register_mcp_adapters()` should append N BioMCP specs to TOOL_SPECS."""
    import harness.eval.function_calling_runner as fcr
    # Reset module state for a clean add
    fcr._MCP_SPECS_LOADED = False
    fcr.MCP_ADAPTERS.clear()
    before = len(fcr.TOOL_SPECS)
    added = await asyncio.wait_for(fcr.register_mcp_adapters(timeout=240), timeout=300)
    after = len(fcr.TOOL_SPECS)
    assert added >= 10, f"expected >=10 MCP specs, added {added}"
    assert after - before == added
    # Dispatch lookup by prefix
    assert "mcp_biomcp_" in fcr.MCP_ADAPTERS
    # Cleanup — stop the subprocess
    adapter = fcr.MCP_ADAPTERS["mcp_biomcp_"]
    await adapter.stop()
