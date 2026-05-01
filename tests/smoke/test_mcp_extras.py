"""Smoke tests for extra MCP servers (PubMed / GEO / UniProt).

Live round-trip is OFF by default because these servers cold-start
slowly (uvx package pulls) and some need site-specific config/keys.
Opt in with HARNESS_RUN_EXTRA_MCP=1.
"""

from __future__ import annotations

import os
import shutil

import pytest

pytest.importorskip("mcp")
_HAS_UVX = shutil.which("uvx") is not None
_LIVE = os.environ.get("HARNESS_RUN_EXTRA_MCP") == "1"


def test_all_three_registered():
    from harness.adapters import ADAPTER_REGISTRY
    for name in ("PubMedMCPAdapter", "GEOMCPAdapter", "UniProtMCPAdapter"):
        assert name in ADAPTER_REGISTRY


def test_specs_registered():
    from harness.adapters.mcp_base import MCP_SERVER_REGISTRY
    for name in ("pubmed_mcp", "geo_mcp", "uniprot_mcp"):
        assert name in MCP_SERVER_REGISTRY
    assert MCP_SERVER_REGISTRY["pubmed_mcp"].modality == "literature"
    assert MCP_SERVER_REGISTRY["geo_mcp"].modality == "genomics"
    assert MCP_SERVER_REGISTRY["uniprot_mcp"].modality == "protein"


def test_registry_yaml_presets():
    import importlib.resources
    import yaml
    text = importlib.resources.files("harness.adapters").joinpath(
        "mcp_registry.yaml"
    ).read_text()
    data = yaml.safe_load(text)
    assert "literature" in data["presets"]
    assert "protein" in data["presets"]
    full = set(data["presets"]["full"])
    assert {"biomcp", "tooluniverse", "dicom_mcp",
              "pubmed_mcp", "geo_mcp", "uniprot_mcp"} <= full


def test_adapter_basic_pubmed():
    from harness.adapters.mcp_extra_adapters import PubMedMCPAdapter
    a = PubMedMCPAdapter()
    assert a.available is True
    caps = a.capabilities()
    assert "pubmed_advanced_search" in caps


def test_adapter_basic_geo():
    from harness.adapters.mcp_extra_adapters import GEOMCPAdapter
    a = GEOMCPAdapter()
    assert a.available is True
    assert "geo_dataset_search" in a.capabilities()


def test_adapter_basic_uniprot():
    from harness.adapters.mcp_extra_adapters import UniProtMCPAdapter
    a = UniProtMCPAdapter()
    assert a.available is True
    assert "uniprot_lookup" in a.capabilities()


@pytest.mark.skipif(not (_HAS_UVX and _LIVE), reason="live opt-in not set")
@pytest.mark.slow
async def test_live_one_extra_server_list_tools():
    """Opt-in live test — pick one server and verify list_tools works."""
    import asyncio
    from harness.adapters.mcp_extra_adapters import UniProtMCPAdapter
    a = UniProtMCPAdapter()
    try:
        specs = await asyncio.wait_for(a.list_tool_specs(), timeout=360)
    finally:
        await a.stop()
    # Any non-zero count means the protocol works end-to-end
    assert len(specs) >= 1
