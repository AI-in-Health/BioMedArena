"""Smoke tests for dicom-mcp adapter.

dicom-mcp is a Python 3.12+ package; we spawn it via
`uvx --python 3.12 --from dicom-mcp dicom-mcp <config.yaml>` so our
main venv stays on 3.11.15.

Without a live PACS node the server still starts and enumerates its 11
tools — we only test up to `list_tools` + `get_attribute_presets`
(which does not require PACS connectivity). Full DIMSE round-trips
would require Orthanc or dcm4chee running locally.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

pytest.importorskip("mcp")

_HAS_UVX = shutil.which("uvx") is not None
_REPO = Path(__file__).resolve().parent.parent.parent
_CONFIG = _REPO / "vendors" / "dicom_mcp" / "config.yaml"


# ======================================================================
# Wiring
# ======================================================================


def test_dicom_mcp_adapter_in_registry():
    from harness.adapters import ADAPTER_REGISTRY
    assert "DicomMCPAdapter" in ADAPTER_REGISTRY


def test_dicom_mcp_spec_registered():
    from harness.adapters.mcp_base import MCP_SERVER_REGISTRY
    assert "dicom_mcp" in MCP_SERVER_REGISTRY
    spec = MCP_SERVER_REGISTRY["dicom_mcp"]
    assert spec.command == "uvx"
    assert "dicom-mcp" in spec.args
    assert spec.tool_prefix == "mcp_dicom_"
    assert spec.modality == "imaging"


def test_default_config_file_exists():
    """The placeholder config YAML must ship with the repo."""
    assert _CONFIG.exists(), f"missing {_CONFIG}"
    text = _CONFIG.read_text()
    assert "nodes:" in text
    assert "current_node:" in text


def test_dicom_mcp_adapter_basic():
    from harness.adapters.dicom_mcp_adapter import DicomMCPAdapter
    a = DicomMCPAdapter()
    assert a.available is True
    assert a.name == "mcp_dicom_mcp"
    caps = a.capabilities()
    assert "pacs_query" in caps
    assert "dicom_study_query" in caps


def test_dicom_mcp_custom_config_path():
    """Passing config['dicom_mcp_config_path'] should route to that YAML."""
    from harness.adapters.dicom_mcp_adapter import DicomMCPAdapter
    a = DicomMCPAdapter(config={"dicom_mcp_config_path": "/tmp/fake.yaml"})
    # The spec args for this instance should reference the custom path
    assert any("/tmp/fake.yaml" in arg for arg in a.spec.args)


def test_mcp_registry_yaml_imaging_preset():
    import importlib.resources
    import yaml
    text = importlib.resources.files("harness.adapters").joinpath(
        "mcp_registry.yaml"
    ).read_text()
    data = yaml.safe_load(text)
    assert "dicom_mcp" in data["presets"]["imaging"]
    assert "dicom_mcp" in data["presets"]["full"]


# ======================================================================
# Live MCP round-trip (slow)
# ======================================================================


@pytest.mark.skipif(not _HAS_UVX, reason="uvx not on PATH")
@pytest.mark.slow
async def test_dicom_mcp_list_tools():
    """Server should enumerate 11 DICOM tools without needing a live PACS."""
    from harness.adapters.dicom_mcp_adapter import DicomMCPAdapter
    a = DicomMCPAdapter()
    try:
        specs = await asyncio.wait_for(a.list_tool_specs(), timeout=300)
    finally:
        await a.stop()
    names = [s["function"]["name"] for s in specs]
    assert all(n.startswith("mcp_dicom_") for n in names)
    stripped = {n.removeprefix("mcp_dicom_") for n in names}
    # The 11 known tools
    expected_subset = {
        "list_dicom_nodes", "switch_dicom_node", "verify_connection",
        "query_patients", "query_studies", "query_series", "query_instances",
        "get_attribute_presets",
    }
    missing = expected_subset - stripped
    assert not missing, f"expected tools missing: {missing}"
    assert len(specs) >= 10


@pytest.mark.skipif(not _HAS_UVX, reason="uvx not on PATH")
@pytest.mark.slow
async def test_dicom_mcp_get_attribute_presets():
    """`get_attribute_presets` returns static data — no PACS needed."""
    from harness.adapters.dicom_mcp_adapter import DicomMCPAdapter
    a = DicomMCPAdapter()
    try:
        # Populate cache
        await asyncio.wait_for(a.list_tool_specs(), timeout=300)
        text = await asyncio.wait_for(
            a.call_tool("mcp_dicom_get_attribute_presets", {}),
            timeout=60,
        )
    finally:
        await a.stop()
    assert not text.startswith("[mcp_error:"), f"presets returned error: {text[:200]}"
    assert len(text) > 20


@pytest.mark.skipif(not _HAS_UVX, reason="uvx not on PATH")
@pytest.mark.slow
async def test_dicom_mcp_list_dicom_nodes():
    """list_dicom_nodes reflects the placeholder config."""
    from harness.adapters.dicom_mcp_adapter import DicomMCPAdapter
    a = DicomMCPAdapter()
    try:
        await asyncio.wait_for(a.list_tool_specs(), timeout=300)
        text = await asyncio.wait_for(
            a.call_tool("mcp_dicom_list_dicom_nodes", {}),
            timeout=60,
        )
    finally:
        await a.stop()
    assert not text.startswith("[mcp_error:")
    # Our placeholder node is named 'main'
    assert "main" in text.lower()
