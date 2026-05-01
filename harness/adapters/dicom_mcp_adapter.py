"""dicom-mcp adapter — DICOM/PACS access via MCP.

Exposes 11 DICOM tools for querying a real PACS node via DIMSE protocol:
list_dicom_nodes, switch_dicom_node, verify_connection, query_patients,
query_studies, query_series, query_instances, move_series, move_study,
get_attribute_presets, extract_pdf_text_from_dicom.

Complements the existing `DicomAdapter` (local DICOM file metadata +
pixel stats via pydicom) by adding **network-attached PACS** access.

Python requirement: dicom-mcp depends on Python>=3.12. We spawn it via
`uvx --python 3.12 ...` so the main venv stays on 3.11.15.

Config: a YAML file listing one or more DICOM nodes (host/port/AE).
For smoke tests we point at a local placeholder (127.0.0.1:4242); live
workflows point at a real Orthanc / dcm4chee instance.

Routing: `harness/adapters/mcp_base.py` handles stdio + JSON-schema
conversion. This module just declares the server spec.
"""

from __future__ import annotations

from pathlib import Path

from harness.adapters.mcp_base import (
    MCPAdapter, MCPServerSpec, register_mcp_server,
)


_REPO = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG = _REPO / "vendors" / "dicom_mcp" / "config.yaml"


def _make_dicom_mcp_spec(config_path: Path = _DEFAULT_CONFIG) -> MCPServerSpec:
    """Build the MCPServerSpec. Exposed as a function so callers can
    override the config path (e.g. to point at a real PACS YAML)."""
    return MCPServerSpec(
        name="dicom_mcp",
        command="uvx",
        args=[
            "--python", "3.12",
            "--from", "dicom-mcp",
            "dicom-mcp",
            str(config_path),
            "--transport", "stdio",
        ],
        env={},
        description=(
            "dicom-mcp — DICOM / PACS network access over MCP. 11 tools for "
            "querying patients, studies, series, and instances from a "
            "configured DICOM node; verifying connectivity; moving data; "
            "and extracting encapsulated PDF reports. Complements the "
            "local-file DicomAdapter (pydicom)."
        ),
        modality="imaging",
        tool_prefix="mcp_dicom_",
        allowed_tools=set(),  # all 11 are worth exposing
    )


DICOM_MCP_SPEC = register_mcp_server(_make_dicom_mcp_spec())


class DicomMCPAdapter(MCPAdapter):
    """Adapter bound to the dicom-mcp PACS server."""

    default_spec = DICOM_MCP_SPEC

    def __init__(self, config: dict | None = None, **kwargs):
        # If caller supplied a custom PACS config path in `config`,
        # rebuild the spec with that path before delegating.
        chosen_spec = DICOM_MCP_SPEC
        if config and config.get("dicom_mcp_config_path"):
            custom_path = Path(config["dicom_mcp_config_path"])
            chosen_spec = _make_dicom_mcp_spec(custom_path)

        super().__init__(
            spec=chosen_spec,
            config=config,
            startup_timeout=kwargs.pop("startup_timeout", 240),
            per_tool_timeout=kwargs.pop("per_tool_timeout", 60.0),
            **kwargs,
        )

    def capabilities(self) -> list[str]:
        return [
            "pacs_query",
            "dicom_patient_query",
            "dicom_study_query",
            "dicom_series_query",
            "dicom_move",
            "dicom_connectivity_check",
            "pdf_from_dicom_extraction",
            "mcp_protocol",
        ]
