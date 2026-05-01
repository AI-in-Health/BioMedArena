"""DICOM TOOL_SPECS — stateless metadata + pixel stats readers.

Deliberately kept thin (no full image pipelines); the adapter
`harness.adapters.dicom_adapter.DicomAdapter` handles heavier ops.
"""

from __future__ import annotations

import json as _json
from typing import Any


def _tool_read_metadata(path: str) -> str:
    from harness.adapters.dicom_adapter import _read_metadata_sync
    return _json.dumps(_read_metadata_sync(path), default=str)[:8000]


def _tool_pixel_stats(path: str) -> str:
    from harness.adapters.dicom_adapter import _read_pixel_stats_sync
    return _json.dumps(_read_pixel_stats_sync(path), default=str)[:4000]


DICOM_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_dicom_metadata",
            "description": (
                "Read a DICOM file's header (no pixel data): modality, patient ID, "
                "study/series UIDs, dimensions, photometric interpretation. Use to "
                "answer questions about a specific DICOM image's metadata."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to .dcm file"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dicom_pixel_stats",
            "description": (
                "Load a DICOM file's pixel array and return summary stats (shape, "
                "dtype, min, max, mean, std). Useful as a sanity check before more "
                "expensive imaging analysis. Slower than read_dicom_metadata."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                },
                "required": ["path"],
            },
        },
    },
]


DICOM_TOOL_NAMES = {"read_dicom_metadata", "dicom_pixel_stats"}


def handle_dicom_tool(name: str, args: dict[str, Any]) -> str:
    try:
        if name == "read_dicom_metadata":
            return _tool_read_metadata(args["path"])
        if name == "dicom_pixel_stats":
            return _tool_pixel_stats(args["path"])
        return f"[unknown DICOM tool: {name}]"
    except Exception as exc:
        return f"[{name} error: {exc}]"
