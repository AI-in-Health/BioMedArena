"""DICOM adapter — pydicom + highdicom imaging metadata reader.

Routing: adapter (heavy file I/O + pixel array).

Operations supported via context['operation']:
    metadata     (default) — read DICOM header, return patient/study/series dict
    stats                   — pixel-array summary stats (min/max/mean/shape)
    series_organize         — scan a directory tree and group files into series
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from harness.adapter_base import AdapterBase


class DicomAdapter(AdapterBase):
    name = "dicom"
    modality = "imaging"
    description = (
        "DICOM file metadata and pixel-array reader. Operations: metadata "
        "(default), stats (pixel array summary), series_organize (group "
        "files by SeriesInstanceUID). Requires context['dicom_path'] "
        "pointing at a .dcm file or a directory."
    )

    def __init__(self, config: dict | None = None, **kwargs: Any):
        self._config = config or {}
        self._llm = kwargs.get("llm")
        try:
            import pydicom  # noqa: F401
            self._ok = True
        except ImportError:
            self._ok = False
            self.mark_unavailable("pydicom not installed")

    def capabilities(self) -> list[str]:
        return [
            "dicom_read",
            "dicom_metadata",
            "pixel_array_stats",
            "series_grouping",
            "radiology_imaging",
        ]

    async def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._ok:
            return self.result(answer=self.unavailable_reason, confidence=0.0)

        ctx = context or {}
        path = ctx.get("dicom_path")
        op = ctx.get("operation", "metadata")
        if not path:
            return self.result(
                answer="DicomAdapter needs context['dicom_path'] (.dcm file or directory).",
                confidence=0.1,
            )

        p = Path(path)
        if not p.exists():
            return self.result(
                answer=f"DICOM path does not exist: {path}",
                confidence=0.0,
            )

        try:
            if op == "metadata":
                data = await asyncio.to_thread(_read_metadata_sync, str(p))
            elif op == "stats":
                data = await asyncio.to_thread(_read_pixel_stats_sync, str(p))
            elif op == "series_organize":
                data = await asyncio.to_thread(_series_organize_sync, str(p))
            else:
                return self.result(
                    answer=f"Unknown operation: {op}. Supported: metadata, stats, series_organize",
                    confidence=0.1,
                )
        except Exception as exc:
            return self.result(answer=f"DICOM error: {type(exc).__name__}: {exc}", confidence=0.0)

        summary_lines = [f"{k}: {v}" for k, v in data.items() if not k.startswith("_")]
        return self.result(
            answer="\n".join(summary_lines),
            evidence=[f"DICOM {op} on {p.name}"],
            confidence=0.9,
            raw=data,
        )


# ======================================================================
# Sync workers (pydicom is synchronous)
# ======================================================================


def _read_metadata_sync(path: str) -> dict[str, Any]:
    """Read a DICOM file's header and return the key identifiers."""
    import pydicom
    ds = pydicom.dcmread(path, stop_before_pixels=True)

    def _get(attr: str, default: Any = None) -> Any:
        v = getattr(ds, attr, default)
        return str(v) if v is not None else None

    return {
        "modality": _get("Modality"),
        "manufacturer": _get("Manufacturer"),
        "patient_id": _get("PatientID"),
        "patient_name": _get("PatientName"),
        "patient_birthdate": _get("PatientBirthDate"),
        "patient_sex": _get("PatientSex"),
        "study_date": _get("StudyDate"),
        "study_uid": _get("StudyInstanceUID"),
        "series_uid": _get("SeriesInstanceUID"),
        "series_description": _get("SeriesDescription"),
        "rows": _get("Rows"),
        "columns": _get("Columns"),
        "bits_allocated": _get("BitsAllocated"),
        "photometric": _get("PhotometricInterpretation"),
    }


def _read_pixel_stats_sync(path: str) -> dict[str, Any]:
    """Load pixel array and return summary stats (no values)."""
    import numpy as np
    import pydicom
    ds = pydicom.dcmread(path)
    if not hasattr(ds, "pixel_array"):
        return {"error": "no pixel data"}
    arr = ds.pixel_array
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": round(float(np.mean(arr)), 3),
        "std": round(float(np.std(arr)), 3),
    }


def _series_organize_sync(dir_path: str) -> dict[str, Any]:
    """Walk a directory of DICOM files and group by SeriesInstanceUID."""
    import pydicom
    root = Path(dir_path)
    if not root.is_dir():
        # Treat a single file as one series
        meta = _read_metadata_sync(str(root))
        return {
            "num_series": 1,
            "num_files": 1,
            "series": [{
                "series_uid": meta.get("series_uid"),
                "modality": meta.get("modality"),
                "description": meta.get("series_description"),
                "files": [root.name],
            }],
        }

    series_map: dict[str, list[str]] = {}
    series_meta: dict[str, dict[str, Any]] = {}
    total = 0
    for f in root.rglob("*"):
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            uid = str(getattr(ds, "SeriesInstanceUID", "unknown"))
            series_map.setdefault(uid, []).append(str(f.relative_to(root)))
            if uid not in series_meta:
                series_meta[uid] = {
                    "modality": str(getattr(ds, "Modality", "?")),
                    "description": str(getattr(ds, "SeriesDescription", "")),
                }
            total += 1
        except Exception:
            # Not a DICOM file — skip
            continue

    return {
        "num_series": len(series_map),
        "num_files": total,
        "series": [
            {
                "series_uid": uid,
                "modality": series_meta[uid]["modality"],
                "description": series_meta[uid]["description"],
                "num_files": len(files),
                "files_preview": files[:3],
            }
            for uid, files in series_map.items()
        ],
    }
