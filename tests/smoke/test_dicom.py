"""Smoke tests for pydicom + highdicom.

Generates a tiny synthetic DICOM file in a temp dir to exercise the
whole metadata + pixel-stats + series-organize path without needing
real imaging data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pydicom")
pytest.importorskip("highdicom")


@pytest.fixture
def synthetic_dicom(tmp_path: Path) -> Path:
    """Create a minimal DICOM Secondary Capture file."""
    import numpy as np
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import (
        ExplicitVRLittleEndian,
        SecondaryCaptureImageStorage,
        generate_uid,
    )

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "OT"
    ds.Manufacturer = "BioMedArena-Test"
    ds.PatientName = "Test^Patient"
    ds.PatientID = "P0001"
    ds.PatientBirthDate = "19700101"
    ds.PatientSex = "O"
    ds.StudyDate = "20260416"
    ds.SeriesDescription = "Synthetic smoke test"

    # 16x16 grayscale image
    arr = np.arange(256, dtype=np.uint16).reshape(16, 16) * 100
    ds.Rows, ds.Columns = 16, 16
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.PixelData = arr.tobytes()

    out = tmp_path / "synthetic.dcm"
    pydicom.dcmwrite(str(out), ds, write_like_original=False)
    return out


# ======================================================================


def test_read_metadata(synthetic_dicom):
    from harness.adapters.dicom_adapter import _read_metadata_sync
    m = _read_metadata_sync(str(synthetic_dicom))
    assert m["patient_id"] == "P0001"
    assert m["modality"] == "OT"
    assert str(m["rows"]) == "16"
    assert str(m["columns"]) == "16"
    assert m["photometric"] == "MONOCHROME2"


def test_pixel_stats(synthetic_dicom):
    from harness.adapters.dicom_adapter import _read_pixel_stats_sync
    s = _read_pixel_stats_sync(str(synthetic_dicom))
    assert s["shape"] == [16, 16]
    assert s["min"] == 0.0
    assert s["max"] == 25500.0  # 255 * 100


def test_series_organize(synthetic_dicom, tmp_path):
    from harness.adapters.dicom_adapter import _series_organize_sync
    r = _series_organize_sync(str(tmp_path))
    assert r["num_series"] == 1
    assert r["num_files"] == 1


def test_dicom_tools_in_specs():
    from harness.eval.function_calling_runner import TOOL_SPECS
    names = [t["function"]["name"] for t in TOOL_SPECS]
    assert "read_dicom_metadata" in names
    assert "dicom_pixel_stats" in names


def test_registry_has_dicom():
    from harness.adapters import ADAPTER_REGISTRY
    assert "DicomAdapter" in ADAPTER_REGISTRY


def test_handle_dicom_tool(synthetic_dicom):
    from harness.tools.dicom_tools import handle_dicom_tool
    out = handle_dicom_tool("read_dicom_metadata", {"path": str(synthetic_dicom)})
    assert "P0001" in out
    assert "MONOCHROME2" in out


def test_dicom_tool_unknown():
    from harness.tools.dicom_tools import handle_dicom_tool
    out = handle_dicom_tool("not_a_dicom_tool", {})
    assert "unknown" in out.lower()


@pytest.mark.asyncio
async def test_adapter_run_metadata(synthetic_dicom):
    from harness.adapters.dicom_adapter import DicomAdapter
    a = DicomAdapter()
    assert a.available
    r = await a.run("read this", context={
        "dicom_path": str(synthetic_dicom),
        "operation": "metadata",
    })
    assert r["confidence"] > 0.8
    assert "P0001" in r["answer"]


@pytest.mark.asyncio
async def test_adapter_missing_path():
    from harness.adapters.dicom_adapter import DicomAdapter
    a = DicomAdapter()
    r = await a.run("x", context={})
    assert r["confidence"] < 0.2
    assert "dicom_path" in r["answer"].lower()
