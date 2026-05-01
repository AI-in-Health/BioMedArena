"""Smoke tests for molfeat + scvi-tools + MONAI libraries."""

from __future__ import annotations

import pytest


pytest.importorskip("rdkit")


def test_molfeat_tool_registered():
    from harness.eval.function_calling_runner import TOOL_SPECS
    names = {t["function"]["name"] for t in TOOL_SPECS}
    assert "molfeat_featurize" in names


def test_monai_tools_registered():
    from harness.eval.function_calling_runner import TOOL_SPECS
    names = {t["function"]["name"] for t in TOOL_SPECS}
    assert "medical_image_metadata" in names
    assert "medical_image_normalize" in names


def test_scvi_importable():
    """scvi-tools is installed; no TOOL_SPECS (training-heavy) but
    coverage claim requires the import to succeed."""
    import scvi
    assert hasattr(scvi, "__version__")
    # SCVI model class should be importable
    from scvi.model import SCVI  # noqa: F401


@pytest.mark.slow
def test_molfeat_ecfp_aspirin():
    """Compute Morgan fingerprint for aspirin via molfeat."""
    from harness.tools.molfeat_tools import _featurize_sync
    r = _featurize_sync("CC(=O)Oc1ccccc1C(=O)O", featurizer="ecfp")
    assert r["dim"] == 2048  # default Morgan length
    assert r["nnz"] > 5  # sparse but non-trivial
    assert len(r["first10"]) == 10


@pytest.mark.slow
def test_molfeat_maccs():
    from harness.tools.molfeat_tools import _featurize_sync
    r = _featurize_sync("CCO", featurizer="maccs")
    assert r["dim"] >= 160  # MACCS is 166 bits
    assert r["dim"] <= 170


def test_handle_molfeat_unknown():
    from harness.tools.molfeat_tools import handle_molfeat_tool
    r = handle_molfeat_tool("not_a_tool", {"smiles": "CCO"})
    assert "unknown" in r.lower()


@pytest.mark.slow
def test_monai_metadata_png(tmp_path):
    """Write a tiny PNG and load it via MONAI."""
    import numpy as np
    from PIL import Image
    arr = (np.random.rand(32, 32) * 255).astype(np.uint8)
    p = tmp_path / "dummy.png"
    Image.fromarray(arr).save(str(p))
    from harness.tools.monai_tools import _metadata_sync
    r = _metadata_sync(str(p))
    assert r["shape"][0] == 32
    assert r["dtype"] is not None
    assert r["min"] <= r["max"]


def test_handle_monai_unknown():
    from harness.tools.monai_tools import handle_monai_tool
    r = handle_monai_tool("not_a_monai_tool", {"path": "/tmp/x"})
    assert "unknown" in r.lower()
