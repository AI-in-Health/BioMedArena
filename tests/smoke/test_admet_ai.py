"""Smoke tests for ADMET-AI native prediction.

admet-ai wraps an ensemble of chemprop graph-neural-net models trained on
Therapeutics Data Commons. First prediction pays the cold-start cost
(~0.5-1s for model load, ~0.3s per SMILES after). Subsequent predictions
reuse the cached class-level `ChemistryAdapter._admet_model`.
"""

from __future__ import annotations

import pytest

pytest.importorskip("admet_ai")
pytest.importorskip("rdkit")


# ======================================================================
# Wiring
# ======================================================================


def test_admet_tools_in_specs():
    from harness.eval.function_calling_runner import TOOL_SPECS
    names = [t["function"]["name"] for t in TOOL_SPECS]
    assert "admet_predict_native" in names
    assert "molecular_property_predict" in names


def test_chemistry_adapter_admet_capabilities():
    from harness.adapters.chemistry_adapter import ChemistryAdapter
    a = ChemistryAdapter()
    caps = a.capabilities()
    assert "admet_prediction_native" in caps
    assert "molecular_property_prediction" in caps


# ======================================================================
# Live inference (slow — loads chemprop ensemble)
# ======================================================================


@pytest.mark.slow
def test_admet_predict_aspirin_via_adapter():
    """End-to-end: aspirin SMILES → predictions dict with known keys."""
    from harness.adapters.chemistry_adapter import ChemistryAdapter
    a = ChemistryAdapter()
    assert a._admet_ok
    resp = a.admet_predict_native("CC(=O)Oc1ccccc1C(=O)O")
    assert resp["ok"] is True, f"predict failed: {resp}"
    preds = resp["predictions"]
    assert isinstance(preds, dict)
    # Expected physicochemical keys (stable across admet_ai versions)
    for key in ["molecular_weight", "logP", "QED", "Lipinski"]:
        assert key in preds, f"{key} missing from predictions"
    # Aspirin is a small molecule, MW should be < 200
    assert 100 < preds["molecular_weight"] < 250, (
        f"MW {preds['molecular_weight']} not plausible for aspirin"
    )


@pytest.mark.slow
def test_admet_model_cached_class_level():
    """Two adapter instances must share the same cached ADMETModel."""
    from harness.adapters.chemistry_adapter import ChemistryAdapter
    a = ChemistryAdapter()
    b = ChemistryAdapter()
    m1 = a._get_admet_model()
    m2 = b._get_admet_model()
    assert m1 is m2


@pytest.mark.slow
def test_handle_admet_tool_predict():
    """Native tool dispatch returns a compact summary string."""
    from harness.tools.chemistry_tools import handle_admet_tool
    out = handle_admet_tool(
        "admet_predict_native", {"smiles": "CC(=O)Oc1ccccc1C(=O)O"}
    )
    # Should contain at least one known property with a value
    assert "molecular_weight=" in out or "logP=" in out
    assert "error" not in out.lower() or "error=" not in out.lower()


@pytest.mark.slow
def test_handle_admet_tool_specific_property():
    from harness.tools.chemistry_tools import handle_admet_tool
    out = handle_admet_tool(
        "molecular_property_predict",
        {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "property_name": "logP"},
    )
    # Either the property was found ("logP=value") or we got a friendly
    # fallback listing available properties (stable API contract)
    assert "logP" in out


def test_handle_admet_unknown_tool():
    """Unknown tool name returns a clean error string, no exception."""
    from harness.tools.chemistry_tools import handle_admet_tool
    out = handle_admet_tool("not_a_real_admet_tool", {"smiles": "CCO"})
    assert "unknown" in out.lower()
