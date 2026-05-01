"""Smoke tests for rdkit + datamol chemistry integration."""

from __future__ import annotations

import pytest


# Skip entire module if rdkit isn't installed
pytest.importorskip("rdkit")


# ======================================================================
# Direct tool functions
# ======================================================================


def test_canonicalise_aspirin():
    from harness.tools.chemistry_tools import _canonicalise_sync
    canonical = _canonicalise_sync("CC(=O)Oc1ccccc1C(=O)O")
    assert canonical is not None
    # RDKit canonical form for aspirin
    assert "O=C" in canonical or "C(=O)" in canonical


def test_canonicalise_invalid():
    from harness.tools.chemistry_tools import _canonicalise_sync
    assert _canonicalise_sync("not a molecule") is None


def test_descriptors_aspirin():
    """Aspirin: MW 180.16, LogP ~1.19."""
    from harness.tools.chemistry_tools import _descriptors_sync
    d = _descriptors_sync("CC(=O)Oc1ccccc1C(=O)O")
    assert 179.0 < d["MW"] < 181.0, f"Unexpected MW: {d['MW']}"
    assert 1.0 < d["LogP"] < 1.4, f"Unexpected LogP: {d['LogP']}"
    assert d["HBA"] >= 3  # 3-4 H-bond acceptors
    assert d["HBD"] == 1  # one -OH
    assert d["lipinski_pass"] is True
    assert d["lipinski_violations"] == 0


def test_fingerprint_morgan_has_expected_size():
    from harness.tools.chemistry_tools import _fingerprint_sync
    bits = _fingerprint_sync("CC(=O)Oc1ccccc1C(=O)O", fp_type="morgan", n_bits=2048)
    assert len(bits) == 2048
    assert sum(bits) > 0


def test_similarity_self_identity():
    from harness.tools.chemistry_tools import _similarity_sync
    sim = _similarity_sync(
        "CC(=O)Oc1ccccc1C(=O)O",
        "CC(=O)Oc1ccccc1C(=O)O",
    )
    assert sim == 1.0


def test_similarity_different():
    """Aspirin vs caffeine — should be dissimilar."""
    from harness.tools.chemistry_tools import _similarity_sync
    sim = _similarity_sync(
        "CC(=O)Oc1ccccc1C(=O)O",
        "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    )
    assert 0.0 <= sim < 0.5  # known to be quite different


def test_substructure_match_benzene():
    """Aspirin contains a benzene ring."""
    from harness.tools.chemistry_tools import _substructure_match_sync
    r = _substructure_match_sync("c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O")
    assert r["match"] is True
    assert r["num_matches"] >= 1


# ======================================================================
# Dispatcher (TOOL_SPECS route)
# ======================================================================


def test_handle_mol_descriptors():
    from harness.tools.chemistry_tools import handle_mol_tool
    result = handle_mol_tool("mol_descriptors", {"smiles": "CC(=O)Oc1ccccc1C(=O)O"})
    assert "MW=" in result
    assert "lipinski_pass=True" in result


def test_handle_mol_similarity():
    from harness.tools.chemistry_tools import handle_mol_tool
    result = handle_mol_tool("mol_similarity", {
        "smiles_a": "CC(=O)Oc1ccccc1C(=O)O",
        "smiles_b": "CC(=O)Oc1ccccc1C(=O)O",
    })
    assert "tanimoto=1.0" in result


def test_handle_unknown_tool():
    from harness.tools.chemistry_tools import handle_mol_tool
    result = handle_mol_tool("not_a_real_tool", {})
    assert "unknown chemistry tool" in result


# ======================================================================
# Adapter integration
# ======================================================================


@pytest.mark.asyncio
async def test_chemistry_adapter_single():
    from harness.adapters.chemistry_adapter import ChemistryAdapter
    adapter = ChemistryAdapter()
    assert adapter.available

    result = await adapter.run(
        "Analyse aspirin drug-likeness",
        context={"smiles": "CC(=O)Oc1ccccc1C(=O)O"},
    )
    assert result["confidence"] > 0.9
    assert "MW=" in result["answer"]
    assert "Lipinski=PASS" in result["answer"]


@pytest.mark.asyncio
async def test_chemistry_adapter_batch():
    from harness.adapters.chemistry_adapter import ChemistryAdapter
    adapter = ChemistryAdapter()
    result = await adapter.run(
        "Analyse aspirin and caffeine",
        context={"smiles_list": [
            "CC(=O)Oc1ccccc1C(=O)O",                       # aspirin
            "Cn1cnc2c1c(=O)n(C)c(=O)n2C",                  # caffeine
        ]},
    )
    lines = result["answer"].split("\n")
    assert len(lines) == 2


@pytest.mark.asyncio
async def test_chemistry_adapter_missing_smiles():
    from harness.adapters.chemistry_adapter import ChemistryAdapter
    adapter = ChemistryAdapter()
    result = await adapter.run("no smiles given", context={})
    assert result["confidence"] < 0.2
    assert "needs" in result["answer"].lower() or "context" in result["answer"].lower()


# ======================================================================
# Registry check
# ======================================================================


def test_chemistry_adapter_in_registry():
    from harness.adapters import ADAPTER_REGISTRY
    assert "ChemistryAdapter" in ADAPTER_REGISTRY


def test_chemistry_tools_in_tool_specs():
    from harness.eval.function_calling_runner import TOOL_SPECS
    tool_names = [t["function"]["name"] for t in TOOL_SPECS]
    for expected in ["mol_from_smiles", "mol_descriptors", "mol_fingerprint",
                      "mol_similarity", "mol_substructure_match"]:
        assert expected in tool_names, f"TOOL_SPECS missing {expected}"
