"""Smoke tests for protein adapter: ESM-2 embed + AlphaFold-DB lookup.

ESM model load triggers a ~30MB weight download on first run — pytest
timeouts account for this.
"""

from __future__ import annotations

import pytest

pytest.importorskip("esm")


# ======================================================================
# Registry / wiring
# ======================================================================


def test_protein_adapter_in_registry():
    from harness.adapters import ADAPTER_REGISTRY
    assert "ProteinAdapter" in ADAPTER_REGISTRY


def test_alphafold_tool_in_specs():
    from harness.eval.function_calling_runner import TOOL_SPECS
    names = [t["function"]["name"] for t in TOOL_SPECS]
    assert "alphafold_db_lookup" in names


# ======================================================================
# AlphaFold-DB tool (live HTTP)
# ======================================================================


def test_alphafold_lookup_tp53():
    """Human TP53 = UniProt P04637 should have an AlphaFold-DB entry."""
    from harness.tools.protein_tools import _alphafold_db_sync
    r = _alphafold_db_sync("P04637")
    # Tolerant: AF-DB API might change fields, but ok + uniprot_id should be stable
    assert r["ok"] is True, f"AlphaFold-DB lookup failed: {r}"
    assert r["uniprot_id"] == "P04637"
    # TP53 is 393 aa — sequence_length may be None on schema drift but shouldn't error
    assert "pdb_url" in r or "cif_url" in r


def test_alphafold_lookup_invalid():
    from harness.tools.protein_tools import _alphafold_db_sync
    r = _alphafold_db_sync("NOT_A_REAL_ID_99999")
    assert r["ok"] is False


def test_handle_protein_tool():
    from harness.tools.protein_tools import handle_protein_tool
    out = handle_protein_tool("alphafold_db_lookup", {"uniprot_id": "P04637"})
    assert "P04637" in out
    assert "pdb" in out.lower() or "cif" in out.lower()


def test_handle_unknown_protein_tool():
    from harness.tools.protein_tools import handle_protein_tool
    assert "unknown" in handle_protein_tool("not_a_tool", {}).lower()


# ======================================================================
# ESM embed (slow — model load)
# ======================================================================


@pytest.mark.asyncio
async def test_embed_short_peptide():
    """Short peptide → ESM-2 small residue embeddings of expected shape."""
    from harness.adapters.protein_adapter import ProteinAdapter
    adapter = ProteinAdapter()
    assert adapter.available
    # First call downloads model weights (~30MB) — give it time
    result = await adapter.run("embed", context={
        "sequence": "MVLSPADKTNVKAAW",
        "operation": "embed",
    })
    raw = result["raw"]
    assert raw["sequence_length"] == 15
    # esm2_t6_8M_UR50D has 320-dim embeddings; 15 residues -> shape (15, 320)
    assert raw["embedding_shape"] == [15, 320]
    assert result["confidence"] > 0.8


@pytest.mark.asyncio
async def test_embed_missing_sequence():
    from harness.adapters.protein_adapter import ProteinAdapter
    a = ProteinAdapter()
    r = await a.run("embed", context={"operation": "embed"})
    assert r["confidence"] < 0.2
    assert "sequence" in r["answer"].lower()


@pytest.mark.asyncio
async def test_alphafold_through_adapter():
    from harness.adapters.protein_adapter import ProteinAdapter
    a = ProteinAdapter()
    r = await a.run("lookup", context={
        "operation": "alphafold_db", "uniprot_id": "P04637",
    })
    assert r["confidence"] > 0.5
    assert "P04637" in r["answer"]


# ======================================================================
# Configurable ESM size
# ======================================================================


def test_esm_default_size_from_config():
    """Config can override the default ESM size per adapter instance."""
    from harness.adapters.protein_adapter import ProteinAdapter, ESM_MODEL_MAP
    assert set(ESM_MODEL_MAP.keys()) == {"8M", "650M", "3B"}
    a_default = ProteinAdapter()
    assert a_default._default_size == "8M"
    a_sized = ProteinAdapter(config={"protein": {"esm_model_size": "650M"}})
    assert a_sized._default_size == "650M"
    # Invalid size silently falls back to 8M
    a_bad = ProteinAdapter(config={"protein": {"esm_model_size": "nonsense"}})
    assert a_bad._default_size == "8M"


def test_resolve_size_override():
    """Per-call context override wins over adapter default."""
    from harness.adapters.protein_adapter import ProteinAdapter
    a = ProteinAdapter(config={"protein": {"esm_model_size": "8M"}})
    assert a._resolve_size(None) == "8M"
    assert a._resolve_size("650M") == "650M"
    assert a._resolve_size("not_a_size") == "8M"  # invalid override ignored


@pytest.mark.asyncio
async def test_embed_reports_size_in_evidence():
    """Embed result surfaces both the model name and the size alias."""
    from harness.adapters.protein_adapter import ProteinAdapter
    a = ProteinAdapter()
    r = await a.run("embed", context={
        "sequence": "MVLSPAD", "operation": "embed",
    })
    raw = r["raw"]
    assert raw["size"] == "8M"
    assert raw["model"] == "esm2_t6_8M_UR50D"
    assert any("size=8M" in e for e in r["evidence"])
