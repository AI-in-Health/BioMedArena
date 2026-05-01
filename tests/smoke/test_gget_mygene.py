"""Smoke tests for gget + mygene.

Most tests are live HTTP calls — marked `live` for CI to optionally skip.
Registry + TOOL_SPECS checks run offline.
"""

from __future__ import annotations

import pytest

pytest.importorskip("gget")
pytest.importorskip("mygene")


# ======================================================================
# Registry / wiring (offline)
# ======================================================================


def test_gget_tools_in_specs():
    from harness.eval.function_calling_runner import TOOL_SPECS
    names = [t["function"]["name"] for t in TOOL_SPECS]
    for expected in ["gget_search", "gget_info", "gget_seq", "mygene_query"]:
        assert expected in names


# ======================================================================
# mygene live calls
# ======================================================================


def test_mygene_query_brca1():
    """BRCA1 should have Entrez gene ID 672."""
    from harness.tools.gget_tools import _mygene_query_sync
    result = _mygene_query_sync("BRCA1", species="human",
                                  fields=["symbol", "name", "entrezgene"])
    assert result["ok"] is True
    assert len(result["hits"]) >= 1
    hit = result["hits"][0]
    # Entrez gene ID for BRCA1 is 672
    assert str(hit.get("entrezgene")) == "672"
    assert hit.get("symbol") == "BRCA1"


def test_handle_mygene_tool():
    from harness.tools.gget_tools import handle_gget_tool
    out = handle_gget_tool("mygene_query", {"query": "TP53", "species": "human", "size": 1})
    assert "TP53" in out
    # Entrez gene ID for TP53 is 7157
    assert "7157" in out


# ======================================================================
# gget live calls
# ======================================================================


def test_gget_search_brca1():
    """gget.search for BRCA1 should return Ensembl IDs containing BRCA1."""
    from harness.tools.gget_tools import _gget_search_sync
    result = _gget_search_sync("BRCA1", species="human", limit=3)
    assert result["ok"] is True
    assert len(result["matches"]) >= 1
    # Known: BRCA1 Ensembl stable ID starts with ENSG00000012048
    any_brca1 = any(
        "ENSG00000012048" in str(m.get("ensembl_id") or m)
        or "BRCA1" in str(m.get("gene_name") or m.get("symbol") or m)
        for m in result["matches"]
    )
    assert any_brca1, f"No BRCA1 hit in matches: {result['matches']}"


def test_handle_gget_search():
    from harness.tools.gget_tools import handle_gget_tool
    out = handle_gget_tool("gget_search", {"query": "TP53", "species": "human", "limit": 2})
    assert "ok" in out
    # TP53's Ensembl ID
    assert "ENSG00000141510" in out or "TP53" in out


# ======================================================================
# Dispatcher (unknown tool)
# ======================================================================


def test_unknown_tool():
    from harness.tools.gget_tools import handle_gget_tool
    result = handle_gget_tool("not_a_real_gget_tool", {})
    assert "unknown" in result.lower()
