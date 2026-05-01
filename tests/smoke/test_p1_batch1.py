"""Smoke tests for biopython + lifelines + fhir libraries.

Fast tests only — no network required. Live FHIR calls are covered
by a separate @slow test that skips unless FHIR_TEST_BASE_URL is set.
"""

from __future__ import annotations

import os
import pytest


# ======================================================================
# TOOL_SPECS registration
# ======================================================================


def test_all_p1_tools_registered():
    from harness.eval.function_calling_runner import TOOL_SPECS
    names = {t["function"]["name"] for t in TOOL_SPECS}
    expected = {
        "fasta_parse", "dna_reverse_complement", "dna_translate", "genbank_parse",
        "kaplan_meier_fit", "logrank_two_sample",
    }
    assert expected <= names, f"missing: {expected - names}"


# ======================================================================
# biopython
# ======================================================================


def test_fasta_parse():
    pytest.importorskip("Bio")
    from harness.tools.biopython_tools import _fasta_parse_sync
    fasta = ">seq1 toy\nACGTACGT\n>seq2 other\nGGGGCCCC\n"
    out = _fasta_parse_sync(fasta)
    assert len(out) == 2
    assert out[0]["id"] == "seq1"
    assert out[0]["sequence"] == "ACGTACGT"
    assert out[1]["length"] == 8


def test_reverse_complement():
    pytest.importorskip("Bio")
    from harness.tools.biopython_tools import _reverse_complement_sync
    assert _reverse_complement_sync("ACGT") == "ACGT"  # palindrome
    assert _reverse_complement_sync("AATG") == "CATT"


def test_translate():
    pytest.importorskip("Bio")
    from harness.tools.biopython_tools import _translate_sync
    # ATG AAA TAA → M K (stop)
    out = _translate_sync("ATGAAATAA", to_stop=True)
    assert out == "MK"


def test_translate_with_frame():
    pytest.importorskip("Bio")
    from harness.tools.biopython_tools import _translate_sync
    # Same seq, frame 1 → different protein
    out0 = _translate_sync("ATGAAATAA", frame=0, to_stop=True)
    out1 = _translate_sync("ATGAAATAA", frame=1, to_stop=True)
    assert out0 != out1


def test_handle_biopython_tool_dispatch():
    pytest.importorskip("Bio")
    from harness.tools.biopython_tools import handle_biopython_tool
    r = handle_biopython_tool("dna_reverse_complement", {"sequence": "AATG"})
    assert r == "CATT"
    r2 = handle_biopython_tool("unknown_tool", {})
    assert "unknown" in r2.lower()


# ======================================================================
# lifelines — survival
# ======================================================================


def test_kaplan_meier_fit():
    pytest.importorskip("lifelines")
    from harness.tools.survival_tools import _km_fit_sync
    durations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    events = [1, 1, 0, 1, 1, 0, 1, 0, 1, 1]
    out = _km_fit_sync(durations, events, summary_times=[2, 5, 8])
    assert len(out["survival_probability"]) == 3
    assert 0.0 <= out["survival_probability"][0] <= 1.0
    assert out["survival_probability"][0] >= out["survival_probability"][-1]


def test_logrank_two_sample():
    pytest.importorskip("lifelines")
    from harness.tools.survival_tools import _logrank_sync
    # Two clearly different groups
    out = _logrank_sync(
        [1, 2, 3, 4, 5], [1, 1, 1, 1, 1],     # group A — all die fast
        [20, 22, 25, 30, 40], [1, 1, 1, 1, 0],  # group B — live long
    )
    assert out["p_value"] < 0.05
    assert out["significant_at_0p05"] is True


def test_handle_survival_tool_dispatch():
    pytest.importorskip("lifelines")
    from harness.tools.survival_tools import handle_survival_tool
    r = handle_survival_tool(
        "kaplan_meier_fit",
        {"durations": [1, 2, 3, 4, 5], "events": [1, 1, 0, 1, 1]},
    )
    assert "median=" in r


# ======================================================================
# FHIR adapter
# ======================================================================


def test_fhir_adapter_registered():
    from harness.adapters import ADAPTER_REGISTRY
    assert "FHIRAdapter" in ADAPTER_REGISTRY


def test_fhir_adapter_basic():
    pytest.importorskip("fhirpy")
    pytest.importorskip("fhir.resources")
    from harness.adapters.fhir_adapter import FHIRAdapter
    a = FHIRAdapter()
    # deps installed → adapter should be available regardless of server
    assert a.available is True
    assert a.modality == "ehr"
    caps = a.capabilities()
    assert "fhir_search" in caps
    assert "patient_summary" in caps


def test_fhir_adapter_custom_base_url():
    pytest.importorskip("fhirpy")
    pytest.importorskip("fhir.resources")
    from harness.adapters.fhir_adapter import FHIRAdapter
    a = FHIRAdapter(config={"fhir": {"base_url": "https://example.com/fhir",
                                       "token": "abc123"}})
    assert a._base_url == "https://example.com/fhir"
    assert a._token == "abc123"


def test_fhir_client_cached():
    """Multiple calls with same base_url reuse the client instance."""
    pytest.importorskip("fhirpy")
    pytest.importorskip("fhir.resources")
    from harness.adapters.fhir_adapter import FHIRAdapter
    a = FHIRAdapter()
    c1 = a._get_client()
    c2 = a._get_client()
    assert c1 is c2


# ======================================================================
# Live FHIR round-trip (opt-in, slow)
# ======================================================================


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("FHIR_TEST_BASE_URL"),
    reason="FHIR_TEST_BASE_URL not set",
)
def test_fhir_live_search():
    from harness.adapters.fhir_adapter import FHIRAdapter
    a = FHIRAdapter(config={"fhir": {
        "base_url": os.environ["FHIR_TEST_BASE_URL"],
    }})
    records = a.search("Patient", _count=2)
    assert isinstance(records, list)
