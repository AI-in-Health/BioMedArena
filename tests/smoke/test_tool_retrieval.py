"""Smoke tests for the Tool Retrieval subsystem.

Tests use the deterministic hashing-trigram embedder so they run
offline without hitting the OpenAI API. The separate (opt-in) live
test asserts that OpenAI embeddings are used when a key is present.
"""

from __future__ import annotations

import os
import pickle
import time

import numpy as np
import pytest

# Ensure .env is loaded before evaluating skipif decorators
import harness  # noqa: F401  (triggers _load_env_once)


# ======================================================================
# Module imports
# ======================================================================


def test_context_module_imports():
    from harness.context import (
        ToolRetriever, CORE_TOOL_NAMES, score_query_domain,
        TokenBudgetTracker, MODE_BUDGETS, BudgetExceededError,
    )
    assert "pubmed_search" in CORE_TOOL_NAMES
    assert "calculator_eval" in CORE_TOOL_NAMES
    assert "python_exec" in CORE_TOOL_NAMES
    assert "compute_calculator" in CORE_TOOL_NAMES
    assert "code_search" in CORE_TOOL_NAMES
    assert "gene_lookup" in CORE_TOOL_NAMES
    assert "clinvar_lookup" in CORE_TOOL_NAMES
    assert len(CORE_TOOL_NAMES) == 7
    assert "light" in MODE_BUDGETS


# ======================================================================
# Domain heuristic
# ======================================================================


def test_score_query_domain_chemistry():
    from harness.context import score_query_domain
    w = score_query_domain("Is the SMILES CC(=O)O a valid molecule with good ADMET?")
    assert "chemistry" in w
    assert w["chemistry"] > 0


def test_score_query_domain_genomics():
    from harness.context import score_query_domain
    w = score_query_domain("What is the clinical significance of the BRCA1 variant?")
    assert "genomics" in w
    assert w["genomics"] > 0


def test_score_query_domain_imaging():
    from harness.context import score_query_domain
    w = score_query_domain("Fetch the DICOM series from the PACS for this patient.")
    assert "imaging" in w


def test_score_query_domain_multidomain():
    from harness.context import score_query_domain
    w = score_query_domain("Find the DICOM CT scan for a lung cancer patient with EGFR mutation.")
    # Two domains should fire
    assert len(w) >= 2
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-5)


def test_score_query_domain_no_match():
    from harness.context import score_query_domain
    assert score_query_domain("What is the capital of France?") == {}


# ======================================================================
# Retriever — offline (hashing trigram) path
# ======================================================================


def _dummy_specs() -> list[dict]:
    """Use the real TOOL_SPECS from FunctionCallingRunner so the test
    exercises the actual production tool pool."""
    from harness.eval.function_calling_runner import TOOL_SPECS
    return list(TOOL_SPECS)


@pytest.fixture
def retriever(tmp_path):
    """Retriever with the full production tool pool + isolated cache."""
    from harness.context.tool_retrieval import ToolRetriever, _hashing_trigram_embed
    return ToolRetriever(
        tools=_dummy_specs(),
        embed_fn=_hashing_trigram_embed,
        cache_path=tmp_path / "emb_cache.pkl",
    )


def test_retrieve_always_includes_core_tools(retriever):
    from harness.context import CORE_TOOL_NAMES
    picked = retriever.retrieve("Random query", top_k=5)
    names = {t["function"]["name"] for t in picked}
    pool_names = {t["function"]["name"] for t in _dummy_specs()}
    for core in CORE_TOOL_NAMES:
        if core in pool_names:
            assert core in names, f"core tool {core} missing from retrieval"


def test_retrieve_chemistry_query(retriever):
    picked = retriever.retrieve(
        "Calculate the Lipinski properties of this SMILES: CC(=O)Oc1ccccc1C(=O)O",
        top_k=15,
    )
    names = [t["function"]["name"] for t in picked]
    # At least one chem-related tool beyond the core
    chem = [
        n for n in names if (n.startswith(("mol_", "admet_", "molecular_", "tdc_")))
    ]
    assert chem, f"no chemistry tool in top-15 for chemistry query: {names}"


def test_retrieve_genomics_query(retriever):
    picked = retriever.retrieve(
        "Look up the gene BRCA1 and its variants in ClinVar.",
        top_k=15,
    )
    names = [t["function"]["name"] for t in picked]
    # gene_lookup / clinvar_lookup are core (always present), but we
    # should also see at least one non-core genomics tool like
    # omim_lookup, orphanet_lookup or gget_*
    non_core_genomics = [
        n for n in names
        if any(n.startswith(p) for p in ("omim_", "gget_", "mygene_", "orphanet_"))
    ]
    assert non_core_genomics or any(n.startswith("mcp_") for n in names), (
        f"no non-core genomics tool in top-15: {names}"
    )


def test_retrieve_imaging_query(retriever):
    picked = retriever.retrieve(
        "Open the DICOM series and get the pixel statistics.",
        top_k=15,
    )
    names = [t["function"]["name"] for t in picked]
    dicom_tools = [n for n in names if "dicom" in n.lower()]
    assert dicom_tools, f"no DICOM tool in top-15 for imaging query: {names}"


def test_retrieve_size_bounded(retriever):
    """Final output length ≤ (|core in pool|) + top_k."""
    from harness.context import CORE_TOOL_NAMES
    top_k = 8
    picked = retriever.retrieve("What genes are on chromosome 17?", top_k=top_k)
    pool_names = {t["function"]["name"] for t in _dummy_specs()}
    core_in_pool = len(CORE_TOOL_NAMES & pool_names)
    assert len(picked) <= core_in_pool + top_k


def test_retrieve_deduplicates(retriever):
    """No duplicate tool specs in the output list."""
    picked = retriever.retrieve("drug interactions for warfarin", top_k=10)
    names = [t["function"]["name"] for t in picked]
    assert len(names) == len(set(names))


# ======================================================================
# Retriever cache
# ======================================================================


def test_retrieval_cache_persists(tmp_path):
    """Second call with the same tools must be fast (cache hit)."""
    from harness.context.tool_retrieval import ToolRetriever, _hashing_trigram_embed
    cache = tmp_path / "cache.pkl"
    # Wrap the embedder to count calls
    call_counter = {"n": 0}

    def _embed(texts):
        call_counter["n"] += len(texts)
        return _hashing_trigram_embed(texts)

    r1 = ToolRetriever(_dummy_specs(), embed_fn=_embed, cache_path=cache)
    _ = r1.retrieve("gene lookup", top_k=5)
    assert call_counter["n"] > 0  # initial embed pass
    first_calls = call_counter["n"]

    # Fresh retriever, same cache file → should only embed the query,
    # no tool re-embeds. (Tool descriptions haven't changed.)
    r2 = ToolRetriever(_dummy_specs(), embed_fn=_embed, cache_path=cache)
    t0 = time.monotonic()
    _ = r2.retrieve("SMILES CC(=O)O ADMET", top_k=5)
    dt = time.monotonic() - t0
    assert dt < 0.5, f"2nd retrieve took {dt:.2f}s (should be < 0.5s on cache)"
    # We embedded just the query text on the 2nd run
    new_embeds = call_counter["n"] - first_calls
    assert new_embeds <= 2, f"expected ≤2 new embeds on cache hit, got {new_embeds}"
    # Cache file written
    assert cache.exists()
    with open(cache, "rb") as f:
        data = pickle.load(f)
    assert len(data) == len(_dummy_specs())


# ======================================================================
# Token budget tracker
# ======================================================================


def test_budget_tracker_counts():
    from harness.context import MODE_BUDGETS, TokenBudgetTracker
    t = TokenBudgetTracker(mode="light")
    t.observe_input(messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ])
    assert t.input_used > 0
    t.observe_output(text="Paris is the capital.")
    assert t.output_used > 0
    snap = t.snapshot()
    assert snap["input_budget"] == MODE_BUDGETS["light"].input_tokens
    assert snap["output_budget"] == MODE_BUDGETS["light"].output_tokens


def test_budget_tracker_degrade_action():
    from harness.context import TokenBudgetTracker, MODE_BUDGETS
    t = TokenBudgetTracker(mode="simple_llm")
    assert t.degrade_action() == "ok"
    # Simulate near-full input
    t.input_used = MODE_BUDGETS["simple_llm"].input_tokens - 100
    assert t.degrade_action() in ("compress", "ok")
    t.input_used = MODE_BUDGETS["simple_llm"].input_tokens + 10
    assert t.degrade_action() == "truncate"
    t.input_used = int(MODE_BUDGETS["simple_llm"].input_tokens * 1.5)
    assert t.degrade_action() == "force_answer"


def test_budget_tracker_ensure_room_hard_raises():
    from harness.context import TokenBudgetTracker, BudgetExceededError
    t = TokenBudgetTracker(mode="simple_llm")
    t.input_used = 7_900
    with pytest.raises(BudgetExceededError):
        t.ensure_room(required_input=1_000, hard=True)


def test_budget_tracker_truncate():
    from harness.context import TokenBudgetTracker
    t = TokenBudgetTracker(mode="light")
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "t1"},
        {"role": "assistant", "content": "a2"},
        {"role": "tool", "content": "t2"},
        {"role": "assistant", "content": "a3"},
    ]
    out = t.truncate(msgs, keep_last=2)
    roles = [m["role"] for m in out]
    # Must keep: system, first user, last 2
    assert roles[0] == "system"
    assert "user" in roles
    assert roles[-1] == "assistant"
    assert len(out) < len(msgs)


# ======================================================================
# Integration into FunctionCallingRunner
# ======================================================================


def test_function_calling_runner_retrieval_flag_wired():
    """Runner.__init__ accepts enable_retrieval and related kwargs."""
    from harness.eval.function_calling_runner import FunctionCallingRunner
    from harness.llm_client import LLMClient

    class _DummyLLM(LLMClient):  # type: ignore[misc]
        def __init__(self):  # skip real init
            self.provider = "openai"

    llm = _DummyLLM()
    r = FunctionCallingRunner(
        llm=llm,
        enable_retrieval=True,
        retrieval_top_k=12,
        enable_budget_tracking=True,
        budget_mode="light",
    )
    assert r.enable_retrieval is True
    assert r.retrieval_top_k == 12
    assert r.enable_budget_tracking is True
    assert r.budget_mode == "light"
    assert r.last_retrieval_log is None  # not yet run


# ======================================================================
# Live OpenAI embeddings (opt-in, slow)
# ======================================================================


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set",
)
def test_retrieval_live_openai(tmp_path):
    """Verify OpenAI embeddings actually get used when a key is set."""
    from harness.context.tool_retrieval import ToolRetriever
    r = ToolRetriever(
        tools=_dummy_specs()[:20],
        cache_path=tmp_path / "openai_cache.pkl",
        # default embed_fn → OpenAI path
    )
    picked = r.retrieve(
        "Predict BBB permeability and hERG toxicity for aspirin.",
        top_k=5,
    )
    names = [t["function"]["name"] for t in picked]
    assert r.fallback_used is False
    # At least one chemistry/admet tool should rank in the non-core top 5
    assert any("admet" in n or "mol_" in n or "tdc_" in n for n in names), names
