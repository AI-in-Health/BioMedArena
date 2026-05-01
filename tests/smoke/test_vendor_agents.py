"""Smoke tests for vendor-agent adapters.

All tests are offline — live Biomni A1 execution would need the
~11 GB data lake and a live LLM call, which we don't run in CI.
"""

from __future__ import annotations

import pytest


def test_all_three_registered():
    from harness.adapters import ADAPTER_REGISTRY
    assert "BiomniAdapter" in ADAPTER_REGISTRY
    assert "TxAgentStubAdapter" in ADAPTER_REGISTRY


# ======================================================================
# Biomni
# ======================================================================


def test_biomni_importable():
    pytest.importorskip("biomni")
    import biomni  # noqa: F401
    from biomni.agent import A1  # noqa: F401


def test_biomni_adapter_basic():
    pytest.importorskip("biomni")
    from harness.adapters.biomni_adapter import BiomniAdapter
    a = BiomniAdapter()
    assert a.available is True
    assert a.modality == "reasoning"
    caps = a.capabilities()
    assert "biomni_delegation" in caps
    assert "langgraph_react" in caps


def test_biomni_adapter_config_override():
    pytest.importorskip("biomni")
    from harness.adapters.biomni_adapter import BiomniAdapter
    a = BiomniAdapter(config={
        "biomni": {
            "data_path": "/tmp/fake_biomni",
            "llm": "gpt-4o",
            "source": "OpenAI",
            "use_tool_retriever": False,
        },
    })
    assert a._data_path == "/tmp/fake_biomni"
    assert a._llm == "gpt-4o"
    assert a._source == "OpenAI"
    assert a._use_tool_retriever is False


async def test_biomni_refuses_without_data_path(monkeypatch):
    """Without data_path and without BIOMNI_ALLOW_DOWNLOAD, run() must
    return a refusal rather than triggering the 11GB download."""
    pytest.importorskip("biomni")
    from harness.adapters.biomni_adapter import BiomniAdapter
    monkeypatch.delenv("BIOMNI_DATA_PATH", raising=False)
    monkeypatch.delenv("BIOMNI_ALLOW_DOWNLOAD", raising=False)
    a = BiomniAdapter()
    r = await a.run("test query")
    assert r["confidence"] == 0.0
    assert "data_path" in r["answer"] or "BIOMNI_ALLOW_DOWNLOAD" in r["answer"]


# ======================================================================
# TxAgent stub
# ======================================================================


def test_txagent_stub_basic():
    from harness.adapters.txagent_stub_adapter import TxAgentStubAdapter
    a = TxAgentStubAdapter()
    assert a.available is True
    caps = a.capabilities()
    assert "txagent_style_delegation" in caps
    assert "tooluniverse_first" in caps


async def test_txagent_stub_runs_without_llm():
    from harness.adapters.txagent_stub_adapter import TxAgentStubAdapter
    a = TxAgentStubAdapter()
    r = await a.run("What is the mechanism of warfarin?")
    # Without an LLM attached the stub should return a deterministic
    # fallback describing what it WOULD have done.
    assert r["confidence"] <= 0.2
    assert "tooluniverse" in r["answer"].lower() or "txagent" in r["answer"].lower()
