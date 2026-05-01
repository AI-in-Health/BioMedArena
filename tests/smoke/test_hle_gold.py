"""Smoke test for HLE-Gold loader (requires HF_TOKEN + accepted terms)."""

import os

import pytest


def test_hle_gold_loads():
    if not (os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        pytest.skip("No HF_TOKEN")

    from harness.eval.bench_hle_gold import load_hle_gold_tasks

    try:
        tasks = load_hle_gold_tasks(limit=5)
    except RuntimeError as exc:
        if "gated" in str(exc).lower():
            pytest.skip(f"HLE-Gold gated: {exc}")
        raise

    assert tasks, "HLE-Gold loader returned 0 tasks"
    for t in tasks:
        assert "question" in t and t["question"]
        assert t["answer_type"] in ("multipleChoice", "openText")
        assert t["category"] == "HLE-Gold"
        assert t["scorer_kind"] in ("mcq_exact", "llm_judge")
        ctx = t["context"]
        assert ctx["source"] == "futurehouse/hle-gold-bio-chem"
        assert ctx["is_biomed"] or ctx["is_chemistry"]


def test_hle_gold_include_chemistry_false():
    if not (os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        pytest.skip("No HF_TOKEN")
    from harness.eval.bench_hle_gold import load_hle_gold_tasks
    try:
        biomed_only = load_hle_gold_tasks(limit=150, include_chemistry=False)
    except RuntimeError as exc:
        if "gated" in str(exc).lower():
            pytest.skip(f"HLE-Gold gated: {exc}")
        raise
    assert biomed_only, "empty biomed-only load"
    for t in biomed_only:
        assert t["context"]["is_biomed"]
        assert not t["context"]["is_chemistry"]


def test_hle_gold_mcq_only_flag():
    if not (os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        pytest.skip("No HF_TOKEN")
    from harness.eval.bench_hle_gold import load_hle_gold_tasks
    try:
        mcq = load_hle_gold_tasks(limit=150, include_mcq_only=True)
    except RuntimeError as exc:
        if "gated" in str(exc).lower():
            pytest.skip(f"HLE-Gold gated: {exc}")
        raise
    assert mcq, "empty mcq-only load"
    for t in mcq:
        assert t["answer_type"] == "multipleChoice"
        assert t["scorer_kind"] == "mcq_exact"
