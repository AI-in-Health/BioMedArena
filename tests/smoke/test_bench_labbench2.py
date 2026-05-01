"""Smoke tests for the LAB-Bench 2 loader + scorer.

No LLM calls are made here — this is loader + scoring-logic only.

HF auth: live subset loading requires accepting the dataset terms at
https://huggingface.co/datasets/EdisonScientific/labbench2 AND an HF_TOKEN
(or `huggingface-cli login`). Tests that need live HF access use the
`hf_auth_or_skip` fixture; the rest run offline.
"""

from __future__ import annotations

import os

import pytest


# ---- Fixtures -------------------------------------------------------

@pytest.fixture
def hf_auth_or_skip():
    """Skip when HF credentials or gated access are unavailable."""
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )
    if not token:
        # Try the cached huggingface-cli login as a secondary source.
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            token = None
    if not token:
        pytest.skip(
            "No HF auth — set HF_TOKEN or run `huggingface-cli login`, "
            "and accept terms at "
            "https://huggingface.co/datasets/EdisonScientific/labbench2"
        )
    try:
        from datasets import load_dataset
        # Streaming probe: cheap, validates access.
        iter(load_dataset(
            "EdisonScientific/labbench2", "seqqa2",
            split="train", streaming=True, token=token,
        ))
    except Exception as exc:
        msg = str(exc).lower()
        if "gated" in msg or "access" in msg or "401" in msg or "403" in msg:
            pytest.skip(f"labbench2 gated access not granted: {exc}")
        raise
    return token


@pytest.fixture
def synthetic_v2_row():
    """A v2-schema row matching the live probe's 16 fields."""
    return {
        "id": "lb2_test_0001",
        "tag": "seqqa2",
        "version": "1.0",
        "question": "Design primers to amplify a 200 bp amplicon.",
        "ideal": "ACGT primer sequence",
        "files": [],
        "sources": ["pubmed:12345"],
        "key_passage": None,
        "canary": False,
        "is_opensource": True,
        "ground_truth": "ACGT",
        "prompt_suffix": None,
        "type": "sequence",
        "mode": "text",
        "validator_params": {"regex": "ACGT"},
        "answer_regex": r"ACGT",
    }


# ---- Offline tests --------------------------------------------------

def test_subset_taxonomy_counts():
    from harness.eval.bench_labbench2 import (
        ALL_SUBSETS, TEXT_ONLY_SUBSETS, FILE_REF_TEXT_SUBSETS,
        MULTIMODAL_SUBSETS,
    )
    assert len(ALL_SUBSETS) == 15
    assert len(TEXT_ONLY_SUBSETS) == 7
    assert len(FILE_REF_TEXT_SUBSETS) == 4
    assert len(MULTIMODAL_SUBSETS) == 4
    # No overlap between text-only and multimodal.
    assert not (set(TEXT_ONLY_SUBSETS) & MULTIMODAL_SUBSETS)
    # Every multimodal tag is in ALL_SUBSETS.
    assert MULTIMODAL_SUBSETS <= set(ALL_SUBSETS)


def test_resolve_subsets_defaults():
    from harness.eval.bench_labbench2 import _resolve_subsets
    assert _resolve_subsets(None, include_multimodal=False) == [
        "litqa3", "patentqa", "trialqa", "dbqa2", "suppqa2", "figqa2", "tableqa2",
    ]


def test_resolve_subsets_all_uses_official_all_config():
    from harness.eval.bench_labbench2 import _resolve_subsets
    resolved = _resolve_subsets("all", include_multimodal=False)
    assert resolved == ["all"]
    resolved_mm = _resolve_subsets("all", include_multimodal=True)
    assert resolved_mm == ["all"]


def test_resolve_subsets_rejects_unknown():
    from harness.eval.bench_labbench2 import _resolve_subsets
    with pytest.raises(ValueError, match="Unknown labbench2 subset"):
        _resolve_subsets(["nosuchsubset"], include_multimodal=False)


def test_resolve_subsets_rejects_explicit_multimodal():
    from harness.eval.bench_labbench2 import _resolve_subsets
    with pytest.raises(ValueError, match="multimodal"):
        _resolve_subsets(["figqa2-img"], include_multimodal=False)


def test_normalize_row_basic(synthetic_v2_row):
    from harness.eval.bench_labbench2 import _normalize_v2_task
    t = _normalize_v2_task(synthetic_v2_row, "seqqa2", 0)
    assert t["answer_type"] == "openText"
    assert t["scorer_kind"] == "labbench2_regex"
    assert t["category"] == "LAB-Bench-2/seqqa2"
    assert t["answer"] == "ACGT primer sequence"
    assert t["context"]["benchmark"] == "labbench2"
    assert t["context"]["tag"] == "seqqa2"
    assert t["context"]["has_files"] is False
    assert t["scorer_params"]["answer_regex"] == r"ACGT"


def test_normalize_row_schema_drift_raises(synthetic_v2_row):
    from harness.eval.bench_labbench2 import _normalize_v2_task
    bad = dict(synthetic_v2_row)
    bad.pop("ideal")
    with pytest.raises(KeyError, match="missing expected fields"):
        _normalize_v2_task(bad, "seqqa2", 7)


def test_normalize_row_with_files_preserves_refs(synthetic_v2_row):
    from harness.eval.bench_labbench2 import _normalize_v2_task
    row = dict(synthetic_v2_row)
    row["files"] = ["gs://labbench2/foo.pdb", "gs://labbench2/bar.fasta"]
    t = _normalize_v2_task(row, "seqqa2", 3)
    assert t["context"]["has_files"] is True
    assert t["context"]["file_refs"] == row["files"]


def test_scorer_regex_match(synthetic_v2_row):
    from harness.eval.bench_labbench2 import _normalize_v2_task
    from harness.eval.labbench2_scorer import score_labbench2_regex

    t = _normalize_v2_task(synthetic_v2_row, "seqqa2", 0)
    result = score_labbench2_regex("The primer is ACGT yes.", t)
    assert result["correct"] is True
    assert result["score"] == 1.0
    assert result["method"] == "regex"
    assert result["matched_text"] == "ACGT"


def test_scorer_regex_miss(synthetic_v2_row):
    from harness.eval.bench_labbench2 import _normalize_v2_task
    from harness.eval.labbench2_scorer import score_labbench2_regex

    t = _normalize_v2_task(synthetic_v2_row, "seqqa2", 0)
    result = score_labbench2_regex("irrelevant prediction", t)
    assert result["correct"] is False
    assert result["method"] == "regex"


def test_scorer_broken_regex_falls_back_to_substring(synthetic_v2_row):
    from harness.eval.bench_labbench2 import _normalize_v2_task
    from harness.eval.labbench2_scorer import score_labbench2_regex

    row = dict(synthetic_v2_row)
    row["answer_regex"] = "[unclosed"  # invalid regex
    t = _normalize_v2_task(row, "seqqa2", 0)
    # prediction contains the full ideal as substring
    pred = "the model thinks: ACGT primer sequence confirmed"
    result = score_labbench2_regex(pred, t)
    assert result["correct"] is True
    assert result["method"] == "substring_fallback"
    assert result["regex_error"] is not None


def test_scorer_no_regex_no_gold():
    from harness.eval.labbench2_scorer import score_labbench2_regex

    # A synthetic task with no regex and no gold — scorer must not crash.
    task = {
        "scorer_params": {
            "answer_regex": None,
            "ideal": "",
            "ground_truth": "",
        },
    }
    result = score_labbench2_regex("anything", task)
    assert result["correct"] is False
    assert result["method"] == "no_gold"


# ---- Live HF tests (skipped without auth) ---------------------------

def test_loader_seqqa2_sample(hf_auth_or_skip):
    """Load a small live sample of seqqa2 and sanity-check shape."""
    from harness.eval.bench_labbench2 import load_labbench2_tasks

    tasks = load_labbench2_tasks(subsets=["seqqa2"], limit=5)
    assert len(tasks) <= 5
    for t in tasks:
        assert t["answer_type"] == "openText"
        assert t["scorer_kind"] == "labbench2_regex"
        assert t["category"].startswith("LAB-Bench-2/")
        assert "id" in t and t["id"]
        assert "question" in t and t["question"]


def test_loader_default_text_only_baseline(hf_auth_or_skip):
    """Default path samples the shared 7-tag text-only baseline."""
    from harness.eval.bench_labbench2 import load_labbench2_tasks

    tasks = load_labbench2_tasks(limit=30)
    assert tasks
    assert {t["context"]["source"] for t in tasks} == {"EdisonScientific/labbench2"}
    assert {t["raw_subject"] for t in tasks} <= {
        "litqa3", "patentqa", "trialqa", "dbqa2", "suppqa2", "figqa2", "tableqa2",
    }
    assert all(not t["context"]["has_files"] for t in tasks)


def test_loader_skip_with_files(hf_auth_or_skip):
    """skip_with_files=True must yield only tasks with has_files=False."""
    from harness.eval.bench_labbench2 import load_labbench2_tasks

    tasks = load_labbench2_tasks(
        subsets=["seqqa2"], limit=10, skip_with_files=True
    )
    for t in tasks:
        assert t["context"]["has_files"] is False
