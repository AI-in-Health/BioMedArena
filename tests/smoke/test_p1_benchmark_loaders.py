"""Smoke tests for benchmark loaders.

Official-source loaders fail closed when upstream data is unavailable, so
hermetic CI only asserts importability and schema for any rows that load.
"""

from __future__ import annotations

import pytest


# ======================================================================
# Individual loaders
# ======================================================================


def test_bixbench_loader_importable():
    from harness.eval import load_bixbench_tasks
    tasks = load_bixbench_tasks(limit=2)
    assert isinstance(tasks, list)
    for t in tasks:
        assert {"question", "answer", "metadata", "scorer_kind"} <= t.keys()


# ======================================================================
# Combined Medical-QA
# ======================================================================


def test_medical_qa_loader_importable():
    from harness.eval import load_medical_qa_tasks
    tasks = load_medical_qa_tasks(limit=3)
    assert isinstance(tasks, list)
    for t in tasks:
        assert t["metadata"].get("dataset") in {"medqa", "medmcqa", "pubmedqa"}


def test_medical_qa_source_filter():
    from harness.eval import load_medical_qa_tasks
    # If HuggingFace is unavailable the loader returns [] rather than
    # offline fallback data. This test verifies the kwarg remains accepted.
    tasks = load_medical_qa_tasks(sources=["pubmedqa"], limit=5)
    assert isinstance(tasks, list)


def test_medical_qa_shape_invariant():
    from harness.eval import load_medical_qa_tasks
    tasks = load_medical_qa_tasks(limit=3)
    required = {"id", "question", "choices", "answer", "metadata", "scorer_kind"}
    for t in tasks:
        assert required <= t.keys()


def test_new_loaders_exported_from_eval_module():
    import harness.eval as E
    for name in (
        "load_bixbench_tasks",
        "load_medical_qa_tasks",
        "load_super_chemistry_tasks",
    ):
        assert hasattr(E, name), f"missing: {name}"
