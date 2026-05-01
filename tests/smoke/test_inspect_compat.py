"""Smoke tests for inspect-ai compatibility skeleton."""

from __future__ import annotations

import pytest


def test_module_imports():
    from harness.eval.inspect_compat import (
        inspect_ai_available, from_inspect_task, to_inspect_task,
        scorer_to_metric, round_trip, register_scorer,
    )
    # Availability probe must not raise regardless of install status
    _ = inspect_ai_available()


def test_from_inspect_task_dict_minimal():
    from harness.eval.inspect_compat import from_inspect_task
    inspect_spec = {
        "name": "toy",
        "dataset": [
            {"input": "What is 2+2?", "target": "4"},
            {"input": "What is 3+3?", "target": "6"},
        ],
    }
    out = from_inspect_task(inspect_spec)
    assert len(out) == 2
    assert out[0]["question"] == "What is 2+2?"
    assert out[0]["answer"] == "4"
    # Heuristic: numeric answer → scorer_kind='numeric'
    assert out[0]["scorer_kind"] == "numeric"


def test_from_inspect_task_mcq_heuristic():
    from harness.eval.inspect_compat import from_inspect_task
    inspect_spec = {
        "dataset": [{
            "input": "Which vitamin is produced in skin by UVB?",
            "choices": ["A", "B", "C", "D"],
            "target": "D",
        }],
    }
    out = from_inspect_task(inspect_spec)
    assert out[0]["choices"] == ["A", "B", "C", "D"]
    assert out[0]["scorer_kind"] == "mcq"


def test_from_inspect_task_chat_messages_flattened():
    from harness.eval.inspect_compat import from_inspect_task
    # Inspect-AI allows the input to be a list of chat messages
    inspect_spec = {
        "dataset": [{
            "input": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Define sepsis."},
            ],
            "target": "Life-threatening organ dysfunction from infection.",
        }],
    }
    out = from_inspect_task(inspect_spec)
    assert "Define sepsis" in out[0]["question"]
    assert "You are helpful" in out[0]["question"]


def test_from_inspect_task_raises_on_subtasks():
    from harness.eval.inspect_compat import from_inspect_task
    with pytest.raises(NotImplementedError):
        from_inspect_task({
            "dataset": [{"input": "q", "target": "a"}],
            "subtasks": [{"name": "other"}],
        })


def test_to_inspect_task_shape():
    from harness.eval.inspect_compat import to_inspect_task
    out = to_inspect_task(
        [{"question": "q1", "answer": "a1", "choices": ["A", "B"]}],
        name="export_test", scorer="mcq",
    )
    assert out["name"] == "export_test"
    assert out["scorer"] == "mcq"
    assert len(out["dataset"]) == 1
    s = out["dataset"][0]
    assert s["input"] == "q1"
    assert s["target"] == "a1"
    assert s["choices"] == ["A", "B"]


def test_round_trip_preserves_data():
    """Our format → inspect dict → our format = idempotent on core fields."""
    from harness.eval.inspect_compat import round_trip
    original = [
        {"question": "Which gene is on Chr17?", "choices": ["BRCA1", "TP53", "EGFR", "KRAS"],
         "answer": "BRCA1", "metadata": {"source": "demo"}, "scorer_kind": "mcq"},
        {"question": "Compute 1+1.", "answer": "2",
         "metadata": {}, "choices": None, "scorer_kind": "numeric"},
    ]
    recovered = round_trip(original)
    assert len(recovered) == len(original)
    for o, r in zip(original, recovered):
        assert r["question"] == o["question"]
        assert r["answer"] == o["answer"]
        # choices preserved (or both None-ish)
        assert (r["choices"] or None) == (o["choices"] or None)
        # scorer_kind re-inferred, but should match for these 2 cases
        assert r["scorer_kind"] == o["scorer_kind"]


def test_scorer_to_metric_exact():
    from harness.eval.inspect_compat import scorer_to_metric
    fn = scorer_to_metric("exact")
    assert fn("Paris", "paris") == 1.0
    assert fn("Paris", "London") == 0.0


def test_scorer_to_metric_mcq():
    from harness.eval.inspect_compat import scorer_to_metric
    fn = scorer_to_metric("mcq")
    assert fn("The answer is B.", "B") == 1.0
    assert fn("A) is wrong, answer is C", "C") == 1.0
    assert fn("D", "A") == 0.0


def test_scorer_to_metric_unknown_raises():
    from harness.eval.inspect_compat import scorer_to_metric
    with pytest.raises(NotImplementedError):
        scorer_to_metric("custom_weighted_f1_with_calibration")


def test_register_scorer():
    from harness.eval.inspect_compat import register_scorer, scorer_to_metric
    register_scorer("always_zero", lambda p, g: 0.0)
    fn = scorer_to_metric("always_zero")
    assert fn("x", "x") == 0.0
