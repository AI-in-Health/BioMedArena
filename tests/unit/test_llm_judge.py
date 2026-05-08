"""Unit tests for ``harness.eval.llm_judge``.

Tests the routing logic, helper functions, and ``score_with_fallback``
with the judge call mocked out.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from harness.eval import llm_judge as J
from harness.eval.llm_judge import (
    OPEN_ANSWER_TYPES,
    is_open_ended,
    judge_enabled,
    pick_judge_model,
    score_with_fallback,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_is_open_ended_accepts_task_dict():
    assert is_open_ended({"answer_type": "openText"}) is True
    assert is_open_ended({"answer_type": "openended"}) is True
    assert is_open_ended({"answer_type": "freeText"}) is True
    assert is_open_ended({"answer_type": "multipleChoice"}) is False
    assert is_open_ended({"answer_type": "numeric"}) is False
    assert is_open_ended({"answer_type": ""}) is False


def test_is_open_ended_accepts_string():
    assert is_open_ended("openText") is True
    assert is_open_ended("multipleChoice") is False
    assert is_open_ended(None) is False


def test_open_answer_types_set():
    assert "opentext" in OPEN_ANSWER_TYPES
    assert "openended" in OPEN_ANSWER_TYPES
    assert "freetext" in OPEN_ANSWER_TYPES


def test_pick_judge_model_always_claude(monkeypatch):
    """Judge model is always claude-sonnet-4-5 regardless of target."""
    monkeypatch.delenv("BIOAGENT_JUDGE_MODEL", raising=False)
    for target in [
        "gemini-2.5-flash",
        "claude-sonnet-4-5",
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "gpt-4o",
        "",  # empty / unknown
    ]:
        assert pick_judge_model(target) == "claude-sonnet-4-5", (
            f"Target {target!r} should map to claude-sonnet-4-5"
        )


def test_pick_judge_model_default_arg(monkeypatch):
    """``pick_judge_model`` works with no argument (default empty string)."""
    monkeypatch.delenv("BIOAGENT_JUDGE_MODEL", raising=False)
    assert pick_judge_model() == "claude-sonnet-4-5"


def test_pick_judge_model_env_override(monkeypatch):
    """Judge model can be pinned per run for comparable rejudging."""
    monkeypatch.setenv("BIOAGENT_JUDGE_MODEL", "gemini-2.5-flash")
    assert pick_judge_model("gpt-4o") == "gemini-2.5-flash"


def test_judge_enabled_default():
    old = os.environ.pop("BIOAGENT_LLM_JUDGE", None)
    try:
        assert judge_enabled() is True
        os.environ["BIOAGENT_LLM_JUDGE"] = "1"
        assert judge_enabled() is True
        os.environ["BIOAGENT_LLM_JUDGE"] = "0"
        assert judge_enabled() is False
    finally:
        if old is None:
            os.environ.pop("BIOAGENT_LLM_JUDGE", None)
        else:
            os.environ["BIOAGENT_LLM_JUDGE"] = old


# ---------------------------------------------------------------------------
# score_with_fallback routing
# ---------------------------------------------------------------------------


class _FakeJudge:
    """Stands in for ``LLMJudge.judge``; returns the verdict pre-set."""

    def __init__(self, verdict: bool):
        self._verdict = verdict
        self.calls = 0
        # Pretend to have an LLMClient with a model attribute.
        self.llm = type("L", (), {"model": "fake-judge"})()

    async def judge(self, question, expected, predicted):
        self.calls += 1
        return {"correct": self._verdict, "reasoning": "r"}


@pytest.fixture(autouse=True)
def reset_judge_singleton():
    old = J._default_judge
    J._default_judge = None
    yield
    J._default_judge = old


@pytest.mark.asyncio
async def test_mcq_primary_correct_skips_judge():
    """Primary says correct -> no judge invocation."""
    fake = _FakeJudge(verdict=False)
    with patch.object(J, "_get_judge_for", return_value=fake):
        with patch("harness.eval.scoring.score_question", return_value=True):
            result = await score_with_fallback(
                {"answer_type": "numeric", "answer": "42", "question": "Q"},
                "42",
                target_backbone="claude-sonnet-4-6",
            )
    assert result["correct"] is True
    assert result["details"]["judge_invoked"] is False
    assert fake.calls == 0


@pytest.mark.asyncio
async def test_mcq_judge_promotes_incorrect_to_correct():
    """Primary says incorrect + judge says correct -> final correct."""
    fake = _FakeJudge(verdict=True)
    with patch.object(J, "_get_judge_for", return_value=fake):
        with patch("harness.eval.scoring.score_question", return_value=False):
            result = await score_with_fallback(
                {"answer_type": "numeric", "answer": "42.0", "question": "Q"},
                "42 approximately",
                target_backbone="claude-sonnet-4-6",
            )
    assert result["correct"] is True
    assert result["details"]["judge_invoked"] is True
    assert result["details"]["judge_verdict"] is True
    assert "llm_judge_fallback" in result["method"]


@pytest.mark.asyncio
async def test_multiple_choice_primary_correct_is_authoritative():
    """MCQ uses deterministic primary scoring when the letter match succeeds."""
    fake = _FakeJudge(verdict=False)
    with patch.object(J, "_get_judge_for", return_value=fake):
        with patch("harness.eval.scoring.score_question", return_value=True):
            result = await score_with_fallback(
                {"answer_type": "multipleChoice", "answer": "A",
                 "question": "Q"},
                "A",
                target_backbone="gpt-4o",
            )
    assert result["correct"] is True
    assert fake.calls == 0
    assert result["details"]["judge_invoked"] is False
    assert result["details"]["primary_verdict"] is True
    assert result["method"] == "primary:multipleChoice"


@pytest.mark.asyncio
async def test_open_ended_judge_is_authoritative_correct():
    """Open-ended: primary INCORRECT, judge CORRECT -> final CORRECT."""
    fake = _FakeJudge(verdict=True)
    with patch.object(J, "_get_judge_for", return_value=fake):
        with patch("harness.eval.scoring.score_question", return_value=False):
            result = await score_with_fallback(
                {"answer_type": "openText", "answer": "x", "question": "Q"},
                "paraphrase of x",
                target_backbone="claude-sonnet-4-6",
            )
    assert result["correct"] is True
    assert result["details"]["is_open_ended"] is True
    assert result["details"]["judge_invoked"] is True
    assert result["method"].startswith("llm_judge_primary")
    # Primary verdict preserved as metadata
    assert result["details"]["primary_verdict"] is False


@pytest.mark.asyncio
async def test_open_ended_judge_is_authoritative_incorrect():
    """Open-ended: primary CORRECT, judge INCORRECT -> final INCORRECT."""
    fake = _FakeJudge(verdict=False)
    with patch.object(J, "_get_judge_for", return_value=fake):
        with patch("harness.eval.scoring.score_question", return_value=True):
            result = await score_with_fallback(
                {"answer_type": "openText", "answer": "x", "question": "Q"},
                "wrong answer",
                target_backbone="claude-sonnet-4-6",
            )
    assert result["correct"] is False
    assert result["details"]["judge_invoked"] is True
    assert result["details"]["primary_verdict"] is True


@pytest.mark.asyncio
async def test_empty_candidate_skips_judge():
    fake = _FakeJudge(verdict=True)
    with patch.object(J, "_get_judge_for", return_value=fake):
        with patch("harness.eval.scoring.score_question", return_value=False):
            result = await score_with_fallback(
                {"answer_type": "numeric", "answer": "42"},
                "",
                target_backbone="claude-sonnet-4-6",
            )
    assert result["correct"] is False
    assert result["details"]["judge_invoked"] is False
    assert fake.calls == 0


@pytest.mark.asyncio
async def test_judge_disabled_env_var():
    old = os.environ.get("BIOAGENT_LLM_JUDGE")
    os.environ["BIOAGENT_LLM_JUDGE"] = "0"
    try:
        fake = _FakeJudge(verdict=True)
        with patch.object(J, "_get_judge_for", return_value=fake):
            with patch(
                "harness.eval.scoring.score_question", return_value=False,
            ):
                result = await score_with_fallback(
                    {"answer_type": "numeric", "answer": "42"},
                    "41",
                )
        assert result["correct"] is False
        assert result["details"]["judge_invoked"] is False
    finally:
        if old is None:
            os.environ.pop("BIOAGENT_LLM_JUDGE", None)
        else:
            os.environ["BIOAGENT_LLM_JUDGE"] = old
