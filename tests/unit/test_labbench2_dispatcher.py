"""Verify labbench2_regex scorer is wired into score_question's dispatcher.

The entry point in this project is
`harness.eval.scoring.score_question(predicted, expected, answer_type, context)`.
Callers (benchmark_suite.py) pass the task's
`context` dict; the loader duplicates `scorer_kind` and `scorer_params`
into that context so the dispatcher can branch on them.
"""

from __future__ import annotations

from harness.eval.scoring import score_question


def _make_lb2_context(answer_regex: str | None = None, ideal: str = "", ground_truth: str = ""):
    return {
        "benchmark": "labbench2",
        "scorer_kind": "labbench2_regex",
        "scorer_params": {
            "answer_regex": answer_regex,
            "validator_params": {},
            "ground_truth": ground_truth,
            "ideal": ideal,
        },
    }


def test_labbench2_regex_route_primary_regex_hit():
    """Regex-primary path: a matching prediction must return True."""
    ctx = _make_lb2_context(answer_regex=r"target_\w+", ideal="target_answer")
    result = score_question(
        "The answer is target_answer.", "target_answer", "openText", ctx
    )
    assert result is True


def test_labbench2_regex_route_primary_regex_miss():
    """Regex-primary path: a non-matching prediction must return False."""
    ctx = _make_lb2_context(answer_regex=r"target_\w+", ideal="target_answer")
    result = score_question(
        "totally wrong garbage xyzzy 98765", "target_answer", "openText", ctx
    )
    assert result is False


def test_labbench2_regex_route_substring_fallback_hit():
    """No regex → fall back to substring match on ideal."""
    ctx = _make_lb2_context(answer_regex=None, ideal="ciprofloxacin")
    result = score_question(
        "the drug is ciprofloxacin.", "ciprofloxacin", "openText", ctx
    )
    assert result is True


def test_labbench2_regex_route_substring_fallback_miss():
    ctx = _make_lb2_context(answer_regex=None, ideal="ciprofloxacin")
    result = score_question(
        "the drug is amoxicillin.", "ciprofloxacin", "openText", ctx
    )
    assert result is False


def test_non_labbench2_openText_falls_through_to_default():
    """Tasks without scorer_kind='labbench2_regex' must NOT be captured.

    Uses score_open_text semantics — exact canonical-case match passes.
    """
    # No context at all — default path.
    result_exact = score_question("42 is the answer", "42 is the answer", "openText", None)
    assert result_exact is True

    # Context exists but scorer_kind is unset — still default path.
    ctx = {"benchmark": "medcalc", "lower_limit": 40, "upper_limit": 44}
    result_numeric = score_question("42", "42", "exactNumeric", ctx)
    assert result_numeric is True


def test_non_labbench2_multiple_choice_falls_through():
    """A multipleChoice task with unrelated context still routes to MCQ scorer."""
    ctx = {"benchmark": "gpqa_bio"}
    # score_multiple_choice extracts letters from free text
    assert score_question("The answer is B.", "B", "multipleChoice", ctx) is True
    assert score_question("The answer is A.", "B", "multipleChoice", ctx) is False
