"""Regex-based scorer for biomedical lab tasks.

LAB-Bench 2 rows carry `answer_regex` + `validator_params` rather
than a fixed letter choice or free-text gold. This module dispatches:

    1. Primary: compile `answer_regex` against the prediction
       (IGNORECASE | DOTALL). Match → correct.
    2. Fallback: case-insensitive substring match on
       `ideal` (or `ground_truth`) if no regex is present or if the
       regex fails to compile.

Returns a dict in the project's standard scorer shape:
    {correct: bool, score: float, method: str, ...}

This file is intentionally standalone so the main session can wire
it into the central scoring dispatcher (harness/eval/scoring.py)
without this session touching scoring.py. A ready-to-apply wiring
snippet is documented in the module docstring below.

To integrate into harness.eval.scoring:

    # in the scoring dispatcher, e.g. score_question:
    if task.get("scorer_kind") == "labbench2_regex":
        from harness.eval.labbench2_scorer import score_labbench2_regex
        return score_labbench2_regex(prediction, task)

No other code in scoring.py needs to change.
"""

from __future__ import annotations

import re
from typing import Any


def _safe_regex_search(
    pattern: str | None, text: str
) -> tuple[bool, str | None, str | None]:
    """Search `text` for `pattern` safely.

    Returns (ok, matched_text, regex_error).

    - ok is True if the pattern compiled and matched.
    - matched_text is re.Match.group(0) or None.
    - regex_error is the re.error message if compilation failed, else None.
    """
    if not pattern:
        return False, None, None
    try:
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    except re.error as exc:
        return False, None, str(exc)
    if m is None:
        return False, None, None
    return True, m.group(0), None


def _substring_match(gold: str, prediction: str) -> bool:
    if not gold:
        return False
    return gold.strip().lower() in prediction.lower()


def score_labbench2_regex(
    prediction: str, task: dict[str, Any]
) -> dict[str, Any]:
    """Score a prediction against its regex / gold.

    Args:
        prediction: model output (already extracted from chat /
            tool-call response by the caller).
        task: a task dict produced by
            harness/eval/bench_labbench2.py::load_labbench2_tasks.
            Expected to carry scorer_params={answer_regex,
            validator_params, ground_truth, ideal}.

    Returns:
        {
            correct: bool,
            score: 1.0 | 0.0,
            method: "regex" | "substring_fallback" | "no_gold",
            matched_text: str | None,
            regex_error: str | None,  # only when regex failed to compile
        }
    """
    pred = (prediction or "").strip()
    params = task.get("scorer_params") or {}
    regex = params.get("answer_regex")
    gold = params.get("ideal") or params.get("ground_truth") or ""

    # Primary: regex
    ok, matched, regex_error = _safe_regex_search(regex, pred)
    if ok:
        return {
            "correct": True,
            "score": 1.0,
            "method": "regex",
            "matched_text": matched,
            "regex_error": None,
        }
    if regex and regex_error is None:
        # Regex compiled but didn't match.
        return {
            "correct": False,
            "score": 0.0,
            "method": "regex",
            "matched_text": None,
            "regex_error": None,
        }

    # Fallback: substring on gold
    if gold:
        matched_substr = _substring_match(gold, pred)
        return {
            "correct": matched_substr,
            "score": 1.0 if matched_substr else 0.0,
            "method": "substring_fallback",
            "matched_text": gold if matched_substr else None,
            "regex_error": regex_error,
        }

    return {
        "correct": False,
        "score": 0.0,
        "method": "no_gold",
        "matched_text": None,
        "regex_error": regex_error,
    }
