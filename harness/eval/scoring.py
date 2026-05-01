"""Robust scoring for benchmark evaluation — answer extraction and comparison."""

from __future__ import annotations

import re
import string

_NUMERIC_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

_ANSWER_TYPE_ALIASES = {
    "mcq": "multipleChoice",
    "multiplechoice": "multipleChoice",
    "multiple_choice": "multipleChoice",
    "exact": "exactMatch",
    "exactmatch": "exactMatch",
    "exact_match": "exactMatch",
    "numeric": "exactNumeric",
    "exactnumeric": "exactNumeric",
    "exact_numeric": "exactNumeric",
    "opentext": "openText",
    "open_text": "openText",
    "openended": "openText",
    "open_ended": "openText",
    "freetext": "openText",
    "free_text": "openText",
}


def normalize_answer_type(answer_type: str) -> str:
    """Return the canonical public answer type name when known."""
    key = re.sub(r"[\s-]+", "_", str(answer_type or "").strip().lower())
    return _ANSWER_TYPE_ALIASES.get(key, str(answer_type or "").strip())


def normalize_text(text: str) -> str:
    """Lowercase, strip whitespace, collapse spaces, remove leading/trailing punct."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(string.punctuation + " ")
    return text


def extract_answer_from_response(response: str, answer_type: str) -> str:
    """Extract the final answer from an LLM response.

    For multipleChoice: extract the letter (A-Z).
    For exactMatch: extract the answer after common patterns, or return full text.
    """
    answer_type = normalize_answer_type(answer_type)
    if answer_type == "multipleChoice":
        return _extract_mc_answer(response)
    return _extract_exact_answer(response)


def _is_english_word(letter: str, text: str, letter_end: int) -> bool:
    """Check if a single-letter match is an English word (pronoun/article), not an MC answer.

    "I" followed by lowercase or contraction ('m, 'll, ...) → pronoun.
    "A" followed by lowercase → article.
    """
    upper = letter.upper()
    if upper not in ("I", "A"):
        return False
    after = text[letter_end:letter_end + 30].lstrip()
    if not after:
        return False
    # "I" + lowercase or contraction → pronoun ("I don't", "I'm", "I need")
    if upper == "I" and (after[0].islower() or after[0] == "'"):
        return True
    # "A" + lowercase → article ("A specific", "A real")
    if upper == "A" and after[0].islower():
        return True
    return False


def _extract_mc_answer(response: str) -> str:
    """Extract a multiple-choice letter (A-Z) from an LLM response."""
    LETTERS = "A-Za-z"

    # 1. "The answer is (X)" / "The answer is X"
    m = re.search(rf"[Tt]he\s+answer\s+is\s*\(?([{LETTERS}])\)?", response)
    if m:
        return m.group(1).upper()

    # 2. "Answer: X" / "Answer: (X)"
    m = re.search(rf"[Aa]nswer:\s*\(?([{LETTERS}])\)?", response)
    if m:
        return m.group(1).upper()

    # 3. Boxed answer: \boxed{X}
    m = re.search(rf"\\boxed\{{([{LETTERS}])\}}", response)
    if m:
        return m.group(1).upper()

    # 4. "**X**" at end of response (bold letter)
    m = re.search(rf"\*\*([{LETTERS}])\*\*\s*\.?\s*$", response)
    if m:
        return m.group(1).upper()

    # 5. Last standalone letter in the response (excluding English pronouns/articles)
    filtered = []
    for m in re.finditer(rf"(?:^|\s|\()([{LETTERS}])(?:\)|\s|\.|\,|$)", response):
        letter = m.group(1)
        if not _is_english_word(letter, response, m.end(1)):
            filtered.append(letter)
    if filtered:
        return filtered[-1].upper()

    # 6. Fallback: first non-English-word letter anywhere
    for m in re.finditer(rf"[{LETTERS}]", response):
        letter = m.group(0)
        if not _is_english_word(letter, response, m.end()):
            return letter.upper()

    return normalize_text(response)


def _extract_exact_answer(response: str) -> str:
    """Extract an exact answer from a free-form response."""
    # 1. "The answer is ..." pattern
    m = re.search(r"the\s+answer\s+is\s+(.+?)(?:\n|$)", response, re.DOTALL | re.IGNORECASE)
    if m:
        return normalize_text(m.group(1))

    # 2. "Answer: ..." pattern
    m = re.search(r"answer:\s*(.+?)(?:\n|$)", response, re.DOTALL | re.IGNORECASE)
    if m:
        return normalize_text(m.group(1))

    # 3. Boxed answer: \boxed{...}
    m = re.search(r"\\boxed\{(.+?)\}", response)
    if m:
        return normalize_text(m.group(1))

    # 4. Last line of response (often contains the final answer)
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    if lines:
        last = lines[-1]
        # Remove common prefixes
        for prefix in ["therefore, ", "thus, ", "so, ", "hence, ", "finally, "]:
            if last.lower().startswith(prefix):
                last = last[len(prefix):]
        return normalize_text(last)

    return normalize_text(response)


def score_exact_match(predicted: str, expected: str) -> bool:
    """Score an exact-match question conservatively.

    Exact-match labels are common for classification and structured HF
    benchmarks. Avoid substring-only matches such as expected ``0`` vs
    predicted ``10`` or expected ``active`` vs predicted ``inactive``.
    """
    pred = normalize_text(predicted)
    exp = normalize_text(expected)

    if not pred or not exp:
        return False

    # Direct match
    if pred == exp:
        return True

    # Numeric tolerance: if both parse as numbers, allow 1% relative tolerance
    try:
        pred_num = float(pred.replace(",", ""))
        exp_num = float(exp.replace(",", ""))
        if exp_num != 0:
            return abs(pred_num - exp_num) / abs(exp_num) < 0.01
        return abs(pred_num - exp_num) < 1e-6
    except (ValueError, ZeroDivisionError):
        pass

    # Short labels/classes must be exact. This prevents false positives
    # like expected "0" in predicted "10" and expected "active" in
    # predicted "inactive".
    if len(exp) <= 3 or len(exp.split()) == 1:
        return False

    # For longer phrase answers, allow the expected phrase as a whole-word
    # span in a larger final sentence.
    return re.search(rf"(?<!\w){re.escape(exp)}(?!\w)", pred) is not None


def score_multiple_choice(predicted: str, expected: str) -> bool:
    """Score a multiple-choice question by comparing extracted letters."""
    pred_letter = predicted.strip().upper()
    exp_letter = expected.strip().upper()

    # Both should be single letters at this point
    if len(pred_letter) == 1 and len(exp_letter) == 1:
        return pred_letter == exp_letter

    # Fallback: normalize and compare
    return normalize_text(predicted) == normalize_text(expected)


# Numeric tokens that appear in units / normalisation constants in
# medical formulas but are never themselves the answer. When the model
# does not emit a `The answer is X` trailer, `score_numeric_with_tolerance`
# must not fall back to one of these. For example, CKD-EPI answers of
# the shape "GFR = 127 mL/min/1.73 m²" should be scored against the
# reported GFR, not the 1.73 unit denominator.
_MEDCALC_CONSTANT_TOKENS: frozenset[float] = frozenset({
    1.73,    # CKD-EPI BSA normalisation denominator
    1.0,     # identity multiplier appearing in many formula presentations
    100.0,   # percent denominator
    3.14,    # pi
    3.14159,
    2.718,   # e
    2.71828,
})


def extract_numeric_answer(response: str) -> tuple[float | None, str]:
    """Extract the numeric answer the model intended to emit.

    Strategy, in order:
      1. `The answer is [X]` / `The answer: [X]` / `answer is X` trailer.
      2. `Answer: X` prefix.
      3. `\\boxed{X}` (LaTeX).
      4. Fallback: largest-magnitude numeric token in the final 200 chars
         that is NOT in `_MEDCALC_CONSTANT_TOKENS`. Largest-magnitude is
         a reasonable prior for clinical calculators whose outputs are
         typically 1–2 orders of magnitude larger than adjacent unit
         constants.

    Returns `(value, reason)` where `reason` is one of:
      "primary_trailer" | "answer_prefix" | "boxed" |
      "fallback_last_filtered" | "extraction_failed"

    `value is None` signals extraction_failed — the caller should treat
    this distinctly from "wrong", because it typically means the model
    broke format rather than miscomputed.
    """
    if not isinstance(response, str) or not response.strip():
        return None, "extraction_failed"

    cleaned = response.replace(",", "")

    # 1. Primary: "The answer is [number]" / "Answer is: number"
    m = re.search(
        rf"[Tt]he\s+answer\s*(?:is|:|=)\s*\[?\s*({_NUMERIC_RE})\s*\]?",
        cleaned,
    )
    if m:
        try:
            return float(m.group(1)), "primary_trailer"
        except ValueError:
            pass

    # 2. "Answer: X" prefix (no "the")
    m = re.search(
        rf"(?:^|\n)\s*[Aa]nswer\s*[:=]\s*\[?\s*({_NUMERIC_RE})\s*\]?",
        cleaned,
    )
    if m:
        try:
            return float(m.group(1)), "answer_prefix"
        except ValueError:
            pass

    # 3. \boxed{X}
    m = re.search(rf"\\boxed\{{\s*({_NUMERIC_RE})\s*\}}", cleaned)
    if m:
        try:
            return float(m.group(1)), "boxed"
        except ValueError:
            pass

    # 4. Fallback: last numeric token in the tail after filtering out
    #    known constants (1.73, 1.0, 100, π, e) and year-like integers
    #    (1900..2100). "Last-after-filter" reliably selects the scalar
    #    immediately preceding the unit string in constructions like
    #    "... 130.65 mL/min/1.73 m²" or "... 2021 CKD-EPI ... 11.25 mL/min".
    tail = cleaned[-200:]
    candidates: list[float] = []
    for tok in re.findall(_NUMERIC_RE, tail):
        try:
            v = float(tok)
        except ValueError:
            continue
        if v in _MEDCALC_CONSTANT_TOKENS:
            continue
        # Drop standalone year-like integers (publication years, formula
        # years e.g. "2021 CKD-EPI"). Clinical scalars with '.' fractions
        # are never rejected by this rule.
        if "." not in tok and 1900 <= v <= 2100 and v == int(v):
            continue
        candidates.append(v)
    if not candidates:
        return None, "extraction_failed"
    return candidates[-1], "fallback_last_filtered"


def score_numeric_with_tolerance(predicted: str, expected: str, rel_tol: float = 0.05,
                                   lower: str | None = None, upper: str | None = None) -> bool:
    """Score numeric answer with relative tolerance or explicit bounds.

    Extraction uses `extract_numeric_answer` which:
      - prefers the `The answer is X` trailer,
      - falls back to the largest-magnitude numeric token in the tail,
      - explicitly drops known-constant tokens (1.73, 1.0, 100, π, e)
        that appear in unit strings and formula presentations.
    """
    pred_num, _reason = extract_numeric_answer(predicted)
    if pred_num is None:
        return False

    # Check against bounds if provided
    if lower is not None and upper is not None:
        try:
            lo = float(str(lower).replace(",", ""))
            up = float(str(upper).replace(",", ""))
            return lo <= pred_num <= up
        except (ValueError, TypeError):
            pass

    # Relative tolerance check
    try:
        exp_num = float(str(expected).replace(",", ""))
        if exp_num == 0:
            return abs(pred_num) < 1e-6
        return abs(pred_num - exp_num) / abs(exp_num) <= rel_tol
    except (ValueError, ZeroDivisionError):
        return False


def score_open_text(predicted: str, expected: str) -> bool:
    """Soft scoring for open-ended text (HAL Harness benchmarks).

    Uses keyword overlap heuristic — substantial overlap = pass.
    For rigorous scoring, use llm_judge.score() (separate module).
    """
    pred_norm = normalize_text(predicted)
    exp_norm = normalize_text(expected)

    if not exp_norm:
        return False

    # Direct substring match
    if exp_norm in pred_norm:
        return True

    # Token overlap: ≥60% of expected tokens present in predicted
    exp_tokens = set(exp_norm.split())
    pred_tokens = set(pred_norm.split())
    if not exp_tokens:
        return False

    # Filter out very short tokens (function words)
    meaningful = {t for t in exp_tokens if len(t) > 3}
    if not meaningful:
        return exp_tokens.issubset(pred_tokens)
    overlap = len(meaningful & pred_tokens) / len(meaningful)
    return overlap >= 0.6


def score_question(predicted: str, expected: str, answer_type: str, context: dict | None = None) -> bool:
    """Score a question based on its type.

    answer_type:
        multipleChoice — letter match (A-J)
        exactMatch     — normalized string or numeric tolerance
        exactNumeric   — numeric with ±5% tolerance or explicit bounds (MedCalc)
        openText       — soft semantic match for open-ended benchmark tasks
    """
    # Benchmark-specific scorer dispatch via context.scorer_kind.
    # Must run BEFORE answer_type routing so "openText" tasks carrying
    # labbench2_regex don't fall through to soft semantic matching.
    answer_type = normalize_answer_type(answer_type)
    ctx = context or {}
    scorer_kind = ctx.get("scorer_kind")
    if scorer_kind == "labbench2_regex":
        from harness.eval.labbench2_scorer import score_labbench2_regex
        # labbench2_scorer expects a task-shaped dict with
        # scorer_params; re-assemble from context.
        task_like = {
            "answer": expected,
            "scorer_params": ctx.get("scorer_params") or {},
        }
        return bool(score_labbench2_regex(predicted, task_like).get("correct"))

    extracted = extract_answer_from_response(predicted, answer_type)

    if answer_type == "multipleChoice":
        return score_multiple_choice(extracted, expected)

    if answer_type == "exactNumeric":
        ctx = context or {}
        return score_numeric_with_tolerance(
            predicted, expected,
            lower=ctx.get("lower_limit"),
            upper=ctx.get("upper_limit"),
        )

    if answer_type == "openText":
        return score_open_text(predicted, expected)

    return score_exact_match(extracted, expected)
