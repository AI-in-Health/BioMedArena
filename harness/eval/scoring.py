"""Robust scoring for benchmark evaluation — answer extraction and comparison."""

from __future__ import annotations

import json
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
    "tokensequence": "tokenSequence",
    "token_sequence": "tokenSequence",
    "sequence_labels": "tokenSequence",
    "relationset": "relationSet",
    "relation_set": "relationSet",
    "multilabel": "multiLabel",
    "multi_label": "multiLabel",
    "multilabelclassification": "multiLabel",
    "multi_label_classification": "multiLabel",
    "ranking": "ranking",
    "ranked_list": "ranking",
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
    direct = re.sub(r"[^A-Za-z]", "", str(response or "").strip()).upper()
    if direct and re.fullmatch(r"[A-J]{1,10}", direct):
        return direct

    LETTERS = "A-Za-z"
    answer_value = rf"([{LETTERS}](?:[\s,;/]+[{LETTERS}])+|[{LETTERS}]{{1,10}})"

    # 1. "The answer is (X)" / "The answer is X"
    m = re.search(rf"[Tt]he\s+answer\s+is\s*\(?{answer_value}\)?", response)
    if m:
        return _clean_mc_match(m.group(1))

    # 2. "Answer: X" / "Answer: (X)"
    m = re.search(rf"[Aa]nswer:\s*\(?{answer_value}\)?", response)
    if m:
        return _clean_mc_match(m.group(1))

    # 2b. Chinese final-answer patterns used by CMB-style prompts.
    m = re.search(rf"答案\s*(?:是|为|:|：)\s*\(?{answer_value}\)?", response)
    if m:
        return _clean_mc_match(m.group(1))

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


def _clean_mc_match(value: str) -> str:
    """Normalize a final-answer MCQ match without turning English words into multi-selects."""
    letters = re.sub(r"[^A-Za-z]", "", value).upper()
    if len(letters) > 1 and any(letter not in "ABCDEFGHIJ" for letter in letters):
        return letters[0]
    return letters


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

    pred_set = re.sub(r"[^A-Z]", "", pred_letter)
    exp_set = re.sub(r"[^A-Z]", "", exp_letter)
    if pred_set and exp_set and (len(pred_set) > 1 or len(exp_set) > 1):
        return sorted(pred_set) == sorted(exp_set)

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
    stripped = cleaned.strip()

    # Direct scalar outputs such as "0", "1", or "1.0" are common for
    # binary classification mirrors and should not be filtered as formula
    # constants by the free-text fallback.
    if re.fullmatch(_NUMERIC_RE, stripped):
        try:
            return float(stripped), "direct_scalar"
        except ValueError:
            pass

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


def _parse_json_answer(value: str):
    if not isinstance(value, str):
        return value
    text = value.strip()
    for candidate in (text, _extract_exact_answer(text)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
    return text


def _bio_spans(tags: list[str]) -> set[tuple[str, int, int]]:
    spans: set[tuple[str, int, int]] = set()
    current_type = ""
    start = -1
    for idx, raw_tag in enumerate(tags + ["O"]):
        tag = str(raw_tag)
        if tag.startswith("B-"):
            if current_type:
                spans.add((current_type, start, idx))
            current_type = tag[2:]
            start = idx
        elif tag.startswith("I-") and current_type == tag[2:]:
            continue
        else:
            if current_type:
                spans.add((current_type, start, idx))
            current_type = ""
            start = -1
            if tag.startswith("I-"):
                current_type = tag[2:]
                start = idx
    return spans


def _f1(pred_items: set, exp_items: set) -> float:
    if not pred_items and not exp_items:
        return 1.0
    if not pred_items or not exp_items:
        return 0.0
    tp = len(pred_items & exp_items)
    precision = tp / len(pred_items)
    recall = tp / len(exp_items)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def token_label_f1(predicted: str, expected: str) -> float:
    """Entity-level F1 for BIO/IOB token labels."""
    pred = _parse_json_answer(predicted)
    exp = _parse_json_answer(expected)
    if isinstance(pred, str):
        pred = [part for part in re.split(r"[\s,]+", pred.strip()) if part]
    if isinstance(exp, str):
        exp = [part for part in re.split(r"[\s,]+", exp.strip()) if part]
    if not isinstance(pred, list) or not isinstance(exp, list):
        return 0.0
    return _f1(_bio_spans([str(x) for x in pred]), _bio_spans([str(x) for x in exp]))


def _parse_multilabel_value(value: str) -> set[str] | None:
    parsed = _parse_json_answer(value)
    if isinstance(parsed, dict):
        if "labels" in parsed:
            parsed = parsed["labels"]
        elif "label" in parsed:
            parsed = parsed["label"]
    if isinstance(parsed, list):
        labels: set[str] = set()
        vector_like = all(str(item).strip() in {"0", "1", "0.0", "1.0", "False", "True", "false", "true"} for item in parsed)
        for idx, item in enumerate(parsed):
            text = str(item).strip()
            if vector_like:
                if text.lower() in {"1", "1.0", "true"}:
                    labels.add(str(idx))
            elif text:
                labels.add(normalize_text(text))
        return labels
    if isinstance(parsed, str):
        text = parsed.strip()
        if not text:
            return set()
        parts = [part for part in re.split(r"[\n,;|]+", text) if part.strip()]
        if len(parts) > 1:
            return _parse_multilabel_value(json.dumps([part.strip() for part in parts]))
        return {normalize_text(text)}
    return None


def multilabel_f1(predicted: str, expected: str) -> float:
    """Instance-level F1 for multilabel classification outputs."""
    pred_set = _parse_multilabel_value(predicted)
    exp_set = _parse_multilabel_value(expected)
    if pred_set is None or exp_set is None:
        return 0.0
    return _f1(pred_set, exp_set)


def _relation_key(item) -> tuple[str, str, str] | None:
    if not isinstance(item, dict):
        return None
    relation_type = normalize_text(str(item.get("type") or item.get("relation") or "relation"))
    arg1 = normalize_text(str(item.get("arg1_id") or item.get("arg1") or item.get("head") or ""))
    arg2 = normalize_text(str(item.get("arg2_id") or item.get("arg2") or item.get("tail") or ""))
    if not arg1 or not arg2:
        return None
    return relation_type, arg1, arg2


def relation_set_f1(predicted: str, expected: str) -> float:
    """Micro F1 for JSON relation sets."""
    pred = _parse_json_answer(predicted)
    exp = _parse_json_answer(expected)
    if isinstance(pred, dict):
        pred = [pred]
    if isinstance(exp, dict):
        exp = [exp]
    if not isinstance(pred, list) or not isinstance(exp, list):
        return 0.0
    pred_set = {key for item in pred if (key := _relation_key(item))}
    exp_set = {key for item in exp if (key := _relation_key(item))}
    return _f1(pred_set, exp_set)


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
    if scorer_kind == "token_f1" or answer_type == "tokenSequence":
        return token_label_f1(predicted, expected) >= 0.999
    if scorer_kind == "relation_f1" or answer_type == "relationSet":
        return relation_set_f1(predicted, expected) >= 0.999
    if scorer_kind == "multilabel_f1" or answer_type == "multiLabel":
        return multilabel_f1(predicted, expected) >= 0.999
    if scorer_kind == "retrieval_hit" or answer_type == "ranking":
        return retrieval_hit(predicted, expected)
    if scorer_kind == "smiles_topk_canonical_match":
        return smiles_topk_canonical_match(predicted, expected)
    if scorer_kind == "smiles_validity_plus_tanimoto":
        return smiles_validity_tanimoto_match(predicted, expected) >= 0.999
    if scorer_kind == "pio_span_f1":
        return token_label_f1(predicted, expected) >= 0.999 or score_exact_match(predicted, expected)

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


def retrieval_hit(predicted: str, expected: str) -> bool:
    """Return True when a ranked-id prediction contains any relevant id."""
    expected_ids = _parse_id_list(expected)
    predicted_ids = _parse_id_list(predicted)
    if not expected_ids or not predicted_ids:
        return False
    return bool(set(expected_ids) & set(predicted_ids))


def smiles_topk_canonical_match(predicted: str, expected: str) -> bool:
    expected_set = {_canonical_smiles(item) for item in _parse_smiles_candidates(expected)}
    predicted_set = {_canonical_smiles(item) for item in _parse_smiles_candidates(predicted)}
    expected_set.discard("")
    predicted_set.discard("")
    if expected_set and predicted_set:
        return bool(expected_set & predicted_set)
    return normalize_text(predicted) == normalize_text(expected)


def smiles_validity_tanimoto_match(predicted: str, expected: str) -> float:
    pred_candidates = [_canonical_smiles(item) for item in _parse_smiles_candidates(predicted)]
    exp_candidates = [_canonical_smiles(item) for item in _parse_smiles_candidates(expected)]
    pred_candidates = [item for item in pred_candidates if item]
    exp_candidates = [item for item in exp_candidates if item]
    if not pred_candidates or not exp_candidates:
        return 1.0 if normalize_text(predicted) == normalize_text(expected) else 0.0
    if set(pred_candidates) & set(exp_candidates):
        return 1.0
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
    except ImportError:
        return 0.0
    best = 0.0
    for pred in pred_candidates:
        pred_mol = Chem.MolFromSmiles(pred)
        if pred_mol is None:
            continue
        pred_fp = AllChem.GetMorganFingerprintAsBitVect(pred_mol, 2, nBits=2048)
        for exp in exp_candidates:
            exp_mol = Chem.MolFromSmiles(exp)
            if exp_mol is None:
                continue
            exp_fp = AllChem.GetMorganFingerprintAsBitVect(exp_mol, 2, nBits=2048)
            best = max(best, float(DataStructs.TanimotoSimilarity(pred_fp, exp_fp)))
    return best


def _parse_smiles_candidates(value: str) -> list[str]:
    parsed = _parse_json_answer(value)
    if isinstance(parsed, dict):
        for key in ("smiles", "product", "products", "prediction", "predictions", "answer"):
            if key in parsed:
                return _parse_smiles_candidates(json.dumps(parsed[key]))
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    text = str(parsed or value or "").strip()
    if not text:
        return []
    parts = [part.strip() for part in re.split(r"[\n,;]+", text) if part.strip()]
    return parts or [text]


def _canonical_smiles(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        from rdkit import Chem
    except ImportError:
        return text
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return ""
    return Chem.MolToSmiles(mol, canonical=True)


def _parse_id_list(value: str) -> list[str]:
    parsed = _parse_json_answer(value)
    if isinstance(parsed, dict):
        for key in ("relevant_doc_ids", "ranked_doc_ids", "doc_ids", "ids"):
            if key in parsed:
                parsed = parsed[key]
                break
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    text = str(parsed or value or "").strip()
    if not text:
        return []
    ids = re.findall(r"(?:doc(?:ument)?[_\s-]?id|corpus[_\s-]?id)\s*[:=]\s*([A-Za-z0-9_.:-]+)", text, re.I)
    if ids:
        return ids
    return [part.strip() for part in re.split(r"[\s,;]+", text) if part.strip()]
