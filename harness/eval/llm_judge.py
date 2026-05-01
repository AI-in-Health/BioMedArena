"""LLM-as-Judge scoring for medical/scientific QA.

When string-matching scoring is too strict (open-ended answers, paraphrases,
code-describing-code), use an LLM to evaluate semantic equivalence.

Default judge model: claude-sonnet-4-5, but configurable via BIOAGENT_JUDGE_MODEL.
Judgments are cached on (question_id, predicted_hash) to avoid re-evaluation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from harness.llm_client import LLMClient

logger = logging.getLogger(__name__)


JUDGE_SYSTEM = (
    "You are a strict but fair evaluator of medical/scientific AI answers. "
    "Given a question, the expected reference answer, and a model's predicted answer, "
    "determine whether the predicted answer is semantically equivalent to the expected answer.\n\n"
    "Consider these correctness criteria:\n"
    "- Semantic equivalence: different wording but same meaning = CORRECT\n"
    "  (e.g. 'acute MI' vs 'acute myocardial infarction' = CORRECT)\n"
    "- Numeric: within ±5% relative tolerance, or matching after rounding = CORRECT\n"
    "- Open-ended / patches / code: does the prediction capture the same root cause "
    "  or core fix approach as the reference? = CORRECT\n"
    "- Multiple-choice: same letter or same option content = CORRECT\n"
    "- Hallucinated specifics that contradict the reference = INCORRECT\n\n"
    "First give a brief reasoning, then on the LAST line write your verdict as "
    "exactly one of these two words: CORRECT or INCORRECT"
)


JUDGE_PROMPT_TEMPLATE = """Question:
{question}

Expected reference answer:
{expected}

Predicted answer:
{predicted}

Is the predicted answer correct?"""


def _extract_verdict(text: str) -> bool:
    """Extract CORRECT/INCORRECT verdict from judge's plain-text response.

    Robust against models that don't follow the exact prompt format —
    handles synonyms like WRONG, RIGHT, YES, NO, TRUE, FALSE, etc.

    Strategy (in priority order):
    1. Pure JSON with a `correct` field
    2. Last non-empty line is an exact verdict keyword
    3. JSON parse for structured judge outputs
    4. Scan lines in reverse for verdict keywords
    5. Loose keyword search in full text
    """
    import re

    # Negative keywords checked FIRST (many contain positive substrings,
    # e.g. INCORRECT contains CORRECT)
    _NEG = r'\b(?:INCORRECT|WRONG|FALSE|NO|INACCURATE|DOES\s+NOT\s+MATCH|NOT\s+CORRECT|NOT\s+EQUIVALENT)\b'
    _POS = r'\b(?:CORRECT|RIGHT|TRUE|YES|ACCURATE|MATCHES|EQUIVALENT)\b'

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            import json
            obj = json.loads(stripped)
            if isinstance(obj, dict) and "correct" in obj:
                return bool(obj["correct"])
        except Exception:
            pass

    # 1. Strict: last non-empty line
    if lines:
        last = lines[-1]
        if re.search(_NEG, last, re.IGNORECASE):
            return False
        if re.search(_POS, last, re.IGNORECASE):
            # Skip if it looks like a JSON key (e.g. "correct": false)
            if not re.search(r'["\']correct["\']\s*:', last, re.IGNORECASE):
                return True

    # 2. Try JSON parse before loose line scans. JSON reasoning strings
    # often contain words like "not correct"; the structured `correct`
    # field should remain authoritative.
    try:
        import json
        obj = json.loads(text)
        if isinstance(obj, dict) and "correct" in obj:
            return bool(obj["correct"])
    except Exception:
        pass

    # 3. Scan lines in reverse for verdict keywords
    for line in reversed(lines):
        if re.search(_NEG, line, re.IGNORECASE):
            return False
        if re.search(_POS, line, re.IGNORECASE):
            if re.search(r'["\']correct["\']\s*:', line, re.IGNORECASE):
                continue
            return True

    # 4. Loose keyword search (last resort)
    if re.search(_NEG, text, re.IGNORECASE):
        return False
    if re.search(_POS, text, re.IGNORECASE):
        return True
    return False


class LLMJudge:
    """Async LLM-as-Judge with caching."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        cache_path: str = "data/cache/judgments.jsonl",
        provider: str = "gemini",
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        fallback_llm: LLMClient | None = None,
    ):
        if llm is not None:
            self.llm = llm
        else:
            self.llm = LLMClient(provider=provider, model=model, api_key=api_key)
        self._fallback_llm = fallback_llm
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec
        except Exception as exc:
            logger.warning("judge cache load failed: %s", exc)

    def _persist(self, key: str, record: dict[str, Any]) -> None:
        self._cache[key] = record
        try:
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("judge cache write failed: %s", exc)

    @staticmethod
    def _make_key(question: str, expected: str, predicted: str) -> str:
        h = hashlib.sha256()
        h.update(question.encode("utf-8"))
        h.update(b"||")
        h.update(expected.encode("utf-8"))
        h.update(b"||")
        h.update(predicted.encode("utf-8"))
        return h.hexdigest()

    async def judge(
        self,
        question: str,
        expected: str,
        predicted: str,
    ) -> dict[str, Any]:
        """Returns {"correct": bool, "reasoning": str, "from_cache": bool}."""
        # Truncate very long inputs so judge prompt stays sane.
        # predicted uses a higher limit because multi-agent responses
        # can be long and the final answer is typically near the end.
        q = (question or "")[:3000]
        e = (expected or "")[:1500]
        p = (predicted or "")[:4000]

        key = self._make_key(q, e, p)
        if key in self._cache:
            rec = self._cache[key]
            return {**rec, "from_cache": True}

        prompt = JUDGE_PROMPT_TEMPLATE.format(question=q, expected=e, predicted=p)
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ]

        # Try primary model, then fallback on empty response (e.g. content filter)
        for llm_client in (self.llm, self._fallback_llm):
            if llm_client is None:
                continue
            try:
                raw = await llm_client.chat(messages=messages, temperature=0.0)
                raw = (raw or "").strip()
                if raw:
                    correct = _extract_verdict(raw)
                    reasoning = raw[:300]
                    record = {"key": key, "correct": correct, "reasoning": reasoning}
                    self._persist(key, record)
                    return {**record, "from_cache": False}
                # Empty response — try fallback
                logger.warning(
                    "judge returned empty response (model=%s), trying fallback",
                    llm_client.model,
                )
            except Exception as exc:
                logger.warning("judge call failed (model=%s): %s: %s",
                               llm_client.model, type(exc).__name__, exc)

        # All models exhausted
        return {"correct": False, "reasoning": "judge_error: empty_response",
                "from_cache": False, "error": True}

    async def judge_batch(
        self,
        items: list[tuple[str, str, str]],
        max_concurrent: int = 5,
    ) -> list[dict[str, Any]]:
        """Judge a list of (question, expected, predicted) tuples in parallel."""
        sem = asyncio.Semaphore(max_concurrent)

        async def _one(q, e, p):
            async with sem:
                return await self.judge(q, e, p)

        return list(await asyncio.gather(*[_one(q, e, p) for q, e, p in items]))


# Singleton convenience
_default_judge: LLMJudge | None = None


def _judge_provider_and_key() -> tuple[str, str | None]:
    """Pick provider and api_key for the judge.

    Defaults to Claude/Anthropic unless the run pins a different judge with
    ``BIOAGENT_JUDGE_PROVIDER`` and ``BIOAGENT_JUDGE_MODEL``.
    """
    import os
    provider_override = os.environ.get("BIOAGENT_JUDGE_PROVIDER", "").strip().lower()
    if provider_override == "gemini":
        return "gemini", os.environ.get("GEMINI_API_KEY")
    if provider_override == "anthropic":
        return "anthropic", os.environ.get("ANTHROPIC_API_KEY")
    return "anthropic", os.environ.get("ANTHROPIC_API_KEY")


def get_judge() -> LLMJudge:
    global _default_judge
    if _default_judge is None:
        provider, api_key = _judge_provider_and_key()
        _default_judge = LLMJudge(
            provider=provider,
            model="claude-sonnet-4-5" if provider == "anthropic"
                   else "gemini-2.5-flash",
            api_key=api_key,
        )
    return _default_judge


# ---------------------------------------------------------------------------
# 2-tier routing: primary scorer + LLM judge fallback / primary
# ---------------------------------------------------------------------------
#
# Two routes depending on ``task["answer_type"]``:
#
#   Route A (open-ended): judge is the PRIMARY scorer; its verdict
#     is authoritative. The deterministic scorer still runs; its
#     result is metadata only, preserved in
#     ``scorer_result.details.primary_verdict``.
#
#   Route B (MCQ / exactMatch / exactNumeric / other structured): primary
#     deterministic scorer runs first; judge is invoked as FALLBACK only
#     when primary says incorrect and candidate is non-empty. Judge can only
#     PROMOTE (incorrect -> correct), never demote.


OPEN_ANSWER_TYPES = {"opentext", "openended", "freetext"}

# Answer types where the LLM judge is the PRIMARY (authoritative) scorer.
# Deterministic scoring still runs for metadata, but judge verdict wins.
JUDGE_PRIMARY_ANSWER_TYPES = OPEN_ANSWER_TYPES


def judge_is_primary(task_or_answer_type: Any) -> bool:
    """Return True if the task should use LLM-judge as the primary scorer."""
    if isinstance(task_or_answer_type, dict):
        answer_type = str(task_or_answer_type.get("answer_type", ""))
    else:
        answer_type = str(task_or_answer_type or "")
    return answer_type.lower() in JUDGE_PRIMARY_ANSWER_TYPES


def use_judge_as_primary(task_or_answer_type: Any) -> bool:
    """Backward-compat alias for ``judge_is_primary``."""
    return judge_is_primary(task_or_answer_type)


def is_open_ended(task_or_answer_type: Any) -> bool:
    """Backward-compat alias for ``judge_is_primary``."""
    return judge_is_primary(task_or_answer_type)


DEFAULT_JUDGE_MODEL = "claude-sonnet-4-5"

def pick_judge_model(target_backbone: str = "") -> str:
    """Return the LLM-as-judge model.

    Policy: use one configured judge regardless of target backbone. A
    consistent judge minimizes target-dependent variability and keeps
    evaluation comparisons stable across runs.

    The ``target_backbone`` parameter is kept in the signature for
    backward compatibility and future policy changes but is currently
    unused.
    """
    import os
    return os.environ.get("BIOAGENT_JUDGE_MODEL", DEFAULT_JUDGE_MODEL).strip() or DEFAULT_JUDGE_MODEL


def judge_enabled() -> bool:
    """Return True unless ``BIOAGENT_LLM_JUDGE=0`` is set."""
    import os
    return os.environ.get("BIOAGENT_LLM_JUDGE", "1") != "0"


def _get_judge_for(target_backbone: str | None) -> LLMJudge:
    """Return (and cache) an ``LLMJudge`` instance for the correct
    judge model given a target backbone."""
    global _default_judge
    import os
    desired = pick_judge_model(target_backbone)
    provider, api_key = _judge_provider_and_key()
    actual_model = desired
    if (
        _default_judge is None
        or _default_judge.llm.model != actual_model
        or _default_judge.llm.provider != provider
    ):
        _default_judge = LLMJudge(
            provider=provider,
            model=actual_model,
            api_key=api_key,
        )
    return _default_judge


async def score_with_fallback(
    task: dict,
    prediction: str,
    target_backbone: str | None = None,
) -> dict:
    """Score ``prediction`` against ``task`` with 2-tier judge routing.

    Returns a dict::

        {
            "correct": bool,
            "method": str,
            "details": {
                "primary_verdict": bool,
                "primary_method": str,
                "is_open_ended": bool,
                "judge_primary": bool,
                "judge_model": str | None,
                "judge_raw": str | None,
                "judge_invoked": bool,
                "judge_verdict": bool | None,
                "judge_error": str | None,
            },
        }
    """
    from harness.eval.scoring import score_question

    answer_type = str(task.get("answer_type", ""))
    gold = task.get("answer", "")
    context = task.get("context")

    # Always run primary so details.primary_* is populated.
    try:
        primary_correct = bool(score_question(
            prediction, gold, answer_type, context,
        ))
        primary_method = f"primary:{answer_type or 'default'}"
    except Exception as exc:  # noqa: BLE001
        primary_correct = False
        primary_method = f"primary_error:{type(exc).__name__}"

    is_open = is_open_ended(task)
    judge_primary = use_judge_as_primary(task)
    judge_invoked = False
    judge_verdict: bool | None = None
    judge_model: str | None = None
    judge_raw: str | None = None
    judge_error: str | None = None
    final_correct = primary_correct
    method = primary_method

    candidate = str(prediction or "")
    can_invoke = (
        judge_enabled()
        and candidate.strip() != ""
        and (judge_primary or not primary_correct)
    )

    if can_invoke:
        try:
            judge = _get_judge_for(target_backbone)
            judge_model = judge.llm.model
            result = await judge.judge(
                question=str(task.get("question", "")),
                expected=str(gold),
                predicted=candidate,
            )
            judge_invoked = True
            if result.get("error"):
                # Judge failed (empty response, safety refusal, invalid JSON, etc.)
                # Fall back to primary scorer — do NOT override with False
                judge_error = str(result.get("reasoning", "judge_error"))[:200]
                judge_verdict = None
                method = f"{primary_method}+judge_failed"
                task_id = task.get("id", "?")
                print(f"judge failed [task={task_id}]: {judge_error}")
            else:
                judge_verdict = bool(result.get("correct"))
                judge_raw = str(result.get("reasoning", ""))[:500]
                if judge_primary:
                    # Judge is authoritative for open-ended answers only.
                    final_correct = judge_verdict
                    method = f"llm_judge_primary({judge_model})"
                elif judge_verdict:
                    # Fallback: judge can only promote (exactMatch, exactNumeric)
                    final_correct = True
                    method = f"{primary_method}+llm_judge_fallback({judge_model})"
                # else: fallback type + judge says INCORRECT -> keep primary verdict
        except Exception as exc:  # noqa: BLE001
            judge_error = f"{type(exc).__name__}: {exc}"[:200]
            task_id = task.get("id", "?")
            print(f"judge failed [task={task_id}]: {judge_error}")

    return {
        "correct": bool(final_correct),
        "method": method,
        "details": {
            "primary_verdict": primary_correct,
            "primary_method": primary_method,
            "is_open_ended": is_open,
            "judge_primary": judge_primary,
            "judge_invoked": judge_invoked,
            "judge_verdict": judge_verdict,
            "judge_model": judge_model,
            "judge_raw": judge_raw,
            "judge_error": judge_error,
        },
    }
