"""Inspect-AI bidirectional compatibility skeleton.

Goal: bidirectional compatibility with the UK AISI Inspect-AI ecosystem.
The scaffold below supports round-trip conversion of the most common
task shape (dataset + solver + scorer → our BenchmarkSuite task dict,
and back). Advanced features (subtasks, multi-turn eval, sample
metadata propagation, streaming scorer output) raise
``NotImplementedError`` — to be completed in a future pass.

Why not install inspect-ai in main venv
---------------------------------------
inspect-ai's click / fsspec pins would downgrade our versions; Gate 3
triggered. We therefore use fully lazy imports — the converter works
in two modes:

  (a) If inspect_ai IS installed (via uvx / separate env), we bind to
      the real types and the converter is fully exercised.
  (b) If it's not (the default in CI), type hints fall back to `Any`
      and the converter operates on dict-shaped inspect-task specs,
      which is what users typically hand-author anyway.

Supported conversions (scaffold, to be extended)
----------------------------------------------------
    from_inspect_task(inspect_task_or_dict)
        -> dict (our BenchmarkSuite task format: {question, choices,
        answer, metadata, scorer_kind})
    to_inspect_task(our_task)
        -> dict (inspect-ai Task spec shape, importable into inspect-ai)
    scorer_to_metric(inspect_scorer_or_name)
        -> callable (str pred, str gold) -> float compatible with our
        harness/eval/metrics.

End-to-end MCQ example lives in tests/smoke/test_inspect_compat.py.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Availability probe
# ---------------------------------------------------------------------------


def inspect_ai_available() -> bool:
    """True if `inspect_ai` can be imported right now."""
    try:
        import inspect_ai  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Inspect-AI → our harness
# ---------------------------------------------------------------------------


def _sample_to_harness_dict(sample: Any) -> dict[str, Any]:
    """Convert a single Inspect-AI Sample (or dict-shaped sample) to
    our BenchmarkSuite per-question dict.

    Our format:
        {
            "question": str,
            "choices": list[str] | None,
            "answer":  str,
            "metadata": dict,
            "scorer_kind": "mcq" | "exact" | "numeric" | "judged"
        }
    """
    # Duck-typed accessor so both real Sample objects and plain dicts work
    def _get(s, key, default=None):
        if isinstance(s, dict):
            return s.get(key, default)
        return getattr(s, key, default)

    question = _get(sample, "input") or _get(sample, "question") or ""
    if isinstance(question, list):
        # Inspect's `input` can be chat-message list — flatten to text
        parts = []
        for m in question:
            c = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            if isinstance(c, str):
                parts.append(c)
        question = "\n".join(parts)

    choices = _get(sample, "choices") or _get(sample, "options")
    answer = _get(sample, "target") or _get(sample, "answer") or ""
    metadata = _get(sample, "metadata") or {}

    # Heuristic scorer hint
    if choices:
        scorer_kind = "mcq"
    elif isinstance(answer, (int, float)) or (
        isinstance(answer, str) and answer.replace(".", "", 1).replace("-", "", 1).isdigit()
    ):
        scorer_kind = "numeric"
    else:
        scorer_kind = "exact"

    return {
        "question": str(question),
        "choices": list(choices) if choices else None,
        "answer": str(answer),
        "metadata": dict(metadata) if metadata else {},
        "scorer_kind": scorer_kind,
    }


def from_inspect_task(inspect_task_or_dict: Any) -> list[dict[str, Any]]:
    """Convert an Inspect-AI Task (object or dict) into a list of
    BenchmarkSuite-formatted per-question dicts.

    Supports:
        - dataset: list[Sample] | iterable
        - dict-shaped tasks with 'dataset' key

    Raises NotImplementedError for features we haven't implemented yet
    (subtasks, solver chains beyond single-step, multi-turn dialog
    evaluation, custom datasource loaders).
    """
    task = inspect_task_or_dict
    dataset = None
    if isinstance(task, dict):
        dataset = task.get("dataset") or task.get("samples")
        if task.get("subtasks"):
            raise NotImplementedError(
                "Inspect-AI subtasks conversion: not implemented"
            )
        if task.get("solver") and isinstance(task["solver"], list) and len(task["solver"]) > 1:
            logger.info(
                "Inspect-AI multi-step solver detected — keeping first step only "
                "in the skeleton converter; full chain support is a future addition."
            )
    else:
        # Real inspect_ai.Task object — probe attributes
        dataset = getattr(task, "dataset", None) or getattr(task, "samples", None)
        if getattr(task, "subtasks", None):
            raise NotImplementedError(
                "Inspect-AI subtasks conversion: not implemented"
            )

    if dataset is None:
        raise ValueError("Inspect-AI task has no dataset/samples field")

    samples: Iterable[Any]
    if hasattr(dataset, "samples"):
        samples = dataset.samples  # MemoryDataset / CSVDataset style
    elif isinstance(dataset, (list, tuple)):
        samples = dataset
    elif hasattr(dataset, "__iter__"):
        samples = dataset
    else:
        raise ValueError(
            f"Cannot extract samples from dataset of type {type(dataset).__name__}"
        )

    return [_sample_to_harness_dict(s) for s in samples]


# ---------------------------------------------------------------------------
# Our harness → Inspect-AI
# ---------------------------------------------------------------------------


def _harness_dict_to_inspect_sample(q: dict[str, Any]) -> dict[str, Any]:
    """Produce a dict shape that inspect-ai.dataset.Sample(**d)
    accepts."""
    out: dict[str, Any] = {
        "input": q.get("question", ""),
        "target": q.get("answer", ""),
    }
    if q.get("choices"):
        out["choices"] = list(q["choices"])
    if q.get("metadata"):
        out["metadata"] = dict(q["metadata"])
    return out


def to_inspect_task(harness_tasks: list[dict[str, Any]] | dict[str, Any],
                       name: str = "harness_export",
                       scorer: str = "exact") -> dict[str, Any]:
    """Produce an Inspect-AI-compatible Task spec dict from our harness
    task list.

    Scorer argument is a string name (`exact`, `match`, `mcq`) — the
    caller can resolve it to a real Inspect-AI scorer via
    `scorer_to_metric` or import directly from `inspect_ai.scorer`.
    """
    if isinstance(harness_tasks, dict):
        harness_tasks = [harness_tasks]
    samples = [_harness_dict_to_inspect_sample(q) for q in harness_tasks]
    return {
        "name": name,
        "dataset": samples,
        "solver": "generate()",  # placeholder; caller overrides
        "scorer": scorer,
    }


# ---------------------------------------------------------------------------
# Scorer bridge
# ---------------------------------------------------------------------------


# Map inspect-ai scorer names to our (pred, gold) -> float functions.
# We implement a minimum useful set here; callers can extend via
# `register_scorer(name, fn)`.
_SCORER_REGISTRY: dict[str, Callable[[str, str], float]] = {}


def register_scorer(name: str, fn: Callable[[str, str], float]) -> None:
    _SCORER_REGISTRY[name] = fn


def _exact_match(pred: str, gold: str) -> float:
    return 1.0 if (pred or "").strip().lower() == (gold or "").strip().lower() else 0.0


def _mcq_match(pred: str, gold: str) -> float:
    """Extract the final MCQ letter choice from a prediction.

    Precedence (most to least reliable):
      1. "The answer is [X]" / "answer: X" / "answer is X" at end.
      2. Last standalone A-E letter in the text.
      3. First char of the prediction.
    """
    import re
    p = (pred or "").upper().strip()
    # Strong signal: "answer is <X>" or "answer: <X>"
    m = re.search(r"ANSWER\s*(?:IS|:)?\s*[\[\(]?([A-E])[\]\)]?", p)
    if m:
        pred_letter = m.group(1)
    else:
        # Fall back to the LAST standalone A-E letter (LLMs usually end
        # with their choice rather than mention a wrong one last).
        matches = re.findall(r"\b([A-E])\b", p)
        if matches:
            pred_letter = matches[-1]
        else:
            pred_letter = p[:1]
    gold_letter = (gold or "").strip().upper()[:1]
    return 1.0 if pred_letter == gold_letter else 0.0


def _includes(pred: str, gold: str) -> float:
    return 1.0 if (gold or "").strip().lower() in (pred or "").strip().lower() else 0.0


register_scorer("exact", _exact_match)
register_scorer("match", _exact_match)
register_scorer("mcq", _mcq_match)
register_scorer("choice", _mcq_match)
register_scorer("includes", _includes)


def scorer_to_metric(inspect_scorer: Any) -> Callable[[str, str], float]:
    """Resolve an inspect-ai scorer (or scorer name) to a plain
    `(pred, gold) -> float` metric function usable by our BenchmarkSuite.
    """
    # String → registry lookup
    if isinstance(inspect_scorer, str):
        fn = _SCORER_REGISTRY.get(inspect_scorer.lower())
        if fn is None:
            raise NotImplementedError(
                f"scorer '{inspect_scorer}' not in skeleton registry; "
                f"known: {sorted(_SCORER_REGISTRY)}"
            )
        return fn

    # Callable already — treat as a drop-in metric if signature matches
    if callable(inspect_scorer):
        return inspect_scorer

    # Real inspect_ai.scorer.Scorer instances — return a wrapper that
    # extracts `.value` from the Score object. Only works if inspect_ai
    # is installed at call time.
    if inspect_ai_available():
        try:
            from inspect_ai.scorer import Score  # noqa: F401
        except Exception:
            Score = None  # type: ignore
        else:
            def _wrapped(pred: str, gold: str) -> float:
                # Inspect scorers take TaskState + Target, not raw strings.
                # A full bridge requires TaskState construction; not
                # implemented yet.
                raise NotImplementedError(
                    "Real inspect_ai.Scorer bridging requires TaskState "
                    "construction — not implemented."
                )
            return _wrapped

    raise TypeError(
        f"Cannot convert scorer of type {type(inspect_scorer).__name__} to metric"
    )


# ---------------------------------------------------------------------------
# Round-trip helper (used by the smoke test)
# ---------------------------------------------------------------------------


def round_trip(harness_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert our format → inspect dict → our format again. The
    returned list must be data-equivalent to the input modulo dict
    ordering."""
    inspect_spec = to_inspect_task(harness_tasks)
    return from_inspect_task(inspect_spec)
