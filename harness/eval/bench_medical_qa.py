"""Combined Medical-QA loader — MedQA + MedMCQA + PubMedQA.

These three HF datasets form the baseline "medical knowledge" triplet
for clinical LLM evaluation:

- MedQA (USMLE-style, 1273 MCQ test)
- MedMCQA (AIIMS/NEET entrance, 4183 MCQ test)
- PubMedQA (yes/no/maybe over abstracts, 500 expert-labeled test)

Single entry point `load_medical_qa_tasks(sources=...)` returns a
unified task list with a `source` metadata tag so downstream
benchmark reports can split results per sub-bench.

Each task dict:
  {id, question, choices, answer, answer_type, metadata.source, scorer_kind}

`scorer_kind` is 'mcq' for MedQA/MedMCQA and 'exact' (yes/no/maybe)
for PubMedQA.
"""

from __future__ import annotations

import warnings
from typing import Any, Iterable


def _return_unavailable(reason: str) -> list[dict[str, Any]]:
    warnings.warn(
        "medical QA official HuggingFace source unavailable; returning no "
        f"tasks instead of offline fallback data ({reason}).",
        RuntimeWarning,
        stacklevel=3,
    )
    return []


_LETTERS = ["A", "B", "C", "D", "E"]


def _normalise_medqa(ds) -> list[dict[str, Any]]:
    out = []
    for i, row in enumerate(ds):
        q = row.get("question") or row.get("sent1") or ""
        opts = row.get("options") or row.get("choices")
        if isinstance(opts, dict):
            # MedQA "options" are a dict {A: "...", B: "...", ...}
            choices = [opts[k] for k in sorted(opts)]
        else:
            choices = list(opts) if opts else None
        answer = row.get("answer") or row.get("answer_idx") or ""
        if isinstance(answer, int):
            answer = _LETTERS[answer] if 0 <= answer < 5 else str(answer)
        # MedQA 'answer' can be the choice text; convert to letter
        if choices and answer in choices:
            answer = _LETTERS[choices.index(answer)]
        out.append({
            "id": f"medqa_{i}",
            "question": q,
            "choices": choices,
            "answer": str(answer),
            "answer_type": "multipleChoice",
            "metadata": {"source": "medqa_hf", "dataset": "medqa"},
            "scorer_kind": "mcq",
        })
    return out


def _normalise_medmcqa(ds) -> list[dict[str, Any]]:
    out = []
    for i, row in enumerate(ds):
        q = row.get("question") or ""
        choices = [row.get(f"op{k}") for k in "abcd" if row.get(f"op{k}")]
        answer_idx = row.get("cop")
        answer = _LETTERS[int(answer_idx)] if answer_idx is not None else ""
        out.append({
            "id": f"medmcqa_{i}",
            "question": q,
            "choices": choices if choices else None,
            "answer": str(answer),
            "answer_type": "multipleChoice",
            "metadata": {
                "source": "medmcqa_hf", "dataset": "medmcqa",
                "subject": row.get("subject_name"),
                "topic": row.get("topic_name"),
            },
            "scorer_kind": "mcq",
        })
    return out


def _normalise_pubmedqa(ds) -> list[dict[str, Any]]:
    out = []
    for i, row in enumerate(ds):
        contexts = row.get("context") or row.get("contexts") or ""
        if isinstance(contexts, dict):
            contexts = "\n".join(contexts.get("contexts", []))
        elif isinstance(contexts, list):
            contexts = "\n".join(contexts)
        q = row.get("question") or ""
        full_q = f"Context: {contexts}\n\nQuestion: {q}" if contexts else q
        answer = row.get("final_decision") or row.get("answer") or ""
        out.append({
            "id": f"pubmedqa_{i}",
            "question": full_q,
            "choices": ["yes", "no", "maybe"],
            "answer": str(answer).lower(),
            "answer_type": "exactMatch",
            "metadata": {"source": "pubmedqa_hf", "dataset": "pubmedqa",
                          "pubid": row.get("pubid")},
            "scorer_kind": "exact",
        })
    return out


_HF_SOURCES: dict[str, tuple[str, str | None, str, Any]] = {
    "medqa": (
        "GBaker/MedQA-USMLE-4-options",
        None,
        "test",
        _normalise_medqa,
    ),
    "medmcqa": (
        "openlifescienceai/medmcqa",
        None,
        "validation",
        _normalise_medmcqa,
    ),
    "pubmedqa": (
        "qiaojin/PubMedQA",
        "pqa_labeled",
        "train",
        _normalise_pubmedqa,
    ),
}


def load_medical_qa_tasks(limit: int | None = None,
                             sources: Iterable[str] | None = None,
                             per_source_limit: int | None = None,
                             ) -> list[dict[str, Any]]:
    """Load a unified Medical-QA task list.

    Parameters
    ----------
    limit
        Total cap across sources (None = all).
    sources
        Which sub-datasets to include. Default all three: 'medqa',
        'medmcqa', 'pubmedqa'.
    per_source_limit
        Cap per sub-dataset BEFORE applying `limit`. Useful for
        balanced sampling in pilots.
    """
    wanted = list(sources) if sources else ["medqa", "medmcqa", "pubmedqa"]
    try:
        from datasets import load_dataset
    except ImportError:
        return _return_unavailable("datasets package not installed")

    all_tasks: list[dict[str, Any]] = []
    for src in wanted:
        if src not in _HF_SOURCES:
            continue
        hf_name, config, split, norm = _HF_SOURCES[src]
        try:
            if config:
                ds = load_dataset(hf_name, config, split=split)
            else:
                ds = load_dataset(hf_name, split=split)
        except Exception:
            # Sub-source failure is non-fatal — continue with the others
            continue
        sub = norm(ds)
        if per_source_limit:
            sub = sub[:per_source_limit]
        all_tasks.extend(sub)

    if not all_tasks:
        return _return_unavailable("all selected sub-sources failed")
    return all_tasks[:limit] if limit else all_tasks
