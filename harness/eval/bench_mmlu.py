"""MMLU loader.

Source: ``cais/mmlu`` (HuggingFace, MIT).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

_HF_REPO = "cais/mmlu"
_LETTERS = ["A", "B", "C", "D"]
_DEFAULT_MEDICAL_SUBJECTS = [
    "clinical_knowledge",
    "college_medicine",
    "professional_medicine",
    "medical_genetics",
    "anatomy",
    "college_biology",
    "nutrition",
    "virology",
]


def _format_mcq(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"{_LETTERS[i]}. {choice}" for i, choice in enumerate(choices))
    return f"{question}\n\nOptions:\n{opts}"


def _normalise_subjects(subjects: Iterable[str] | str | None) -> list[str]:
    if subjects is None or subjects == "medical":
        return list(_DEFAULT_MEDICAL_SUBJECTS)
    if subjects == "all":
        # Keep this list explicit so MMLU runs are reproducible and do not
        # depend on remote config listing APIs.
        return [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics", "college_medicine",
            "college_physics", "computer_security", "conceptual_physics",
            "econometrics", "electrical_engineering", "elementary_mathematics",
            "formal_logic", "global_facts", "high_school_biology",
            "high_school_chemistry", "high_school_computer_science",
            "high_school_european_history", "high_school_geography",
            "high_school_government_and_politics", "high_school_macroeconomics",
            "high_school_mathematics", "high_school_microeconomics",
            "high_school_physics", "high_school_psychology",
            "high_school_statistics", "high_school_us_history",
            "high_school_world_history", "human_aging", "human_sexuality",
            "international_law", "jurisprudence", "logical_fallacies",
            "machine_learning", "management", "marketing", "medical_genetics",
            "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
            "philosophy", "prehistory", "professional_accounting",
            "professional_law", "professional_medicine",
            "professional_psychology", "public_relations", "security_studies",
            "sociology", "us_foreign_policy", "virology", "world_religions",
        ]
    if isinstance(subjects, str):
        return [s.strip() for s in subjects.split(",") if s.strip()]
    return [str(s).strip() for s in subjects if str(s).strip()]


def load_mmlu_tasks(
    *,
    limit: int | None = None,
    subjects: Iterable[str] | str | None = "medical",
    split: str = "test",
    cache_dir: str | Path = "data/cache/huggingface",
) -> list[dict[str, Any]]:
    """Load MMLU tasks.

    Defaults to the medically relevant MMLU subjects for this biomedical
    harness. Pass ``subjects="all"`` to sweep the full benchmark.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        logger.warning("MMLU requires datasets: %s", exc)
        return []

    tasks: list[dict[str, Any]] = []
    for subject in _normalise_subjects(subjects):
        try:
            ds = load_dataset(_HF_REPO, subject, split=split, cache_dir=str(cache_dir))
        except Exception as exc:
            logger.warning("MMLU subject %s failed: %s", subject, exc)
            continue
        for i, row in enumerate(ds):
            choices = [str(c) for c in (row.get("choices") or [])]
            if not choices:
                continue
            answer_idx = row.get("answer")
            try:
                answer = _LETTERS[int(answer_idx)]
            except Exception:
                answer = str(answer_idx)
            question = str(row.get("question") or "")
            tasks.append({
                "id": f"mmlu_{subject}_{i}",
                "question": _format_mcq(question, choices),
                "choices": choices,
                "answer": answer,
                "answer_type": "multipleChoice",
                "category": f"MMLU/{subject}",
                "raw_subject": subject,
                "context": {
                    "source": _HF_REPO,
                    "subject": subject,
                    "split": split,
                },
                "metadata": {
                    "source": "mmlu_hf",
                    "subject": subject,
                    "split": split,
                },
                "scorer_kind": "mcq",
            })
            if limit and len(tasks) >= limit:
                return tasks
    return tasks
