"""Chemistry reasoning benchmark loader.

Expert-curated, reasoning-intensive chemistry multiple-choice problems.
Text-only evaluation (images referenced in questions are not loaded).
"""

from __future__ import annotations

from typing import Any


def load_superchem_tasks(
    limit: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load chemistry reasoning questions (English, text-only).

    Args:
        limit: cap number returned
        seed: for reproducibility
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    try:
        # Use data_files to avoid encrypted PDF zip causing load failure
        ds = load_dataset(
            "ZehuaZhao/SUPERChem",
            data_files="**/*.parquet",
            split="train",
        )
    except Exception:
        return []

    tasks: list[dict[str, Any]] = []

    for row in ds:
        qid = row.get("uuid") or f"superchem_{len(tasks):04d}"
        question = row.get("question_en") or ""
        options = row.get("options_en") or {}
        answer_list = row.get("answer_en") or []
        explanation = row.get("explanation_en") or ""

        if not question or not isinstance(options, dict) or not answer_list:
            continue

        # Build MCQ: options_en is {"A": "...", "B": "...", ...}
        sorted_keys = sorted(options.keys())
        opts_block = "\n".join(
            f"{k}) {options[k]}" for k in sorted_keys if options[k]
        )

        full_question = f"{question}\n\n{opts_block}"

        # answer_en is ["B"] or ["A", "C"] for multi-select
        correct_letter = ", ".join(answer_list)

        tasks.append({
            "id": str(qid),
            "question": full_question,
            "answer": correct_letter,
            "answer_type": "multipleChoice",
            "category": "SUPERChem/chemistry",
            "context": {
                "explanation": explanation,
                "question_type": row.get("question_type", ""),
            },
        })
        if limit and len(tasks) >= limit:
            break

    return tasks
