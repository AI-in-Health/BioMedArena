"""Graduate-level QA loader across multiple disciplines.

PhD-level multiple-choice questions. This loader filters to
specific fields (Biology, Pharmacy by default) for biomedical evaluation.
"""

from __future__ import annotations

from typing import Any


# Fields to include by default (Biology + Pharmacy)
DEFAULT_FIELDS = ("Biology", "Pharmacy")


def load_supergpqa_tasks(
    limit: int | None = None,
    seed: int = 42,
    fields: tuple[str, ...] | list[str] = DEFAULT_FIELDS,
) -> list[dict[str, Any]]:
    """Load graduate-level QA questions filtered by field.

    Args:
        limit: cap number returned
        seed: for reproducibility
        fields: tuple of field names to include (e.g. ("Biology", "Pharmacy"))
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    try:
        ds = load_dataset("m-a-p/SuperGPQA", split="train")
    except Exception:
        return []

    fields_set = set(fields)
    tasks: list[dict[str, Any]] = []

    for row in ds:
        field = row.get("field") or ""
        if field not in fields_set:
            continue

        qid = row.get("uuid") or f"supergpqa_{len(tasks):04d}"
        question = row.get("question") or ""
        options = row.get("options") or []
        answer_letter = row.get("answer_letter") or ""

        if not question or not options or not answer_letter:
            continue

        # Build MCQ: options is a list like ["opt A text", "opt B text", ...]
        letters = [chr(ord("A") + i) for i in range(len(options))]
        opts_block = "\n".join(
            f"{l}) {opt}" for l, opt in zip(letters, options) if opt
        )
        full_question = f"{question}\n\n{opts_block}"

        tasks.append({
            "id": str(qid),
            "question": full_question,
            "answer": answer_letter,
            "answer_type": "multipleChoice",
            "category": f"SuperGPQA/{field}/{row.get('subfield', '')}",
            "context": {
                "discipline": row.get("discipline", ""),
                "field": field,
                "subfield": row.get("subfield", ""),
                "difficulty": row.get("difficulty", ""),
                "is_calculation": row.get("is_calculation", False),
            },
        })
        if limit and len(tasks) >= limit:
            break

    return tasks
