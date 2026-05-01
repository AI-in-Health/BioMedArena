"""Biomedical QA benchmark loader.

Uses a parquet-formatted, RAG-friendly mirror: each question carries a
free-text answer and the IDs of relevant PubMed passages.

Suited for PubMed retrieval evaluation.
"""

from __future__ import annotations

from typing import Any


def load_bioasq_tasks(
    limit: int | None = None,
    question_types: list[str] | None = None,  # kept for API compat (unused with this mirror)
    split: str = "test",
) -> list[dict[str, Any]]:
    """Load biomedical QA tasks."""
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    try:
        ds = load_dataset(
            "enelpol/rag-mini-bioasq",
            "question-answer-passages",
            split=split,
        )
    except Exception:
        return []

    tasks: list[dict[str, Any]] = []
    for row in ds:
        qid = row.get("id", f"bioasq_{len(tasks):04d}")
        question = row.get("question", "")
        answer = row.get("answer", "")
        passage_ids = row.get("relevant_passage_ids", []) or []

        if not (question and answer):
            continue

        # All answers are free-text, score with LLM-judge for fairness
        tasks.append({
            "id": str(qid),
            "question": question,
            "answer": str(answer),
            "answer_type": "openText",
            "category": "BioASQ",
            "context": {
                "relevant_pmids": passage_ids,
            },
        })
        if limit and len(tasks) >= limit:
            break

    return tasks
