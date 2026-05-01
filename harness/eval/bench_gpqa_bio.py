"""GPQA loader — all GPQA questions by default.

Source: Idavidrein/gpqa
HuggingFace: https://huggingface.co/datasets/Idavidrein/gpqa
License: CC BY 4.0

GPQA is a graduate-level multiple-choice benchmark covering biology, physics,
and chemistry. The default uses gpqa_main, the 448-question official main set
available from the current Hugging Face dataset. gpqa_diamond contains the
198-question highest-quality subset.
"""

from __future__ import annotations

import random
from typing import Any


def load_gpqa_bio_tasks(
    limit: int | None = None,
    variant: str = "gpqa_main",
    seed: int = 42,
    domain_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Load GPQA questions.

    Args:
        limit: cap number returned
        variant: 'gpqa_main' | 'gpqa_diamond'
        seed: for option-shuffling reproducibility
        domain_filter: optional High-level domain substring filter
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    variants = [variant]
    if variant in {"gpqa_extended", "gpqa_experts"}:
        variants.append("gpqa_main")

    ds = None
    last_exc: Exception | None = None
    for candidate in variants:
        try:
            ds = load_dataset("Idavidrein/gpqa", candidate, split="train")
            variant = candidate
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    if ds is None:
        raise RuntimeError(
            "Unable to load Idavidrein/gpqa. Accept the dataset terms on "
            "Hugging Face or check network/cache availability."
        ) from last_exc

    rng = random.Random(seed)
    tasks: list[dict[str, Any]] = []

    for row in ds:
        high_domain = row.get("High-level domain") or row.get("high_level_domain") or ""
        if domain_filter and domain_filter.lower() not in high_domain.lower():
            continue

        qid = row.get("Record ID") or row.get("id") or f"gpqa_{len(tasks):04d}"
        question = row.get("Question") or ""
        correct = row.get("Correct Answer") or ""
        incorrect = [
            row.get("Incorrect Answer 1", ""),
            row.get("Incorrect Answer 2", ""),
            row.get("Incorrect Answer 3", ""),
        ]
        incorrect = [i for i in incorrect if i]
        subdomain = row.get("Subdomain") or ""

        if not (question and correct and incorrect):
            continue

        # Shuffle options for a fair MCQ
        options = [correct] + incorrect
        rng2 = random.Random(f"{seed}_{qid}")
        rng2.shuffle(options)
        correct_idx = options.index(correct)
        letters = ["A", "B", "C", "D", "E"][: len(options)]
        correct_letter = letters[correct_idx]

        opts_block = "\n".join(f"{l}) {opt}" for l, opt in zip(letters, options))
        full_question = f"{question}\n\n{opts_block}"

        tasks.append({
            "id": str(qid),
            "question": full_question,
            "answer": correct_letter,
            "answer_type": "multipleChoice",
            "category": f"GPQA/{high_domain or 'Unknown'}/{subdomain or 'Unknown'}",
            "raw_subject": subdomain,
            "context": {
                "high_level_domain": high_domain,
                "subdomain": subdomain,
                "correct_text": correct,
                "variant": variant,
            },
        })
        if limit and len(tasks) >= limit:
            break

    return tasks
