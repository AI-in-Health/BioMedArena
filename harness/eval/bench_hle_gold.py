"""Expert bio+chem QA loader.

Expert-verified biology + chemistry questions that survived a human-expert
re-review for ground-truth correctness.

Schema:
    id              str     question UUID
    question        str     full question text (MCQ options inline, no
                            separate `choices` list)
    image, image_preview    (always empty on train split)
    answer          str     gold (single letter A/B/C/D/E for
                            multipleChoice, or free-text exactMatch
                            for exactMatch)
    answer_type     str     'multipleChoice' | 'exactMatch'
    author_name     str
    rationale       str     expert-written solution (~1-3 paragraphs)
    raw_subject     str     granular subject (e.g. 'Bioinformatics')
    category        str     'Biology/Medicine' or 'Chemistry'
    canary          str     watermark; passed through for scorer audits

All rows are biomed+chem; no additional subject filter is strictly
necessary. We still expose `include_chemistry` for scope control.
"""
from __future__ import annotations

import os
import warnings
from typing import Any, Optional


def load_hle_gold_tasks(
    limit: Optional[int] = None,
    include_chemistry: bool = True,
    include_mcq_only: bool = False,
) -> list[dict[str, Any]]:
    """Load expert biology + chemistry tasks.

    Args:
        limit: Max tasks to return after filtering. None = all available.
        include_chemistry: When False, keep only Biology/Medicine rows.
        include_mcq_only: When True, keep only rows with
            ``answer_type == "multipleChoice"``.

    Returns:
        List of normalized task dicts with keys:
            id, question, answer, answer_type, category, raw_subject,
            context, scorer_kind, scorer_params.
    """
    from datasets import load_dataset

    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN required for HLE-Gold (gated dataset). "
            "Set HF_TOKEN in .env and accept terms at "
            "https://huggingface.co/datasets/futurehouse/hle-gold-bio-chem"
        )

    ds = load_dataset(
        "futurehouse/hle-gold-bio-chem",
        split="train",
        token=hf_token,
    )

    tasks: list[dict[str, Any]] = []
    for row in ds:
        category = str(row.get("category") or "").strip()
        cat_lower = category.lower()
        is_biomed = "biology" in cat_lower or "medicine" in cat_lower
        is_chem = "chemistry" in cat_lower

        # Subject filter: always include biomed; chemistry is opt-out.
        if not is_biomed and not (include_chemistry and is_chem):
            continue

        answer_type_raw = str(row.get("answer_type") or "").strip()
        is_mcq = answer_type_raw == "multipleChoice"
        if include_mcq_only and not is_mcq:
            continue

        task_id = row.get("id") or f"hle_gold_{len(tasks)}"
        question = str(row.get("question") or "")
        gold = str(row.get("answer") or "")

        # Normalize answer_type to the harness's two accepted values.
        # - multipleChoice stays the same (scorer treats as letter-match).
        # - exactMatch -> openText so the generic open-text scorer runs;
        #   store the exact gold in scorer_params.ground_truth and flag
        #   llm_judge so Brief-E-time grading can upgrade the signal.
        answer_type = "multipleChoice" if is_mcq else "openText"
        scorer_kind = "mcq_exact" if is_mcq else "llm_judge"

        task = {
            "id": task_id,
            "question": question,
            "answer": gold,
            "answer_type": answer_type,
            "category": "HLE-Gold",
            "raw_subject": str(row.get("raw_subject") or ""),
            "context": {
                "source": "futurehouse/hle-gold-bio-chem",
                "category": category,
                "raw_subject": row.get("raw_subject"),
                "is_biomed": is_biomed,
                "is_chemistry": is_chem,
                "author_name": row.get("author_name"),
                # Keep the first ~500 chars of the expert rationale so
                # an LLM-judge pass can leverage it. Full text is too
                # large for per-call context.
                "rationale": (str(row.get("rationale") or "")[:500]
                              if row.get("rationale") else None),
                "hle_answer_type": answer_type_raw,
                "canary": row.get("canary"),
            },
            "scorer_kind": scorer_kind,
            "scorer_params": {
                "ground_truth": gold,
                "answer_type": answer_type_raw,
            },
        }
        tasks.append(task)

    if limit is not None:
        tasks = tasks[:limit]

    if not tasks:
        warnings.warn(
            "HLE-Gold loader returned 0 tasks — check include_chemistry / "
            "include_mcq_only flags.",
            RuntimeWarning,
            stacklevel=2,
        )

    return tasks
