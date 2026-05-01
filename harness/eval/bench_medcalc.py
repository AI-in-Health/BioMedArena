"""Clinical calculation benchmark loader.

Test instances over multiple calculators. Each has:
- Patient Note (clinical vignette)
- Question (what to compute)
- Ground Truth Answer (numeric)
- Lower/Upper Limit (acceptance range, typically +/-5%)
- Relevant Entities (structured inputs)
- Ground Truth Explanation
"""

from __future__ import annotations

import os
import warnings
from typing import Any

_HF_REPO = "ncbi/MedCalc-Bench-v1.2"


def _return_unavailable(reason: str) -> list[dict[str, Any]]:
    warnings.warn(
        "medcalc real dataset unavailable; returning no tasks. "
        f"Reason: {reason}",
        RuntimeWarning,
        stacklevel=3,
    )
    return []


def load_medcalc_tasks(
    limit: int | None = None,
    split: str = "test",
    revision: str | None = None,
    require_online: bool = True,
) -> list[dict[str, Any]]:
    """Load clinical calculation test tasks."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        return _return_unavailable(str(exc))

    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    source_info = {"revision": revision or "main", "sha": None}
    if require_online:
        from harness.eval.hf_source import verify_hf_dataset_online
        verified = verify_hf_dataset_online(
            _HF_REPO,
            token=hf_token,
            revision=revision,
        )
        if verified is None:
            return _return_unavailable(
                f"upstream {_HF_REPO} could not be verified online"
            )
        source_info = verified

    try:
        kwargs: dict[str, Any] = {"split": split}
        if hf_token:
            kwargs["token"] = hf_token
        if revision:
            kwargs["revision"] = revision
        ds = load_dataset(_HF_REPO, **kwargs)
    except Exception as exc:
        return _return_unavailable(str(exc))

    tasks = []
    for row in ds:
        row_idx = row.get("Row Number", len(tasks))
        try:
            row_idx_fmt = f"{int(row_idx):04d}"
        except (TypeError, ValueError):
            row_idx_fmt = str(row_idx)
        calc_id = row.get("Calculator ID", "")
        calc_name = row.get("Calculator Name", "")
        category = row.get("Category", "")
        patient_note = row.get("Patient Note", "")
        question_text = row.get("Question", "")
        gt_answer = str(row.get("Ground Truth Answer", ""))
        lower = row.get("Lower Limit", gt_answer)
        upper = row.get("Upper Limit", gt_answer)
        relevant = row.get("Relevant Entities", "{}")

        # Combine patient note + question
        full_question = f"{patient_note}\n\nQuestion: {question_text}"

        tasks.append({
            "id": f"mcb_{row_idx_fmt}",
            "question": full_question,
            "answer": gt_answer,
            "answer_type": "exactNumeric",
            "category": f"MedCalc/{category}",
            "context": {
                "calculator_id": calc_id,
                "calculator_name": calc_name,
                "relevant_entities": relevant,
                "lower_limit": str(lower),
                "upper_limit": str(upper),
                "source": _HF_REPO,
                "split": split,
                "revision": source_info["revision"],
                "source_sha": source_info["sha"],
            },
            "raw_subject": calc_name,
        })
        if limit and len(tasks) >= limit:
            break

    return tasks
