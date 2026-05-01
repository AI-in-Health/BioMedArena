"""Expert-level medical reasoning benchmark loader.

Text subset: test questions across multiple specialties and body systems.
Tasks: Diagnosis, Treatment, Basic Science.
Multiple choice with up to 10 options (A-J).
"""

from __future__ import annotations

import warnings
from typing import Any


# Silent-fallback guard (Task B.1).
_WARN_SILENT_FALLBACK = True


def _return_fallback(limit):
    if _WARN_SILENT_FALLBACK:
        n = len(_FALLBACK_TASKS[:limit]) if limit else len(_FALLBACK_TASKS)
        warnings.warn(
            f"medxpertqa loader returned fallback data ({n} tasks). "
            "Real dataset unavailable — exclude from matrix or fix HF path.",
            RuntimeWarning, stacklevel=3,
        )
    return _FALLBACK_TASKS[:limit] if limit else list(_FALLBACK_TASKS)


def load_medxpertqa_tasks(limit: int | None = None, subset: str = "Text") -> list[dict[str, Any]]:
    """Load expert medical QA text test subset."""
    try:
        from datasets import load_dataset
    except ImportError:
        return _return_fallback(limit)

    try:
        ds = load_dataset("TsinghuaC3I/MedXpertQA", subset, split="test")
    except Exception:
        return _return_fallback(limit)

    tasks = []
    for row in ds:
        qid = row.get("id", f"mxq_{len(tasks):04d}")
        q = row.get("question", "")
        options = row.get("options", {}) or {}
        label = row.get("label", "")
        task_type = row.get("medical_task", "")
        body_system = row.get("body_system", "")
        qtype = row.get("question_type", "")

        # Format options into question text
        opts_text = "\n".join(f"{k}) {v}" for k, v in sorted(options.items()))
        full_question = f"{q}\n\n{opts_text}"

        tasks.append({
            "id": qid,
            "question": full_question,
            "answer": label.upper().strip(),
            "answer_type": "multipleChoice",
            "category": f"MedXpert/{task_type}/{body_system}",
            "raw_subject": f"{task_type}|{qtype}",
            "context": {"options": options, "task": task_type, "body_system": body_system},
        })
        if limit and len(tasks) >= limit:
            break

    return tasks


_FALLBACK_TASKS: list[dict[str, Any]] = [
    {
        "id": "mxq_fb_001", "category": "MedXpert/Diagnosis/General",
        "answer_type": "multipleChoice",
        "question": "A 65-year-old presents with sudden severe chest pain radiating to back. BP right arm 180/95, left arm 140/80. What is the most likely diagnosis?\nA) Acute MI\nB) PE\nC) Aortic dissection\nD) Pneumothorax\nE) Pericarditis",
        "answer": "C",
    },
]
