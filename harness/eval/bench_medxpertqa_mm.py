"""MedXpertQA MM (multi-modal / VQA) loader.

HuggingFace: TsinghuaC3I/MedXpertQA, subset "MM"
Paper: https://huggingface.co/papers/2501.18362

2000 visual question-answering tasks with:
- question text
- options (A-J)
- label (correct letter)
- images (list of image filenames, stored in data/medxpertqa_mm/images/)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


# Default location where images.zip was extracted
DEFAULT_IMAGE_DIR = Path("data/medxpertqa_mm/images")


def load_medxpertqa_mm_tasks(
    limit: int | None = None,
    image_dir: str | Path = DEFAULT_IMAGE_DIR,
    require_images: bool = True,
) -> list[dict[str, Any]]:
    """Load MedXpertQA MM test subset with image paths resolved."""
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    try:
        ds = load_dataset("TsinghuaC3I/MedXpertQA", "MM", split="test")
    except Exception:
        return []

    img_dir = Path(image_dir)
    tasks = []
    for row in ds:
        qid = row.get("id", f"mm_{len(tasks):04d}")
        q = row.get("question", "")
        options = row.get("options", {}) or {}
        label = row.get("label", "")
        images = row.get("images", []) or []
        task_type = row.get("medical_task", "")
        body_system = row.get("body_system", "")
        qtype = row.get("question_type", "")

        # Resolve image paths
        image_paths = []
        for img_name in images:
            p = img_dir / img_name
            if p.exists():
                image_paths.append(str(p))
        if require_images and not image_paths:
            continue

        # Format the question text with options
        opts_text = "\n".join(f"{k}) {v}" for k, v in sorted(options.items()))
        full_question = f"{q}\n\n{opts_text}"

        tasks.append({
            "id": qid,
            "question": full_question,
            "answer": label.upper().strip(),
            "answer_type": "multipleChoice",
            "category": f"VQA/{task_type}/{body_system}",
            "raw_subject": f"{task_type}|{qtype}",
            "context": {
                "options": options,
                "task": task_type,
                "body_system": body_system,
                "image_paths": image_paths,
                "input_type": "image+text",
                "multimodal": True,
                "source": "https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA",
                "image_cache_dir": str(img_dir),
            },
        })
        if limit and len(tasks) >= limit:
            break

    return tasks
