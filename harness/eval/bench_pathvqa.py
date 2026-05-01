"""PathVQA loader — pathology visual question answering.

Source: flaviagiammarino/path-vqa
HuggingFace: https://huggingface.co/datasets/flaviagiammarino/path-vqa
License: MIT

32,795 questions on pathology images (H&E stained slides, IHC, etc).
Demonstrates the harness's agentic pipeline advantage on multimodal:
    VLM sees image → retrieval augments with pathology knowledge → reasoning

Best medical VLMs reach only ~30-35% on PathVQA — plenty of room for agents.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


DEFAULT_IMAGE_CACHE = Path("data/cache/pathvqa_images")


def _save_pil_image(img, out_dir: Path, qid: str) -> str | None:
    """Save a PIL Image to disk and return the path."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Hash the image bytes for dedup
        import io
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = buf.getvalue()
        h = hashlib.sha256(data).hexdigest()[:16]
        out_path = out_dir / f"{h}.png"
        if not out_path.exists():
            out_path.write_bytes(data)
        return str(out_path)
    except Exception:
        return None


def load_pathvqa_tasks(
    limit: int | None = None,
    split: str = "test",
    image_cache_dir: str | Path = DEFAULT_IMAGE_CACHE,
    include_open_ended: bool = False,
) -> list[dict[str, Any]]:
    """Load PathVQA tasks.

    Args:
        limit: cap number returned
        split: 'train' | 'validation' | 'test' (default)
        image_cache_dir: where to materialize image files for the LLM
        include_open_ended: if False, only closed-ended (yes/no) are kept
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    try:
        ds = load_dataset("flaviagiammarino/path-vqa", split=split)
    except Exception:
        return []

    cache_dir = Path(image_cache_dir)
    tasks: list[dict[str, Any]] = []

    for i, row in enumerate(ds):
        question = row.get("question", "")
        answer = str(row.get("answer", "")).strip()
        image = row.get("image")

        if not question or not answer or image is None:
            continue

        # PathVQA has both open-ended and closed-ended (yes/no) questions.
        # Filter to closed-ended for reliable scoring unless opted in.
        is_closed = answer.lower() in {"yes", "no"}
        if not include_open_ended and not is_closed:
            continue

        img_path = _save_pil_image(image, cache_dir, str(i))
        if img_path is None:
            continue

        tasks.append({
            "id": f"pvq_{i:05d}",
            "question": question,
            "answer": answer.lower() if is_closed else answer,
            "answer_type": "exactMatch" if is_closed else "openText",
            "category": "PathVQA/" + ("closed" if is_closed else "open"),
            "context": {
                "image_paths": [img_path],
                "input_type": "image+text",
                "multimodal": True,
                "source": "https://huggingface.co/datasets/flaviagiammarino/path-vqa",
                "image_cache_dir": str(cache_dir),
            },
        })
        if limit and len(tasks) >= limit:
            break

    return tasks
