"""SUPERChem loader.

Source: ``ZehuaZhao/SUPERChem`` (HuggingFace, 500 rows).

Bilingual (EN + ZH), all multiple-choice, all ``field=chemistry``. Up to
26 options per question (A–Z), ~47 % of tasks include images referenced
in the question text as ``<MultiModal>![...](...)`` markdown. The
dataset ships a single shared image blob pool in
``question_images``: a dict of ``path → image-bytes`` keyed on the same
paths the markdown references.

We materialise referenced images to ``data/cache/super_chemistry/images/``
lazily on first load so vision models can pick them up via
``task.context.image_paths``.

Returned schema (per task)::

    {
        "id":           str,               # uuid from the dataset
        "question":     str,               # question_{en|zh} with image refs preserved
        "answer":       str,               # single letter A–Z
        "answer_type":  "multipleChoice",
        "category":     "chemistry",
        "context":      {
            "image_paths": [str, ...],     # absolute image paths for vision
            "options":     dict[str,str],  # letter → text (None values dropped)
            "language":    "en" | "zh",
        },
        "raw_subject":  "chemistry",
    }

The question body already renders the options inline via the SUPERChem
upstream format; we additionally append a plain ``A. ... B. ...`` list
so text-only models see the choices without needing markdown parsing.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

_HF_REPO = "ZehuaZhao/SUPERChem"
_HF_PARQUET = "SUPERChem-500.parquet"
_IMAGE_DIR = Path("data/cache/super_chemistry/images")


def _format_options(options: dict[str, Any]) -> tuple[dict[str, str], str]:
    """Return (letter→text map, ``A. ...\\nB. ...\\n...`` string).

    Drops entries whose value is ``None`` or empty. Keys are kept in
    alphabetical letter order so the prompt stays deterministic.
    """
    cleaned: dict[str, str] = {}
    for letter in sorted(options):
        v = options[letter]
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        cleaned[letter] = str(v)
    formatted = "\n".join(f"{k}. {v}" for k, v in cleaned.items())
    return cleaned, formatted


def _materialise_images(
    image_blob: dict[str, bytes] | None,
    referenced_paths: Iterable[str],
) -> list[str]:
    """Write any referenced images to disk; return absolute paths.

    ``image_blob`` is the row-level ``question_images`` struct in
    SUPERChem — keyed on ``/media/uploads/<file>.{png,jpg}``. Returns an
    empty list when the question has no image references or the blob is
    missing them (benign for text-only prompts).
    """
    if not image_blob:
        return []
    _IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    out: list[str] = []
    for ref in referenced_paths:
        data = image_blob.get(ref)
        if not data:
            continue
        # Dedupe by content hash so the same image isn't rewritten every load.
        sha = hashlib.sha1(data).hexdigest()[:16]
        ext = Path(ref).suffix or ".png"
        target = _IMAGE_DIR / f"{sha}{ext}"
        if not target.exists():
            try:
                target.write_bytes(data)
            except Exception as exc:
                logger.warning("SUPERChem image write failed %s: %s", target, exc)
                continue
        out.append(str(target.resolve()))
    return out


def _extract_image_refs(text: str) -> list[str]:
    """Pull ``/media/uploads/...`` paths out of the ``<MultiModal>`` markdown."""
    import re
    return re.findall(r"\]\((/media/uploads/[^)]+)\)", text or "")


def load_super_chemistry_tasks(
    *,
    language: str = "en",
    limit: int | None = None,
    seed: int | None = None,
    include_images: bool = True,
    require_images: bool = False,
    skip_with_images: bool = False,
) -> list[dict[str, Any]]:
    """Load the SUPERChem benchmark.

    Args:
        language: ``"en"`` (default) or ``"zh"`` — selects which set of
            columns to use (``question_en`` vs ``question_zh``, etc.).
        limit: optionally cap the number of returned tasks.
        seed: when set, shuffle the rows with this RNG seed before
            applying ``limit`` so the cap picks a random subset rather
            than the dataset's natural prefix.
        include_images: when True (default), materialise referenced
            images to ``data/cache/super_chemistry/images/`` and expose them
            via ``task.context.image_paths``. Set to False for a
            pure-text workload (image markdown refs stay in the
            question body regardless).
        require_images: when True, skip image-referenced rows whose image
            blobs cannot be materialised.
        skip_with_images: when True, skip any row that references images.
    """
    if language not in ("en", "zh"):
        raise ValueError(f"language must be 'en' or 'zh' (got {language!r})")

    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
    except ImportError as exc:
        logger.warning("SUPERChem requires huggingface_hub + pyarrow: %s", exc)
        return []

    try:
        path = hf_hub_download(_HF_REPO, _HF_PARQUET, repo_type="dataset")
    except Exception as exc:
        logger.warning("SUPERChem download failed: %s", exc)
        return []

    try:
        df = pq.read_table(path).to_pandas()
    except Exception as exc:
        logger.warning("SUPERChem parquet read failed: %s", exc)
        return []

    # Seeded random shuffle for reproducible subsets. Applied before
    # ``limit`` so a seeded limit=50 always picks the same 50 tasks.
    if seed is not None:
        df = df.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)

    q_col = f"question_{language}"
    opt_col = f"options_{language}"
    ans_col = f"answer_{language}"

    tasks: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        question_text = row.get(q_col) or ""
        options_raw = row.get(opt_col) or {}
        answers = row.get(ans_col)
        # answer is a numpy array / list; SUPERChem uses single-element
        # lists for every row. Guard against the occasional multi-answer.
        if answers is None:
            continue
        try:
            ans_letter = str(answers[0]).strip().upper()
        except (IndexError, TypeError):
            continue
        if not ans_letter or not ans_letter[0].isalpha():
            continue

        opt_map, opt_block = _format_options(dict(options_raw))
        if not opt_map:
            continue

        # Append the plain A./B./... option list so text-only models see
        # the choices even when the upstream body renders them inline.
        full_question = (
            f"{question_text}\n\n"
            f"Options:\n{opt_block}"
        )

        refs = _extract_image_refs(question_text)
        if skip_with_images and refs:
            continue

        image_paths: list[str] = []
        if include_images:
            if refs:
                blob = row.get("question_images")
                # question_images is a struct (pandas dict); may be None.
                image_paths = _materialise_images(blob, refs)
        if require_images and refs and len(image_paths) < len(refs):
            continue

        tasks.append({
            "id": str(row.get("uuid") or len(tasks)),
            "question": full_question,
            "answer": ans_letter[0],   # first letter only; dataset guarantees single-letter
            "answer_type": "multipleChoice",
            "category": "chemistry",
            "raw_subject": "chemistry",
            "context": {
                "image_paths": image_paths,
                "options": opt_map,
                "language": language,
                "n_options": len(opt_map),
            },
        })
        if limit and len(tasks) >= limit:
            break

    logger.info(
        "SUPERChem loaded %d tasks (language=%s, images materialised for %d)",
        len(tasks), language,
        sum(1 for t in tasks if t["context"]["image_paths"]),
    )
    return tasks
