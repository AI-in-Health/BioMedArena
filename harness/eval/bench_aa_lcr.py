"""AA-LCR loader.

Source: ``ArtificialAnalysis/AA-LCR`` (HuggingFace).

Artificial Analysis Long Context Reasoning contains 100 text-only
questions that require reasoning over multiple real-world documents. The
dataset publishes a CSV with question/answer metadata and a zip archive
containing extracted document text. This loader downloads both assets and
injects the selected document set into each task context.
"""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HF_REPO = "ArtificialAnalysis/AA-LCR"
_HF_CSV = "AA-LCR_Dataset.csv"
_TEXT_ZIP = "extracted_text/AA-LCR_extracted-text.zip"


def _split_semicolon_field(value: Any) -> list[str]:
    """Split HF CSV semicolon fields into clean non-empty strings."""
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def _category_path(value: Any) -> str:
    """Normalize category labels to the extracted_text directory names."""
    return str(value or "").strip().replace(" ", "_")


def _read_document_set(
    zip_path: str | Path,
    *,
    document_category: str,
    document_set_id: str,
    filenames: list[str],
) -> tuple[list[dict[str, str]], list[str]]:
    """Read a question's referenced document files from the AA-LCR zip.

    Returns ``(documents, missing_filenames)`` where each document dict has
    ``filename`` and ``text`` keys. Missing files are reported in context
    instead of raising so a partial upstream packaging issue does not crash
    an entire matrix run.
    """
    base = f"lcr/{_category_path(document_category)}/{document_set_id}/"
    documents: list[dict[str, str]] = []
    missing: list[str] = []

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for filename in filenames:
            member = base + filename
            if member not in names:
                missing.append(filename)
                continue
            text = zf.read(member).decode("utf-8", errors="replace").strip()
            documents.append({"filename": filename, "text": text})

    return documents, missing


def _load_remote_assets(
    *,
    split: str,
    cache_dir: Path,
    include_documents: bool,
) -> tuple[Any, str | None, str]:
    """Load AA-LCR from HF after confirming the remote repo is reachable."""
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi().repo_info(_HF_REPO, repo_type="dataset")
    revision = str(info.sha)
    ds = load_dataset(
        _HF_REPO,
        split=split,
        revision=revision,
        cache_dir=str(cache_dir),
    )

    zip_path: str | None = None
    if include_documents:
        zip_path = hf_hub_download(
            _HF_REPO,
            _TEXT_ZIP,
            repo_type="dataset",
            revision=revision,
            cache_dir=str(cache_dir),
        )
    return ds, zip_path, revision


def format_aa_lcr_prompt(task: dict[str, Any]) -> str:
    """Format an AA-LCR prompt with all source documents inline."""
    question = str(task.get("question", "")).strip()
    ctx = task.get("context") or {}
    documents = ctx.get("documents") or []

    parts: list[str] = [
        "Answer the question using only the provided documents.",
        "",
        "## Question",
        question,
    ]

    if documents:
        parts.extend(["", "## Documents"])
        for i, doc in enumerate(documents, 1):
            filename = str(doc.get("filename") or f"document_{i}")
            text = str(doc.get("text") or "").strip()
            parts.extend([
                "",
                f"### Document {i}: {filename}",
                text,
            ])

    parts.extend([
        "",
        "Reason across the documents as needed. Return only the final answer.",
    ])
    return "\n".join(parts)


def load_aa_lcr_tasks(
    *,
    limit: int | None = None,
    split: str = "test",
    include_documents: bool = True,
    cache_dir: str | Path = "data/cache/huggingface",
) -> list[dict[str, Any]]:
    """Load AA-LCR tasks from HuggingFace.

    Args:
        limit: optional maximum number of tasks to return.
        split: dataset split, defaults to ``test`` (the only published split).
        include_documents: when True, download/read the extracted-text zip
            and attach the referenced document set to ``task["context"]``.
            Set False for cheap schema smoke tests.
        cache_dir: HuggingFace cache directory. Defaults inside the repo's
            ignored ``data/cache`` tree so sandboxed runs can write locks and
            reuse downloaded data.
    """
    cache_path = Path(cache_dir)
    source_mode = "hf"
    source_revision = ""
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
        ds, zip_path, source_revision = _load_remote_assets(
            split=split,
            cache_dir=cache_path,
            include_documents=include_documents,
        )
    except Exception as exc:
        logger.warning(
            "AA-LCR remote load failed; returning no tasks: %s",
            exc,
        )
        return []

    tasks: list[dict[str, Any]] = []
    for row in ds:
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not question or not answer:
            continue

        filenames = _split_semicolon_field(row.get("data_source_filenames"))
        urls = _split_semicolon_field(row.get("data_source_urls"))
        documents: list[dict[str, str]] = []
        missing_documents: list[str] = []

        if include_documents and zip_path:
            documents, missing_documents = _read_document_set(
                zip_path,
                document_category=str(row.get("document_category") or ""),
                document_set_id=str(row.get("document_set_id") or ""),
                filenames=filenames,
            )

        qid = row.get("question_id") or row.get("Unnamed: 0") or len(tasks) + 1
        document_category = str(row.get("document_category") or "")
        document_set_id = str(row.get("document_set_id") or "")

        tasks.append({
            "id": f"aa_lcr_{qid}",
            "question": question,
            "answer": answer,
            "answer_type": "openText",
            "category": f"AA-LCR/{document_category}",
            "raw_subject": document_category,
            "context": {
                "benchmark": "aa_lcr",
                "source": _HF_REPO,
                "source_mode": source_mode,
                "source_revision": source_revision,
                "document_category": document_category,
                "document_set_id": document_set_id,
                "data_source_filenames": filenames,
                "data_source_urls": urls,
                "input_tokens": row.get("input_tokens"),
                "documents": documents,
                "missing_documents": missing_documents,
            },
            "metadata": {
                "source": "aa_lcr_hf",
                "source_mode": source_mode,
                "source_revision": source_revision,
                "document_category": document_category,
                "document_set_id": document_set_id,
                "input_tokens": row.get("input_tokens"),
                "include_documents": include_documents,
            },
            "scorer_kind": "llm_judge",
            "scorer_params": {
                "ground_truth": answer,
                "judge_prompt": "aa_lcr_equality",
            },
        })
        if limit and len(tasks) >= limit:
            break

    logger.info("AA-LCR loaded %d tasks (source_mode=%s, include_documents=%s)",
                len(tasks), source_mode, include_documents)
    return tasks
