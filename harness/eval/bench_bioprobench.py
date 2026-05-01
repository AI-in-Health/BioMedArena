"""BioProBench loader.

Source: BioProBench/BioProBench on Hugging Face.
Official page: https://huggingface.co/datasets/BioProBench/BioProBench

BioProBench covers biological protocol understanding and reasoning. The
official test files are task-specific JSON files, so this loader reads those
files directly instead of using the generic datasets builder, which currently
struggles with mixed schemas across files.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

_HF_REPO = "BioProBench/BioProBench"
_TASK_FILES = {
    "pqa": "PQA_test.json",
    "ord": "ORD_test.json",
    "err": "ERR_test.json",
    "gen": "GEN_test.json",
}


def _iter_json(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        yield from data


def _format_options(options: list[Any]) -> tuple[str, str]:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = []
    answer_map: dict[str, str] = {}
    for idx, option in enumerate(options):
        letter = letters[idx]
        text = str(option)
        lines.append(f"{letter}. {text}")
        answer_map[text] = letter
    return "\n".join(lines), json.dumps(answer_map)


def _make_pqa(row: dict[str, Any]) -> dict[str, Any]:
    options = [str(x) for x in row.get("choices") or []]
    opts_block, answer_map_json = _format_options(options)
    answer = str(row.get("answer") or "")
    answer_map = json.loads(answer_map_json)
    return {
        "id": str(row.get("id") or ""),
        "question": f"{row.get('question', '')}\n\n{opts_block}",
        "answer": answer_map.get(answer, answer),
        "answer_type": "multipleChoice",
        "category": f"BioProBench/PQA/{row.get('type') or 'unknown'}",
        "raw_subject": row.get("type") or "pqa",
        "context": {"source": _HF_REPO, "task": "pqa", "answer_text": answer},
    }


def _make_ord(row: dict[str, Any]) -> dict[str, Any]:
    steps = [str(x) for x in row.get("wrong_steps") or []]
    correct = [str(x) for x in row.get("correct_steps") or []]
    step_block = "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(steps))
    return {
        "id": str(row.get("id") or ""),
        "question": (
            f"{row.get('question', '')}\n\nSteps to order:\n{step_block}\n\n"
            "Return the correct order as the step titles separated by semicolons."
        ),
        "answer": "; ".join(correct),
        "answer_type": "openText",
        "category": f"BioProBench/ORD/{row.get('type') or 'unknown'}",
        "raw_subject": row.get("type") or "ord",
        "context": {"source": _HF_REPO, "task": "ord", "correct_steps": correct},
    }


def _make_err(row: dict[str, Any]) -> dict[str, Any]:
    ctx = row.get("context") or {}
    prior = ctx.get("prior_step") or ""
    next_step = ctx.get("next_step") or ""
    corrupted = row.get("corrupted_text")
    corrected = row.get("corrected_text") or ""
    if row.get("is_correct"):
        answer = "No error"
    else:
        answer = corrected
    return {
        "id": str(row.get("id") or ""),
        "question": (
            "Check this biological protocol step. If it contains an error, "
            "provide the corrected step; otherwise answer 'No error'.\n\n"
            f"Purpose: {ctx.get('purpose') or ''}\n"
            f"Prior step: {prior}\n"
            f"Step: {corrupted or corrected}\n"
            f"Next step: {next_step}"
        ),
        "answer": answer,
        "answer_type": "openText",
        "category": f"BioProBench/ERR/{row.get('type') or 'unknown'}",
        "raw_subject": row.get("type") or "err",
        "context": {
            "source": _HF_REPO,
            "task": "err",
            "is_correct": row.get("is_correct"),
            "error_description": row.get("error_description"),
        },
    }


def _make_gen(row: dict[str, Any]) -> dict[str, Any]:
    output = row.get("output") or []
    if isinstance(output, list):
        answer = "\n".join(str(step) for step in output)
    else:
        answer = str(output)
    return {
        "id": str(row.get("id") or ""),
        "question": (
            f"{row.get('system_prompt') or ''}\n\n"
            f"{row.get('instruction') or ''}\n\n"
            f"{row.get('input') or ''}"
        ).strip(),
        "answer": answer,
        "answer_type": "openText",
        "category": f"BioProBench/GEN/{row.get('type') or 'unknown'}",
        "raw_subject": row.get("type") or "gen",
        "context": {"source": _HF_REPO, "task": "gen"},
    }


_MAKERS = {
    "pqa": _make_pqa,
    "ord": _make_ord,
    "err": _make_err,
    "gen": _make_gen,
}


def load_bioprobench_tasks(
    limit: int | None = None,
    tasks: list[str] | None = None,
    cache_dir: str | Path = "data/cache/huggingface",
) -> list[dict[str, Any]]:
    """Load official BioProBench test tasks.

    Args:
        limit: optional total cap across selected task files.
        tasks: subset of ``pqa``, ``ord``, ``err``, ``gen``. Defaults to all.
        cache_dir: Hugging Face cache directory.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return []

    selected = tasks or list(_TASK_FILES)
    out: list[dict[str, Any]] = []
    for task in selected:
        if task not in _TASK_FILES:
            raise ValueError(f"unknown BioProBench task {task!r}; choose {sorted(_TASK_FILES)}")
        try:
            path = hf_hub_download(
                _HF_REPO,
                _TASK_FILES[task],
                repo_type="dataset",
                cache_dir=str(cache_dir),
            )
        except Exception:
            continue
        maker = _MAKERS[task]
        for row in _iter_json(path):
            item = maker(row)
            if not item["id"]:
                item["id"] = f"bioprobench_{len(out):06d}"
            out.append(item)
            if limit and len(out) >= limit:
                return out
    return out
