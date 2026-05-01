"""HealthBench loader.

Source: ``openai/healthbench`` (HuggingFace, MIT).

The HF ``datasets`` auto-loader currently fails because HealthBench ships
multiple jsonl files with different schemas. This loader reads the official
jsonl files directly through ``huggingface_hub`` and normalizes each
conversation into the harness task shape.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

_HF_REPO = "openai/healthbench"
_FILES = {
    "full": "2025-05-07-06-14-12_oss_eval.jsonl",
    "consensus": "consensus_2025-05-09-20-00-46.jsonl",
    "hard": "hard_2025-05-08-21-00-10.jsonl",
}


def _format_prompt(messages: list[dict[str, Any]]) -> str:
    lines = ["Respond to the final user message in this health conversation."]
    lines.append("")
    lines.append("## Conversation")
    for msg in messages:
        role = str(msg.get("role") or "user").upper()
        content = str(msg.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _rubric_answer(row: dict[str, Any]) -> str:
    ideal = row.get("ideal_completions_data") or {}
    if isinstance(ideal, dict) and ideal.get("ideal_completion"):
        return str(ideal["ideal_completion"])
    rubrics = row.get("rubrics") or []
    criteria = []
    for item in rubrics[:12]:
        criterion = str((item or {}).get("criterion") or "").strip()
        points = (item or {}).get("points")
        if criterion:
            criteria.append(f"[{points}] {criterion}")
    return "Rubric criteria:\n" + "\n".join(criteria)


def _iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_healthbench_tasks(
    *,
    limit: int | None = None,
    subset: str = "hard",
    cache_dir: str | Path = "data/cache/huggingface",
) -> list[dict[str, Any]]:
    """Load HealthBench tasks.

    Args:
        limit: optional maximum number of examples.
        subset: one of ``full``, ``consensus``, or ``hard``. The default
            is ``hard`` because it is smaller and intended as a challenging
            target subset.
        cache_dir: HuggingFace cache directory.
    """
    if subset not in _FILES:
        raise ValueError(f"unknown HealthBench subset {subset!r}; choose {sorted(_FILES)}")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        logger.warning("HealthBench requires huggingface_hub: %s", exc)
        return []

    try:
        path = hf_hub_download(
            _HF_REPO,
            _FILES[subset],
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
    except Exception as exc:
        logger.warning("HealthBench download failed: %s", exc)
        return []

    tasks: list[dict[str, Any]] = []
    for row in _iter_jsonl(path):
        prompt_id = str(row.get("prompt_id") or f"healthbench_{len(tasks)}")
        rubrics = row.get("rubrics") or []
        tags = row.get("example_tags") or []
        tasks.append({
            "id": prompt_id,
            "question": _format_prompt(row.get("prompt") or []),
            "answer": _rubric_answer(row),
            "answer_type": "openText",
            "category": f"HealthBench/{subset}",
            "raw_subject": "health",
            "context": {
                "source": _HF_REPO,
                "subset": subset,
                "rubrics": rubrics,
                "example_tags": tags,
                "canary": row.get("canary"),
            },
            "metadata": {
                "source": "healthbench_hf",
                "subset": subset,
                "n_rubrics": len(rubrics),
                "example_tags": tags,
            },
            "scorer_kind": "llm_judge",
            "scorer_params": {
                "ground_truth": _rubric_answer(row),
                "rubrics": rubrics,
            },
        })
        if limit and len(tasks) >= limit:
            break

    return tasks
