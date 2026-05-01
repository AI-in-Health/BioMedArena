"""Inject benchmark-specific context into model prompts.

Some benchmarks store supplementary retrieval content in task["context"]
(e.g., key_passage + sources for retrieval-augmented tasks). This module
normalizes that context into the prompt sent to the model.

Currently handles:
  - Retrieval-augmented tasks: key_passage + sources injection

Tasks without retrieval context fall through to question-only (preserves
existing behavior for all other benchmarks).
"""
from __future__ import annotations


def format_task_prompt(task: dict) -> str:
    """Return the text prompt that should be sent to the model.

    For tasks with retrieval-style context (sources + key_passage),
    returns question + formatted context.

    For all other benchmarks, returns task["question"] unchanged.
    """
    question = str(task.get("question", "")).strip()
    ctx = task.get("context") or {}

    if not isinstance(ctx, dict):
        return question

    if _has_retrieval_context(ctx):
        return _format_retrieval_prompt(question, ctx)

    return question


def _has_retrieval_context(ctx: dict) -> bool:
    """Detect retrieval-augmented style context.

    Returns True when the context dict contains key_passage or sources
    fields, regardless of which benchmark produced them.
    """
    return bool(ctx.get("key_passage") or ctx.get("sources"))


def _format_retrieval_prompt(question: str, ctx: dict) -> str:
    """Format a retrieval-augmented task with sources + key_passage injected."""
    sources = ctx.get("sources") or []
    key_passage = (ctx.get("key_passage") or "").strip()

    # Normalise sources into a list of non-empty strings.
    source_lines: list[str] = []
    if isinstance(sources, list):
        for src in sources:
            s = str(src).strip()
            if s:
                source_lines.append(s)
    elif isinstance(sources, str):
        s = sources.strip()
        if s:
            source_lines.append(s)

    if not source_lines and not key_passage:
        # Nothing to inject — return bare question.
        return question

    parts: list[str] = [question]

    if source_lines:
        parts.append("")
        parts.append("## Relevant Sources")
        for i, src in enumerate(source_lines, 1):
            parts.append(f"{i}. {src}")

    if key_passage:
        parts.append("")
        parts.append("## Key Passage (from source)")
        parts.append(key_passage)
        parts.append("")
        parts.append(
            "Use the key passage above to answer the question precisely. "
            "The passage contains specific facts needed for the answer."
        )

    return "\n".join(parts)
