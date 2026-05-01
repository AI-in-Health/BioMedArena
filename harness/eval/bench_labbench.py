"""Biomedical lab agent benchmark loader.

MCQ tasks across multiple subsets, suitable for agentic evaluation:
    LitQA2, DbQA, SeqQA, FigQA, TableQA, ProtocolQA, SuppQA,
    CloningScenarios.

`openai_tri` subset: {LitQA2, ProtocolQA, CloningScenarios}. Within the
tri-subset, ProtocolQA is exposed in OPEN-ENDED form (options hidden,
LLM-judge scoring) per upstream convention; the other two remain MCQ.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any


DEFAULT_SUBSETS = ["LitQA2", "DbQA", "SeqQA", "ProtocolQA", "SuppQA", "CloningScenarios"]

# Canonical tri-subset exposed for compact evaluations.
# Counts come from the HF dataset: LitQA2=199, ProtocolQA=108,
# CloningScenarios=33 → 340 tasks total.
OPENAI_TRI_SUBSETS = ["LitQA2", "ProtocolQA", "CloningScenarios"]

# Subset(s) that are rendered as open-ended within `openai_tri` — their
# gold answer is the free-text `ideal`, not an option letter, and the
# caller must route these through LLM-judge (see harness/eval/llm_judge.py).
_OPEN_IN_OPENAI_TRI = frozenset({"ProtocolQA"})


def _seed_for_task(task_id: str, subset: str) -> int:
    """Deterministic 64-bit seed per (subset, id). Same id → same permutation."""
    h = hashlib.sha256(f"{subset}:{task_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def _shuffled_mcq(ideal: str, distractors: list[str], seed: int) -> tuple[list[str], list[str], str]:
    """Return (options, letters, ideal_letter) after deterministic shuffle."""
    options = [ideal] + list(distractors)[:4]
    idx = list(range(len(options)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    letters = ["A", "B", "C", "D", "E"][: len(options)]
    shuffled = [options[i] for i in idx]
    # The new position of the ideal (which was originally at index 0).
    ideal_new_pos = idx.index(0)
    return shuffled, letters, letters[ideal_new_pos]


def load_labbench_tasks(
    limit: int | None = None,
    subsets: list[str] | None = None,
    subset_preset: str | None = None,
) -> list[dict[str, Any]]:
    """Load biomedical lab tasks.

    Args:
        limit: cap number returned
        subsets: optional filter, e.g. ["LitQA2", "DbQA"]. Ignored when
            subset_preset is set. Default: text-only subsets
            (skips FigQA/TableQA which need images).
        subset_preset: named preset. Currently supported:
            "openai_tri" → {LitQA2, ProtocolQA, CloningScenarios} = 340.
            ProtocolQA is rendered as OPEN-ENDED (no MCQ options) so
            the caller can route it to LLM-judge; LitQA2 and
            CloningScenarios remain MCQ with shuffled options.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    if subset_preset == "openai_tri":
        subsets = list(OPENAI_TRI_SUBSETS)
        open_ended = _OPEN_IN_OPENAI_TRI
    else:
        subsets = subsets or DEFAULT_SUBSETS
        open_ended = frozenset()

    tasks: list[dict[str, Any]] = []

    for subset in subsets:
        try:
            ds = load_dataset("futurehouse/lab-bench", subset, split="train")
        except Exception:
            continue

        for row in ds:
            qid = row.get("id", f"lab_{subset}_{len(tasks):04d}")
            question = row.get("question", "")
            ideal = row.get("ideal", "")
            distractors = row.get("distractors", []) or []
            subtask = row.get("subtask", subset)

            if subset in open_ended:
                # OPEN-ENDED: hide the MCQ options entirely; the gold
                # answer is the free-text `ideal`, to be scored by
                # LLM-judge. For ProtocolQA, prepend the protocol text
                # so the question is self-contained.
                protocol = row.get("protocol") or ""
                prefix = f"Protocol:\n{protocol}\n\n" if protocol else ""
                full_question = f"{prefix}{question}\n\nAnswer in your own words."
                tasks.append({
                    "id": str(qid),
                    "question": full_question,
                    "answer": ideal,
                    "answer_type": "openText",
                    "category": f"LAB-Bench/{subset}",
                    "raw_subject": subtask,
                    "context": {
                        "subset": subset,
                        "subtask": subtask,
                        "ideal": ideal,
                        "num_options": 0,
                        "form": "open_ended",
                        "scorer_hint": "llm_judge",
                    },
                })
                if limit and len(tasks) >= limit:
                    return tasks
                continue

            # MCQ: deterministic per-task shuffle.
            seed = _seed_for_task(str(qid), subset)
            shuffled, letters, ideal_letter = _shuffled_mcq(ideal, distractors, seed)
            opts_block = "\n".join(f"{l}) {opt}" for l, opt in zip(letters, shuffled))
            full_question = f"{question}\n\n{opts_block}"

            tasks.append({
                "id": str(qid),
                "question": full_question,
                "answer": ideal_letter,
                "answer_type": "multipleChoice",
                "category": f"LAB-Bench/{subset}",
                "raw_subject": subtask,
                "context": {
                    "subset": subset,
                    "subtask": subtask,
                    "ideal": ideal,
                    "ideal_letter": ideal_letter,
                    "num_options": len(shuffled),
                    "shuffle_seed": seed,
                    "form": "mcq",
                },
            })
            if limit and len(tasks) >= limit:
                return tasks

    return tasks
