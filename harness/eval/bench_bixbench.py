"""Bioinformatics analysis benchmark loader.

The public loader uses the official FutureHouse BixBench dataset and
normalizes rows into deterministic closed-book MCQ tasks.
"""

from __future__ import annotations

import hashlib
import random
import os
import warnings
from typing import Any

def _seed_for(task_id: str) -> int:
    h = hashlib.sha256(f"bixbench:{task_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def _shuffled(ideal: str, distractors: list[str], seed: int) -> tuple[list[str], str]:
    options = [str(ideal)] + [str(d) for d in (distractors or [])][:4]
    idx = list(range(len(options)))
    random.Random(seed).shuffle(idx)
    letters = ["A", "B", "C", "D", "E"][: len(options)]
    shuffled_opts = [options[i] for i in idx]
    ideal_pos = idx.index(0)
    return shuffled_opts, letters[ideal_pos]


def load_bixbench_tasks(limit: int | None = None,
                          split: str = "train",
                          form: str = "mcq",
                          revision: str | None = None,
                          require_online: bool = True) -> list[dict[str, Any]]:
    """Load bioinformatics analysis tasks.

    Parameters
    ----------
    limit
        Max tasks to return (None = all).
    split
        The dataset publishes a 'train' split with 205 closed-book
        MCQ-style rows. 'test' is accepted as an alias and maps to
        'train'.
    form
        'mcq' (default) — deterministic shuffle of ideal + distractors.
        'open' — MCQ options hidden, free-text ideal answer.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    # Real dataset only publishes 'train'.
    hf_split = "train" if split in ("train", "test") else split

    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    source_info = {"revision": revision or "main", "sha": None}
    if require_online:
        from harness.eval.hf_source import verify_hf_dataset_online
        verified = verify_hf_dataset_online(
            "futurehouse/BixBench",
            token=hf_token,
            revision=revision,
        )
        if verified is None:
            warnings.warn(
                "BixBench upstream could not be verified online; returning "
                "no tasks instead of using a local cache.",
                RuntimeWarning,
                stacklevel=2,
            )
            return []
        source_info = verified

    try:
        kwargs: dict[str, Any] = {"split": hf_split}
        if hf_token:
            kwargs["token"] = hf_token
        if revision:
            kwargs["revision"] = revision
        ds = load_dataset("futurehouse/BixBench", **kwargs)
    except Exception:
        return []

    tasks: list[dict[str, Any]] = []
    for row in ds:
        question = row.get("question") or ""
        ideal = row.get("ideal")
        distractors = row.get("distractors") or []
        if not question or ideal is None:
            continue

        # Prefer the stable short_id/question_id if present, else the HF row id.
        qid = (row.get("question_id") or row.get("short_id")
               or row.get("id") or f"bixbench_{len(tasks)}")

        if form == "open":
            tasks.append({
                "id": str(qid),
                "question": question + "\n\nAnswer in your own words.",
                "choices": None,
                "answer": str(ideal),
                "answer_type": "openText",
                "category": f"BixBench/{row.get('categories','')}",
                "metadata": {
                    "capsule_uuid": row.get("capsule_uuid"),
                    "hypothesis": row.get("hypothesis"),
                    "categories": row.get("categories"),
                    "paper": row.get("paper"),
                    "eval_mode": row.get("eval_mode"),
                    "form": "open",
                    "source": "bixbench_hf",
                    "dataset": "futurehouse/BixBench",
                    "split": hf_split,
                    "revision": source_info["revision"],
                    "source_sha": source_info["sha"],
                },
                "scorer_kind": "open",
            })
        else:
            seed = _seed_for(str(qid))
            shuffled_opts, ideal_letter = _shuffled(ideal, distractors, seed)
            letters = ["A", "B", "C", "D", "E"][: len(shuffled_opts)]
            opts_block = "\n".join(f"{l}) {o}" for l, o in zip(letters, shuffled_opts))
            full_question = f"{question}\n\n{opts_block}"

            tasks.append({
                "id": str(qid),
                "question": full_question,
                "choices": shuffled_opts,
                "answer": ideal_letter,
                "answer_type": "multipleChoice",
                "category": f"BixBench/{row.get('categories','')}",
                "metadata": {
                    "capsule_uuid": row.get("capsule_uuid"),
                    "hypothesis": row.get("hypothesis"),
                    "categories": row.get("categories"),
                    "paper": row.get("paper"),
                    "eval_mode": row.get("eval_mode"),
                    "ideal_letter": ideal_letter,
                    "ideal": str(ideal),
                    "shuffle_seed": seed,
                    "form": "mcq",
                    "source": "bixbench_hf",
                    "dataset": "futurehouse/BixBench",
                    "split": hf_split,
                    "revision": source_info["revision"],
                    "source_sha": source_info["sha"],
                },
                "scorer_kind": "mcq",
            })
        if limit and len(tasks) >= limit:
            break

    return tasks
