"""Biomedical lab benchmark loader (v2).

This benchmark is FUNDAMENTALLY DIFFERENT from v1:
    - v1: MCQ with `ideal` + `distractors` → letter answers (A/B/C/D)
    - v2: open-ended with `answer_regex` + `validator_params` → regex /
      validator-based scoring. No distractors, no letter positions,
      no MCQ shuffle concern.

Row schema (16 fields, confirmed via live probe):
    id, tag, version, question, ideal, files, sources, key_passage,
    canary, is_opensource, ground_truth, prompt_suffix, type, mode,
    validator_params, answer_regex

Known tags (15 dataset configs + an `all` super-config):
    BASELINE TEXT-ONLY (7, default; 821 rows):
        litqa3, patentqa, trialqa, dbqa2, suppqa2, figqa2, tableqa2
    FILE-REF TEXT (4, opt-in):
        seqqa2, sourcequality, protocolqa2, cloning
    FIG/TABLE base (2, included in the default text-only baseline):
        figqa2, tableqa2
    MULTIMODAL (4, opt-in):
        figqa2-img, figqa2-pdf, tableqa2-img, tableqa2-pdf

Design:
    - Strict schema validation — missing required fields raise
      KeyError rather than silently returning empty strings.
    - answer_type is always "openText" (no MCQ conversion).
    - scorer_kind="labbench2_regex" directs the scorer dispatcher to
      harness/eval/labbench2_scorer.py (primary: regex; fallback:
      case-insensitive substring on `ideal`/`ground_truth`).
    - Default loading uses the 7-tag text-only subset and skips
      file-bearing rows. Pass
      subsets="all", skip_with_files=False, include_multimodal=True to
      explicitly load the complete official benchmark.
    - `canary` rows are observed-and-logged rather than skipped
      (their purpose in the dataset isn't documented yet — the
      conservative default is to include them; the schema probe
      will surface prevalence).

Merge-safety: any new tool specs live here as LABBENCH2_TOOL_SPECS
and are registered via register_tool_specs() rather than by editing
harness/eval/function_calling_runner.py directly.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any


log = logging.getLogger(__name__)


# ---- Subset taxonomy ------------------------------------------------

# All 15 concrete subset tags (excluding the "all" super-config).
ALL_SUBSETS: tuple[str, ...] = (
    "cloning",
    "dbqa2",
    "figqa2",
    "figqa2-img",
    "figqa2-pdf",
    "litqa3",
    "patentqa",
    "protocolqa2",
    "seqqa2",
    "sourcequality",
    "suppqa2",
    "tableqa2",
    "tableqa2-img",
    "tableqa2-pdf",
    "trialqa",
)

# Default text-only subset (821 rows).
TEXT_ONLY_SUBSETS: tuple[str, ...] = (
    "litqa3",
    "patentqa",
    "trialqa",
    "dbqa2",
    "suppqa2",
    "figqa2",
    "tableqa2",
)

FILE_REF_TEXT_SUBSETS: tuple[str, ...] = (
    "seqqa2",
    "sourcequality",
    "protocolqa2",
    "cloning",
)

# Image/PDF variants require explicit opt-in.
MULTIMODAL_SUBSETS: frozenset[str] = frozenset({
    "figqa2-img",
    "figqa2-pdf",
    "tableqa2-img",
    "tableqa2-pdf",
})


# ---- Schema validation ----------------------------------------------

# Minimum fields every row must provide. If these are missing, the
# loader raises rather than silently synthesising defaults.
_REQUIRED_FIELDS: frozenset[str] = frozenset({
    "id", "question", "ideal", "tag"
})

# One-shot warning state keyed by (subset, reason) so large subsets
# don't spam the log.
_warned_keys: set[tuple[str, str]] = set()


def _warn_once(subset: str, reason: str, message: str) -> None:
    key = (subset, reason)
    if key in _warned_keys:
        return
    _warned_keys.add(key)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


# ---- Tool specs (merge-safety pattern) ------------------------------

LABBENCH2_TOOL_SPECS: list[dict[str, Any]] = []


def register_tool_specs() -> None:
    """Merge-safe hook for extending the central TOOL_SPECS list.

    No-op when LABBENCH2_TOOL_SPECS is empty. Callers (e.g.
    harness/eval/__init__.py) invoke this to register LAB-Bench 2's
    tool specs WITHOUT editing function_calling_runner.py.
    """
    if not LABBENCH2_TOOL_SPECS:
        return
    try:
        from harness.eval.function_calling_runner import TOOL_SPECS
    except Exception:
        return
    TOOL_SPECS.extend(LABBENCH2_TOOL_SPECS)


# ---- Public loader --------------------------------------------------

def _resolve_subsets(
    subsets: list[str] | str | None,
    include_multimodal: bool,
) -> list[str]:
    """Normalize the `subsets` argument.

    - None → the 7-tag text-only baseline.
    - "all" → official `all` config.
    - list → validated against ALL_SUBSETS.
    """
    if subsets is None:
        resolved = list(TEXT_ONLY_SUBSETS)
    elif isinstance(subsets, str):
        if subsets == "all":
            resolved = ["all"]
        elif subsets in ALL_SUBSETS:
            resolved = [subsets]
        else:
            raise ValueError(
                f"Unknown labbench2 subset string: {subsets!r}. "
                f"Use 'all', a specific tag, or a list. "
                f"Known tags: {list(ALL_SUBSETS)}"
            )
    else:
        unknown = [s for s in subsets if s not in ALL_SUBSETS]
        if unknown:
            raise ValueError(
                f"Unknown labbench2 subset(s): {unknown}. "
                f"Known tags: {list(ALL_SUBSETS)}"
            )
        resolved = list(subsets)

    if "all" in resolved:
        return resolved

    if not include_multimodal:
        mm_requested = [s for s in resolved if s in MULTIMODAL_SUBSETS]
        if mm_requested:
            # Silent drop for convenience when the caller passed
            # "all" — explicit requests still raise so the user
            # notices they asked for something they can't get.
            if subsets in (None, "all"):
                resolved = [s for s in resolved if s not in MULTIMODAL_SUBSETS]
            else:
                raise ValueError(
                    f"Subsets {mm_requested} are multimodal / "
                    "figure / table variants. Pass "
                    "include_multimodal=True to opt in, or drop "
                    "them from the subsets= list."
                )
    return resolved


def _normalize_v2_task(
    row: dict[str, Any], subset: str, idx: int
) -> dict[str, Any]:
    """Normalize a benchmark row into the harness's canonical shape.

    LAB-Bench 2 tasks are open-ended; scoring is regex / validator
    based and dispatched via scorer_kind="labbench2_regex".
    """
    # Strict schema check — no silent fallback for required fields.
    missing = _REQUIRED_FIELDS - set(row.keys())
    if missing:
        raise KeyError(
            f"labbench2/{subset} row {idx} missing expected fields: "
            f"{sorted(missing)}. Actual keys: {sorted(row.keys())}. "
            "Schema drift detected — run "
            "scripts/labbench2_schema_probe.py for a live snapshot "
            "and update _REQUIRED_FIELDS if appropriate."
        )

    qid = str(row.get("id") or f"labbench2_{subset}_{idx:04d}")
    has_files = bool(row.get("files"))
    is_canary = bool(row.get("canary"))

    if is_canary:
        # `canary` rows are kept-but-flagged. Log prevalence per-
        # subset once so we can audit after loading.
        _warn_once(
            subset,
            "canary_present",
            f"labbench2/{subset}: encountered canary=True row(s). "
            "Keeping them in the task list (purpose undocumented); "
            "verify via the schema probe if prevalence looks high.",
        )

    return {
        "id": qid,
        "question": row["question"],
        # Gold free-text answer. Actual correctness judged by the
        # regex scorer, not by string-equality against this field.
        "answer": row["ideal"],
        "answer_type": "openText",
        "category": f"LAB-Bench-2/{subset}",
        "raw_subject": subset,
        "context": {
            "benchmark": "labbench2",
            "subset": subset,
            "tag": row.get("tag"),
            "version": row.get("version"),
            "type": row.get("type"),
            "mode": row.get("mode"),
            "sources": row.get("sources"),
            "key_passage": row.get("key_passage"),
            "prompt_suffix": row.get("prompt_suffix"),
            "is_opensource": row.get("is_opensource"),
            "canary": is_canary,
            "has_files": has_files,
            "file_refs": row.get("files"),
            "form": "open_ended",
            "scorer_hint": "labbench2_regex",
            # Duplicated into context so score_question(..., context=...)
            # can dispatch — its caller doesn't see the top-level
            # scorer_kind / scorer_params keys.
            "scorer_kind": "labbench2_regex",
            "scorer_params": {
                "answer_regex": row.get("answer_regex"),
                "validator_params": row.get("validator_params"),
                "ground_truth": row.get("ground_truth"),
                "ideal": row["ideal"],
            },
        },
        "scorer_kind": "labbench2_regex",
        "scorer_params": {
            "answer_regex": row.get("answer_regex"),
            "validator_params": row.get("validator_params"),
            "ground_truth": row.get("ground_truth"),
            "ideal": row["ideal"],
        },
    }


def load_labbench2_tasks(
    limit: int | None = None,
    subsets: list[str] | str | None = None,
    skip_with_files: bool = True,
    include_multimodal: bool = False,
    revision: str | None = None,
    require_hf_token: bool = True,
) -> list[dict[str, Any]]:
    """Load biomedical lab tasks (v2).

    Args:
        limit: cap total tasks returned across all subsets.
        subsets: which subset tags to load. None → the 7-tag text-only
            baseline used in the formal model comparison. "all" → official
            EdisonScientific/labbench2 `all` config. A list → explicit tag
            list.
        skip_with_files: when True, skip rows whose `files` field is
            non-empty. The default is True because all currently compared
            models are text-only.
        include_multimodal: when False, the figqa2* and tableqa2* configs
            are unavailable when requested explicitly. The default is False
            for text-only runs.

    Returns:
        A list of canonical task dicts. All LAB-Bench 2 tasks are
        open-ended (answer_type="openText") and carry
        scorer_kind="labbench2_regex" so the scoring dispatcher
        routes to harness/eval/labbench2_scorer.py.

    Raises:
        KeyError: row is missing required schema fields.
        ValueError: caller requested an unknown or multimodal subset
            without include_multimodal=True.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        warnings.warn(
            "labbench2 loader: `datasets` not installed; returning []. "
            "pip install datasets to enable.",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    resolved_subsets = _resolve_subsets(subsets, include_multimodal)
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    )
    if require_hf_token and not hf_token:
        warnings.warn(
            "labbench2 is gated and require_hf_token=True; returning no tasks "
            "so cached/unauthenticated data is not mistaken for current "
            "upstream LAB-Bench 2.",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    from harness.eval.hf_source import verify_hf_dataset_online
    source_info = verify_hf_dataset_online(
        "EdisonScientific/labbench2",
        token=hf_token,
        revision=revision,
    )
    if source_info is None:
        warnings.warn(
            "labbench2 upstream could not be verified online; returning no "
            "tasks instead of using a local cache.",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    tasks: list[dict[str, Any]] = []
    per_subset_skipped_files: dict[str, int] = {}

    for subset in resolved_subsets:
        try:
            kwargs: dict[str, Any] = {"split": "train"}
            if hf_token:
                kwargs["token"] = hf_token
            if revision:
                kwargs["revision"] = revision
            ds = load_dataset("EdisonScientific/labbench2", subset, **kwargs)
        except Exception as exc:
            warnings.warn(
                f"labbench2 loader: failed to load subset '{subset}': "
                f"{type(exc).__name__}: {str(exc)[:200]}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        subset_skipped = 0
        for idx, row in enumerate(ds):
            if skip_with_files and row.get("files"):
                subset_skipped += 1
                continue
            task = _normalize_v2_task(row, subset, idx)
            task["context"].update({
                "source": "EdisonScientific/labbench2",
                "split": "train",
                "revision": source_info["revision"],
                "source_sha": source_info["sha"],
                "requires_hf_token": require_hf_token,
            })
            tasks.append(task)
            if limit and len(tasks) >= limit:
                per_subset_skipped_files[subset] = subset_skipped
                log.info(
                    "labbench2: hit limit=%d; skipped %s file-bearing rows",
                    limit, per_subset_skipped_files,
                )
                return tasks

        per_subset_skipped_files[subset] = subset_skipped

    if any(per_subset_skipped_files.values()):
        log.info(
            "labbench2: skipped file-bearing rows per subset: %s",
            per_subset_skipped_files,
        )

    return tasks
