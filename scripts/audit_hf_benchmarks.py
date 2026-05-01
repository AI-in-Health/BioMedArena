#!/usr/bin/env python3
"""Audit generic Hugging Face benchmark registrations.

For each ``hf_*`` dataset spec, this probes a small sample in an isolated
process so slow or incompatible HF datasets cannot block the full audit.
The output is a JSON file with both source schema information and the
normalized harness task shape used for scoring.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    candidates = [REPO_ROOT / ".env"]
    extra_env_file = os.environ.get("BIOAGENT_ENV_FILE")
    if extra_env_file:
        candidates.insert(0, Path(extra_env_file).expanduser())
    for candidate in candidates:
        try:
            if candidate.exists():
                load_dotenv(candidate, override=False)
        except OSError:
            continue


def _probe_dataset(dataset_key: str, sample_limit: int, queue: mp.Queue) -> None:
    try:
        from datasets import load_dataset

        from harness.eval.bench_hf_benchmark import (
            _choose_split,
            _load_dataset,
            load_hf_benchmark_tasks,
        )
        from harness.eval.hf_benchmark_registry import HF_BENCHMARK_SPECS

        spec = HF_BENCHMARK_SPECS[dataset_key]
        split_name = spec.split
        raw_columns: list[str] = []
        raw_preview: dict[str, Any] = {}
        raw_split = split_name

        try:
            ds = _load_dataset(
                load_dataset,
                spec,
                split_name,
                Path("data/cache/huggingface"),
                prefer_streaming=bool(spec.extra.get("streaming")),
            )
            if isinstance(ds, dict):
                raw_split = _choose_split(ds, split_name)
                ds = ds[raw_split]
            iterator = iter(ds)
            first = next(iterator, None)
            if isinstance(first, dict):
                raw_columns = list(first.keys())
                raw_preview = {
                    key: _short_value(value)
                    for key, value in list(first.items())[:20]
                }
        except Exception as exc:  # noqa: BLE001
            raw_preview = {"raw_probe_error": f"{type(exc).__name__}: {exc}"}

        tasks = load_hf_benchmark_tasks(dataset_key=dataset_key, limit=sample_limit)
        normalized = []
        for task in tasks:
            context = task.get("context") or {}
            normalized.append({
                "id": task.get("id"),
                "question_preview": str(task.get("question", ""))[:500],
                "answer_preview": str(task.get("answer", ""))[:300],
                "answer_type": task.get("answer_type"),
                "scorer_kind": task.get("scorer_kind") or context.get("scorer_kind"),
                "context_answer_type": context.get("answer_type"),
                "choices_count": len(task.get("choices") or []),
            })

        queue.put({
            "dataset_key": dataset_key,
            "repo": spec.repo,
            "config": spec.config,
            "domain": spec.domain,
            "task_type": spec.task_type,
            "declared_split": split_name,
            "resolved_split": raw_split,
            "streaming": bool(spec.extra.get("streaming")),
            "status": "ok" if tasks else "empty",
            "n_tasks": len(tasks),
            "raw_columns": raw_columns,
            "raw_preview": raw_preview,
            "normalized_samples": normalized,
        })
    except Exception as exc:  # noqa: BLE001
        queue.put({
            "dataset_key": dataset_key,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })


def _short_value(value: Any, max_chars: int = 300) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except TypeError:
            text = str(value)
    if isinstance(text, str) and len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def audit_dataset(dataset_key: str, sample_limit: int, timeout_s: int) -> dict[str, Any]:
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_probe_dataset, args=(dataset_key, sample_limit, queue))
    start = time.monotonic()
    proc.start()
    proc.join(timeout_s)
    elapsed = round(time.monotonic() - start, 2)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return {
            "dataset_key": dataset_key,
            "status": "timeout",
            "timeout_s": timeout_s,
            "elapsed_s": elapsed,
        }
    if not queue.empty():
        record = queue.get()
    else:
        record = {
            "dataset_key": dataset_key,
            "status": "error",
            "error": f"probe process exited with code {proc.exitcode} and no record",
        }
    record["elapsed_s"] = elapsed
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/hf_benchmark_audit.json")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--timeout-s", type=int, default=45)
    parser.add_argument("--only-prefix", default="")
    parser.add_argument("--keys", default="",
                        help="Comma-separated dataset keys to audit.")
    parser.add_argument("--start", type=int, default=0,
                        help="0-based start index after filtering.")
    parser.add_argument("--end", type=int, default=0,
                        help="Exclusive end index after filtering; 0 means no end.")
    parser.add_argument("--max", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    _load_env()

    from harness.eval.hf_benchmark_registry import HF_BENCHMARK_SPECS

    keys = list(HF_BENCHMARK_SPECS)
    if args.keys:
        requested = [key.strip() for key in args.keys.split(",") if key.strip()]
        keys = [key for key in requested if key in HF_BENCHMARK_SPECS]
    if args.only_prefix:
        keys = [key for key in keys if key.startswith(args.only_prefix)]
    if args.start:
        keys = keys[args.start:]
    if args.end:
        end = max(args.end - args.start, 0) if args.start else args.end
        keys = keys[:end]
    if args.max:
        keys = keys[:args.max]

    output = Path(args.output)
    records = []
    done_keys: set[str] = set()
    if args.resume and output.exists():
        try:
            existing = json.loads(output.read_text())
            records = list(existing.get("records") or [])
            done_keys = {str(r.get("dataset_key")) for r in records}
        except (json.JSONDecodeError, OSError):
            records = []
            done_keys = set()

    for idx, key in enumerate(keys, 1):
        if key in done_keys:
            print(f"[{idx}/{len(keys)}] {key} -> skipped (resume)", flush=True)
            continue
        print(f"[{idx}/{len(keys)}] {key}", flush=True)
        rec = audit_dataset(key, sample_limit=args.limit, timeout_s=args.timeout_s)
        records.append(rec)
        print(
            f"  -> {rec.get('status')} n={rec.get('n_tasks', 0)} "
            f"elapsed={rec.get('elapsed_s')}s",
            flush=True,
        )
        _write_output(output, records)

    _write_output(output, records)
    print(json.dumps(_summarize(records), indent=2), flush=True)
    print(f"Wrote: {output}", flush=True)
    return 0


def _summarize(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(records),
        "ok": sum(1 for r in records if r.get("status") == "ok"),
        "empty": sum(1 for r in records if r.get("status") == "empty"),
        "timeout": sum(1 for r in records if r.get("status") == "timeout"),
        "error": sum(1 for r in records if r.get("status") == "error"),
    }


def _write_output(output: Path, records: list[dict[str, Any]]) -> None:
    payload = {"summary": _summarize(records), "records": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
