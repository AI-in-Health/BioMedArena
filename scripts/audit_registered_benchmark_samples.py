#!/usr/bin/env python3
"""Probe registered benchmarks end-to-end with a small gold-answer sample."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _task_context(task: dict[str, Any]) -> dict[str, Any]:
    context = dict(task.get("context") or {})
    for key in ("scorer_kind", "scorer_params"):
        if key in task:
            context[key] = task[key]
    return context


def _short(value: Any, max_chars: int = 280) -> str:
    text = str(value or "")
    return text[:max_chars]


def _probe(key: str, limit: int, queue: mp.Queue) -> None:
    try:
        from harness.cli import BENCHMARKS, _load_tasks
        from harness.eval.benchmark_readiness import registered_benchmark_metadata
        from harness.eval.official_metrics import compute_official_metrics
        from harness.eval.scoring import score_question

        tasks = _load_tasks(key, limit=limit, seed=13)
        meta = registered_benchmark_metadata(key, BENCHMARKS[key])
        samples = []
        records = []
        issues = []
        for task in tasks:
            answer = str(task.get("answer", ""))
            answer_type = task.get("answer_type") or (task.get("context") or {}).get("answer_type") or "exactMatch"
            context = _task_context(task)
            try:
                self_score = bool(score_question(answer, answer, answer_type, context))
            except Exception as exc:  # noqa: BLE001
                self_score = False
                issues.append(f"self_score_error:{task.get('id')}:{type(exc).__name__}:{exc}")
            samples.append({
                "id": task.get("id"),
                "answer_type": answer_type,
                "scorer_kind": task.get("scorer_kind") or context.get("scorer_kind"),
                "self_score_gold": self_score,
                "answer_preview": _short(answer),
                "question_preview": _short(task.get("question", ""), 500),
                "official_metric": context.get("official_metric"),
            })
            records.append({
                "id": task.get("id"),
                "expected": answer,
                "predicted": answer,
                "predicted_raw": answer,
                "context": context,
            })
        official = compute_official_metrics(key, records)
        if not tasks:
            issues.append("loader_returned_no_tasks")
        if any(not sample["self_score_gold"] for sample in samples):
            issues.append("gold_answer_failed_self_score")
        queue.put({
            "key": key,
            "status": "ok" if tasks and not issues else "issue",
            "issues": issues,
            "description": f"{meta.get('domain', 'core')} {meta.get('task_type', '')}".strip(),
            "source": meta.get("source", ""),
            "source_url": meta.get("source_url", ""),
            "official_split": meta.get("official_split", ""),
            "declared_count": meta.get("total_count", "unknown"),
            "n_loaded": len(tasks),
            "readiness": meta.get("readiness", ""),
            "readiness_reason": meta.get("readiness_reason", ""),
            "needs_hf_token": bool(meta.get("needs_hf_token")),
            "needs_external_env": bool(meta.get("needs_external_env")),
            "input_modality": meta.get("input_modality", ""),
            "official_metrics_self_gold": official,
            "samples": samples,
        })
    except Exception as exc:  # noqa: BLE001
        queue.put({
            "key": key,
            "status": "error",
            "issues": [f"{type(exc).__name__}: {exc}"],
            "n_loaded": 0,
        })


def _audit_one(key: str, limit: int, timeout_s: int) -> dict[str, Any]:
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_probe, args=(key, limit, queue))
    start = time.monotonic()
    proc.start()
    proc.join(timeout_s)
    elapsed = round(time.monotonic() - start, 2)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        return {
            "key": key,
            "status": "timeout",
            "issues": [f"timeout_after_{timeout_s}s"],
            "n_loaded": 0,
            "elapsed_s": elapsed,
        }
    record = queue.get() if not queue.empty() else {
        "key": key,
        "status": "error",
        "issues": [f"probe exited {proc.exitcode} without result"],
        "n_loaded": 0,
    }
    record["elapsed_s"] = elapsed
    return record


def _load_keys(path: str | None) -> list[str]:
    from harness.cli import BENCHMARKS
    from harness.eval.hf_benchmark_registry import HF_DEPRECATED_ALIASES, HF_REMOVED_NONBENCHMARK_KEYS

    if path and Path(path).exists():
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        records = payload.get("records", [])
        keys = [str(record["key"]) for record in records if record.get("key") in BENCHMARKS]
    else:
        keys = sorted(BENCHMARKS)
    return [
        key for key in keys
        if key not in HF_DEPRECATED_ALIASES and key not in HF_REMOVED_NONBENCHMARK_KEYS
    ]


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(records),
        "summary": dict(Counter(record.get("status", "") for record in records)),
        "readiness_summary": dict(Counter(record.get("readiness", "") for record in records)),
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "key", "status", "readiness", "n_loaded", "source", "official_split",
        "input_modality", "needs_hf_token", "needs_external_env", "issues",
        "readiness_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({
                field: "; ".join(record.get(field, [])) if field == "issues" else record.get(field, "")
                for field in fields
            })


def _write_md(path: Path, records: list[dict[str, Any]]) -> None:
    summary = _summary(records)
    lines = [
        "# Registered Benchmark Sample Audit",
        "",
        f"Total: {summary['total']}",
        "",
        "## Status",
        "",
    ]
    for key, value in sorted(summary["summary"].items()):
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Readiness", ""]
    for key, value in sorted(summary["readiness_summary"].items()):
        lines.append(f"- `{key or 'unknown'}`: {value}")
    lines += [
        "",
        "## Table",
        "",
        "| Benchmark | Status | Readiness | Loaded | Source | Issues | Reason |",
        "|---|---|---|---:|---|---|---|",
    ]
    for record in records:
        source = record.get("source", "")
        if record.get("source_url"):
            source = f"[{source}]({record['source_url']})"
        issues = "; ".join(record.get("issues") or [])
        lines.append(
            f"| `{record['key']}` | `{record.get('status', '')}` | `{record.get('readiness', '')}` | "
            f"{record.get('n_loaded', 0)} | {source} | {issues} | {record.get('readiness_reason', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys-from", default="results/registered_benchmark_sample_audit.json")
    parser.add_argument("--out-json", default="results/registered_benchmark_sample_audit.json")
    parser.add_argument("--out-csv", default="results/registered_benchmark_sample_audit.csv")
    parser.add_argument("--out-md", default="results/registered_benchmark_sample_audit.md")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--timeout-s", type=int, default=75)
    parser.add_argument("--keys", default="", help="Comma-separated benchmark keys to audit.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    keys = [key.strip() for key in args.keys.split(",") if key.strip()] if args.keys else _load_keys(args.keys_from)
    out_json = Path(args.out_json)
    records = []
    done = set()
    if args.resume and out_json.exists():
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        records = list(payload.get("records") or [])
        done = {record.get("key") for record in records}

    for idx, key in enumerate(keys, start=1):
        if key in done:
            print(f"[{idx}/{len(keys)}] {key} -> resume", flush=True)
            continue
        print(f"[{idx}/{len(keys)}] {key}", flush=True)
        record = _audit_one(key, args.limit, args.timeout_s)
        records.append(record)
        print(f"  -> {record.get('status')} loaded={record.get('n_loaded')} elapsed={record.get('elapsed_s')}s", flush=True)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps({**_summary(records), "records": records}, indent=2, ensure_ascii=False), encoding="utf-8")

    payload = {**_summary(records), "records": records}
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(Path(args.out_csv), records)
    _write_md(Path(args.out_md), records)
    print(json.dumps(_summary(records), indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
