#!/usr/bin/env python3
"""Generate benchmark split/count/readiness audit files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.cli import BENCHMARKS
from harness.eval.benchmark_readiness import (
    probe_hf_count,
    registered_benchmark_metadata,
)
from harness.eval.hf_benchmark_registry import HF_BENCHMARK_SPECS
from harness.eval.hf_benchmark_registry import HF_DEPRECATED_ALIASES, HF_REMOVED_NONBENCHMARK_KEYS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--probe-counts", action="store_true")
    parser.add_argument(
        "--keys-from",
        default="results/registered_benchmark_sample_audit.json",
        help="Optional sample-audit JSON whose record order defines the benchmark set.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key_path = Path(args.keys_from) if args.keys_from else None
    if key_path and key_path.exists():
        payload = json.loads(key_path.read_text(encoding="utf-8"))
        keys = [record["key"] for record in payload.get("records", [])]
    else:
        keys = sorted(BENCHMARKS)
    keys = [
        key for key in keys
        if key not in HF_DEPRECATED_ALIASES and key not in HF_REMOVED_NONBENCHMARK_KEYS
    ]
    rows = []
    for key in keys:
        cli_meta = BENCHMARKS[key]
        meta = registered_benchmark_metadata(key, cli_meta)
        count = None
        count_status = "not_probed"
        split = meta.get("official_split", "")
        if args.probe_counts and key in HF_BENCHMARK_SPECS:
            count, count_status = probe_hf_count(key)
            if count is not None:
                split = count_status
        rows.append({
            "benchmark": key,
            "description": f"{meta.get('domain', '')} {meta.get('task_type', '')}".strip(),
            "source": meta.get("source", ""),
            "source_url": meta.get("source_url", ""),
            "official_split": split,
            "total_count": count if count is not None else "unknown",
            "count_status": count_status,
            "sampled_count": 5,
            "gated": "yes" if meta.get("needs_hf_token") else "no",
            "needs_hf_token": "yes" if meta.get("needs_hf_token") else "no",
            "needs_external_env": "yes" if meta.get("needs_external_env") else "no",
            "input_modality": meta.get("input_modality", ""),
            "readiness": meta.get("readiness", ""),
            "readiness_reason": meta.get("readiness_reason", ""),
        })

    csv_path = out_dir / "benchmark_readiness_audit.csv"
    fieldnames = [
        "benchmark", "description", "source", "source_url", "official_split",
        "total_count", "count_status", "sampled_count", "gated",
        "needs_hf_token", "needs_external_env", "input_modality",
        "readiness", "readiness_reason",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["readiness"] for row in rows)
    md_path = out_dir / "benchmark_readiness_audit.md"
    lines = [
        "# Benchmark Readiness Audit",
        "",
        f"Total registered benchmarks: {len(rows)}",
        "",
        "## Readiness Summary",
        "",
    ]
    for key, value in sorted(counts.items()):
        lines.append(f"- `{key}`: {value}")
    lines += [
        "",
        "## Table",
        "",
        "| Benchmark | Source | Split | Count | HF token | External env | Modality | Readiness | Reason |",
        "|---|---|---|---:|---|---|---|---|---|",
    ]
    for row in rows:
        source = f"[{row['source']}]({row['source_url']})" if row["source_url"] else row["source"]
        lines.append(
            f"| `{row['benchmark']}` | {source} | {row['official_split']} | "
            f"{row['total_count']} | {row['needs_hf_token']} | "
            f"{row['needs_external_env']} | {row['input_modality']} | "
            f"`{row['readiness']}` | {row['readiness_reason']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
