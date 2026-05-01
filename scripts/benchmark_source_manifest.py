#!/usr/bin/env python3
"""Print the official source manifest for the formal benchmark loaders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


BENCHMARK_SOURCES = {
    "medcalc": {
        "loader": "load_medcalc_tasks",
        "sources": ["ncbi/MedCalc-Bench-v1.2:test"],
        "expected_size": 1100,
        "access": "public",
        "assets": "none",
        "strictness": "online source verification required",
    },
    "labbench2": {
        "loader": "load_labbench2_tasks",
        "sources": [
            "EdisonScientific/labbench2:{litqa3,patentqa,trialqa,dbqa2,suppqa2,figqa2,tableqa2}/train"
        ],
        "expected_size": 821,
        "access": "gated_hf",
        "assets": "text-only rows; file-bearing rows skipped by default",
        "strictness": "HF token and online source verification required; pass subsets='all' for full 1912-row benchmark",
    },
    "gpqa_bio": {
        "loader": "load_gpqa_bio_tasks",
        "sources": ["Idavidrein/gpqa:gpqa_diamond/train"],
        "expected_size": 198,
        "access": "gated_hf",
        "assets": "none",
        "strictness": "default scope is full GPQA Diamond for this benchmark key",
    },
    "hle_gold": {
        "loader": "load_hle_gold_tasks",
        "sources": ["futurehouse/hle-gold-bio-chem:train"],
        "expected_size": 149,
        "access": "gated_hf",
        "assets": "none",
        "strictness": "HF token required by upstream gated dataset",
    },
    "super_chemistry": {
        "loader": "load_super_chemistry_tasks",
        "sources": ["ZehuaZhao/SUPERChem:SUPERChem-500.parquet"],
        "expected_size": 500,
        "access": "public",
        "assets": "optional local image cache",
        "strictness": "online source verification required; pass include_images=True and require_images=True to validate image materialisation",
    },
    "bixbench": {
        "loader": "load_bixbench_tasks",
        "sources": ["futurehouse/BixBench"],
        "expected_size": 205,
        "access": "public",
        "assets": "none",
        "strictness": "online source verification required",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output file.")
    args = parser.parse_args()

    if args.json:
        text = json.dumps(BENCHMARK_SOURCES, indent=2, sort_keys=True)
    else:
        rows = [
            "| Benchmark | Access | Expected size | Official source(s) | Assets | Strictness |",
            "|---|---|---:|---|---|---|",
        ]
        for name, meta in BENCHMARK_SOURCES.items():
            rows.append(
                "| {name} | {access} | {expected_size} | {sources} | {assets} | {strictness} |".format(
                    name=name,
                    access=meta["access"],
                    expected_size=meta["expected_size"],
                    sources="<br>".join(meta["sources"]),
                    assets=meta["assets"],
                    strictness=meta["strictness"],
                )
            )
        text = "\n".join(rows)

    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
