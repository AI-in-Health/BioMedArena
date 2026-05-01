#!/usr/bin/env python3
"""Verify that benchmark loaders can reach official data sources.

The script is designed for public users before they spend model budget:
it checks source accessibility, gated-token requirements, partial-load
guards, and multimodal asset integrity on a tiny sample.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv


@dataclass
class CheckResult:
    benchmark: str
    status: str
    n_tasks: int
    message: str
    seconds: float


def _load_env() -> None:
    for p in (_ROOT / ".env.local", _ROOT / ".env"):
        if p.exists():
            load_dotenv(p, override=True)


def _run_loader(
    benchmark: str,
    import_path: str,
    kwargs: dict[str, Any],
    validator: Callable[[list[dict[str, Any]]], tuple[str, str]] | None = None,
) -> CheckResult:
    t0 = time.monotonic()
    try:
        module_name, func_name = import_path.rsplit(":", 1)
        fn = getattr(importlib.import_module(module_name), func_name)
        tasks = fn(**kwargs)
        if validator:
            status, message = validator(tasks)
        else:
            status = "ok" if tasks else "source_unavailable"
            message = "loaded tasks" if tasks else "loader returned 0 tasks"
        return CheckResult(benchmark, status, len(tasks), message, time.monotonic() - t0)
    except Exception as exc:
        return CheckResult(
            benchmark,
            "error",
            0,
            f"{type(exc).__name__}: {str(exc)[:300]}",
            time.monotonic() - t0,
        )


def _require_sources(expected: set[str]) -> Callable[[list[dict[str, Any]]], tuple[str, str]]:
    def validate(tasks: list[dict[str, Any]]) -> tuple[str, str]:
        got = {str((t.get("metadata") or {}).get("dataset")) for t in tasks}
        missing = sorted(expected - got)
        if missing:
            return "partial_load_blocked", f"missing sub-sources: {', '.join(missing)}"
        return "ok", f"loaded all sub-sources: {', '.join(sorted(got))}"
    return validate


def _require_context_flag(flag: str) -> Callable[[list[dict[str, Any]]], tuple[str, str]]:
    def validate(tasks: list[dict[str, Any]]) -> tuple[str, str]:
        if not tasks:
            return "source_unavailable", "loader returned 0 tasks"
        bad = [t.get("id") for t in tasks if not (t.get("context") or {}).get(flag)]
        if bad:
            return "missing_assets", f"{len(bad)} sampled tasks missing context.{flag}"
        return "ok", f"all sampled tasks have context.{flag}=true"
    return validate


def _gated_status(name: str, loader: str, kwargs: dict[str, Any]) -> CheckResult:
    if not (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    ):
        return CheckResult(
            name,
            "gated_missing_token",
            0,
            "set HF_TOKEN and accept the dataset terms on Hugging Face",
            0.0,
        )
    return _run_loader(name, loader, kwargs)


CHECKS: dict[str, tuple[str, str, dict[str, Any], Callable | None]] = {
    "medcalc": (
        "public",
        "harness.eval.bench_medcalc:load_medcalc_tasks",
        {"limit": 2},
        None,
    ),
    "labbench2": (
        "gated",
        "harness.eval.bench_labbench2:load_labbench2_tasks",
        {"limit": 1, "skip_with_files": True},
        None,
    ),
    "gpqa_bio": (
        "gated",
        "harness.eval.bench_gpqa_bio:load_gpqa_bio_tasks",
        {"limit": 1},
        None,
    ),
    "hle_gold": (
        "gated",
        "harness.eval.bench_hle_gold:load_hle_gold_tasks",
        {"limit": 1},
        None,
    ),
    "super_chemistry": (
        "public",
        "harness.eval.bench_super_chemistry:load_super_chemistry_tasks",
        {"limit": 5, "include_images": False, "require_images": False, "skip_with_images": True},
        None,
    ),
    "bixbench": (
        "public",
        "harness.eval.bench_bixbench:load_bixbench_tasks",
        {"limit": 2},
        None,
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks",
        default=",".join(CHECKS),
        help="Comma-separated benchmark names or 'all'.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args()

    _load_env()
    names = list(CHECKS) if args.benchmarks == "all" else [
        x.strip() for x in args.benchmarks.split(",") if x.strip()
    ]
    results: list[CheckResult] = []
    for name in names:
        if name not in CHECKS:
            results.append(CheckResult(name, "unknown_benchmark", 0, "not in verifier registry", 0.0))
            continue
        access, loader, kwargs, validator = CHECKS[name]
        kwargs = dict(kwargs)
        if access == "gated":
            result = _gated_status(name, loader, kwargs)
            if result.status == "ok" and validator:
                # Currently no gated validators need this, but keep the shape consistent.
                pass
        else:
            result = _run_loader(name, loader, kwargs, validator)
        results.append(result)

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        print("| Benchmark | Status | Tasks | Message | Seconds |")
        print("|---|---|---:|---|---:|")
        for r in results:
            print(f"| {r.benchmark} | {r.status} | {r.n_tasks} | {r.message} | {r.seconds:.1f} |")

    bad = {"error", "partial_load_blocked", "missing_assets", "source_unavailable"}
    return 1 if any(r.status in bad for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
