#!/usr/bin/env python3
"""Offline release gate for BioMedArena.

This script intentionally avoids live LLM/HuggingFace/API calls. It checks the
parts of the public release contract that can be verified deterministically in
a clean checkout. Live gates (HF full audit, MCP round trips, provider calls)
remain separate release-environment checks.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _tool_name(spec: dict[str, Any]) -> str:
    if isinstance(spec.get("function"), dict):
        return str(spec["function"].get("name") or "")
    return str(spec.get("name") or "")


def _run_quick_suite() -> tuple[bool, dict[str, Any]]:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_quick_suite.py"), "--json"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return False, {"returncode": proc.returncode, "stderr": proc.stderr.strip()}
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return False, {"error": f"invalid quick suite JSON: {exc}", "stdout": proc.stdout}
    return bool(payload.get("summary", {}).get("ok")), payload


def check_quick_suite() -> list[str]:
    ok, payload = _run_quick_suite()
    if not ok:
        return [f"quick_suite failed: {payload}"]
    summary = payload["summary"]
    problems = []
    if summary.get("tasks") != 20:
        problems.append(f"quick_suite task count is {summary.get('tasks')}, expected 20")
    if summary.get("scored") != 20:
        problems.append(f"quick_suite scored {summary.get('scored')}/20")
    expected_types = {
        "multipleChoice": 5,
        "exactMatch": 5,
        "exactNumeric": 5,
        "openText": 5,
    }
    if summary.get("answer_type_counts") != expected_types:
        problems.append(f"quick_suite answer type counts drifted: {summary.get('answer_type_counts')}")
    return problems


def check_registry_counts() -> list[str]:
    from harness.cli import BACKBONES, BENCHMARKS, MODES
    from harness.tools import TOOL_SPECS

    problems = []
    thresholds = {
        "benchmarks": (len(BENCHMARKS), 100),
        "tools": (len(TOOL_SPECS), 75),
        "modes": (len(MODES), 4),
        "backbones": (len(BACKBONES), 3),
    }
    for name, (actual, minimum) in thresholds.items():
        if actual < minimum:
            problems.append(f"{name} registry too small: {actual} < {minimum}")
    return problems


def check_hf_metadata() -> list[str]:
    from harness.eval.hf_benchmark_registry import (
        HF_BENCHMARK_SPECS,
        HF_VERIFIED_BENCHMARK_KEYS,
        hf_verified_metadata,
        validate_hf_metadata,
    )

    problems = validate_hf_metadata()
    missing_specs = sorted(set(HF_VERIFIED_BENCHMARK_KEYS) - set(HF_BENCHMARK_SPECS))
    if missing_specs:
        problems.append(f"HF verified keys missing specs: {missing_specs[:10]}")
    metadata = hf_verified_metadata()
    if len(metadata) != len(HF_VERIFIED_BENCHMARK_KEYS):
        problems.append(
            f"HF metadata count {len(metadata)} != verified keys {len(HF_VERIFIED_BENCHMARK_KEYS)}"
        )
    if len(metadata) < 100:
        problems.append(f"HF verified metadata below 100 entries: {len(metadata)}")
    return problems


def check_tool_registry() -> list[str]:
    from harness.tools import TOOL_SPECS
    from harness.tools.tool_categories import (
        OPTIONAL_TOOL_CATEGORY_ENTRIES,
        TOOL_CATEGORIES,
        uncategorised_tools,
    )

    names = [_tool_name(spec) for spec in TOOL_SPECS]
    problems = []
    if any(not name for name in names):
        problems.append("one or more tool specs have no public name")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        problems.append(f"duplicate tool names: {duplicates}")
    for spec in TOOL_SPECS:
        name = _tool_name(spec)
        fn = spec.get("function", spec)
        if not isinstance(fn, dict):
            problems.append(f"{name}: tool spec is not a dict")
            continue
        if not fn.get("description"):
            problems.append(f"{name}: missing description")
        if "parameters" not in fn:
            problems.append(f"{name}: missing parameters schema")
    uncat = uncategorised_tools(TOOL_SPECS)
    if uncat:
        problems.append(f"uncategorised tools: {uncat}")
    stale = sorted(set(TOOL_CATEGORIES) - set(names) - set(OPTIONAL_TOOL_CATEGORY_ENTRIES))
    if stale:
        problems.append(f"stale non-optional category entries: {stale}")
    return problems


def check_docs_consistency() -> list[str]:
    from harness.cli import BACKBONES, BENCHMARKS, MODES
    from harness.tools import TOOL_SPECS

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    expected_fragments = [
        f"{len(BENCHMARKS)} registered",
        f"{len(TOOL_SPECS)} tools",
        f"{len(MODES)} modes",
        f"{len(BACKBONES)} registered",
    ]
    problems = [
        f"README missing current count phrase: {frag}"
        for frag in expected_fragments
        if frag not in readme
    ]
    for rel in (
        "docs/benchmark_datasets.md",
        "docs/tools_and_skills.md",
    ):
        if rel in readme and not (REPO_ROOT / rel).exists():
            problems.append(f"README references missing file: {rel}")
    return problems


def check_security_packaging() -> list[str]:
    problems = []
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if tracked.returncode != 0:
        return [f"git ls-files failed: {tracked.stderr.strip()}"]
    paths = tracked.stdout.splitlines()
    if any(path.startswith(".worktrees/") for path in paths):
        problems.append("tracked .worktrees content")
    if any(path.startswith("data/cache/") for path in paths):
        problems.append("tracked data/cache content")
    return problems


CHECKS = {
    "quick_suite": check_quick_suite,
    "registry_counts": check_registry_counts,
    "hf_metadata": check_hf_metadata,
    "tool_registry": check_tool_registry,
    "docs_consistency": check_docs_consistency,
    "security_packaging": check_security_packaging,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print JSON verdict.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on any failure.")
    args = parser.parse_args()

    results: dict[str, dict[str, Any]] = {}
    for name, fn in CHECKS.items():
        try:
            problems = fn()
        except Exception as exc:
            problems = [f"check crashed: {exc}"]
        results[name] = {"ok": not problems, "problems": problems}

    ok = all(item["ok"] for item in results.values())
    payload = {"ok": ok, "checks": results}
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("BioMedArena release gate")
        for name, result in results.items():
            status = "PASS" if result["ok"] else "FAIL"
            print(f"  {name:<20} {status}")
            for problem in result["problems"]:
                print(f"    - {problem}")
        print(f"  status               {'PASS' if ok else 'FAIL'}")
    return 1 if args.strict and not ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
