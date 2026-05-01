#!/usr/bin/env python3
"""Offline install smoke for BioMedArena-Eval quick suite.

This does not call an LLM. It validates that the package can import,
the quick-suite benchmark is registered, all 20 built-in tasks load, and
the deterministic scorers accept canonical answer formats.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.cli import BENCHMARKS, MODES, BACKBONES
from harness.eval.bench_quick_suite import load_quick_suite_tasks
from harness.eval.scoring import score_question
from harness.tools import TOOL_SPECS

EXPECTED_ANSWER_TYPES = {
    "multipleChoice": 5,
    "exactMatch": 5,
    "exactNumeric": 5,
    "openText": 5,
}


def _canonical_prediction(task: dict) -> str:
    answer = str(task["answer"])
    answer_type = task["answer_type"]
    if answer_type == "multipleChoice":
        return f"The answer is {answer}."
    if answer_type == "exactNumeric":
        return f"The answer is {answer}."
    if answer_type == "exactMatch":
        return f"The answer is {answer}."
    return answer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true",
                        help="Print machine-readable JSON only.")
    args = parser.parse_args()

    tasks = load_quick_suite_tasks()
    records = []
    answer_type_counts = {k: 0 for k in EXPECTED_ANSWER_TYPES}
    for task in tasks:
        answer_type = str(task["answer_type"])
        if answer_type in answer_type_counts:
            answer_type_counts[answer_type] += 1
        pred = _canonical_prediction(task)
        correct = score_question(
            pred,
            str(task["answer"]),
            answer_type,
            task.get("context") or {},
        )
        records.append({
            "id": task["id"],
            "answer_type": task["answer_type"],
            "correct": bool(correct),
        })

    scored = sum(1 for r in records if r["correct"])
    quick_suite_registered = "quick_suite" in BENCHMARKS
    registry_ready = (
        quick_suite_registered
        and len(MODES) > 0
        and len(BACKBONES) > 0
        and len(TOOL_SPECS) > 0
    )
    task_shape_ready = (
        len(tasks) == 20
        and answer_type_counts == EXPECTED_ANSWER_TYPES
        and all({"id", "question", "answer", "answer_type"}.issubset(task) for task in tasks)
    )

    summary = {
        "ok": registry_ready and task_shape_ready and scored == len(tasks),
        "tasks": len(tasks),
        "scored": scored,
        "answer_type_counts": answer_type_counts,
        "benchmarks_registered": len(BENCHMARKS),
        "quick_suite_registered": quick_suite_registered,
        "modes_registered": len(MODES),
        "backbones_registered": len(BACKBONES),
        "tools_registered": len(TOOL_SPECS),
    }

    if args.json:
        print(json.dumps({"summary": summary, "records": records}, indent=2))
    else:
        print("BioMedArena-Eval quick suite")
        print(f"  benchmark registered: {summary['quick_suite_registered']}")
        print(f"  tasks loaded:         {summary['tasks']}")
        print(f"  scorer checks:        {summary['scored']}/{summary['tasks']}")
        print(f"  benchmarks:           {summary['benchmarks_registered']}")
        print(f"  modes:                {summary['modes_registered']}")
        print(f"  backbones:            {summary['backbones_registered']}")
        print(f"  tools:                {summary['tools_registered']}")
        print("  status:               PASS" if summary["ok"] else "  status:               FAIL")

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
