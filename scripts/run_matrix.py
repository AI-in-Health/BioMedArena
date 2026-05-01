#!/usr/bin/env python3
"""Benchmark matrix runner.

Reads a matrix YAML (default ``configs/matrix_default.yaml``) and iterates
``backbones × modes × benchmarks``. Per cell: load tasks, evaluate via
``BenchmarkSuite``, write per-cell JSON to ``data/runs/matrix_<ts>/``.

Two run modes:

  Default (fresh run):
      python scripts/run_matrix.py --config configs/matrix_default.yaml

  Patch mode (append specific benchmarks to the latest existing run):
      python scripts/run_matrix.py --patch --benchmarks hle_gold,genotex

Cost monitor hard-stop and per-task / per-cell timeouts are honoured.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

_WORKTREE_ROOT = Path(__file__).resolve().parent.parent
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))

import yaml
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_env() -> None:
    """Load ``.env`` with ``override=True`` (a shell may export empty keys)."""
    for candidate in [
        _WORKTREE_ROOT / ".env",
        Path.cwd() / ".env",
    ]:
        if candidate.exists():
            load_dotenv(candidate, override=True)
            return


def _load_loader(name: str):
    """Return a loader function from ``harness/eval/bench_*.py`` by name."""
    short = name.replace("load_", "").replace("_tasks", "")
    try:
        mod = importlib.import_module(f"harness.eval.bench_{short}")
        return getattr(mod, name)
    except (ImportError, AttributeError) as exc:
        raise ImportError(f"Could not locate loader {name}: {exc}") from exc


def _make_temp_config(base_config_path: str, backbone: dict) -> Path:
    """Materialize a temp config.yaml with the desired backbone wired in."""
    base = yaml.safe_load(Path(base_config_path).read_text())
    base["llm"] = {
        "provider": backbone["provider"],
        "model": backbone["model"],
        "api_key": "${" + backbone["api_key_env"] + "}",
    }
    tmp = Path(f"/tmp/matrix_cfg_{backbone['id']}.yaml")
    tmp.write_text(yaml.safe_dump(base))
    return tmp


async def _run_cell(
    backbone: dict,
    mode: str,
    bench: dict,
    base_config_path: str,
    per_task_s: int,
    per_cell_s: int,
    out_path: Path,
    run_dir: Path | None = None,
    seed: int = 42,
) -> dict:
    """Run one ``(backbone, mode, benchmark)`` cell and persist a JSON file.

    When ``run_dir`` is provided, per-task trace JSON is written to
    ``<run_dir>/traces/<cell_id>/<task_id>.json`` and a single 22-column
    row is appended to ``<run_dir>/results.csv``.
    """
    from harness.eval.benchmark_suite import BenchmarkSuite
    from harness.cost_monitor import snapshot
    from harness.eval.results_writer import aggregate_cell, append_cell_row

    loader = _load_loader(bench["loader"])
    tasks = loader(**(bench.get("kwargs") or {}))

    cell_id = f"{backbone['id']}__{mode}__{bench['name']}"
    t_cell_start = time.monotonic()
    cost_before = snapshot()["total_usd"]

    # Collect per-task traces for aggregation into the CSV row.
    collected_traces: list = []
    traces_dir = (
        run_dir / "traces" / cell_id if run_dir is not None else None
    )

    def _trace_sink(task: dict, recorder) -> None:
        collected_traces.append(recorder)
        if traces_dir is not None:
            try:
                recorder.dump(traces_dir / f"{task.get('id', 'anon')}.json")
            except Exception:
                pass

    if not tasks:
        return {
            "cell": cell_id,
            "backbone": backbone["id"],
            "mode": mode,
            "benchmark": bench["name"],
            "summary": {
                "n_tasks": 0,
                "n_correct": 0,
                "n_error": 0,
                "accuracy": 0.0,
                "total_cost_usd": 0.0,
                "wall_s": 0.0,
                "note": "loader returned 0 tasks",
            },
            "questions": [],
        }

    cfg_path = _make_temp_config(base_config_path, backbone)
    suite = BenchmarkSuite(config_path=str(cfg_path))

    try:
        metrics = await asyncio.wait_for(
            suite.eval_tasks(
                bench["name"],
                [dict(t) for t in tasks],
                mode,
                benchmark_key=bench.get("benchmark_key", bench["name"]),
                backbone_id=backbone["id"],
                trace_sink=_trace_sink,
            ),
            timeout=per_cell_s,
        )
    except asyncio.TimeoutError:
        return {
            "cell": cell_id,
            "backbone": backbone["id"],
            "mode": mode,
            "benchmark": bench["name"],
            "summary": {
                "n_tasks": len(tasks),
                "n_correct": 0,
                "n_error": len(tasks),
                "accuracy": 0.0,
                "total_cost_usd": snapshot()["total_usd"] - cost_before,
                "wall_s": time.monotonic() - t_cell_start,
                "note": f"per-cell timeout {per_cell_s}s",
            },
            "questions": [],
        }
    except Exception as exc:
        return {
            "cell": cell_id,
            "backbone": backbone["id"],
            "mode": mode,
            "benchmark": bench["name"],
            "summary": {
                "n_tasks": len(tasks),
                "n_correct": 0,
                "n_error": len(tasks),
                "accuracy": 0.0,
                "total_cost_usd": snapshot()["total_usd"] - cost_before,
                "wall_s": time.monotonic() - t_cell_start,
                "note": (
                    f"cell-level exception: "
                    f"{type(exc).__name__}: {str(exc)[:200]}"
                ),
            },
            "questions": [],
        }

    cost_after = snapshot()["total_usd"]
    n_correct = sum(1 for q in metrics.per_question if q.task_success)
    n_error = sum(1 for q in metrics.per_question if q.error)

    result = {
        "cell": cell_id,
        "backbone": backbone["id"],
        "mode": mode,
        "benchmark": bench["name"],
        "summary": {
            "n_tasks": len(metrics.per_question),
            "n_correct": n_correct,
            "n_error": n_error,
            "accuracy": metrics.task_success_rate,
            "total_cost_usd": cost_after - cost_before,
            "wall_s": time.monotonic() - t_cell_start,
        },
        "questions": [
            {
                "id": q.question_id,
                "expected": q.expected,
                "predicted": q.predicted,
                "task_success": q.task_success,
                "latency_s": q.latency_s,
                "error": q.error,
                "tools": q.tool_calls_made,
            }
            for q in metrics.per_question
        ],
    }
    out_path.write_text(json.dumps(result, indent=2, default=str))

    # Append the 22-column row for this cell to results.csv.
    if run_dir is not None and collected_traces:
        row = aggregate_cell(
            collected_traces,
            benchmark=bench["name"],
            backbone=backbone["id"],
            mode=mode,
            seed=seed,
            n_error_override=n_error,
        )
        append_cell_row(run_dir / "results.csv", row)

    return result


def _iter_planned_cells(config: dict):
    """Yield ``(backbone, mode, bench)`` triples according to config rules."""
    backbones = config["backbones"]
    modes_default = config.get("modes_default") or config.get("modes") or []
    for bench in config["benchmarks"]:
        bench_modes = bench.get("modes") or modes_default
        bench_bb_restr = bench.get("backbone_restriction")
        for backbone in backbones:
            if bench_bb_restr and backbone["id"] not in bench_bb_restr:
                continue
            for mode in bench_modes:
                yield backbone, mode, bench


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _run_fresh(
    config_path: str,
    run_name: str | None = None,
    output_dir: str | None = None,
    limit_override: int | None = None,
    only: str | None = None,
) -> int:
    _load_env()
    config = yaml.safe_load(Path(config_path).read_text())
    base_config = config.get("base_config", "config.yaml")

    hard_stop = float(config["cost_monitor"]["hard_stop_usd"])
    soft_warn = float(config["cost_monitor"]["soft_warn_usd"])
    per_task_s = int(config["timeout"]["per_task_s"])
    per_cell_s = int(config["timeout"]["per_cell_s"])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    effective_run_name = run_name or config.get("run_name") or "matrix"
    if output_dir:
        run_dir = Path(output_dir)
    else:
        run_dir = Path(f"data/runs/{effective_run_name}_{ts}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Apply --limit-override to every benchmark entry.
    if limit_override is not None:
        for b in config.get("benchmarks", []):
            b_kwargs = b.setdefault("kwargs", {})
            b_kwargs["limit"] = int(limit_override)

    # Apply --only filter: keep cells matching benchmark,backbone,mode.
    only_filter: set[tuple[str, str, str]] | None = None
    if only:
        only_filter = set()
        for triple in only.split(";"):
            parts = [p.strip() for p in triple.split(",") if p.strip()]
            if len(parts) == 3:
                only_filter.add(tuple(parts))

    from harness.cost_monitor import snapshot, _LEDGER_PATH
    print(f"Cost ledger: {_LEDGER_PATH}")
    spend_at_start = snapshot()["total_usd"]
    print(f"Ledger total at start: ${spend_at_start:.4f}")

    planned = list(_iter_planned_cells(config))
    if only_filter:
        planned = [
            (bb, m, b) for (bb, m, b) in planned
            if (b["name"], bb["id"], m) in only_filter
        ]
    total_cells = len(planned)
    print(f"Planned cells: {total_cells}")
    seeds = config.get("seeds") or [42]
    seed = int(seeds[0])

    cell_results: list[dict] = []
    budget_stopped = False
    for idx, (backbone, mode, bench) in enumerate(planned, start=1):
        cell_id = f"{backbone['id']}__{mode}__{bench['name']}"
        cur_total = snapshot()["total_usd"]
        cur_session = cur_total - spend_at_start
        print(
            f"\n[{idx}/{total_cells}] {cell_id}  "
            f"(session spend: ${cur_session:.4f}, "
            f"ledger: ${cur_total:.4f})"
        )

        if cur_session >= hard_stop:
            print(
                f"BUDGET HARD STOP — session "
                f"${cur_session:.4f} >= ${hard_stop}"
            )
            budget_stopped = True
            break
        if cur_session >= soft_warn:
            print(f"BUDGET WARN — session ${cur_session:.4f} >= ${soft_warn}")

        out_path = run_dir / f"{cell_id}.json"
        try:
            result = await _run_cell(
                backbone, mode, bench, base_config,
                per_task_s, per_cell_s, out_path,
                run_dir=run_dir, seed=seed,
            )
            cell_results.append(result)
            s = result["summary"]
            print(
                f"  -> n={s['n_tasks']} correct={s['n_correct']} "
                f"err={s['n_error']} acc={s['accuracy']:.1%} "
                f"cost=${s['total_cost_usd']:.4f} "
                f"wall={s['wall_s']:.0f}s "
                f"{s.get('note', '')}"
            )
        except Exception as exc:
            print(f"  CELL FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    final_total = snapshot()["total_usd"]
    summary = {
        "config": config_path,
        "ts": ts,
        "n_cells_completed": len(cell_results),
        "n_cells_planned": total_cells,
        "session_spend_usd": final_total - spend_at_start,
        "ledger_total_usd": final_total,
        "budget_stopped": budget_stopped,
        "cells": [
            {
                "cell": r["cell"],
                "backbone": r.get("backbone"),
                "mode": r.get("mode"),
                "benchmark": r.get("benchmark"),
                "n": r["summary"]["n_tasks"],
                "correct": r["summary"]["n_correct"],
                "acc": r["summary"]["accuracy"],
                "err": r["summary"]["n_error"],
                "cost": r["summary"]["total_cost_usd"],
                "wall_s": r["summary"]["wall_s"],
                "note": r["summary"].get("note", ""),
            }
            for r in cell_results
        ],
    }
    (run_dir / "MATRIX_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(
        f"\nDone. {len(cell_results)}/{total_cells} cells. "
        f"Session spend: ${summary['session_spend_usd']:.4f}. "
        f"Summary: {run_dir}/MATRIX_SUMMARY.json"
    )
    return 0


async def _run_patch(config_path: str, patch_benchmarks: list[str]) -> int:
    _load_env()
    config = yaml.safe_load(Path(config_path).read_text())
    base_config = config.get("base_config", "config.yaml")

    hard_stop = float(config["cost_monitor"]["hard_stop_usd"])
    soft_warn = float(config["cost_monitor"]["soft_warn_usd"])
    per_task_s = int(config["timeout"]["per_task_s"])
    per_cell_s = int(config["timeout"]["per_cell_s"])

    run_dirs = sorted(
        list(Path("data/runs").glob("matrix_*"))
        + list(Path("data/runs").glob("v1_matrix_*"))
    )
    if not run_dirs:
        print(
            "ERROR: no existing matrix_* run dir. Run this script "
            "without --patch first."
        )
        return 2
    run_dir = run_dirs[-1]
    print(f"Appending to: {run_dir}")

    from harness.cost_monitor import snapshot
    spend_at_start = snapshot()["total_usd"]
    print(f"Ledger total at start: ${spend_at_start:.4f}")

    target_benches = [
        b for b in config["benchmarks"] if b["name"] in patch_benchmarks
    ]
    if not target_benches:
        print(f"ERROR: none of {patch_benchmarks} found in config.")
        return 2
    for b in target_benches:
        print(
            f"  patch benchmark: {b['name']}  "
            f"(key={b.get('benchmark_key')}, kwargs={b.get('kwargs')})"
        )

    patch_cfg = dict(config)
    patch_cfg["benchmarks"] = target_benches

    new_results: list[dict] = []
    budget_stopped = False
    for backbone, mode, bench in _iter_planned_cells(patch_cfg):
        cell_id = f"{backbone['id']}__{mode}__{bench['name']}"
        cell_path = run_dir / f"{cell_id}.json"
        cur_total = snapshot()["total_usd"]
        cur_session = cur_total - spend_at_start

        if cell_path.exists():
            print(f"\n[skip-exists] {cell_id}")
            continue

        print(
            f"\n[{cell_id}]  (session spend: ${cur_session:.4f}, "
            f"ledger: ${cur_total:.4f})"
        )
        if cur_session >= hard_stop:
            print(f"BUDGET HARD STOP at ${cur_session:.4f}")
            budget_stopped = True
            break
        if cur_session >= soft_warn:
            print(f"BUDGET WARN at ${cur_session:.4f}")

        try:
            result = await _run_cell(
                backbone, mode, bench, base_config,
                per_task_s, per_cell_s, cell_path,
                run_dir=run_dir,
            )
            new_results.append(result)
            s = result["summary"]
            print(
                f"  -> n={s['n_tasks']} correct={s['n_correct']} "
                f"err={s['n_error']} acc={s['accuracy']:.1%} "
                f"cost=${s['total_cost_usd']:.4f} "
                f"wall={s['wall_s']:.0f}s {s.get('note','')}"
            )
        except Exception as exc:
            print(f"  CELL FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    final_total = snapshot()["total_usd"]
    print(
        f"\nPatch complete. Session spend: "
        f"${final_total - spend_at_start:.4f}. New cells: "
        f"{len(new_results)}."
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a benchmark × backbone × mode matrix."
    )
    parser.add_argument(
        "--config", default="configs/matrix_default.yaml",
        help="Path to matrix YAML config (default: configs/matrix_default.yaml)",
    )
    parser.add_argument(
        "--patch", action="store_true",
        help="Append cells to the latest existing matrix_* run directory "
             "instead of creating a new one.",
    )
    parser.add_argument(
        "--benchmarks", default="",
        help="Comma-separated benchmark names. Required with --patch, "
             "ignored otherwise.",
    )
    parser.add_argument(
        "--run-name", default=None,
        help="Human-readable prefix for the output directory "
             "(default: config 'run_name' field or 'matrix').",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Explicit output directory. Overrides --run-name + timestamp.",
    )
    parser.add_argument(
        "--limit-override", type=int, default=None,
        help="Force every benchmark's `kwargs.limit` to this value "
             "(useful for dry runs).",
    )
    parser.add_argument(
        "--only", default=None,
        help="Filter to specific cells: "
             "'benchmark,backbone,mode[;benchmark,backbone,mode]...'",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.patch:
        names = [n.strip() for n in args.benchmarks.split(",") if n.strip()]
        if not names:
            print("ERROR: --patch requires --benchmarks name1,name2,...")
            sys.exit(2)
        sys.exit(asyncio.run(_run_patch(args.config, names)))
    sys.exit(asyncio.run(_run_fresh(
        args.config,
        run_name=args.run_name,
        output_dir=args.output_dir,
        limit_override=args.limit_override,
        only=args.only,
    )))
