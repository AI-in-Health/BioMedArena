"""Command-line interface for BioMedArena.

Entry point is exposed as ``bioagent`` via ``pyproject.toml``
``[project.scripts]``. Run ``bioagent --help`` for usage.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Static metadata
# ---------------------------------------------------------------------------


BENCHMARKS: dict[str, dict[str, Any]] = {
    "medcalc": {
        "loader": "load_medcalc_tasks",
        "benchmark_key": "medcalc",
    },
    "medxpertqa": {
        "loader": "load_medxpertqa_tasks",
        "benchmark_key": "medxpertqa",
        "kwargs": {"subset": "Text"},
    },
    "labbench": {
        "loader": "load_labbench_tasks",
        "benchmark_key": "labbench",
        "kwargs": {"subsets": ["LitQA2", "CloningScenarios", "ProtocolQA"]},
        "scorer": "labbench_official_evaluator",
    },
    "labbench2": {
        "loader": "load_labbench2_tasks",
        "benchmark_key": "labbench2",
        "kwargs": {"subsets": ["litqa3", "patentqa", "trialqa", "dbqa2",
                               "suppqa2", "figqa2", "tableqa2"],
                   "include_multimodal": False,
                   "skip_with_files": True},
    },
    "bioasq": {
        "loader": "load_bioasq_tasks",
        "benchmark_key": "bioasq",
        "scorer": "bioasq_official",
    },
    "gpqa_bio": {
        "loader": "load_gpqa_bio_tasks",
        "benchmark_key": "hle_bio_med_chem",
    },
    "hle_gold": {
        "loader": "load_hle_gold_tasks",
        "benchmark_key": "hle_gold",
        "kwargs": {"include_chemistry": True},
    },
    "super_chemistry": {
        "loader": "load_super_chemistry_tasks",
        "benchmark_key": "super_chemistry",
        "kwargs": {"language": "en", "include_images": False,
                   "require_images": False, "skip_with_images": False},
    },
    "aa_lcr": {
        "loader": "load_aa_lcr_tasks",
        "benchmark_key": "aa_lcr",
    },
    "quick_suite": {
        "loader": "load_quick_suite_tasks",
        "benchmark_key": "quick_suite",
    },
    "pathvqa": {
        "loader": "load_pathvqa_tasks",
        "benchmark_key": "pathvqa",
    },
    "agentclinic": {
        "loader": "load_agentclinic_tasks",
        "benchmark_key": "agentclinic",
        "scorer": "agentclinic_dialogue_loop",
    },
    "medagentbench": {
        "loader": "load_medagentbench_tasks",
        "benchmark_key": "medagentbench",
    },
    "bixbench": {
        "loader": "load_bixbench_tasks",
        "benchmark_key": "bixbench_closed_book",
        "scorer": "bixbench_agent_runner",
    },
    "genotex": {
        "loader": "load_genotex_tasks",
        "benchmark_key": "genotex",
    },
    "medxpertqa_mm": {
        "loader": "load_medxpertqa_mm_tasks",
        "benchmark_key": "medxpertqa_mm",
        "kwargs": {"require_images": False},
    },
    "rag_essential": {
        "loader": "load_rag_essential_tasks",
        "benchmark_key": "rag_essential",
    },
    "medqa": {
        "loader": "load_medical_qa_tasks",
        "benchmark_key": "medqa",
        "kwargs": {"sources": ["medqa"]},
    },
    "medmcqa": {
        "loader": "load_medical_qa_tasks",
        "benchmark_key": "medmcqa",
        "kwargs": {"sources": ["medmcqa"]},
    },
    "pubmedqa": {
        "loader": "load_medical_qa_tasks",
        "benchmark_key": "pubmedqa",
        "kwargs": {"sources": ["pubmedqa"]},
    },
    "mmlu": {
        "loader": "load_mmlu_tasks",
        "benchmark_key": "mmlu",
    },
    "healthbench": {
        "loader": "load_healthbench_tasks",
        "benchmark_key": "healthbench",
        "scorer": "openai_simple_evals_rubric_judge",
    },
    "bioprobench": {
        "loader": "load_bioprobench_tasks",
        "benchmark_key": "bioprobench",
    },
    "medhelm": {
        "loader": "load_medhelm_tasks",
        "benchmark_key": "medhelm",
    },
    "superchem": {
        "loader": "load_superchem_tasks",
        "benchmark_key": "superchem",
    },
    "supergpqa": {
        "loader": "load_supergpqa_tasks",
        "benchmark_key": "supergpqa",
    },
}

try:
    from harness.eval.hf_benchmark_registry import HF_DEPRECATED_ALIASES, hf_benchmark_cli_entries
    BENCHMARKS.update(hf_benchmark_cli_entries())
    for alias, canonical in HF_DEPRECATED_ALIASES.items():
        if canonical in BENCHMARKS and alias not in BENCHMARKS:
            meta = dict(BENCHMARKS[canonical])
            meta["deprecated_alias_for"] = canonical
            BENCHMARKS[alias] = meta
except Exception:
    # Keep the core CLI usable if optional HF registry imports fail.
    pass


BACKBONES: dict[str, dict[str, str]] = {
    # Anthropic Claude family
    "claude-sonnet-4-6": {
        "provider": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "claude-opus-4-6": {
        "provider": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "claude-sonnet-4": {
        "provider": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "claude-sonnet-4-5": {
        "provider": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "claude-opus-4-5": {
        "provider": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    # OpenAI family
    "gpt-4o": {
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
    },
    # Google Gemini family
    "gemini-2.5-flash": {
        "provider": "gemini",
        "api_key_env": "GEMINI_API_KEY",
    },
    "gemini-2.5-pro": {
        "provider": "gemini",
        "api_key_env": "GEMINI_API_KEY",
    },
    "gemini-3-flash-preview": {
        "provider": "gemini",
        "api_key_env": "GEMINI_API_KEY",
    },
}



MODES = [
    ("simple_llm", "Pure LLM baseline, no tools."),
    ("deep_think",
     "Native thinking/reasoning (Claude Extended Thinking, Gemini "
     "ThinkingConfig, OpenAI reasoning models)."),
    ("light", "Single-turn tool calling with the LLM."),
    ("heavy", "Multi-turn ReAct loop with tool retrieval."),
]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _cmd_list_benchmarks(args: argparse.Namespace) -> int:
    del args
    for name in sorted(BENCHMARKS.keys()):
        print(name)
    return 0


def _cmd_list_backbones(args: argparse.Namespace) -> int:
    del args
    for name in sorted(BACKBONES.keys()):
        meta = BACKBONES[name]
        print(f"{name:<24} provider={meta['provider']:<10} "
              f"api_key_env={meta['api_key_env']}")
    return 0


def _cmd_list_modes(args: argparse.Namespace) -> int:
    del args
    for name, desc in MODES:
        print(f"{name:<20} {desc}")
    return 0


def _ensure_env() -> None:
    """Load ``.env`` from the current working directory if present.

    Uses ``override=True`` because shells may export keys as empty
    strings, which ``override=False`` would keep.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    import os
    preserved = {
        key: os.environ.get(key)
        for key in (
            "BIOAGENT_JUDGE_PROVIDER",
            "BIOAGENT_JUDGE_MODEL",
            "BIOAGENT_LLM_JUDGE",
        )
        if os.environ.get(key) is not None
    }
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    extra_env_file = os.environ.get("BIOAGENT_ENV_FILE")
    if extra_env_file:
        candidates.insert(0, Path(extra_env_file).expanduser())
    for candidate in candidates:
        if candidate.exists():
            try:
                load_dotenv(candidate, override=True)
                os.environ.update(preserved)
                return
            except OSError:
                continue


def _make_temp_config(
    backbone: str,
    max_iterations: int | None = None,
    min_iterations: int | None = None,
) -> Path:
    """Materialize a temp config.yaml with the selected backbone wired in.

    Starts from the repository's ``config.yaml`` (or ``config_claude.yaml``
    if only the claude-specific base is present) and overwrites the
    ``llm`` block. Also injects ``function_calling.max_iterations`` and
    ``function_calling.min_iterations`` when provided via CLI.
    """
    import yaml

    meta = BACKBONES[backbone]
    base_path = Path("config.yaml")
    if not base_path.exists():
        base_path = Path("config_claude.yaml")
    base = yaml.safe_load(base_path.read_text()) if base_path.exists() else {}

    base["llm"] = {
        "provider": meta["provider"],
        "model": backbone,
        "api_key": "${" + meta["api_key_env"] + "}",
    }

    # Inject CLI iteration overrides into function_calling config
    if max_iterations is not None or min_iterations is not None:
        fc = base.setdefault("function_calling", {})
        if max_iterations is not None:
            fc["max_iterations"] = max_iterations
        if min_iterations is not None:
            fc["min_iterations"] = min_iterations

    tmp = Path(f"/tmp/bioagent_cfg_{backbone}.yaml")
    tmp.write_text(yaml.safe_dump(base))
    return tmp


def _load_tasks(benchmark: str, limit: int, seed: int) -> list[dict[str, Any]]:
    import importlib

    meta = BENCHMARKS[benchmark]
    if meta.get("deprecated_alias_for"):
        print(
            f"warning: benchmark {benchmark!r} is deprecated; "
            f"redirecting to {meta['deprecated_alias_for']!r}",
            file=sys.stderr,
        )
    loader_name = meta["loader"]
    short = loader_name.replace("load_", "").replace("_tasks", "")
    module = importlib.import_module(f"harness.eval.bench_{short}")
    loader = getattr(module, loader_name)

    kwargs = dict(meta.get("kwargs") or {})
    kwargs.setdefault("limit", limit)
    tasks = loader(**kwargs)

    if len(tasks) > limit:
        rng = random.Random(seed)
        tasks = rng.sample(list(tasks), limit)
    return list(tasks)


# Infra errors that warrant retry (not the model's fault)
_INFRA_ERROR_PATTERNS = (
    "No module named",          # missing dependency
    "ConnectTimeoutError",      # network timeout
    "ReadTimeoutError",         # network timeout
    "EndpointConnectionError",  # endpoint unreachable
    "ServiceUnavailableException",  # service 503
    "Error code: 403",          # HTTP 403 forbidden
    "Error code: 429",          # HTTP 429 rate limit / too many tokens
    "too many tokens",          # token-rate throttle
    "usage limits",             # API usage/spending limits
    "rate limit",               # generic rate limit
    "APIConnectionError",       # httpx connection error
    "APITimeoutError",          # httpx timeout
    "InternalServerError",      # provider 500
    "overloaded",               # Anthropic overloaded
    "status 529",               # Anthropic overloaded HTTP status
    "error 529",                # Anthropic overloaded HTTP status
    "quota",                    # quota exceeded
    "billing",                  # billing issue
    "ECONNREFUSED",             # connection refused
    "ETIMEDOUT",                # connection timed out
    "exhausted retries",        # llm_client.py retry loop fully exhausted
    "content filtering",        # provider content filter block
    "Error code: 500",          # HTTP 500 server error
)


def _is_infra_error(error_str: str) -> bool:
    """Return True if the error is an infrastructure/transient issue, not model's fault."""
    if not error_str:
        return False
    return any(pat.lower() in error_str.lower() for pat in _INFRA_ERROR_PATTERNS)


def _load_completed(output_path: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Load results from a previous run, split into (done, needs_rejudge, aborted).

    - done: fully complete (model answered + judge OK) — skip entirely
    - needs_rejudge: model answered but judge failed — only re-run judge
    - aborted: infra error during model call — re-run everything
    """
    try:
        data = json.loads(Path(output_path).read_text())
        per_question = data.get("per_question", [])
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return [], [], []

    done = []
    needs_rejudge = []
    aborted = []
    for q in per_question:
        err = q.get("error", "")
        judge_err = q.get("judge_error")
        pred = str(q.get("predicted", "") or "")
        pred_raw = str(q.get("predicted_raw", "") or "")
        if err and _is_infra_error(err):
            # Model call failed due to infra — re-run everything
            aborted.append(q)
        elif _is_infra_error(pred) or _is_infra_error(pred_raw):
            # Prediction IS an error message (runner caught exception
            # internally and returned it as text) — treat as aborted
            aborted.append(q)
        elif judge_err:
            # Model has prediction but judge failed — only re-judge
            needs_rejudge.append(q)
        else:
            # Fully complete
            done.append(q)
    return done, needs_rejudge, aborted


async def _run_once(
    benchmark: str, backbone: str, mode: str, limit: int, seed: int,
    output: str | None, verbose: bool, web_tools: str = "off",
    resume: bool = False,
    max_iterations: int | None = None, min_iterations: int | None = None,
    enable_thinking: bool | None = None,
) -> int:
    _ensure_env()
    if benchmark not in BENCHMARKS:
        print(f"ERROR: unknown benchmark {benchmark!r}. "
              f"Use `bioagent list-benchmarks` to see options.",
              file=sys.stderr)
        return 2
    if backbone not in BACKBONES:
        print(f"ERROR: unknown backbone {backbone!r}. "
              f"Use `bioagent list-backbones` to see options.",
              file=sys.stderr)
        return 2
    valid_modes = {m for m, _ in MODES}
    if mode not in valid_modes and not mode.startswith("self_consistency:"):
        print(f"ERROR: unknown mode {mode!r}. "
              f"Use `bioagent list-modes` to see options.",
              file=sys.stderr)
        return 2

    import os

    meta = BACKBONES[backbone]
    if not os.environ.get(meta["api_key_env"]):
        print(f"ERROR: {meta['api_key_env']} is not set. "
              f"Populate .env or export the env var.", file=sys.stderr)
        return 3

    # Validate web-tools prerequisites
    if web_tools != "off" and not os.environ.get("SERPER_API_KEY"):
        print("ERROR: --web-tools requires SERPER_API_KEY in .env or env.",
              file=sys.stderr)
        return 3

    # Configure web tools mode via environment variable so
    # FunctionCallingRunner and BenchmarkSuite can read it.
    os.environ["BIOAGENT_WEB_TOOLS"] = web_tools

    from harness.eval.benchmark_suite import BenchmarkSuite

    cfg_path = _make_temp_config(
        backbone,
        max_iterations=max_iterations, min_iterations=min_iterations,
    )
    tasks = _load_tasks(benchmark, limit=limit, seed=seed)
    if not tasks:
        print(f"Loader returned 0 tasks for {benchmark}.")
        return 0

    # --resume: load previous results, split into done / needs_rejudge / aborted
    done_results: list[dict] = []
    rejudge_results: list[dict] = []
    if resume and output:
        done, needs_rejudge, aborted = _load_completed(output)
        if done or needs_rejudge or aborted:
            # Tasks to skip (fully complete)
            done_ids = {str(r["id"]) for r in done}
            # Tasks to re-run model (aborted + not started)
            tasks = [t for t in tasks if str(t.get("id", "")) not in done_ids]
            # Also remove rejudge tasks from model-run list (they don't need model re-run)
            rejudge_ids = {str(r["id"]) for r in needs_rejudge}
            tasks = [t for t in tasks if str(t.get("id", "")) not in rejudge_ids]

            n_not_started = len(tasks) - len(aborted)
            print(f"[resume] Previous run: {len(done)} done, "
                  f"{len(needs_rejudge)} needs rejudge, "
                  f"{len(aborted)} aborted (infra error → full retry), "
                  f"{max(0, n_not_started)} not started.")
            done_results = done

            # Phase 1: Re-judge tasks that have predictions but judge failed
            if needs_rejudge:
                print(f"[rejudge] Re-scoring {len(needs_rejudge)} tasks (model predictions kept)...")
                from harness.eval.llm_judge import score_with_fallback
                target_backbone = backbone
                rejudged_count = 0
                for q in needs_rejudge:
                    prediction = q.get("predicted", "") or q.get("predicted_raw", "")
                    if not prediction:
                        # No prediction to judge — keep as-is
                        rejudge_results.append(q)
                        continue
                    # Build a minimal task dict for score_with_fallback
                    task_dict = {
                        "id": q["id"],
                        "question": q.get("question_text", ""),
                        "answer": q["expected"],
                        "answer_type": q.get("answer_type", ""),
                        "context": q.get("context"),
                    }
                    try:
                        score_out = await score_with_fallback(
                            task_dict, prediction, target_backbone=target_backbone,
                        )
                        q["task_success"] = bool(score_out["correct"])
                        q["score_method"] = score_out.get("method", "")
                        new_judge_err = (score_out.get("details") or {}).get("judge_error")
                        q["judge_error"] = new_judge_err
                        if not new_judge_err:
                            rejudged_count += 1
                    except Exception as exc:
                        q["judge_error"] = f"rejudge_error: {exc}"
                    rejudge_results.append(q)
                print(f"[rejudge] Done: {rejudged_count}/{len(needs_rejudge)} successfully re-judged.")

            if not tasks:
                if not needs_rejudge:
                    print(f"[resume] All {len(done)} tasks completed. Nothing to run.")
                    # Still need to write merged results if rejudge happened
                    if not rejudge_results:
                        return 0

    suite = BenchmarkSuite(config_path=str(cfg_path))
    bench_key = BENCHMARKS[benchmark].get("benchmark_key", benchmark)

    # Run model on remaining tasks (aborted + not started)
    new_results: list[dict] = []
    if tasks:
        task_list = [dict(t) for t in tasks]
        # Build task ID → context map so we can save context for rejudge
        task_context_map = {str(t.get("id", "")): t.get("context") for t in task_list}
        metrics = await suite.eval_tasks(
            benchmark, task_list, mode,
            benchmark_key=bench_key,
            enable_thinking=enable_thinking,
        )
        new_results = [
            {
                "id": q.question_id,
                "question_text": q.question_text,
                "answer_type": getattr(q, "_answer_type", ""),
                "expected": q.expected,
                "predicted": q.predicted,
                "predicted_raw": q.predicted_raw,
                "context": task_context_map.get(str(q.question_id)),
                "task_success": q.task_success,
                "tool_calls_made": q.tool_calls_made or [],
                "latency_s": q.latency_s,
                "error": q.error,
                "score_method": getattr(q, "_score_method", ""),
                "judge_error": getattr(q, "_judge_error", None),
            }
            for q in metrics.per_question
        ]

    # Merge: done + rejudged + new
    all_results = done_results + rejudge_results + new_results

    # Print summary (over ALL results including resumed)
    n = len(all_results)
    correct = sum(1 for q in all_results if q.get("task_success"))
    accuracy = correct / n if n else 0.0
    judge_failed = sum(1 for q in all_results if q.get("judge_error"))
    infra_errors = sum(1 for q in all_results
                       if q.get("error") and _is_infra_error(q["error"]))

    # Clean accuracy: exclude aborted (infra error) and needs_rejudge tasks
    clean_results = [
        q for q in all_results
        if not (q.get("error") and _is_infra_error(q["error"]))
        and not _is_infra_error(str(q.get("predicted", "") or ""))
        and not _is_infra_error(str(q.get("predicted_raw", "") or ""))
        and not q.get("judge_error")
    ]
    n_clean = len(clean_results)
    correct_clean = sum(1 for q in clean_results if q.get("task_success"))
    accuracy_clean = correct_clean / n_clean if n_clean else 0.0
    try:
        from harness.eval.official_metrics import compute_official_metrics
        official_metrics = compute_official_metrics(benchmark, clean_results or all_results)
    except Exception as exc:
        official_metrics = {"status": "error", "reason": str(exc)}

    print(f"Benchmark: {benchmark}")
    print(f"Backbone:  {backbone}")
    print(f"Mode:      {mode}")
    n_resumed = len(done_results) + len(rejudge_results)
    print(f"Tasks:     {n}" + (f"  ({n_resumed} resumed + {len(new_results)} new)"
                               if n_resumed else ""))
    print(f"Correct:   {correct}")
    print(f"Accuracy:  {accuracy:.1%}")
    if infra_errors or judge_failed:
        n_excluded = n - n_clean
        print(f"Clean:     {correct_clean}/{n_clean} ({accuracy_clean:.1%})"
              f"  [excluded {n_excluded}: {infra_errors} infra + {judge_failed} judge_failed]")
    if judge_failed:
        print(f"Judge failed: {judge_failed} tasks (fell back to string match)")
    if infra_errors:
        print(f"Infra errors: {infra_errors} tasks (will retry on --resume)")
    if official_metrics.get("status") == "ok":
        metric_bits = [
            f"{k}={v:.4f}" for k, v in official_metrics.items()
            if isinstance(v, float) and k not in {"n_records", "n_scored"}
        ]
        if metric_bits:
            print("Official:  " + ", ".join(metric_bits))
    elif official_metrics.get("status") == "unavailable":
        print(f"Official:  unavailable ({official_metrics.get('reason')})")

    if verbose:
        print("\nPer-task:")
        for q in all_results:
            mark = "OK " if q.get("task_success") else "   "
            err = f"  [error: {str(q.get('error', ''))[:80]}]" if q.get("error") else ""
            print(f"  [{mark}] {q['id']}: "
                  f"expected={str(q.get('expected', ''))[:40]!r} "
                  f"predicted={str(q.get('predicted', ''))[:40]!r}{err}")

    if output:
        payload = {
            "benchmark": benchmark,
            "backbone": backbone,
            "mode": mode,
            "n_tasks": n,
            "n_correct": correct,
            "accuracy": accuracy,
            "n_clean": n_clean,
            "n_correct_clean": correct_clean,
            "accuracy_clean": accuracy_clean,
            "n_infra_errors": infra_errors,
            "n_judge_failed": judge_failed,
            "official_metrics": official_metrics,
            "per_question": all_results,
        }
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nWrote: {output}")

    return 0


def _resolve_cli_args(args: argparse.Namespace) -> tuple[str, str, "bool | None"]:
    """Resolve --tools/--reasoning-mode/--enable-thinking into (mode, web_tools, enable_thinking).

    When the new ``--tools`` interface is used, returns a concrete
    ``enable_thinking`` bool.  When the legacy ``--mode`` interface is
    used, returns ``None`` so each runner keeps its built-in default.
    """
    if getattr(args, "tools", None) is not None:
        # ---------- New unified interface ----------
        if args.tools == "off":
            web_tools = "off"
            if getattr(args, "enable_thinking", None) is not None:
                enable_thinking: bool | None = bool(args.enable_thinking)
                mode = "deep_think" if enable_thinking else "simple_llm"
            else:
                mode = "deep_think"
                enable_thinking = True
        else:
            if getattr(args, "reasoning_mode", None) is None:
                print("ERROR: --reasoning-mode is required when --tools is not 'off'.",
                      file=sys.stderr)
                sys.exit(2)
            mode = ("light" if args.reasoning_mode == "light"
                    else "heavy")
            web_tools = {"biomed": "off", "search": "only",
                         "all": "combined"}[args.tools]
            if getattr(args, "enable_thinking", None) is not None:
                enable_thinking = bool(args.enable_thinking)
            else:
                enable_thinking = (args.reasoning_mode == "heavy")
        # Wrap with self-consistency if requested
        if getattr(args, "self_consistency", False):
            mode = f"self_consistency:{mode}"
        return mode, web_tools, enable_thinking
    # ---------- Legacy --mode / --web-tools interface ----------
    mode = args.mode
    web_tools = args.web_tools
    enable_thinking = None

    # Wrap with self-consistency if requested
    if getattr(args, "self_consistency", False) and not mode.startswith("self_consistency"):
        mode = f"self_consistency:{mode}"
    return mode, web_tools, enable_thinking


def _cmd_run(args: argparse.Namespace) -> int:
    mode, web_tools, enable_thinking = _resolve_cli_args(args)
    return asyncio.run(_run_once(
        benchmark=args.benchmark,
        backbone=args.backbone,
        mode=mode,
        limit=args.limit,
        seed=args.seed,
        output=args.output,
        verbose=args.verbose,
        web_tools=web_tools,
        resume=args.resume,
        max_iterations=args.max_iterations,
        min_iterations=args.min_iterations,
        enable_thinking=enable_thinking,
    ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bioagent",
        description="BioMedArena — evaluate LLM agents on biomedical benchmarks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-benchmarks", help="List registered benchmarks.").set_defaults(
        func=_cmd_list_benchmarks
    )
    sub.add_parser("list-backbones", help="List supported LLM backbones.").set_defaults(
        func=_cmd_list_backbones
    )
    sub.add_parser("list-modes", help="List harness modes.").set_defaults(
        func=_cmd_list_modes
    )

    run = sub.add_parser("run", help="Evaluate one benchmark x backbone x mode cell.")
    run.add_argument("--benchmark", required=True,
                     help="Benchmark name. See `bioagent list-benchmarks`.")
    run.add_argument("--backbone", required=True,
                     help="LLM backbone. See `bioagent list-backbones`.")
    run.add_argument("--mode", default="heavy",
                     help="Harness mode. Default: heavy.")
    run.add_argument("--limit", type=int, default=10,
                     help="Maximum number of tasks to evaluate. Default: 10.")
    run.add_argument("--seed", type=int, default=42,
                     help="Random seed for task selection when loader returns "
                          "more than --limit tasks. Default: 42.")
    run.add_argument("--output",
                     help="Optional path to write a per-task JSON summary.")
    run.add_argument("--verbose", action="store_true",
                     help="Print per-task details.")
    run.add_argument("--web-tools", choices=["off", "only", "combined"],
                     default="off",
                     help="Web search tool mode. "
                          "'off': no Serper/Jina (default, existing behavior). "
                          "'only': use ONLY Serper+Jina as tools. "
                          "'combined': use Serper+Jina together with benchmark category tools.")
    # --- New unified interface (preferred over --mode/--web-tools) ---
    run.add_argument("--tools", choices=["off", "biomed", "search", "all"],
                     default=None,
                     help="Tool configuration. 'off': thinking only (no tools). "
                          "'biomed': domain-specific biomedical tools. "
                          "'search': web search tools (Serper+Jina). "
                          "'all': domain + web search combined. "
                          "When provided, overrides --mode/--web-tools.")
    run.add_argument("--reasoning-mode", choices=["light", "heavy"],
                     default=None,
                     help="Reasoning depth. 'light': single-turn function calling. "
                          "'heavy': multi-turn ReAct loop. "
                          "Required when --tools is not 'off'.")
    run.add_argument("--enable-thinking", type=int, choices=[0, 1],
                     default=None, dest="enable_thinking",
                     help="Override extended thinking (0=off, 1=on). "
                          "Default: ON for heavy/off, OFF for light.")
    run.add_argument("--resume", action="store_true",
                     help="Resume from a partial run. Loads completed task IDs "
                          "from --output file and skips them. Requires --output.")
    run.add_argument("--max-iterations", type=int, default=None,
                     help="Maximum tool-call iterations per task. "
                          "Overrides the default (50) for all benchmarks.")
    run.add_argument("--min-iterations", type=int, default=None,
                     help="Minimum tool-call iterations before the model can "
                          "give a final answer. Default: 0.")
    run.add_argument("--self-consistency", action="store_true",
                     default=False, dest="self_consistency",
                     help="Enable self-consistency voting: run the inner mode N times "
                          "and majority-vote the answer. Configure N via config YAML "
                          "(self_consistency.n, default 5).")
    run.set_defaults(func=_cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
