"""Unified BenchmarkSuite — runs all benchmarks across 3 modes, generates leaderboard."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from harness.eval.metrics import BenchmarkMetrics, QuestionMetric, build_metrics
from harness.eval.scoring import extract_answer_from_response
from harness.llm_client import LLMClient

logger = logging.getLogger(__name__)

# System prompts for the three modes
SIMPLE_SYSTEM = (
    "You are a helpful medical assistant. Answer the question concisely. "
    "If multiple choice, respond with ONLY the letter. "
    "If exact answer, respond with ONLY the answer."
)



class BenchmarkSuite:
    """Run multiple benchmarks across multiple modes and generate leaderboard."""

    def __init__(self, config_path: str = "config.yaml"):
        import os
        import yaml

        self._config_path = config_path
        raw = Path(config_path).read_text()
        for key, val in os.environ.items():
            raw = raw.replace(f"${{{key}}}", val)
        self._config = yaml.safe_load(raw)

        self.llm: LLMClient | None = None
        # Per-benchmark runner cache — keyed by `benchmark_key` so
        # retrieval-heavy benchmarks get the larger iteration budget
        # that calculator benchmarks don't need. See
        # DEFAULT_MAX_ITERATIONS in function_calling_runner.py.
        self._fc_runner_cache: dict[str, Any] = {}

    def _get_llm(self) -> LLMClient:
        if self.llm is None:
            cfg = self._config.get("llm", {})
            self.llm = LLMClient(
                provider=cfg.get("provider", "openai"),
                model=cfg.get("model", "gpt-4o"),
                api_key=cfg.get("api_key"),
            )
        return self.llm

    # ------------------------------------------------------------------
    # Core evaluation for a single mode on a task list
    # ------------------------------------------------------------------

    async def eval_tasks(
        self,
        benchmark_name: str,
        tasks: list[dict[str, Any]],
        mode: str,
        max_concurrent: int = 5,
        benchmark_key: str | None = None,
        backbone_id: str | None = None,
        trace_sink: "callable[[dict, Any], None] | None" = None,
        enable_thinking: bool | None = None,
    ) -> BenchmarkMetrics:
        """Evaluate a list of tasks in a given mode.

        Each task dict must have:
            id, question, answer, answer_type, category
        Optional: context (dict), raw_subject

        ``benchmark_key`` (default = ``benchmark_name``) routes to the
        correct per-benchmark iteration budget via
        ``DEFAULT_MAX_ITERATIONS``.

        ``backbone_id`` tags the TraceRecorder; defaults to the
        configured LLM model string.

        ``trace_sink`` — optional callback ``(task, trace) -> None``
        invoked after each task completes. The runner (e.g.
        ``scripts/run_matrix.py``) uses this to dump per-task trace
        JSON and to collect traces for CSV aggregation. Traces are
        always created per task; when ``trace_sink`` is ``None`` the
        trace is discarded after scoring.
        """
        self._enable_thinking = enable_thinking

        _heavy_modes = {"heavy"}
        sem = asyncio.Semaphore(max_concurrent if mode not in _heavy_modes else 5)
        question_metrics: list[QuestionMetric] = []

        effective_key = benchmark_key or benchmark_name
        effective_backbone = (
            backbone_id
            or (self._config.get("llm", {}) or {}).get("model", "unknown")
        )
        # Stamp every task so _run_function_calling can look up the
        # per-benchmark iteration budget. Caller-supplied task dicts are
        # not copied — just augmented in-place.
        for t in tasks:
            t.setdefault("_benchmark_key", effective_key)

        from harness.trace import TraceRecorder, active_trace

        async def _eval_one(task: dict) -> QuestionMetric:
            async with sem:
                start = time.monotonic()
                recorder = TraceRecorder(
                    task_id=str(task.get("id", "")),
                    benchmark=benchmark_name,
                    backbone=str(effective_backbone),
                    mode=mode,
                )
                try:
                    with active_trace(recorder):
                        result_metric = await _eval_one_inner(task, start)
                finally:
                    recorder.finalize()
                    if trace_sink is not None:
                        try:
                            trace_sink(task, recorder)
                        except Exception:
                            # Sink failures must never abort evaluation.
                            pass
                return result_metric

        async def _eval_one_inner(task: dict, start: float) -> QuestionMetric:
            try:
                if mode == "simple_llm":
                    resp, tools = await self._run_simple(task)
                elif mode == "deep_think":
                    resp, tools = await self._run_deep(task)
                elif mode == "heavy":
                    resp, tools = await self._run_harness(task)
                elif mode == "light":
                    resp, tools = await self._run_function_calling(task)
                elif mode.startswith("self_consistency"):
                    resp, tools = await self._run_self_consistency(task, mode)
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                latency = time.monotonic() - start
                extracted = extract_answer_from_response(resp, task["answer_type"])

                # 2-tier scoring: primary first, judge as fallback
                # (MCQ) or primary (open-ended). Falls back to the
                # deterministic ``score_question`` when the judge is
                # disabled via ``BIOAGENT_LLM_JUDGE=0``.
                from harness.eval.llm_judge import score_with_fallback
                from harness.trace import get_active_trace as _get_tr
                target_backbone = (self._config.get("llm", {}) or {}).get("model")
                score_out = await score_with_fallback(
                    task, resp, target_backbone=target_backbone,
                )
                success = bool(score_out["correct"])
                _tr_score = _get_tr()
                if _tr_score is not None:
                    _tr_score.set_final_answer(resp or "")
                    _tr_score.set_scorer_result(
                        correct=success,
                        method=score_out.get("method", ""),
                        details=score_out.get("details", {}),
                        llm_judge_invoked=bool(
                            (score_out.get("details") or {}).get("judge_invoked")
                        ),
                        llm_judge_verdict=(
                            (score_out.get("details") or {}).get("judge_verdict")
                        ),
                    )

                # Tool call accuracy: fraction of tools that are
                # relevant to category.
                tool_acc = (
                    self._score_tool_relevance(tools, task) if tools else 0.0
                )

                # Reasoning faithfulness: heuristic — does response
                # mention key terms?
                faith = self._score_faithfulness(resp, task, mode)

                qm = QuestionMetric(
                    question_id=task["id"],
                    benchmark=benchmark_name,
                    mode=mode,
                    category=task.get("category", ""),
                    question_text=task["question"][:200],
                    expected=task["answer"],
                    predicted=extracted,
                    predicted_raw=resp[:4000],
                    task_success=success,
                    tool_calls_made=tools,
                    tool_call_accuracy=tool_acc,
                    reasoning_faithfulness=faith,
                    latency_s=round(latency, 2),
                    context=task.get("context") or {},
                )
                # Attach scoring metadata for downstream analysis
                qm._score_method = score_out.get("method", "")
                qm._judge_error = (score_out.get("details") or {}).get("judge_error")
                qm._answer_type = task.get("answer_type", "")
                return qm
            except Exception as exc:
                qm_err = QuestionMetric(
                    question_id=task["id"],
                    benchmark=benchmark_name,
                    mode=mode,
                    category=task.get("category", ""),
                    question_text=task["question"][:200],
                    expected=task["answer"],
                    predicted="",
                    predicted_raw="",
                    task_success=False,
                    latency_s=round(time.monotonic() - start, 2),
                    error=str(exc),
                    context=task.get("context") or {},
                )
                qm_err._answer_type = task.get("answer_type", "")
                return qm_err

        coros = [_eval_one(t) for t in tasks]
        total = len(coros)
        for i, fut in enumerate(asyncio.as_completed(coros)):
            qm = await fut
            question_metrics.append(qm)
            if (i + 1) % max(1, total // 5) == 0 or i + 1 == total:
                acc = sum(1 for q in question_metrics if q.task_success) / len(question_metrics)
                print(f"  [{mode}] {i+1}/{total} done, accuracy: {acc:.1%}")

        return build_metrics(benchmark_name, mode, question_metrics)

    # ------------------------------------------------------------------
    # Three mode implementations
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # simple_llm / deep_think purity contract
    # ------------------------------------------------------------------
    #
    # deep_think is intentionally identical to simple_llm except that it
    # enables the provider's native reasoning budget
    # (Claude extended_thinking / Gemini ThinkingConfig / OpenAI
    # reasoning_effort). The prompts are shared; there is no
    # "think step by step" nudge added at the prompt layer. Any
    # per-benchmark ``system_prompt_hint`` / ``expected_answer_format``
    # applies identically in both modes. See
    # ``tests/unit/test_modes_purity.py``.

    def _build_pure_prompts(
        self, task: dict,
    ) -> tuple[str, str, list[str]]:
        """Shared prompt builder for simple_llm and deep_think.

        Returns ``(system_prompt, user_prompt, image_paths)``.
        """
        from harness.benchmark_config import build_system_prompt, build_user_prompt
        bench_key = task.get("_benchmark_key")
        system_prompt = build_system_prompt(bench_key, SIMPLE_SYSTEM)
        user_prompt = build_user_prompt(bench_key, task)
        image_paths = (task.get("context") or {}).get("image_paths", []) or []
        return system_prompt, user_prompt, image_paths

    async def _run_simple(self, task: dict) -> tuple[str, list[str]]:
        from harness.trace import get_active_trace
        llm = self._get_llm()
        system_prompt, user_prompt, image_paths = self._build_pure_prompts(task)

        # Exactly one model call per task -> iterations=1.
        tr = get_active_trace()
        if tr is not None:
            tr.increment_iteration()

        if image_paths:
            resp = await llm.chat_vision(
                system_prompt=system_prompt,
                user_text=user_prompt,
                image_paths=image_paths,
                temperature=0.0,
                max_tokens=1024,
            )
            return resp, ["vision"]

        resp = await llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        return resp, []

    async def _run_deep(self, task: dict) -> tuple[str, list[str]]:
        """Native thinking mode.

        **Purity contract**: identical to ``_run_simple`` except the
        provider's reasoning budget is enabled (via ``chat_think``). No
        extra CoT instruction is injected at the prompt layer; the
        ``system`` and ``user`` prompts are the same as ``simple_llm``.
        """
        from harness.trace import get_active_trace
        llm = self._get_llm()
        system_prompt, user_prompt, image_paths = self._build_pure_prompts(task)

        # Exactly one model call per task -> iterations=1.
        tr = get_active_trace()
        if tr is not None:
            tr.increment_iteration()

        if image_paths:
            # Vision doesn't expose a thinking-mode API; fall back to
            # the same vision call simple_llm would make. Larger
            # max_tokens because "deep" callers likely want more room.
            resp = await llm.chat_vision(
                system_prompt=system_prompt,
                user_text=user_prompt,
                image_paths=image_paths,
                temperature=0.0,
                max_tokens=4096,
            )
            return resp, ["vision"]

        resp = await llm.chat_think(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            thinking_budget=8192,
            max_tokens=16384,
        )
        return resp, []

    async def _run_function_calling(self, task: dict) -> tuple[str, list[str]]:
        """Multi-hop function-calling: LLM iteratively decides which tools to invoke."""
        # Ablation: bypass tools entirely, fall back to deep_think
        if self._config.get("ablation", {}).get("retrieval_off"):
            resp, _ = await self._run_deep(task)
            return resp, ["retrieval_off:deep_think"]

        from harness.eval.function_calling_runner import (
            FunctionCallingRunner,
            default_max_iterations_for,
        )
        llm = self._get_llm()
        fc_cfg = self._config.get("function_calling", {})
        # Benchmark key is stamped onto the task by `eval_tasks` below;
        # falls back to "" (→ "_default" budget) if called directly.
        bench_key = str(task.get("_benchmark_key") or "")
        # Explicit config override > per-benchmark default.
        explicit = fc_cfg.get("max_iterations")
        if explicit is not None:
            max_iter = int(explicit)
        else:
            max_iter = default_max_iterations_for(bench_key)
        min_iter = fc_cfg.get("min_iterations", 0)
        # Resolve enable_thinking: CLI override > mode default (False for light)
        _et = getattr(self, '_enable_thinking', None)
        _enable_thinking = _et if _et is not None else False
        cache_key = f"{bench_key}|{max_iter}|{min_iter}|{_enable_thinking}"
        runner = self._fc_runner_cache.get(cache_key)
        if runner is None:
            runner = FunctionCallingRunner(
                llm=llm,
                max_iterations=max_iter,
                min_iterations=min_iter,
                per_tool_timeout=fc_cfg.get("per_tool_timeout_s", 60),
                truncate_chars=fc_cfg.get("truncate_tool_result_chars", 0),
                truncate_tokens=fc_cfg.get("truncate_tool_result_tokens", 16000),
                enable_thinking=_enable_thinking,
                thinking_budget=8192,
                benchmark_key=bench_key or None,
            )
            self._fc_runner_cache[cache_key] = runner
        return await runner.run(task)

    async def _run_self_consistency(self, task: dict, mode: str) -> tuple[str, list[str]]:
        """Self-consistency voting. Mode format: 'self_consistency:<inner>' where inner ∈ simple_llm|deep_think|light|heavy."""
        from harness.eval.self_consistency import self_consistent_run

        # Parse inner mode
        if ":" in mode:
            inner_name = mode.split(":", 1)[1]
        else:
            inner_name = "fc"

        inner_map = {
            "simple": self._run_simple,
            "simple_llm": self._run_simple,
            "deep": self._run_deep,
            "deep_think": self._run_deep,
            "harness": self._run_harness,
            "heavy": self._run_harness,
            "fc": self._run_function_calling,
            "light": self._run_function_calling,
        }
        inner = inner_map.get(inner_name, self._run_function_calling)

        sc_cfg = self._config.get("self_consistency", {})
        n = sc_cfg.get("n", 5)
        return await self_consistent_run(
            inner_runner=inner,
            task=task,
            n=n,
            temperatures=sc_cfg.get("temperatures"),
        )

    async def _run_harness(self, task: dict) -> tuple[str, list[str]]:
        """Full harness: deep multi-hop ReAct loop with all available tools.

        Previously a rigid 3-phase triage pipeline that only supported 4 tools.
        Now delegates to FunctionCallingRunner with a higher iteration budget
        and min_iterations to ensure thorough research.
        """
        from harness.eval.function_calling_runner import (
            FunctionCallingRunner,
            default_max_iterations_for,
        )

        llm = self._get_llm()
        bench_key = str(task.get("_benchmark_key") or "")
        fc_cfg = self._config.get("function_calling", {})

        # heavy iteration budget: harness-specific key > CLI shared key > per-benchmark default
        explicit_max = fc_cfg.get("harness_max_iterations") or fc_cfg.get("max_iterations")
        if explicit_max is not None:
            max_iter = int(explicit_max)
        else:
            max_iter = default_max_iterations_for(bench_key)

        min_iter = fc_cfg.get("harness_min_iterations", fc_cfg.get("min_iterations", 10))

        # Resolve enable_thinking: CLI override > mode default (True for heavy)
        _et = getattr(self, '_enable_thinking', None)
        _enable_thinking = _et if _et is not None else True
        cache_key = f"harness|{bench_key}|{max_iter}|{min_iter}|{_enable_thinking}"
        runner = self._fc_runner_cache.get(cache_key)
        if runner is None:
            runner = FunctionCallingRunner(
                llm=llm,
                max_iterations=max_iter,
                min_iterations=int(min_iter),
                per_tool_timeout=fc_cfg.get("per_tool_timeout_s", 60),
                truncate_chars=fc_cfg.get("truncate_tool_result_chars", 0),
                truncate_tokens=fc_cfg.get("truncate_tool_result_tokens", 16000),
                enable_thinking=_enable_thinking,
                thinking_budget=8192,
                enable_mcp=fc_cfg.get("mcp_enabled", False),
                mcp_timeout=fc_cfg.get("mcp_timeout", 60.0),
                enable_retrieval=fc_cfg.get("retrieval_enabled", False),
                retrieval_top_k=fc_cfg.get("retrieval_top_k", 15),
                benchmark_key=bench_key or None,
            )
            self._fc_runner_cache[cache_key] = runner

        return await runner.run(task)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_tool_relevance(tools: list[str], task: dict) -> float:
        """Heuristic: what fraction of tools called match the task domain?"""
        if not tools:
            return 0.0
        cat = task.get("category", "").lower()
        domain_map = {
            "genomics": {"ncbi_tools", "geneagent", "genegpt", "genotex", "genomas", "openbio"},
            "biology": {"ncbi_tools", "geneagent", "genegpt", "genotex", "phenoage"},
            "medicine": {"clinical_calculators", "ncbi_tools", "mdagents", "txagent", "ehragent"},
            "chemistry": {"ncbi_tools", "drugagent", "prompt2pill", "txagent"},
            "ehr": {"ehragent", "colacare", "medagentbench", "clinical_calculators"},
            "clinical": {"mdagents", "txagent", "clinical_calculators", "ehragent"},
        }
        relevant = set()
        for domain, adapters in domain_map.items():
            if domain in cat:
                relevant.update(adapters)
        if not relevant:
            relevant = {"ncbi_tools", "clinical_calculators", "mdagents"}

        matched = sum(1 for t in tools if t in relevant)
        return matched / len(tools)

    @staticmethod
    def _score_faithfulness(response: str, task: dict, mode: str) -> float:
        """Heuristic faithfulness score based on evidence mention in response."""
        if mode == "simple_llm":
            return 0.0  # no reasoning expected

        resp_lower = response.lower()
        score = 0.0

        # Check for reasoning indicators
        reasoning_markers = [
            "because", "therefore", "this suggests", "evidence",
            "mechanism", "pathway", "indicates", "consistent with",
            "step 1", "step 2", "first", "second", "finally",
            "the reason", "this is due to", "based on",
        ]
        markers_found = sum(1 for m in reasoning_markers if m in resp_lower)
        score += min(markers_found / 5, 0.5)  # up to 0.5 for reasoning

        # Check for domain-relevant terms from the question
        q_words = set(task["question"].lower().split())
        technical = {w for w in q_words if len(w) > 6}  # longer words are more technical
        if technical:
            mentioned = sum(1 for w in technical if w in resp_lower)
            score += min(mentioned / max(len(technical), 1), 0.5)  # up to 0.5

        return round(min(score, 1.0), 2)
