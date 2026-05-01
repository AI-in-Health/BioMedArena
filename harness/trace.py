"""Per-task trace recording for harness runs.

``TraceRecorder`` captures every LLM call, every tool call, and
iteration counts for a single task. Traces are dumped to JSON and later
used by ``harness.eval.results_writer`` to populate per-cell metrics
(the 22-column CSV) and by humans for debugging.

One recorder per task (not per cell, not per run). The runner creates a
recorder, attaches it to the LLM client and tool runner, and dumps to
``data/runs/<run>/traces/<cell_name>/<task_id>.json``.

Trace file format::

    {
        "task_id": str,
        "benchmark": str,
        "backbone": str,
        "mode": str,
        "started_at": ISO8601,
        "finished_at": ISO8601,
        "total_latency_s": float,
        "iterations": int,
        "llm_calls": [
            {
                "call_index": int,
                "role": "chat" | "chat_think" | "chat_vision",
                "system": str,
                "messages": list[dict],
                "response_text": str,
                "input_tokens": int,
                "output_tokens": int,
                "cost_usd": float,
                "latency_ms": int,
                "finish_reason": str,
                "error": str | None,
            },
        ],
        "tool_calls": [
            {
                "call_index": int,
                "iteration": int,
                "name": str,
                "arguments": dict,
                "result_preview": str,
                "success": bool,
                "error": str | None,
                "latency_ms": int,
            },
        ],
        "final_answer": str,
        "scorer_result": {
            "correct": bool,
            "method": str,
            "details": dict,
            "llm_judge_invoked": bool,
            "llm_judge_verdict": bool | None,
        },
    }
"""
from __future__ import annotations

import contextvars
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Context-var binding: let LLMClient / tool runners find the active recorder
# without having to thread ``trace`` through every method signature.
# ---------------------------------------------------------------------------


_ACTIVE_TRACE: contextvars.ContextVar["TraceRecorder | None"] = (
    contextvars.ContextVar("bioagent_active_trace", default=None)
)


def get_active_trace() -> "TraceRecorder | None":
    """Return the ``TraceRecorder`` bound to the current async task, if any."""
    return _ACTIVE_TRACE.get()


@contextmanager
def active_trace(recorder: "TraceRecorder | None") -> Iterator[None]:
    """Bind ``recorder`` as the active trace for the enclosed block."""
    token = _ACTIVE_TRACE.set(recorder)
    try:
        yield
    finally:
        _ACTIVE_TRACE.reset(token)


@dataclass
class LLMCallRecord:
    call_index: int
    role: str  # "chat" | "chat_think" | "chat_vision"
    system: str
    messages: list[dict]
    response_text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    finish_reason: str = ""
    error: str | None = None


@dataclass
class ToolCallRecord:
    call_index: int
    iteration: int
    name: str
    arguments: dict
    result_preview: str = ""
    success: bool = True
    error: str | None = None
    latency_ms: int = 0


@dataclass
class ScorerRecord:
    correct: bool
    method: str
    details: dict = field(default_factory=dict)
    llm_judge_invoked: bool = False
    llm_judge_verdict: bool | None = None


class TraceRecorder:
    """Records all LLM and tool calls for a single task."""

    def __init__(
        self, task_id: str, benchmark: str, backbone: str, mode: str,
    ) -> None:
        self.task_id = task_id
        self.benchmark = benchmark
        self.backbone = backbone
        self.mode = mode
        self.started_at = _iso_now()
        self._start_time = time.time()
        self.finished_at: str | None = None
        self.total_latency_s: float = 0.0
        self.iterations: int = 0
        self.llm_calls: list[LLMCallRecord] = []
        self.tool_calls: list[ToolCallRecord] = []
        self.final_answer: str = ""
        self.scorer_result: ScorerRecord | None = None

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_llm_call(
        self,
        role: str,
        system: str,
        messages: list[dict],
        response_text: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: int = 0,
        finish_reason: str = "",
        error: str | None = None,
    ) -> None:
        self.llm_calls.append(LLMCallRecord(
            call_index=len(self.llm_calls),
            role=role,
            system=str(system)[:4000],
            messages=_truncate_messages(messages),
            response_text=str(response_text)[:4000],
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
            cost_usd=float(cost_usd or 0.0),
            latency_ms=int(latency_ms or 0),
            finish_reason=str(finish_reason or ""),
            error=error,
        ))

    def record_tool_call(
        self,
        name: str,
        arguments: dict,
        result: Any = None,
        success: bool = True,
        error: str | None = None,
        latency_ms: int = 0,
    ) -> None:
        self.tool_calls.append(ToolCallRecord(
            call_index=len(self.tool_calls),
            iteration=self.iterations,
            name=str(name),
            arguments=_safe_args(arguments),
            result_preview=str(result)[:500] if result is not None else "",
            success=bool(success),
            error=error,
            latency_ms=int(latency_ms or 0),
        ))

    def increment_iteration(self) -> None:
        self.iterations += 1

    def set_final_answer(self, answer: str) -> None:
        self.final_answer = str(answer or "")[:4000]

    def set_scorer_result(
        self,
        correct: bool,
        method: str,
        details: dict | None = None,
        llm_judge_invoked: bool = False,
        llm_judge_verdict: bool | None = None,
    ) -> None:
        self.scorer_result = ScorerRecord(
            correct=bool(correct),
            method=str(method),
            details=dict(details or {}),
            llm_judge_invoked=bool(llm_judge_invoked),
            llm_judge_verdict=llm_judge_verdict,
        )

    # ------------------------------------------------------------------
    # Finalization / IO
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        if self.finished_at is None:
            self.finished_at = _iso_now()
            self.total_latency_s = time.time() - self._start_time

    def to_dict(self) -> dict:
        if self.finished_at is None:
            self.finalize()
        return {
            "task_id": self.task_id,
            "benchmark": self.benchmark,
            "backbone": self.backbone,
            "mode": self.mode,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_latency_s": round(self.total_latency_s, 3),
            "iterations": self.iterations,
            "llm_calls": [asdict(r) for r in self.llm_calls],
            "tool_calls": [asdict(r) for r in self.tool_calls],
            "final_answer": self.final_answer,
            "scorer_result": (
                asdict(self.scorer_result) if self.scorer_result else None
            ),
        }

    def dump(self, path: Path) -> None:
        """Write the trace to a JSON file at ``path``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    # ------------------------------------------------------------------
    # Aggregation helpers (used by results_writer)
    # ------------------------------------------------------------------

    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.llm_calls)

    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.llm_calls)

    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.llm_calls)

    def n_tool_calls(self) -> int:
        return len(self.tool_calls)

    def n_tool_calls_success(self) -> int:
        return sum(1 for c in self.tool_calls if c.success)

    def tool_call_names(self) -> list[str]:
        return [c.name for c in self.tool_calls]

    def has_runtime_error(self) -> bool:
        return any(c.error for c in self.llm_calls)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate_messages(messages: list[dict]) -> list[dict]:
    """Keep last ~20 messages with string content capped at 2000 chars."""
    out: list[dict] = []
    for m in (messages or [])[-20:]:
        new_m = dict(m) if isinstance(m, dict) else {"raw": str(m)}
        content = new_m.get("content", "")
        if isinstance(content, str) and len(content) > 2000:
            new_m["content"] = content[:2000] + "...[truncated]"
        out.append(new_m)
    return out


def _safe_args(args: Any) -> dict:
    """Cast to dict and truncate long string values."""
    if not isinstance(args, dict):
        try:
            args = dict(args) if args is not None else {}
        except Exception:
            return {"_raw": str(args)[:1000]}
    out: dict[str, Any] = {}
    for k, v in args.items():
        if isinstance(v, str) and len(v) > 1000:
            out[str(k)] = v[:1000] + "...[truncated]"
        else:
            out[str(k)] = v
    return out
