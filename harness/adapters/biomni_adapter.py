"""Biomni adapter — delegate queries to Stanford's Biomni A1 agent.

Biomni is a bioinformatics agent wrapping a LangChain/LangGraph ReAct
loop over a large curated toolbox + a ~11 GB "data lake" of reference
datasets. We use it as a **sub-agent** (our paper claim) rather than
as our primary harness.

Key design decisions
--------------------
- Biomni's `A1(...)` constructor eagerly downloads the data lake on
  first init unless `path` points at an already-populated directory.
  We therefore keep the adapter **lazily-initialised**: `__init__` is
  cheap (just probes importability), and the real A1 instance is
  built on first `.run()` call — and only if the user has set
  `config['biomni']['data_path']` to a real directory (or opted in
  via `BIOMNI_DATA_PATH` env var). Without that, the adapter marks
  itself unavailable at call time to avoid surprise downloads.
- A1 expects an LLM source string (OpenAI / Anthropic / etc.). We map
  from our `config['model']` or fall back to OpenAI + the env key.
- Class-level cache keyed by (data_path, llm, source) so multiple
  adapter instances share one A1.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, ClassVar

from harness.adapter_base import AdapterBase

logger = logging.getLogger(__name__)


class BiomniAdapter(AdapterBase):
    name = "biomni"
    modality = "reasoning"
    description = (
        "Stanford Biomni A1 bioinformatics sub-agent. Delegates a natural-"
        "language biomedical query to a LangGraph-based ReAct agent "
        "running over Biomni's curated toolbox + data lake. Use for "
        "comparison baselines and for delegating complex multi-step "
        "analyses our native harness chooses to offload."
    )

    _a1_cache: ClassVar[dict[str, Any]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, config: dict | None = None, **kwargs: Any):
        self._config = config or {}
        try:
            import biomni  # noqa: F401
            self._deps_ok = True
        except ImportError as exc:
            self._deps_ok = False
            self.mark_unavailable(f"biomni not installed: {exc}")
            return

        # Configuration
        b_cfg = self._config.get("biomni", {})
        self._data_path: str | None = (
            b_cfg.get("data_path") or os.environ.get("BIOMNI_DATA_PATH")
        )
        self._llm: str = b_cfg.get("llm") or "claude-sonnet-4-5"
        self._source: str = b_cfg.get("source") or "Anthropic"
        self._timeout: int = int(b_cfg.get("timeout_seconds", 600))
        # Biomni use_tool_retriever — keep True by default so it picks a
        # relevant subset of its 3000+ tools rather than dumping them all.
        self._use_tool_retriever: bool = bool(b_cfg.get("use_tool_retriever", True))

        if not self._data_path:
            # Not marking unavailable unconditionally — some benchmarks
            # might not need the data lake. The warning surfaces at
            # first `.run()` call instead.
            logger.info(
                "BiomniAdapter: no data_path configured — A1 will attempt "
                "to download the ~11GB data lake on first use. Set "
                "config['biomni']['data_path'] or $BIOMNI_DATA_PATH to "
                "avoid this."
            )

    def capabilities(self) -> list[str]:
        return [
            "biomni_delegation",
            "langgraph_react",
            "sub_agent_call",
            "multi_step_bioinformatics",
        ]

    # -------- A1 lifecycle --------------------------------------------

    def _get_a1(self) -> Any:
        """Lazy A1 construction. Returns None if we can't construct one."""
        if not self._deps_ok:
            return None
        key = f"{self._data_path or '_default'}|{self._source}|{self._llm}"
        with self._lock:
            a1 = self._a1_cache.get(key)
            if a1 is not None:
                return a1
            try:
                from biomni.agent import A1
                a1 = A1(
                    path=self._data_path,
                    llm=self._llm,
                    source=self._source,  # type: ignore[arg-type]
                    use_tool_retriever=self._use_tool_retriever,
                    timeout_seconds=self._timeout,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("BiomniAdapter: A1 init failed: %s", exc)
                return None
            self._a1_cache[key] = a1
            return a1

    # -------- AdapterBase.run -----------------------------------------

    async def run(self, query: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._deps_ok:
            return self.result(
                answer=f"Biomni unavailable: {self.unavailable_reason}",
                confidence=0.0,
            )
        if not self._data_path and not os.environ.get("BIOMNI_ALLOW_DOWNLOAD"):
            return self.result(
                answer=(
                    "BiomniAdapter refused to run: no data_path configured "
                    "and BIOMNI_ALLOW_DOWNLOAD not set. Set "
                    "config['biomni']['data_path'] or export "
                    "BIOMNI_ALLOW_DOWNLOAD=1 to allow the ~11GB download."
                ),
                evidence=["biomni guard"],
                confidence=0.0,
            )

        import asyncio
        a1 = await asyncio.to_thread(self._get_a1)
        if a1 is None:
            return self.result(
                answer="Biomni A1 failed to initialise; see logs.",
                confidence=0.0,
            )
        try:
            # A1.go() is synchronous; run in thread to avoid blocking the
            # event loop.
            result = await asyncio.to_thread(a1.go, query)
        except Exception as exc:  # noqa: BLE001
            return self.result(
                answer=f"Biomni A1 execution failed: {exc}",
                confidence=0.0,
            )

        # A1.go typically returns (log, final_answer) or a dict
        answer: str
        raw: Any = result
        if isinstance(result, tuple) and len(result) >= 2:
            answer = str(result[-1])
        elif isinstance(result, dict):
            answer = str(result.get("output") or result.get("answer") or result)
        else:
            answer = str(result)

        return self.result(
            answer=answer,
            evidence=[f"biomni A1 (llm={self._llm}, source={self._source})"],
            confidence=0.7,
            raw=raw,
        )
