"""Minimal LangGraph ReAct baseline agent.

Self-contained: imports only from third-party (langgraph, langchain_*,
pydantic). No imports from harness.* — this baseline is meant to be a
clean comparison point against the harness's FunctionCallingRunner.

Usage:
    from baselines.langgraph_react import LangGraphReActBaseline

    agent = LangGraphReActBaseline(
        model="claude-sonnet-4-5",
        provider="anthropic",
        tool_specs=my_openai_style_specs,
        dispatch=my_tool_dispatcher,
    )
    answer = await agent.run("What is the creatinine clearance for ...?")
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Callable

from langgraph.prebuilt import create_react_agent

from .tool_adapter import specs_to_tools


_DEFAULT_SYSTEM = (
    "You are an expert medical assistant. You may call tools when they "
    "provide useful facts or computation. When you have enough evidence, "
    "emit a final answer. If the task asks for a single value, end your "
    "response with: The answer is [value]."
)


class LangGraphReActBaseline:
    """Minimal wrapper over `langgraph.prebuilt.create_react_agent`.

    Default behaviour: if no tool_specs are supplied, the agent runs
    pure-LLM (no-tools), which is still a valid baseline comparison.
    If tool_specs are supplied, pass a real dispatch callable; the
    built-in dispatcher is a visible placeholder for skeleton use.
    """

    def __init__(
        self,
        model: str,
        provider: str = "anthropic",
        tool_specs: list[dict[str, Any]] | None = None,
        dispatch: Callable[[str, dict[str, Any]], Any] | None = None,
        system_prompt: str = _DEFAULT_SYSTEM,
        temperature: float = 0.1,
        max_tool_iterations: int = 8,
    ):
        self.model = model
        self.provider = provider
        self.system_prompt = system_prompt
        self.max_tool_iterations = max_tool_iterations
        self._llm = self._build_llm(temperature)
        self._tools = (specs_to_tools(tool_specs, dispatch or _echo_dispatch)
                        if tool_specs else [])
        self._agent = create_react_agent(
            model=self._llm,
            tools=self._tools,
            prompt=system_prompt,
        )

    def _build_llm(self, temperature: float):
        if self.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=self.model,
                temperature=temperature,
                max_tokens=4096,
                api_key=os.environ.get("ANTHROPIC_API_KEY"),
            )
        if self.provider in ("openai", "gemini"):
            from langchain_openai import ChatOpenAI
            kw: dict[str, Any] = {
                "model": self.model,
                "temperature": temperature,
                "api_key": os.environ.get("OPENAI_API_KEY" if self.provider == "openai" else "GEMINI_API_KEY"),
            }
            if self.provider == "gemini":
                kw["base_url"] = "https://generativelanguage.googleapis.com/v1beta/openai/"
            return ChatOpenAI(**kw)
        raise ValueError(f"Unsupported provider: {self.provider}")

    async def run(self, question: str, timeout_s: float = 60.0) -> str:
        """Run the agent on a question. Returns the final text answer.

        `invoke` is sync inside langgraph.prebuilt — we dispatch it to a
        thread so the caller can keep its event loop responsive and so
        asyncio.wait_for can enforce a wall-clock timeout.
        """
        messages = [{"role": "user", "content": question}]

        def _sync():
            state = self._agent.invoke({"messages": messages})
            msgs = state.get("messages", [])
            # Last assistant message is the answer.
            for m in reversed(msgs):
                role = getattr(m, "type", None) or (m.get("role") if isinstance(m, dict) else None)
                if role in ("ai", "assistant"):
                    content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else "")
                    if isinstance(content, list):
                        texts = [blk.get("text", "") if isinstance(blk, dict) else str(blk) for blk in content]
                        return "\n".join(t for t in texts if t)
                    return str(content or "")
            return ""

        return await asyncio.wait_for(asyncio.to_thread(_sync), timeout=timeout_s)


def _echo_dispatch(name: str, args: dict[str, Any]) -> str:
    """Placeholder dispatcher. Real pipelines inject a real tool backend."""
    return f"[stub {name}({args}) — wire a real dispatcher to get a real result]"
