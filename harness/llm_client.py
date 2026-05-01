"""Unified LLM interface supporting OpenAI, Anthropic, Gemini, and local models."""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


class ContextWindowError(RuntimeError):
    """Raised when a provider rejects a request due to context-window overflow.

    The context manager in the harness catches this specific type so it can
    trim conversation history and retry, rather than surfacing a generic error.
    """


# Per-async-task stash used by ``_record_usage`` to hand off numbers to
# the public ``chat``/``chat_think``/``chat_vision``/``chat_with_tools``
# wrappers so they can forward to an active ``TraceRecorder`` without
# threading ``trace`` through every retry branch.
# Tuple: (input_tokens, output_tokens, cost_usd, finish_reason).
_LAST_LLM_USAGE: contextvars.ContextVar[tuple[int, int, float, str]] = (
    contextvars.ContextVar("bioagent_last_llm_usage", default=(0, 0, 0.0, ""))
)


def _extract_system_text(messages: list[dict[str, Any]]) -> str:
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == "system":
            content = m.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def _trace_llm_call(
    role: str, messages: list[dict[str, Any]], response_text: str,
    latency_ms: int, error: str | None = None,
) -> None:
    """Forward an LLM-call record to the active ``TraceRecorder``, if any.

    Pulls ``(in_tok, out_tok, cost, finish)`` from ``_LAST_LLM_USAGE`` so
    provider-specific retry loops don't have to be re-plumbed. Never
    raises — any trace failure is swallowed.
    """
    try:
        from harness.trace import get_active_trace
        trace = get_active_trace()
        if trace is None:
            return
        in_tok, out_tok, cost, finish = _LAST_LLM_USAGE.get()
        trace.record_llm_call(
            role=role,
            system=_extract_system_text(messages),
            messages=messages,
            response_text=response_text or "",
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            latency_ms=latency_ms,
            finish_reason=finish,
            error=error,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# OpenAI ↔ Anthropic message-format translator
#
# Our entire harness speaks OpenAI-style chat messages internally:
#     {"role": "system" | "user" | "assistant" | "tool",
#      "content": str | list[dict],
#      ["tool_calls": [{"id","type":"function","function":{"name","arguments"}}]],
#      ["tool_call_id": str]}   # only on role=="tool"
#
# Anthropic's messages-API needs a different shape:
#   - system goes into a top-level `system` kwarg, not a message.
#   - role is only "user" or "assistant".
#   - Tool RESULTS are user messages whose content is a list of
#     `{"type":"tool_result", "tool_use_id": id, "content": text}` blocks.
#   - Tool CALLS are assistant messages whose content is a list mixing
#     `{"type":"text","text":...}` and
#     `{"type":"tool_use","id":id,"name":name,"input":dict}` blocks.
#
# Consecutive `role="tool"` messages (common when the runner dispatches
# multiple tool calls in parallel) MUST be merged into a single
# `role="user"` Anthropic message with multiple tool_result blocks, or
# the API will complain about broken turn alternation.
# ---------------------------------------------------------------------------


def _openai_to_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Translate an OpenAI-format conversation to Anthropic-format.

    Returns `(system_prompt, anth_messages)`. `system_prompt` is the
    concatenated text of any role="system" messages (Anthropic takes it
    as a top-level kwarg); `None` if no system message.

    Translation rules:
      role="system"       → system kwarg (concatenated if multiple)
      role="user"         → role="user",  content passed through
      role="assistant"
        with content only → role="assistant", content=[text_block]
        with tool_calls   → role="assistant",
                            content=[text_block?, tool_use_block, ...]
      role="tool"         → role="user",
                            content=[tool_result_block]
                            CONSECUTIVE tool messages coalesce into one
                            user message carrying all their result blocks.
    """
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []

    for m in messages:
        role = m.get("role")
        if role == "system":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                system_parts.append(c)
            elif isinstance(c, list):
                # rare: list-of-blocks system message
                for b in c:
                    txt = b.get("text") if isinstance(b, dict) else None
                    if isinstance(txt, str) and txt.strip():
                        system_parts.append(txt)
            continue

        if role == "tool":
            tool_use_id = m.get("tool_call_id") or m.get("id") or ""
            raw_content = m.get("content")
            if isinstance(raw_content, list):
                # Content is already a list of blocks — stringify for safety
                text = "\n".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in raw_content
                )
            else:
                text = "" if raw_content is None else str(raw_content)
            tr_block = {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": text,
            }
            # Coalesce into the previous user message if it already
            # carries tool_result blocks.
            if (out and out[-1]["role"] == "user"
                    and isinstance(out[-1].get("content"), list)
                    and out[-1]["content"]
                    and isinstance(out[-1]["content"][0], dict)
                    and out[-1]["content"][0].get("type") == "tool_result"):
                out[-1]["content"].append(tr_block)
            else:
                out.append({"role": "user", "content": [tr_block]})
            continue

        if role == "assistant":
            # Pass through raw Anthropic blocks (including thinking)
            # when present — needed for thinking+tools message fidelity.
            raw_blocks = m.get("_raw_blocks")
            if raw_blocks:
                out.append({"role": "assistant", "content": raw_blocks})
                continue
            tool_calls = m.get("tool_calls") or []
            content_str = m.get("content")
            if tool_calls:
                blocks: list[dict[str, Any]] = []
                if isinstance(content_str, str) and content_str.strip():
                    blocks.append({"type": "text", "text": content_str})
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    raw_args = fn.get("arguments")
                    if isinstance(raw_args, str):
                        try:
                            input_obj = json.loads(raw_args) if raw_args else {}
                        except json.JSONDecodeError:
                            input_obj = {"__raw_arguments": raw_args}
                    elif isinstance(raw_args, dict):
                        input_obj = raw_args
                    else:
                        input_obj = {}
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id") or "",
                        "name": fn.get("name") or "",
                        "input": input_obj,
                    })
                out.append({"role": "assistant", "content": blocks})
            else:
                # Pure-text assistant message — pass through, but
                # Anthropic rejects empty string content; drop empties.
                if isinstance(content_str, str) and content_str.strip():
                    out.append({"role": "assistant", "content": content_str})
                elif isinstance(content_str, list):
                    out.append({"role": "assistant", "content": content_str})
                # else: drop — nothing meaningful to send
            continue

        if role == "user":
            out.append({"role": "user", "content": m.get("content", "")})
            continue

        # Unknown role — pass through as user for safety
        out.append({"role": "user", "content": m.get("content", "")})

    system = "\n\n".join(system_parts) if system_parts else None
    return system, out


class LLMClient:
    """Thin wrapper that normalises chat-completion calls across providers."""

    def __init__(self, provider: str = "openai", model: str = "gpt-4o", api_key: str | None = None):
        self.provider = provider
        self.model = model
        self.api_key = self._resolve_key(api_key)
        self._client: Any = None

    # ------------------------------------------------------------------
    # Lazy client init
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if self.provider == "openai":
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)
        elif self.provider == "anthropic":
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=self.api_key)
        elif self.provider == "gemini":
            # Use OpenAI-compatible endpoint for Gemini
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        elif self.provider == "local":
            from openai import AsyncOpenAI
            base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")
            self._client = AsyncOpenAI(api_key="not-needed", base_url=base_url)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        return self._client

    # ------------------------------------------------------------------
    # Cost tracking — defensive, never raises
    # ------------------------------------------------------------------

    def _record_usage(self, resp: Any, model_override: str | None = None) -> None:
        """Extract token counts from a provider response and record cost.

        Handles OpenAI-shape (``usage.prompt_tokens`` / ``completion_tokens``),
        Anthropic-shape (``usage.input_tokens`` / ``output_tokens``), and
        google-genai (``usage_metadata.prompt_token_count`` /
        ``candidates_token_count``). Silently swallows any error — cost
        tracking must never break a benchmark run.

        Also stashes the computed ``(in_tok, out_tok, cost_usd, finish_reason)``
        tuple into ``_LAST_LLM_USAGE`` (a ContextVar) so the public
        ``chat``/``chat_think``/etc. wrappers can forward the numbers
        to any active ``TraceRecorder`` without having to thread a
        ``trace`` kwarg through every retry branch.
        """
        try:
            from harness.cost_monitor import PRICING, record
            in_tok = 0
            out_tok = 0
            usage = getattr(resp, "usage", None)
            if usage is not None:
                in_tok = (getattr(usage, "input_tokens", None)
                          or getattr(usage, "prompt_tokens", None) or 0)
                out_tok = (getattr(usage, "output_tokens", None)
                           or getattr(usage, "completion_tokens", None) or 0)
            else:
                meta = getattr(resp, "usage_metadata", None)
                if meta is not None:
                    in_tok = getattr(meta, "prompt_token_count", 0) or 0
                    out_tok = getattr(meta, "candidates_token_count", 0) or 0
            model_name = model_override or self.model
            cost = 0.0
            if in_tok or out_tok:
                cost = record(model_name, int(in_tok), int(out_tok))
            # Capture finish_reason if the response carries one (OpenAI /
            # Gemini shape); Anthropic puts it in ``stop_reason``.
            finish = ""
            try:
                choices = getattr(resp, "choices", None)
                if choices:
                    finish = getattr(choices[0], "finish_reason", "") or ""
                if not finish:
                    finish = getattr(resp, "stop_reason", "") or ""
            except Exception:
                pass
            _LAST_LLM_USAGE.set((int(in_tok), int(out_tok), float(cost), str(finish)))
        except Exception:
            _LAST_LLM_USAGE.set((0, 0, 0.0, ""))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        response_format: str | None = None,
    ) -> str:
        """Send a chat completion and return the assistant message text."""
        client = self._get_client()

        # Reset the per-call usage stash so an active ``TraceRecorder``
        # sees only numbers from the call that actually happens below.
        _LAST_LLM_USAGE.set((0, 0, 0.0, ""))
        t_start = time.monotonic()
        try:
            if self.provider in ("openai", "local", "gemini"):
                content = await self._chat_openai(
                    client, messages, temperature, max_tokens, response_format,
                )
            elif self.provider == "anthropic":
                content = await self._chat_anthropic(
                    client, messages, temperature, max_tokens,
                )
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as exc:
            _trace_llm_call(
                role="chat", messages=messages, response_text="",
                latency_ms=int((time.monotonic() - t_start) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        _trace_llm_call(
            role="chat", messages=messages, response_text=content,
            latency_ms=int((time.monotonic() - t_start) * 1000),
        )
        return content

    async def chat_json(self, messages: list[dict[str, str]], temperature: float = 0.1) -> Any:
        """Chat expecting a JSON response. Returns parsed object."""
        # Anthropic doesn't support response_format="json_object",
        # so only try it for OpenAI/Gemini providers. For all providers,
        # fall back to a plain chat call if the first attempt fails.
        raw = None
        if self.provider in ("openai", "gemini", "local"):
            try:
                raw = await self.chat(messages, temperature=temperature, response_format="json_object")
            except Exception:
                pass
        if raw is None:
            raw = await self.chat(messages, temperature=temperature, response_format=None)
        # Strip markdown code fences if present
        text = (raw or "").strip()
        if not text:
            logger.warning("[chat_json] empty response from %s", self.model)
            return {"error": "empty_response"}
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
            if text.endswith("```"):
                text = text[:-3].strip()
        if not text:
            logger.warning("[chat_json] empty after fence-strip from %s: raw=%r",
                           self.model, raw[:200] if raw else raw)
            return {"error": "empty_response"}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            import ast
            import re

            def _fix_json_escapes(s: str) -> str:
                """Double-escape backslashes that aren't valid JSON escapes.

                Valid JSON escapes: \" \\\\ \\/ \\b \\f \\n \\r \\t \\uXXXX
                Anything else (e.g. \\rho, \\pi, \\theta from LaTeX) is
                invalid and must be escaped as \\\\ + char.
                """
                return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)

            # Try to extract a JSON/dict-like substring from mixed text
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            snippet = match.group() if match else None
            # Try json.loads on extracted snippet
            if snippet:
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    pass
            # Fix invalid escape sequences (e.g. \rho, \theta from LaTeX)
            # and retry json.loads
            for candidate in ([snippet, text] if snippet else [text]):
                try:
                    return json.loads(_fix_json_escapes(candidate))
                except (json.JSONDecodeError, AttributeError):
                    pass
            # Try ast.literal_eval — handles Python-style dicts with
            # single quotes and True/False/None instead of true/false/null
            for candidate in ([text, snippet] if snippet else [text]):
                try:
                    obj = ast.literal_eval(candidate)
                    if isinstance(obj, dict):
                        return obj
                except (ValueError, SyntaxError):
                    pass
            # Last resort: replace single quotes with double quotes
            for candidate in ([text, snippet] if snippet else [text]):
                try:
                    fixed = candidate.replace("'", '"')
                    return json.loads(fixed)
                except (json.JSONDecodeError, AttributeError):
                    pass
            # Ultimate fallback: fix escapes + fix quotes combined
            for candidate in ([text, snippet] if snippet else [text]):
                try:
                    fixed = _fix_json_escapes(candidate).replace("'", '"')
                    return json.loads(fixed)
                except (json.JSONDecodeError, AttributeError):
                    pass
            raise

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    async def _chat_openai(
        self, client: Any, messages: list[dict[str, str]],
        temperature: float, max_tokens: int, response_format: str | None,
    ) -> str:
        # gpt-5.x and o-series reject ``max_tokens`` and require
        # ``max_completion_tokens``; temperature is also restricted on
        # those families. Detect by name and translate kwargs.
        model = self.model
        uses_max_completion = (
            model.startswith("gpt-5")
            or model.startswith(("o1", "o3", "o4", "o5"))
        )
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if uses_max_completion:
            kwargs["max_completion_tokens"] = max_tokens
            # reasoning / gpt-5 families only accept default temperature;
            # omit the parameter entirely unless explicitly non-default.
            # Empirically temperature=0.2 triggers 400 on gpt-5.x.
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
        if response_format == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        # Retry with exponential backoff for rate limits
        for attempt in range(5):
            try:
                resp = await client.chat.completions.create(**kwargs)
                self._record_usage(resp)
                # Some thinking-native models (e.g. Gemini thinking
                # variants) return ``content=None`` when
                # the output-token budget is consumed by internal
                # thinking. Coerce to "" so downstream string ops do
                # not crash; caller sees empty answer and scorer
                # marks incorrect.
                return resp.choices[0].message.content or ""
            except Exception as exc:
                if "429" in str(exc) or "rate" in str(exc).lower():
                    wait = 2 ** attempt
                    await asyncio.sleep(wait)
                    continue
                raise
        # Final attempt without catch
        resp = await client.chat.completions.create(**kwargs)
        self._record_usage(resp)
        return resp.choices[0].message.content or ""

    async def _chat_anthropic(
        self, client: Any, messages: list[dict[str, str]],
        temperature: float, max_tokens: int,
    ) -> str:
        system = None
        filtered: list[dict[str, str]] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered.append(m)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": filtered,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system
        for attempt in range(5):
            try:
                resp = await client.messages.create(**kwargs)
                break
            except Exception as exc:
                if "429" in str(exc) or "rate" in str(exc).lower() or "overloaded" in str(exc).lower():
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
        else:
            raise RuntimeError("chat anthropic: exhausted retries")
        self._record_usage(resp)
        # Defensive extraction: Anthropic returns ``content=[]`` when
        # ``stop_reason="refusal"`` (safety policy), and may return
        # blocks other than ``text`` (e.g. ``thinking``, ``tool_use``)
        # in other modes. Flatten all text blocks and fall back to an
        # empty string so downstream scorers simply mark the task
        # incorrect rather than raising IndexError.
        text_parts: list[str] = []
        for block in (resp.content or []):
            if getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", "") or "")
        return "".join(text_parts)

    # ------------------------------------------------------------------
    # Vision input (multi-modal chat)
    # ------------------------------------------------------------------

    async def chat_vision(
        self,
        system_prompt: str,
        user_text: str,
        image_paths: list[str],
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> str:
        """Chat with images attached. Works with vision-capable models
        (OpenAI gpt-4o, Gemini 2.5-flash, Claude 3.5-sonnet)."""
        _LAST_LLM_USAGE.set((0, 0, 0.0, ""))
        t_start = time.monotonic()
        # Build a lightweight messages preview used only for trace; we
        # never ship the raw image bytes to TraceRecorder.
        trace_preview_messages = [
            {"role": "system", "content": str(system_prompt)[:2000]},
            {"role": "user", "content":
                f"[vision input: {len(image_paths)} image(s)] " + str(user_text)[:1000]},
        ]
        try:
            content = await self._chat_vision_dispatch(
                system_prompt, user_text, image_paths, temperature, max_tokens,
            )
        except Exception as exc:
            _trace_llm_call(
                role="chat_vision", messages=trace_preview_messages,
                response_text="",
                latency_ms=int((time.monotonic() - t_start) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        _trace_llm_call(
            role="chat_vision", messages=trace_preview_messages,
            response_text=content,
            latency_ms=int((time.monotonic() - t_start) * 1000),
        )
        return content

    async def _chat_vision_dispatch(
        self,
        system_prompt: str,
        user_text: str,
        image_paths: list[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        import base64
        import mimetypes

        # Build message content with images as data URIs
        content_parts: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for path in image_paths[:5]:  # cap at 5 images
            try:
                with open(path, "rb") as f:
                    data = f.read()
                mime, _ = mimetypes.guess_type(path)
                if mime is None:
                    mime = "image/jpeg"
                b64 = base64.b64encode(data).decode("ascii")
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
            except Exception:
                continue

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_parts},
        ]

        client = self._get_client()
        if self.provider in ("openai", "local", "gemini"):
            for attempt in range(5):
                try:
                    resp = await client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    self._record_usage(resp)
                    return resp.choices[0].message.content or ""
                except Exception as exc:
                    if "429" in str(exc) or "rate" in str(exc).lower():
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise
            raise RuntimeError("vision openai: exhausted retries")

        if self.provider == "anthropic":
            # Anthropic uses a different content-block format for images.
            anth_content: list[dict[str, Any]] = [
                {"type": "text", "text": user_text},
            ]
            for path in image_paths[:5]:
                try:
                    with open(path, "rb") as f:
                        data = f.read()
                    mime, _ = mimetypes.guess_type(path)
                    if mime is None:
                        mime = "image/jpeg"
                    b64 = base64.b64encode(data).decode("ascii")
                    anth_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            "data": b64,
                        },
                    })
                except Exception:
                    continue
            for attempt in range(5):
                try:
                    resp = await client.messages.create(
                        model=self.model,
                        system=system_prompt,
                        messages=[{"role": "user", "content": anth_content}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    self._record_usage(resp)
                    parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
                    return "\n".join(parts)
                except Exception as exc:
                    if "429" in str(exc) or "rate" in str(exc).lower() or "overloaded" in str(exc).lower():
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise
            raise RuntimeError("vision anthropic: exhausted retries")

        raise ValueError(f"vision not implemented for provider {self.provider}")

    # ------------------------------------------------------------------
    # Tool-calling (function calling)
    # ------------------------------------------------------------------

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = 16384,
        enable_thinking: bool = False,
        thinking_budget: int = 8192,
    ) -> dict[str, Any]:
        """Chat with native tool calling.

        Returns: {"content": str|None, "tool_calls": [{"id", "name", "arguments"}]}
        When ``enable_thinking`` is True (Anthropic only), also
        returns ``"_raw_blocks"`` — the full response content blocks
        (including ``thinking`` blocks) needed for faithful message
        reconstruction on the next turn.
        """
        _LAST_LLM_USAGE.set((0, 0, 0.0, ""))
        t_start = time.monotonic()
        try:
            if self.provider in ("openai", "gemini", "local"):
                result = await self._tools_openai(
                    messages, tools, temperature, max_tokens,
                )
            elif self.provider == "anthropic":
                result = await self._tools_anthropic(
                    messages, tools, temperature, max_tokens,
                    enable_thinking=enable_thinking,
                    thinking_budget=thinking_budget,
                )
            else:
                raise ValueError(
                    f"tool calling not supported for {self.provider}"
                )
        except Exception as exc:
            _trace_llm_call(
                role="chat_with_tools", messages=messages, response_text="",
                latency_ms=int((time.monotonic() - t_start) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        # Record the assistant turn plus a compact summary of any tool
        # calls the model requested. Actual tool execution is recorded
        # separately by the FC runner via ``trace.record_tool_call``.
        summary_parts: list[str] = []
        if result.get("content"):
            summary_parts.append(str(result["content"]))
        if result.get("tool_calls"):
            summary_parts.append(
                "tool_calls=" + json.dumps(
                    [{"name": c["name"], "arguments": c["arguments"]}
                     for c in result["tool_calls"]]
                )[:1500]
            )
        _trace_llm_call(
            role="chat_with_tools", messages=messages,
            response_text="\n".join(summary_parts),
            latency_ms=int((time.monotonic() - t_start) * 1000),
        )
        return result

    async def _tools_openai(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
        temperature: float, max_tokens: int,
    ) -> dict[str, Any]:
        client = self._get_client()
        model = self.model
        uses_max_completion = (
            model.startswith("gpt-5")
            or model.startswith(("o1", "o3", "o4", "o5"))
        )
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        }
        if uses_max_completion:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["temperature"] = temperature
            kwargs["max_tokens"] = max_tokens
        for attempt in range(5):
            try:
                resp = await client.chat.completions.create(**kwargs)
                self._record_usage(resp)
                msg = resp.choices[0].message
                tool_calls = []
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append({
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        })
                return {"content": msg.content, "tool_calls": tool_calls}
            except Exception as exc:
                if "429" in str(exc) or "rate" in str(exc).lower():
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError("tools: exhausted retries")

    async def _tools_anthropic(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
        temperature: float, max_tokens: int,
        enable_thinking: bool = False, thinking_budget: int = 8192,
    ) -> dict[str, Any]:
        client = self._get_client()
        # Convert OpenAI-format tool specs → Anthropic `tools` array
        anth_tools = [
            {
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "input_schema": t["function"]["parameters"],
            }
            for t in tools if t.get("type") == "function"
        ]
        # Convert OpenAI-format conversation → Anthropic-format conversation
        system, anth_messages = _openai_to_anthropic_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": anth_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Anthropic rejects an empty tools list; omit the parameter entirely
        # when there are no tools (e.g. the final-answer call sends tools=[]).
        if anth_tools:
            kwargs["tools"] = anth_tools
        if system:
            kwargs["system"] = system
        if enable_thinking:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            kwargs["temperature"] = 1.0  # required by Anthropic for thinking
            kwargs["max_tokens"] = max(max_tokens, thinking_budget + 4096)
        for attempt in range(5):
            try:
                resp = await client.messages.create(**kwargs)
                break
            except Exception as exc:
                exc_str = str(exc).lower()
                # Rate-limit / overload — back off and retry
                if "429" in str(exc) or "rate" in exc_str or "overloaded" in exc_str:
                    await asyncio.sleep(2 ** attempt)
                    continue
                # Thinking not supported by this model — fall back to
                # non-thinking mode and retry once.
                if enable_thinking and any(pat in exc_str for pat in (
                    "thinking", "not supported", "invalid parameter",
                    "validation", "unknown parameter",
                )):
                    logger.warning(
                        "Thinking not supported by %s, falling back to "
                        "non-thinking mode: %s", self.model, exc,
                    )
                    kwargs.pop("thinking", None)
                    kwargs["temperature"] = temperature  # restore original
                    kwargs["max_tokens"] = max_tokens
                    enable_thinking = False  # prevent _raw_blocks collection
                    continue
                # Context-window overflow — re-raise immediately with a
                # normalised message so the context manager can detect it.
                if any(pat in exc_str for pat in (
                    "context_length_exceeded",
                    "too many tokens",
                    "prompt is too long",
                    "maximum context length",
                    "input length",
                    "reduce the length",
                    "tokens exceeds",
                    "exceeds maximum",
                )):
                    raise ContextWindowError(
                        f"Anthropic context window exceeded: {exc}"
                    ) from exc
                raise
        else:
            raise RuntimeError("tools anthropic: exhausted retries")
        self._record_usage(resp)
        content = ""
        tool_calls = []
        raw_blocks = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "thinking":
                raw_blocks.append({
                    "type": "thinking",
                    "thinking": getattr(block, "thinking", ""),
                    "signature": getattr(block, "signature", ""),
                })
            elif btype == "redacted_thinking":
                raw_blocks.append({
                    "type": "redacted_thinking",
                    "data": getattr(block, "data", ""),
                })
            elif btype == "text":
                content += block.text
                raw_blocks.append({"type": "text", "text": block.text})
            elif btype == "tool_use":
                import json as _json
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": _json.dumps(block.input),
                })
                raw_blocks.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        result: dict[str, Any] = {"content": content or None, "tool_calls": tool_calls}
        if enable_thinking and raw_blocks:
            result["_raw_blocks"] = raw_blocks
        return result

    # ------------------------------------------------------------------
    # Native thinking (reasoning models)
    # ------------------------------------------------------------------

    async def chat_think(
        self,
        messages: list[dict[str, str]],
        thinking_budget: int = 8192,
        max_tokens: int = 16384,
    ) -> str:
        """Use native thinking/reasoning capabilities.

        - OpenAI: uses o-series / gpt-5.x reasoning path
        - Gemini: uses google-genai SDK with ThinkingConfig
        - Anthropic: uses Claude extended thinking (budget_tokens)
        - Others: falls back to regular chat
        """
        _LAST_LLM_USAGE.set((0, 0, 0.0, ""))
        t_start = time.monotonic()
        try:
            content = await self._chat_think_dispatch(
                messages, thinking_budget, max_tokens,
            )
        except Exception as exc:
            _trace_llm_call(
                role="chat_think", messages=messages, response_text="",
                latency_ms=int((time.monotonic() - t_start) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        _trace_llm_call(
            role="chat_think", messages=messages, response_text=content,
            latency_ms=int((time.monotonic() - t_start) * 1000),
        )
        return content

    async def _chat_think_dispatch(
        self, messages: list[dict[str, str]],
        thinking_budget: int, max_tokens: int,
    ) -> str:
        if self.provider == "openai":
            return await self._think_openai(messages, max_tokens)
        elif self.provider == "gemini":
            return await self._think_gemini(messages, thinking_budget, max_tokens)
        elif self.provider == "anthropic":
            return await self._think_anthropic(messages, thinking_budget, max_tokens)
        else:
            # Fallback for "local"/unknown provider: call the openai-shaped
            # primitive directly (not ``self.chat``) so the outer
            # ``chat_think`` wrapper is the only place that records the
            # trace entry for this call.
            client = self._get_client()
            return await self._chat_openai(
                client, messages, 0.0, max_tokens, None,
            )

    async def _think_anthropic(
        self, messages: list[dict[str, str]],
        thinking_budget: int, max_tokens: int,
    ) -> str:
        """Claude extended thinking via thinking={'type':'enabled','budget_tokens':N}."""
        client = self._get_client()
        system = None
        filtered: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered.append(m)

        # Extended thinking requires max_tokens > thinking_budget
        total_tokens = max(max_tokens, thinking_budget + 4096)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": filtered,
            "max_tokens": total_tokens,
            "thinking": {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            },
            # Extended thinking requires temperature=1
            "temperature": 1.0,
        }
        if system:
            kwargs["system"] = system

        for attempt in range(5):
            try:
                resp = await client.messages.create(**kwargs)
                self._record_usage(resp)
                # Response has mixed thinking + text blocks
                # We only return the text blocks (the answer)
                parts: list[str] = []
                for block in resp.content:
                    if getattr(block, "type", "") == "text":
                        parts.append(block.text)
                return "\n".join(parts)
            except Exception as exc:
                if "429" in str(exc) or "rate" in str(exc).lower() or "overloaded" in str(exc).lower():
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError("anthropic thinking: exhausted retries")

    async def _think_openai(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        """OpenAI native reasoning.

        Model routing:
          - ``o1``, ``o3``, ``o4-mini``, ``o5-*`` — reasoning series: no
            temperature, ``system`` role rewritten to ``developer``,
            ``max_completion_tokens`` instead of ``max_tokens``.
          - ``gpt-5.x`` — flagship series: attempt reasoning-aware request
            first (``reasoning_effort="high"``), fall back to standard chat.
          - others — standard chat with a "think step by step" instruction
            prepended.
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=self.api_key)
        model = self.model
        is_reasoning_series = model.startswith(("o1", "o3", "o4", "o5"))
        is_gpt5 = model.startswith("gpt-5")

        async def _call_reasoning() -> str:
            """o-series path: developer role + max_completion_tokens."""
            converted: list[dict[str, str]] = []
            for m in messages:
                if m["role"] == "system":
                    converted.append({"role": "developer", "content": m["content"]})
                else:
                    converted.append(m)
            for attempt in range(5):
                try:
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=converted,
                        max_completion_tokens=max_tokens,
                    )
                    self._record_usage(resp, model_override=model)
                    return resp.choices[0].message.content
                except Exception as exc:
                    if "429" in str(exc) or "rate" in str(exc).lower():
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise
            raise RuntimeError("openai reasoning: exhausted retries")

        async def _call_gpt5() -> str:
            """gpt-5.x path: attempt reasoning_effort first, fall back."""
            # Try the reasoning-aware shape first.
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    reasoning_effort="high",
                )
                self._record_usage(resp, model_override=model)
                return resp.choices[0].message.content
            except TypeError:
                # SDK does not accept reasoning_effort kwarg yet.
                pass
            except Exception as exc:
                msg = str(exc).lower()
                if "reasoning_effort" not in msg and "unknown" not in msg:
                    raise  # re-raise non-parameter errors
            # Fallback: standard chat with a system-prompt hint.
            hinted: list[dict[str, str]] = [
                dict(m) if m["role"] != "system" else
                {"role": "system",
                 "content": m["content"] + "\n\nThink step by step before "
                                            "answering."}
                for m in messages
            ]
            has_system = any(m["role"] == "system" for m in hinted)
            if not has_system:
                hinted.insert(0, {
                    "role": "system",
                    "content": "Think step by step before answering.",
                })
            resp = await client.chat.completions.create(
                model=model,
                messages=hinted,
                max_completion_tokens=max_tokens,
            )
            self._record_usage(resp, model_override=model)
            return resp.choices[0].message.content

        async def _call_standard() -> str:
            """Any other model: regular chat + 'think step by step' hint."""
            hinted: list[dict[str, str]] = []
            saw_system = False
            for m in messages:
                if m["role"] == "system":
                    saw_system = True
                    hinted.append({
                        "role": "system",
                        "content": m["content"] + "\n\nThink step by step "
                                                   "before answering.",
                    })
                else:
                    hinted.append(m)
            if not saw_system:
                hinted.insert(0, {
                    "role": "system",
                    "content": "Think step by step before answering.",
                })
            resp = await client.chat.completions.create(
                model=model,
                messages=hinted,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            self._record_usage(resp, model_override=model)
            return resp.choices[0].message.content

        if is_reasoning_series:
            return await _call_reasoning()
        if is_gpt5:
            return await _call_gpt5()
        return await _call_standard()

    async def _think_gemini(
        self, messages: list[dict[str, str]], thinking_budget: int, max_tokens: int,
    ) -> str:
        """Gemini native thinking via google-genai SDK.

        Uses ``self.model`` (respecting the configured backbone) with
        ``ThinkingConfig``. Defensive against cases where the model
        spends all output tokens on internal thinking and returns no
        visible text parts — in that case, returns an empty string
        rather than crashing.
        """
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)

        system_text = ""
        user_text = ""
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            elif m["role"] == "user":
                user_text = m["content"]

        prompt = f"{system_text}\n\n{user_text}" if system_text else user_text

        record_self = self
        model = self.model

        def _call():
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=thinking_budget,
                    ),
                    max_output_tokens=max_tokens,
                    temperature=0.0,
                ),
            )
            record_self._record_usage(response, model_override=model)
            # Defensive parts extraction: candidates[0].content may be
            # None if the model was filtered or exhausted tokens on
            # thinking; parts may contain both thought and text blocks.
            parts: list[str] = []
            if response.candidates:
                content = response.candidates[0].content
                if content is not None and content.parts is not None:
                    for part in content.parts:
                        # ``thought`` attribute is only set on thinking
                        # blocks; text blocks may not have it at all.
                        is_thought = bool(getattr(part, "thought", False))
                        part_text = getattr(part, "text", None)
                        if not is_thought and part_text:
                            parts.append(part_text)
            return "\n".join(parts)

        return await asyncio.to_thread(_call)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_key(key: str | None) -> str | None:
        if key and key.startswith("${") and key.endswith("}"):
            env_var = key[2:-1]
            return os.environ.get(env_var)
        return key
