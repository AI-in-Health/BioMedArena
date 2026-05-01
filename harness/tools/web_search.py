"""Serper (Google Search) + Jina (Page Reader) web tools.

Provides two tools:
  - serper_search: Google search via Serper API (returns snippets + URLs)
  - jina_read_page: Fetch and summarize a web page via Jina Reader API

Environment variables:
  - SERPER_API_KEY (required for serper_search)
  - JINA_API_KEY (optional, for authenticated Jina Reader access)

These tools are registered in TOOL_SPECS via the web_tools category and
can be toggled with the --web-tools CLI flag.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SERPER_API_URL = "https://google.serper.dev/search"
JINA_READER_URL = "https://r.jina.ai"
MAX_SERPER_RESULTS = 10
DEFAULT_SNIPPET_CHARS = 4000
DEFAULT_PAGE_CHARS = 80000

# Blocked domains — benchmark dataset hosts that would leak ground-truth answers.
# Add domains (without scheme) to prevent data contamination during evaluation.
BLOCKED_DOMAINS: tuple[str, ...] = (
    "huggingface.co",
    "hf.co",
    "datasets-server.huggingface.co",
)

# Blocked URL path patterns — specific papers / pages containing benchmark answers.
# Matched against the full URL (case-insensitive substring match).
BLOCKED_URL_PATTERNS: tuple[str, ...] = (
    # Preprint containing benchmark data as supplementary material.
    # Covers both DOI prefixes (10.64898/... and 10.1101/...) and direct biorxiv paths.
    "2026.01.06.697527",
    # GCS bucket containing benchmark evaluation data.
    "storage.googleapis.com/bixbench-results",
)


def _is_blocked_url(url: str) -> bool:
    """Return True if *url* belongs to a blocked domain or matches a blocked pattern."""
    from urllib.parse import urlparse
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        return False
    if any(host == d or host.endswith("." + d) for d in BLOCKED_DOMAINS):
        return True
    # Check URL path patterns (DOI-based blocking for specific papers)
    url_lower = url.lower()
    return any(pat.lower() in url_lower for pat in BLOCKED_URL_PATTERNS)

EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, output the **full original context** as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" fields**"""

# ---------------------------------------------------------------------------
# Tool specs (OpenAI function-calling schema)
# ---------------------------------------------------------------------------

WEB_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "serper_search",
            "description": (
                "Search the web using Google via Serper API. Returns titles, "
                "URLs, and snippets. Use this to find current information, "
                "verify facts, or discover relevant web pages to read."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (natural language or quoted phrases for exact match).",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-10, default 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "jina_read_page",
            "description": (
                "Fetch and read a web page using Jina Reader API. Returns "
                "structured evidence and summary extracted from the page "
                "relevant to the specified goal. Use after serper_search to "
                "read promising URLs in depth."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the web page to read.",
                    },
                    "goal": {
                        "type": "string",
                        "description": "What information you are looking for on this page.",
                    },
                },
                "required": ["url", "goal"],
            },
        },
    },
]

WEB_TOOL_NAMES = {spec["function"]["name"] for spec in WEB_TOOL_SPECS}


# ---------------------------------------------------------------------------
# Serper Search Client
# ---------------------------------------------------------------------------


class SerperClient:
    """Async Serper Google Search client."""

    def __init__(self, api_key: str | None = None, max_chars: int = DEFAULT_SNIPPET_CHARS):
        self.api_key = api_key or os.environ.get("SERPER_API_KEY", "")
        self._max_chars = max_chars

    async def search(self, query: str, num_results: int = 5) -> str:
        """Execute a search and return JSON results."""
        if not self.api_key:
            return json.dumps({"error": "SERPER_API_KEY not set"})

        query = (query or "").strip()
        if not query:
            return json.dumps({"error": "serper_search query is empty"})
        try:
            num_results = int(num_results)
        except (TypeError, ValueError):
            num_results = 5
        num_results = min(max(num_results, 1), MAX_SERPER_RESULTS)
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": num_results}

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(SERPER_API_URL, json=payload, headers=headers)
                if resp.status_code >= 400:
                    body = resp.text[:500]
                    logger.error(
                        "[serper_search] HTTP %s body=%s",
                        resp.status_code,
                        body,
                    )
                    return json.dumps({
                        "error": f"Serper HTTP {resp.status_code}: {body}",
                    })
                data = resp.json()

            results = {}
            rank = 0
            for item in data.get("organic", []):
                url = item.get("link", "")
                if _is_blocked_url(url):
                    logger.info("[serper_search] filtered blocked result: %s", url)
                    continue
                rank += 1
                if rank > num_results:
                    break
                title = item.get("title", f"Result {rank}")
                snippet = item.get("snippet", "")[:500]
                results[f"{rank}. {title}"] = {"url": url, "snippet": snippet}

            output = json.dumps(results, ensure_ascii=False)
            return output[:self._max_chars]
        except Exception as e:
            logger.error("[serper_search] %s", e)
            return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Jina Page Reader Client
# ---------------------------------------------------------------------------


class JinaClient:
    """Async Jina Reader client with LLM-based summarization."""

    def __init__(
        self,
        api_key: str | None = None,
        max_page_chars: int = DEFAULT_PAGE_CHARS,
        summarizer_llm: Any | None = None,
    ):
        self.api_key = api_key or os.environ.get("JINA_API_KEY", "")
        self._max_page_chars = max_page_chars
        self._summarizer_llm = summarizer_llm

    async def fetch_page(self, url: str) -> str:
        """Fetch page content via Jina Reader API with retries."""
        if _is_blocked_url(url):
            logger.info("[jina_fetch] blocked URL (benchmark data host): %s", url)
            return ""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # Accept plain text
        headers["Accept"] = "text/plain"

        jina_url = f"{JINA_READER_URL}/{url}"

        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.get(jina_url, headers=headers)
                    if resp.status_code == 200:
                        text = resp.text
                        if text and len(text.strip()) > 50:
                            return text[:self._max_page_chars]
                    logger.warning(
                        "[jina_fetch] attempt %d status=%d for %s",
                        attempt + 1, resp.status_code, url,
                    )
            except Exception as e:
                logger.warning("[jina_fetch] attempt %d error for %s: %s", attempt + 1, url, e)
            await asyncio.sleep(0.5 * (attempt + 1))

        return ""

    async def read_page(self, url: str, goal: str) -> str:
        """Fetch a page and return structured evidence+summary."""
        raw_content = await self.fetch_page(url)

        if not raw_content:
            return (
                f"[jina_read_page] Failed to fetch content from {url}. "
                "The page may be unavailable or blocked."
            )

        # If we have a summarizer LLM, use it for evidence extraction
        if self._summarizer_llm is not None:
            summary = await self._summarize_with_llm(raw_content, goal)
            if summary:
                return summary

        # Fallback: return truncated raw content
        truncated = raw_content[:6000]
        return (
            f"Content from {url} (truncated to first 6000 chars):\n\n"
            f"{truncated}"
        )

    async def _summarize_with_llm(self, content: str, goal: str) -> str:
        """Use LLM to extract evidence and summary from page content."""
        # Truncate content to avoid overwhelming the summarizer
        content_for_llm = content[:60000]
        prompt = EXTRACTOR_PROMPT.format(webpage_content=content_for_llm, goal=goal)

        try:
            resp = await self._summarizer_llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=4096,
            )

            parsed = self._parse_json_response(resp)
            if parsed:
                evidence = parsed.get("evidence", "")
                summary = parsed.get("summary", "")
                return (
                    f"Evidence from page:\n{evidence}\n\n"
                    f"Summary:\n{summary}"
                )
            # If JSON parse fails, return the raw LLM response
            return resp[:4000] if resp else ""
        except Exception as e:
            logger.warning("[jina_summarize] LLM error: %s", e)
            return ""

    @staticmethod
    def _parse_json_response(raw: str) -> dict[str, Any] | None:
        """Try to parse JSON from LLM response."""
        if not raw:
            return None
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try extracting JSON from surrounding text
            left = cleaned.find("{")
            right = cleaned.rfind("}")
            if left != -1 and right != -1 and left <= right:
                try:
                    return json.loads(cleaned[left: right + 1])
                except json.JSONDecodeError:
                    pass
        return None


# ---------------------------------------------------------------------------
# Dispatch handler (called by FunctionCallingRunner._call)
# ---------------------------------------------------------------------------

# Module-level singletons (lazy-initialized)
_serper_client: SerperClient | None = None
_jina_client: JinaClient | None = None


def init_web_clients(summarizer_llm: Any | None = None) -> None:
    """Initialize web tool clients. Called once before tool dispatch.

    Args:
        summarizer_llm: An LLMClient instance for Jina page summarization.
            If None, raw content is returned (truncated).
    """
    global _serper_client, _jina_client
    _serper_client = SerperClient()
    _jina_client = JinaClient(summarizer_llm=summarizer_llm)


async def handle_web_tool(name: str, args: dict[str, Any]) -> str:
    """Dispatch a web tool call. Returns result string."""
    global _serper_client, _jina_client

    if name == "serper_search":
        if _serper_client is None:
            _serper_client = SerperClient()
        return await _serper_client.search(
            query=args.get("query", ""),
            num_results=args.get("num_results", 5),
        )

    if name == "jina_read_page":
        if _jina_client is None:
            _jina_client = JinaClient()
        return await _jina_client.read_page(
            url=args.get("url", ""),
            goal=args.get("goal", ""),
        )

    return f"[unknown web tool: {name}]"
