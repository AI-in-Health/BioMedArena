"""Base class for retrieval clients — cached GET with rate limiting."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

import httpx

from harness.adapters.retrieval._cache import get_cache
from harness.adapters.retrieval._rate_limiter import acquire


class BaseRetrievalClient:
    """Common async HTTP GET helper with caching + rate limiting."""

    def __init__(self, timeout: int = 20):
        self._http = httpx.AsyncClient(timeout=timeout, follow_redirects=True)
        self._cache = get_cache()

    async def _get_json(self, url: str, params: dict | None = None) -> dict | None:
        host = urlparse(url).netloc
        path = urlparse(url).path
        cached = self._cache.get(host, path, params)
        if cached is not None:
            return cached

        async with acquire(host):
            try:
                resp = await self._http.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()
                self._cache.set(host, path, params, data)
                return data
            except Exception:
                return None

    async def _get_text(self, url: str, params: dict | None = None) -> str | None:
        host = urlparse(url).netloc
        path = urlparse(url).path
        cached = self._cache.get(host, path, params)
        if cached is not None:
            return cached

        async with acquire(host):
            try:
                resp = await self._http.get(url, params=params)
                resp.raise_for_status()
                text = resp.text
                self._cache.set(host, path, params, text)
                return text
            except Exception:
                return None

    @staticmethod
    def _empty_summary(reason: str = "No results") -> dict[str, Any]:
        return {"summary": "", "evidence": [], "reason": reason}
