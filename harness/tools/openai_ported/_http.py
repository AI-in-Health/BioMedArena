"""Shared async HTTP helper for ported skills.

Reuses the retrieval-level diskcache (24h TTL, keyed by host + path +
params) so that smoke tests and benchmark runs do not hammer upstream
APIs. Retries are kept light — one retry on transient errors only —
because ported skills are treated as best-effort evidence, not hard
requirements.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import urlparse

import httpx

from harness.adapters.retrieval._cache import get_cache


_DEFAULT_TIMEOUT = 20.0
_USER_AGENT = "BioMedArena/openai_ported (+httpx)"


def _cache_key_params(params: dict | None, json_body: Any = None, method: str = "GET") -> dict:
    """Cache key must distinguish method + body + params."""
    return {
        "_method": method,
        "_params": params or {},
        "_body": json.dumps(json_body, sort_keys=True) if json_body is not None else "",
    }


async def request_json(
    url: str,
    *,
    method: str = "GET",
    params: dict | None = None,
    json_body: Any = None,
    headers: dict | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict | list | None:
    """Cached async JSON request. Returns None on non-2xx / network error."""
    cache = get_cache()
    parsed = urlparse(url)
    host = parsed.netloc
    path = parsed.path
    ckey = _cache_key_params(params, json_body, method)
    cached = cache.get(host, path, ckey)
    if cached is not None:
        return cached

    hdrs = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
    if headers:
        hdrs.update(headers)

    last_exc: Exception | None = None
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        for attempt in range(2):
            try:
                if method.upper() == "GET":
                    r = await client.get(url, params=params, headers=hdrs)
                elif method.upper() == "POST":
                    r = await client.post(url, params=params, json=json_body, headers=hdrs)
                else:
                    r = await client.request(method, url, params=params,
                                             json=json_body, headers=hdrs)
                r.raise_for_status()
                try:
                    data = r.json()
                except Exception:
                    data = {"_raw_text": r.text[:4000]}
                cache.set(host, path, ckey, data)
                return data
            except httpx.HTTPStatusError as exc:
                # 4xx: don't retry, cache as None signal to caller via return
                if 400 <= exc.response.status_code < 500:
                    return None
                last_exc = exc
            except Exception as exc:
                last_exc = exc
            if attempt == 0:
                await asyncio.sleep(0.5)
    return None


async def request_sparql(
    endpoint: str,
    query: str,
    *,
    timeout: float = 30.0,
) -> dict | None:
    """POST a SPARQL query, return JSON bindings or None."""
    cache = get_cache()
    parsed = urlparse(endpoint)
    host = parsed.netloc
    path = parsed.path
    ckey = {"_sparql_q": query}
    cached = cache.get(host, path, ckey)
    if cached is not None:
        return cached

    hdrs = {
        "User-Agent": _USER_AGENT,
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        try:
            r = await client.post(endpoint, data={"query": query}, headers=hdrs)
            r.raise_for_status()
            data = r.json()
            cache.set(host, path, ckey, data)
            return data
        except Exception:
            return None
