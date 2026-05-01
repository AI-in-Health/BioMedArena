"""OMIM client (requires free API key from omim.org/api).

Graceful fallback: if no key, returns empty summary without error.
"""

from __future__ import annotations

import os
from typing import Any

from harness.adapters.retrieval._base import BaseRetrievalClient

BASE = "https://api.omim.org/api"


class OMIMClient(BaseRetrievalClient):
    def __init__(self, api_key: str | None = None, timeout: int = 20):
        super().__init__(timeout=timeout)
        self.api_key = api_key or os.environ.get("OMIM_API_KEY")
        self.enabled = bool(self.api_key)

    async def search_entry(self, term: str, limit: int = 5) -> dict | None:
        if not self.enabled:
            return None
        return await self._get_json(
            f"{BASE}/entry/search",
            {"search": term, "limit": limit, "format": "json", "apiKey": self.api_key},
        )

    async def entry(self, mim_number: str | int, include: str = "text") -> dict | None:
        if not self.enabled:
            return None
        return await self._get_json(
            f"{BASE}/entry",
            {"mimNumber": str(mim_number), "include": include, "format": "json", "apiKey": self.api_key},
        )

    async def summary(self, query: str) -> dict[str, Any]:
        if not self.enabled:
            return self._empty_summary("OMIM: API key not configured (set OMIM_API_KEY)")

        search = await self.search_entry(query, limit=3)
        if not search:
            return self._empty_summary(f"OMIM: no results for '{query}'")

        entries = search.get("omim", {}).get("searchResponse", {}).get("entryList", [])
        if not entries:
            return self._empty_summary(f"OMIM: no entries for '{query}'")

        parts = []
        evidence = []
        for e in entries[:2]:
            entry_data = e.get("entry", {})
            mim = entry_data.get("mimNumber", "")
            title = entry_data.get("titles", {}).get("preferredTitle", "")
            parts.append(f"[OMIM #{mim}] {title}")
            evidence.append(f"OMIM #{mim}")

        return {"summary": "\n".join(parts), "evidence": evidence}
