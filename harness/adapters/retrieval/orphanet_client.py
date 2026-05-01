"""Orphanet client (rare diseases, free, no auth).

Orphadata API: http://www.orphadata.org/cgi-bin/
Also fallback to web search for disease names.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote_plus

from harness.adapters.retrieval._base import BaseRetrievalClient


class OrphanetClient(BaseRetrievalClient):
    """Rare disease lookup via Orphanet API."""

    async def disease_search(self, query: str) -> dict | None:
        """Search Orphanet via their public API."""
        # Try the new Orphadata API
        url = f"https://api.orphadata.com/rd-cross-referencing/orphacodes"
        data = await self._get_json(url, {"name": query})
        return data

    async def summary(self, query: str) -> dict[str, Any]:
        data = await self.disease_search(query)
        if not data:
            return self._empty_summary(f"Orphanet: no match for '{query}'")

        results = data.get("data", {}).get("results", []) if isinstance(data, dict) else []
        if not results:
            # Alternate response shapes
            results = data if isinstance(data, list) else []

        if not results:
            return self._empty_summary(f"Orphanet: empty results for '{query}'")

        parts = []
        evidence = []
        for r in results[:3]:
            if isinstance(r, dict):
                orpha = r.get("ORPHAcode", r.get("code", ""))
                name = r.get("Preferred_term", r.get("name", ""))
                parts.append(f"[Orphanet ORPHA:{orpha}] {name}")
                evidence.append(f"Orphanet ORPHA:{orpha}")

        if not parts:
            return self._empty_summary(f"Orphanet: no structured results")

        return {"summary": "\n".join(parts), "evidence": evidence}
