"""DailyMed client (NIH, free, no auth). Structured Product Labels (SPL)."""

from __future__ import annotations

from typing import Any

from harness.adapters.retrieval._base import BaseRetrievalClient

BASE = "https://dailymed.nlm.nih.gov/dailymed/services/v2"


class DailyMedClient(BaseRetrievalClient):
    async def spls_by_name(self, drug: str, pagesize: int = 5) -> dict | None:
        return await self._get_json(f"{BASE}/spls.json", {"drug_name": drug, "pagesize": pagesize, "page": 1})

    async def spl_detail(self, setid: str) -> dict | None:
        return await self._get_json(f"{BASE}/spls/{setid}.json", None)

    async def summary(self, query: str) -> dict[str, Any]:
        data = await self.spls_by_name(query, pagesize=3)
        if not data:
            return self._empty_summary(f"DailyMed: no SPL for '{query}'")

        results = data.get("data", [])
        if not results:
            return self._empty_summary(f"DailyMed: empty results for '{query}'")

        parts = []
        evidence = []
        for spl in results[:3]:
            title = spl.get("title", "")[:200]
            setid = spl.get("setid", "")
            published = spl.get("published_date", "")
            parts.append(f"[DailyMed] {title} (published: {published})")
            evidence.append(f"DailyMed SPL: {setid}")

        return {"summary": "\n".join(parts), "evidence": evidence}
