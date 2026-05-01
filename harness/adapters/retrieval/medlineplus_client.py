"""MedlinePlus Connect client (NIH, free, no auth).

Replaces paid UpToDate with patient/clinician health topics.
API: https://medlineplus.gov/medlineplus-connect/
"""

from __future__ import annotations

import re
from typing import Any

from harness.adapters.retrieval._base import BaseRetrievalClient

# Web search API for MedlinePlus
WSEARCH = "https://wsearch.nlm.nih.gov/ws/query"


class MedlinePlusClient(BaseRetrievalClient):
    """Patient-friendly health topic search via NIH MedlinePlus."""

    async def search(self, term: str, retmax: int = 3) -> str | None:
        # Returns XML
        return await self._get_text(
            WSEARCH,
            {"db": "healthTopics", "term": term, "retmax": retmax},
        )

    async def summary(self, query: str) -> dict[str, Any]:
        xml_text = await self.search(query, retmax=3)
        if not xml_text:
            return self._empty_summary(f"MedlinePlus: no results for '{query}'")

        # Simple regex extraction (avoids XML parser deps)
        titles = re.findall(r'<content name="title"[^>]*>(.*?)</content>', xml_text, re.DOTALL)
        snippets = re.findall(r'<content name="FullSummary"[^>]*>(.*?)</content>', xml_text, re.DOTALL)

        if not titles:
            return self._empty_summary(f"MedlinePlus: empty parse for '{query}'")

        parts = []
        evidence = []
        for i, title in enumerate(titles[:3]):
            # Strip HTML
            clean_title = re.sub(r"<[^>]+>", "", title).strip()
            parts.append(f"[MedlinePlus] {clean_title}")
            evidence.append(f"MedlinePlus: {clean_title[:50]}")
            if i < len(snippets):
                clean = re.sub(r"<[^>]+>", "", snippets[i])[:300]
                if clean.strip():
                    parts.append(f"  {clean.strip()}")

        return {"summary": "\n".join(parts), "evidence": evidence}
