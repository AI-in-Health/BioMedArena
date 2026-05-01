"""RxNav / RxNorm client (NIH NLM, free, no auth).

API docs: https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html
"""

from __future__ import annotations

from typing import Any

from harness.adapters.retrieval._base import BaseRetrievalClient

BASE = "https://rxnav.nlm.nih.gov/REST"


class RxNavClient(BaseRetrievalClient):
    """Drug information lookups via NIH RxNav."""

    async def drug_by_name(self, name: str) -> dict | None:
        return await self._get_json(f"{BASE}/drugs.json", {"name": name})

    async def rxcui_by_name(self, name: str) -> str | None:
        data = await self._get_json(f"{BASE}/rxcui.json", {"name": name})
        if not data:
            return None
        ids = data.get("idGroup", {}).get("rxnormId", [])
        return ids[0] if ids else None

    async def interactions(self, rxcui: str) -> dict | None:
        return await self._get_json(f"{BASE}/interaction/interaction.json", {"rxcui": rxcui})

    async def summary(self, query: str) -> dict[str, Any]:
        """Composite lookup: name → rxcui → interactions."""
        data = await self.drug_by_name(query)
        if not data:
            return self._empty_summary(f"RxNav: no match for '{query}'")

        concepts = []
        for group in data.get("drugGroup", {}).get("conceptGroup", []) or []:
            for cp in group.get("conceptProperties", []) or []:
                name = cp.get("name", "")
                tty = cp.get("tty", "")
                concepts.append(f"{name} ({tty})")
                if len(concepts) >= 5:
                    break
            if len(concepts) >= 5:
                break

        if not concepts:
            return self._empty_summary(f"RxNav: no concepts for '{query}'")

        # Get interactions for first rxcui
        rxcui = await self.rxcui_by_name(query)
        interactions_summary = ""
        evidence: list[str] = [f"RxNorm: {c}" for c in concepts[:3]]

        if rxcui:
            inter = await self.interactions(rxcui)
            if inter:
                pairs = []
                for type_group in (inter.get("interactionTypeGroup") or []):
                    for itype in type_group.get("interactionType", []):
                        for pair in itype.get("interactionPair", [])[:3]:
                            desc = pair.get("description", "")
                            if desc:
                                pairs.append(desc[:150])
                if pairs:
                    interactions_summary = "Interactions: " + "; ".join(pairs[:3])
                    evidence.extend(pairs[:3])

        parts = [f"RxNav drug info for '{query}': " + ", ".join(concepts[:3])]
        if interactions_summary:
            parts.append(interactions_summary)

        return {"summary": "\n".join(parts), "evidence": evidence, "rxcui": rxcui}
