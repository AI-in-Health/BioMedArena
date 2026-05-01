"""OpenFDA client (free, no auth required).

API docs: https://open.fda.gov/apis/
"""

from __future__ import annotations

from typing import Any

from harness.adapters.retrieval._base import BaseRetrievalClient

BASE = "https://api.fda.gov"


class OpenFDAClient(BaseRetrievalClient):
    """FDA adverse events, drug labels, and enforcement actions."""

    async def adverse_events(self, drug: str, limit: int = 5) -> dict | None:
        return await self._get_json(
            f"{BASE}/drug/event.json",
            {"search": f'patient.drug.medicinalproduct:"{drug}"', "limit": limit},
        )

    async def drug_label(self, drug: str) -> dict | None:
        return await self._get_json(
            f"{BASE}/drug/label.json",
            {"search": f'openfda.brand_name:"{drug}" OR openfda.generic_name:"{drug}"', "limit": 3},
        )

    async def summary(self, query: str) -> dict[str, Any]:
        """Return adverse events + label highlights for a drug query."""
        adverse = await self.adverse_events(query, limit=5)
        label = await self.drug_label(query)

        parts: list[str] = []
        evidence: list[str] = []

        if label:
            for result in (label.get("results") or [])[:1]:
                indications = result.get("indications_and_usage", [])
                warnings = result.get("warnings", [])
                if indications:
                    text = indications[0][:300]
                    parts.append(f"[OpenFDA Label] Indications: {text}")
                    evidence.append("OpenFDA drug label")
                if warnings:
                    text = warnings[0][:200]
                    parts.append(f"[OpenFDA Label] Warnings: {text}")

        if adverse:
            reactions: dict[str, int] = {}
            for result in (adverse.get("results") or []):
                for drug_entry in result.get("patient", {}).get("reaction", []):
                    r = drug_entry.get("reactionmeddrapt", "")
                    if r:
                        reactions[r] = reactions.get(r, 0) + 1
            top = sorted(reactions.items(), key=lambda x: -x[1])[:5]
            if top:
                parts.append("[OpenFDA AE] Top reactions: " + ", ".join(f"{r}({c})" for r, c in top))
                evidence.extend([f"OpenFDA: {r}" for r, _ in top[:3]])

        if not parts:
            return self._empty_summary(f"OpenFDA: no data for '{query}'")

        return {"summary": "\n".join(parts), "evidence": evidence}
