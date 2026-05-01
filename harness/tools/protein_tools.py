"""Protein TOOL_SPECS — stateless AlphaFold-DB HTTP wrapper.

ESM embedding is too heavy for a stateless tool (30MB model weights need
to stay loaded), so it lives in the adapter. Only AlphaFold-DB lookup is
exposed here since it's a pure HTTP call.
"""

from __future__ import annotations

import json as _json
from typing import Any


_AFDB = "https://alphafold.ebi.ac.uk/api/prediction"


def _alphafold_db_sync(uniprot_id: str) -> dict[str, Any]:
    """Fetch AlphaFold-DB prediction metadata for a UniProt ID."""
    import httpx
    uid = uniprot_id.strip().upper()
    try:
        with httpx.Client(timeout=15.0, follow_redirects=True) as client:
            resp = client.get(f"{_AFDB}/{uid}")
            if resp.status_code != 200:
                return {
                    "ok": False,
                    "uniprot_id": uid,
                    "error": f"HTTP {resp.status_code}",
                    "status_code": resp.status_code,
                }
            data = resp.json()
    except Exception as exc:
        return {"ok": False, "uniprot_id": uid, "error": f"{type(exc).__name__}: {exc}"}

    if not data:
        return {"ok": False, "uniprot_id": uid, "error": "empty response"}

    # API returns a list (usually length 1)
    entry = data[0] if isinstance(data, list) else data
    # Key fields (tolerant of schema changes)
    return {
        "ok": True,
        "uniprot_id": uid,
        "gene": entry.get("gene"),
        "organism_scientific_name": entry.get("organismScientificName"),
        "sequence_length": entry.get("uniprotEnd") or entry.get("sequenceLength"),
        "plddt_mean": entry.get("globalMetricValue", 0.0),  # often pLDDT mean
        "model_created_date": entry.get("modelCreatedDate"),
        "pdb_url": entry.get("pdbUrl"),
        "cif_url": entry.get("cifUrl"),
        "latest_version": entry.get("latestVersion"),
    }


PROTEIN_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "alphafold_db_lookup",
            "description": (
                "Fetch AlphaFold-DB prediction metadata for a UniProt accession: "
                "pLDDT mean, sequence length, organism, PDB/CIF download URLs, model "
                "version. Returns None (ok=false) if the UniProt ID is not in AF-DB."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "uniprot_id": {
                        "type": "string",
                        "description": "UniProt accession, e.g. 'P04637' for human TP53",
                    },
                },
                "required": ["uniprot_id"],
            },
        },
    },
]


PROTEIN_TOOL_NAMES = {"alphafold_db_lookup"}


def handle_protein_tool(name: str, args: dict[str, Any]) -> str:
    if name == "alphafold_db_lookup":
        try:
            r = _alphafold_db_sync(args["uniprot_id"])
            return _json.dumps(r, default=str)[:8000]
        except Exception as exc:
            return f"[{name} error: {exc}]"
    return f"[unknown protein tool: {name}]"
