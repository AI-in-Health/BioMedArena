"""gget + mygene TOOL_SPECS — stateless HTTP wrappers for genomics APIs.

Per the integration routing rule: these are <100 LOC single-shot API calls,
so they live as TOOL_SPECS entries callable mid-reasoning, not as adapters.

gget: https://github.com/pachterlab/gget
  Covers Ensembl / UniProt / NCBI / AlphaFold / ARCHS4 etc. in one CLI-like API.
mygene: https://github.com/biothings/mygene.py
  BioThings gene info service (backed by NCBI / Ensembl / UniProt aggregate).
"""

from __future__ import annotations

from typing import Any


# ======================================================================
# gget wrappers
# ======================================================================


def _gget_search_sync(query: str, species: str = "human", limit: int = 5) -> dict[str, Any]:
    """gget.search → Ensembl IDs for a gene symbol or disease term."""
    import gget
    try:
        df = gget.search(
            searchwords=[query], species=species, limit=limit,
            verbose=False, wrap_text=False,
        )
    except TypeError:
        # Some gget versions don't accept wrap_text/verbose; fallback
        df = gget.search(searchwords=[query], species=species, limit=limit)

    if df is None or (hasattr(df, "empty") and df.empty):
        return {"ok": False, "query": query, "species": species, "matches": []}

    # gget.search returns a DataFrame
    records = df.to_dict(orient="records")
    # Serialise non-primitive values
    for r in records:
        for k, v in list(r.items()):
            r[k] = str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v
    return {"ok": True, "query": query, "species": species, "matches": records[:limit]}


def _gget_info_sync(ensembl_ids: list[str]) -> dict[str, Any]:
    """gget.info → detailed info (name, description, location) for Ensembl IDs."""
    import gget
    if isinstance(ensembl_ids, str):
        ensembl_ids = [ensembl_ids]
    try:
        df = gget.info(ensembl_ids, verbose=False)
    except TypeError:
        df = gget.info(ensembl_ids)

    if df is None or (hasattr(df, "empty") and df.empty):
        return {"ok": False, "ensembl_ids": ensembl_ids, "info": []}
    records = df.to_dict(orient="records")
    for r in records:
        for k, v in list(r.items()):
            r[k] = str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v
    return {"ok": True, "info": records}


def _gget_seq_sync(ensembl_ids: list[str], translate: bool = False) -> dict[str, Any]:
    """gget.seq → DNA or protein sequence for Ensembl IDs."""
    import gget
    if isinstance(ensembl_ids, str):
        ensembl_ids = [ensembl_ids]
    try:
        result = gget.seq(ensembl_ids, translate=translate, verbose=False)
    except TypeError:
        result = gget.seq(ensembl_ids, translate=translate)

    # gget.seq returns a list of FASTA strings like ['>hdr', 'SEQ']
    if result is None:
        return {"ok": False, "ensembl_ids": ensembl_ids, "sequences": []}

    sequences = []
    if isinstance(result, list):
        # Pair up header + sequence
        for i in range(0, len(result), 2):
            header = result[i] if i < len(result) else ""
            seq = result[i + 1] if i + 1 < len(result) else ""
            sequences.append({
                "header": str(header),
                "sequence_length": len(str(seq)),
                # Truncate long sequences for prompt-safety
                "sequence_preview": str(seq)[:200],
                "translated": bool(translate),
            })
    return {"ok": True, "sequences": sequences}


# ======================================================================
# mygene wrappers
# ======================================================================


def _mygene_query_sync(
    query: str,
    species: str = "human",
    fields: list[str] | None = None,
    size: int = 5,
) -> dict[str, Any]:
    """mygene.info query: flexible search across symbol / name / entrez / ensembl."""
    import mygene
    fields = fields or ["symbol", "name", "entrezgene", "ensembl.gene", "type_of_gene"]
    mg = mygene.MyGeneInfo()
    try:
        result = mg.query(query, species=species, fields=",".join(fields), size=size)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    hits = result.get("hits", []) if isinstance(result, dict) else []
    # Clean nested dicts for prompt-safe serialisation
    cleaned = []
    for h in hits[:size]:
        flat = {k: v for k, v in h.items() if not k.startswith("_")}
        cleaned.append(flat)
    return {"ok": True, "query": query, "total": result.get("total", len(cleaned)) if isinstance(result, dict) else len(cleaned),
            "hits": cleaned}


# ======================================================================
# TOOL_SPECS
# ======================================================================


GGET_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "gget_search",
            "description": (
                "Search Ensembl for genes matching a symbol, disease, or free-text term. "
                "Returns matching Ensembl IDs with short descriptions. Use when you have a "
                "gene name but need the canonical Ensembl ID for downstream queries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Gene symbol or free text"},
                    "species": {"type": "string", "default": "human",
                                "description": "Ensembl species name, e.g. 'human' or 'mouse'"},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gget_info",
            "description": (
                "Fetch detailed Ensembl + NCBI + UniProt info for one or more Ensembl IDs: "
                "official symbol, description, biotype, chromosome location."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ensembl_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ensembl IDs (e.g., ENSG00000012048 for BRCA1)",
                    },
                },
                "required": ["ensembl_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gget_seq",
            "description": (
                "Retrieve DNA or protein sequence for Ensembl gene IDs. Sequence is returned "
                "truncated to 200 chars in a preview field; use `translated: true` for the "
                "protein sequence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ensembl_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "translated": {"type": "boolean", "default": False,
                                    "description": "If true, return protein sequence instead of DNA"},
                },
                "required": ["ensembl_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mygene_query",
            "description": (
                "Search BioThings mygene.info for genes by symbol, name, Entrez ID, or Ensembl "
                "ID. Returns structured records with symbol, name, entrezgene, ensembl.gene, "
                "and type_of_gene. Preferred for quick gene metadata lookups."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "species": {"type": "string", "default": "human"},
                    "fields": {"type": "array", "items": {"type": "string"},
                                "description": "mygene fields to retrieve"},
                    "size": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
]


GGET_TOOL_NAMES = {"gget_search", "gget_info", "gget_seq", "mygene_query"}


def handle_gget_tool(name: str, args: dict[str, Any]) -> str:
    """Synchronous dispatcher — called by FunctionCallingRunner in a thread."""
    import json as _json
    try:
        if name == "gget_search":
            r = _gget_search_sync(
                args["query"],
                species=args.get("species", "human"),
                limit=int(args.get("limit", 5)),
            )
        elif name == "gget_info":
            r = _gget_info_sync(args["ensembl_ids"])
        elif name == "gget_seq":
            r = _gget_seq_sync(
                args["ensembl_ids"],
                translate=bool(args.get("translated", False)),
            )
        elif name == "mygene_query":
            r = _mygene_query_sync(
                args["query"],
                species=args.get("species", "human"),
                fields=args.get("fields"),
                size=int(args.get("size", 5)),
            )
        else:
            return f"[unknown gget tool: {name}]"
    except Exception as exc:
        return f"[{name} error: {exc}]"
    return _json.dumps(r, ensure_ascii=False, default=str)[:15000]
