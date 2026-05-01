"""Handler functions for ported OpenAI life-science skills.

Each handler is a thin wrapper around a public REST / GraphQL / SPARQL
endpoint. All handlers return a string (JSON-encoded dict) compatible
with the function-calling runner's tool-message contract. Results are
capped at ~2 KB to keep the LLM context bounded.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

from harness.tools.openai_ported._http import request_json, request_sparql


# ---------------------------------------------------------------------- util


def _compact(obj: Any, max_items: int = 10, max_chars: int = 2000) -> str:
    """JSON-encode, truncating lists and total payload size."""
    def shrink(x: Any) -> Any:
        if isinstance(x, list):
            return [shrink(v) for v in x[:max_items]]
        if isinstance(x, dict):
            return {k: shrink(v) for k, v in x.items()}
        if isinstance(x, str) and len(x) > 500:
            return x[:500] + "..."
        return x

    s = json.dumps(shrink(obj), default=str, ensure_ascii=False)
    if len(s) > max_chars:
        s = s[:max_chars] + "...<truncated>"
    return s


def _err(source: str, msg: str) -> str:
    return json.dumps({"ok": False, "source": source, "error": msg})


# ------------------------------------------------------------------ bgee (SPARQL)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/bgee-skill/SKILL.md
# API:    https://www.bgee.org/sparql/

async def bgee_sparql(args: dict[str, Any]) -> str:
    query = args.get("query", "").strip()
    if not query:
        return _err("bgee", "missing 'query'")
    data = await request_sparql("https://www.bgee.org/sparql/", query)
    if data is None:
        return _err("bgee", "network or parse error")
    bindings = data.get("results", {}).get("bindings", [])
    return _compact({
        "ok": True, "source": "bgee",
        "vars": data.get("head", {}).get("vars", []),
        "n_bindings": len(bindings),
        "bindings": bindings,
    }, max_items=int(args.get("max_items", 10) or 10))


# ---------------------------------------------------------- bindingdb (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/bindingdb-skill/SKILL.md
# API:    https://bindingdb.org/rest/*

async def bindingdb_ligands(args: dict[str, Any]) -> str:
    mode = args.get("mode", "smiles")
    params = {"response": "application/json"}
    if mode == "pdb":
        pdb = args.get("pdb", "")
        if not pdb:
            return _err("bindingdb", "mode=pdb needs 'pdb'")
        params.update({"pdb": pdb,
                       "cutoff": args.get("cutoff", 100),
                       "identity": args.get("identity", 90)})
        path = "rest/getLigandsByPDBs"
    elif mode == "uniprot":
        up = args.get("uniprot", "")
        if not up:
            return _err("bindingdb", "mode=uniprot needs 'uniprot'")
        params.update({"uniprot": up,
                       "cutoff": args.get("cutoff", 100),
                       "code": args.get("code", "0")})
        path = "rest/getLigandsByUniprots"
    elif mode == "smiles":
        smi = args.get("smiles", "")
        if not smi:
            return _err("bindingdb", "mode=smiles needs 'smiles'")
        params.update({"smiles": smi, "cutoff": args.get("cutoff", 0.85)})
        path = "rest/getLigandsBySmiles"
    else:
        return _err("bindingdb", f"unknown mode: {mode}")
    data = await request_json(f"https://bindingdb.org/{path}", params=params)
    if data is None:
        return _err("bindingdb", "no response")
    return _compact({"ok": True, "source": "bindingdb", "mode": mode, "data": data},
                    max_items=int(args.get("max_items", 10) or 10))


# ------------------------------------------------------------ chebi (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/chebi-skill/SKILL.md
# API:    https://www.ebi.ac.uk/chebi/backend/api/public/

async def chebi_lookup(args: dict[str, Any]) -> str:
    action = args.get("action", "search")
    base = "https://www.ebi.ac.uk/chebi/backend/api/public"
    if action == "search":
        q = args.get("query", "")
        if not q:
            return _err("chebi", "action=search needs 'query'")
        data = await request_json(
            f"{base}/es_search/",
            params={"query": q, "size": int(args.get("size", 10))},
        )
        if data is None:
            return _err("chebi", "no response")
        results = data.get("results") or data.get("hits") or data
        return _compact({"ok": True, "source": "chebi", "action": "search",
                         "results": results},
                        max_items=int(args.get("max_items", 10) or 10))
    if action == "compound":
        cid = args.get("chebi_id", "")
        if not cid:
            return _err("chebi", "action=compound needs 'chebi_id'")
        if not cid.upper().startswith("CHEBI:"):
            cid = f"CHEBI:{cid}"
        data = await request_json(f"{base}/compound/{cid}/")
        if data is None:
            return _err("chebi", f"not found: {cid}")
        return _compact({"ok": True, "source": "chebi", "action": "compound",
                         "compound": data}, max_items=20)
    return _err("chebi", f"unknown action: {action}")


# ------------------------------------------------------------ civic (GraphQL)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/civic-skill/SKILL.md
# API:    https://civicdb.org/api/graphql

async def civic_graphql(args: dict[str, Any]) -> str:
    query = args.get("query", "").strip()
    if not query:
        return _err("civic", "missing 'query'")
    body = {"query": query}
    if "variables" in args:
        body["variables"] = args["variables"]
    data = await request_json(
        "https://civicdb.org/api/graphql", method="POST", json_body=body,
    )
    if data is None:
        return _err("civic", "no response")
    if "errors" in data:
        return _compact({"ok": False, "source": "civic",
                         "errors": data["errors"]}, max_items=5)
    return _compact({"ok": True, "source": "civic", "data": data.get("data")},
                    max_items=int(args.get("max_items", 10) or 10))


# ------------------------------------------------------------ efo / OLS4 (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/efo-ontology-skill/SKILL.md
# API:    https://www.ebi.ac.uk/ols4/api

async def efo_ontology(args: dict[str, Any]) -> str:
    base = "https://www.ebi.ac.uk/ols4/api"
    action = args.get("action", "search")
    ontology = args.get("ontology", "efo")
    if action == "search":
        q = args.get("query", "")
        if not q:
            return _err("efo", "action=search needs 'query'")
        data = await request_json(
            f"{base}/search",
            params={"q": q, "ontology": ontology, "rows": int(args.get("rows", 10))},
        )
        if data is None:
            return _err("efo", "no response")
        docs = data.get("response", {}).get("docs", [])
        return _compact({"ok": True, "source": "efo", "action": "search",
                         "n": len(docs), "docs": docs},
                        max_items=int(args.get("max_items", 10) or 10))
    if action == "term":
        iri = args.get("iri", "")
        if not iri:
            return _err("efo", "action=term needs 'iri'")
        # OLS wants double-url-encoded IRI
        enc = quote(quote(iri, safe=""), safe="")
        data = await request_json(f"{base}/ontologies/{ontology}/terms/{enc}")
        if data is None:
            return _err("efo", f"not found: {iri}")
        return _compact({"ok": True, "source": "efo", "action": "term",
                         "term": data}, max_items=20)
    return _err("efo", f"unknown action: {action}")


# ------------------------------------------------------------ gtex-eqtl (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/gtex-eqtl-skill/SKILL.md
# API:    https://gtexportal.org/api/v2

async def gtex_eqtl(args: dict[str, Any]) -> str:
    """Significant single-tissue eQTLs for a gencodeId or variant.

    Upstream skill resolves rsID → chrN_pos_ref_alt_b38 via Ensembl first;
    we only expose the GTEx call layer directly. Callers can pre-resolve
    with gget/mygene if they have an rsID.
    """
    base = "https://gtexportal.org/api/v2"
    action = args.get("action", "variant")
    params: dict[str, Any] = {}
    if action == "variant":
        vid = args.get("variant_id", "")
        if not vid:
            return _err("gtex", "action=variant needs 'variant_id' (chr_pos_ref_alt_b38)")
        params["variantId"] = vid
        if args.get("tissue"):
            params["tissueSiteDetailId"] = args["tissue"]
        path = "/association/singleTissueEqtl"
    elif action == "gene":
        gid = args.get("gencode_id", "")
        if not gid:
            return _err("gtex", "action=gene needs 'gencode_id' (e.g. ENSG00000135744.8)")
        params["gencodeId"] = gid
        if args.get("tissue"):
            params["tissueSiteDetailId"] = args["tissue"]
        path = "/association/singleTissueEqtl"
    else:
        return _err("gtex", f"unknown action: {action}")
    params["itemsPerPage"] = int(args.get("max_items", 25) or 25)
    data = await request_json(base + path, params=params)
    if data is None:
        return _err("gtex", "no response")
    return _compact({"ok": True, "source": "gtex", "action": action,
                     "data": data.get("data", data)},
                    max_items=int(args.get("max_items", 10) or 10))


# ------------------------------------------------------- opentargets (GraphQL)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/opentargets-skill/SKILL.md
# API:    https://api.platform.opentargets.org/api/v4/graphql

async def opentargets_graphql(args: dict[str, Any]) -> str:
    query = args.get("query", "").strip()
    if not query:
        return _err("opentargets", "missing 'query'")
    body: dict[str, Any] = {"query": query}
    if "variables" in args:
        body["variables"] = args["variables"]
    data = await request_json(
        "https://api.platform.opentargets.org/api/v4/graphql",
        method="POST", json_body=body,
    )
    if data is None:
        return _err("opentargets", "no response")
    if "errors" in data:
        return _compact({"ok": False, "source": "opentargets",
                         "errors": data["errors"]}, max_items=5)
    return _compact({"ok": True, "source": "opentargets", "data": data.get("data")},
                    max_items=int(args.get("max_items", 5) or 5))


# ---------------------------------------------------------- pharmgkb (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/pharmgkb-skill/SKILL.md
# API:    https://api.pharmgkb.org/v1/data

async def pharmgkb_lookup(args: dict[str, Any]) -> str:
    base = "https://api.pharmgkb.org/v1/data"
    action = args.get("action", "gene")
    if action in ("gene", "variant", "chemical", "disease"):
        pid = args.get("id", "")
        if not pid:
            return _err("pharmgkb", f"action={action} needs 'id' (e.g. PA134865140)")
        data = await request_json(f"{base}/{action}/{pid}")
        if data is None:
            return _err("pharmgkb", f"not found: {action}/{pid}")
        return _compact({"ok": True, "source": "pharmgkb", "action": action,
                         "record": data.get("data", data)}, max_items=20)
    if action == "clinicalAnnotation":
        params = {"limit": int(args.get("limit", 10))}
        if args.get("chemical_id"):
            params["relatedChemicals.accessionId"] = args["chemical_id"]
        if args.get("gene_id"):
            params["relatedGenes.accessionId"] = args["gene_id"]
        data = await request_json(f"{base}/clinicalAnnotation", params=params)
        if data is None:
            return _err("pharmgkb", "no response")
        return _compact({"ok": True, "source": "pharmgkb",
                         "action": "clinicalAnnotation",
                         "records": data.get("data", data)},
                        max_items=int(args.get("max_items", 10) or 10))
    return _err("pharmgkb", f"unknown action: {action}")


# ------------------------------------------------------------ reactome (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/reactome-skill/SKILL.md
# API:    https://reactome.org/ContentService

async def reactome_query(args: dict[str, Any]) -> str:
    base = "https://reactome.org/ContentService"
    action = args.get("action", "pathways_for_entity")
    if action == "event":
        eid = args.get("event_id", "")
        if not eid:
            return _err("reactome", "action=event needs 'event_id'")
        data = await request_json(f"{base}/data/query/{eid}")
        if data is None:
            return _err("reactome", f"not found: {eid}")
        return _compact({"ok": True, "source": "reactome", "action": "event",
                         "event": data}, max_items=20)
    if action == "pathways_for_entity":
        ident = args.get("identifier", "")
        if not ident:
            return _err("reactome", "needs 'identifier' (e.g. UniProt ID)")
        params = {"species": args.get("species", "Homo sapiens")}
        data = await request_json(
            f"{base}/data/pathways/low/entity/{ident}", params=params,
        )
        if data is None:
            return _err("reactome", f"not found: {ident}")
        return _compact({"ok": True, "source": "reactome",
                         "action": "pathways_for_entity",
                         "pathways": data},
                        max_items=int(args.get("max_items", 10) or 10))
    if action == "participants":
        eid = args.get("event_id", "")
        if not eid:
            return _err("reactome", "action=participants needs 'event_id'")
        data = await request_json(f"{base}/data/participants/{eid}")
        if data is None:
            return _err("reactome", f"not found: {eid}")
        return _compact({"ok": True, "source": "reactome",
                         "action": "participants",
                         "participants": data},
                        max_items=int(args.get("max_items", 10) or 10))
    return _err("reactome", f"unknown action: {action}")


# ------------------------------------------------------------ uniprot (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/uniprot-skill/SKILL.md
# API:    https://rest.uniprot.org

async def uniprot_lookup(args: dict[str, Any]) -> str:
    base = "https://rest.uniprot.org"
    action = args.get("action", "search")
    if action == "search":
        q = args.get("query", "")
        if not q:
            return _err("uniprot", "action=search needs 'query'")
        fields = args.get("fields", "accession,id,gene_names,organism_name,protein_name")
        params = {"query": q, "fields": fields,
                  "size": int(args.get("size", 10)),
                  "format": "json"}
        data = await request_json(f"{base}/uniprotkb/search", params=params)
        if data is None:
            return _err("uniprot", "no response")
        results = data.get("results", [])
        return _compact({"ok": True, "source": "uniprot", "action": "search",
                         "n": len(results), "results": results},
                        max_items=int(args.get("max_items", 10) or 10))
    if action == "accession":
        acc = args.get("accession", "")
        if not acc:
            return _err("uniprot", "action=accession needs 'accession' (e.g. P04637)")
        data = await request_json(f"{base}/uniprotkb/{acc}", params={"format": "json"})
        if data is None:
            return _err("uniprot", f"not found: {acc}")
        return _compact({"ok": True, "source": "uniprot", "action": "accession",
                         "entry": data}, max_items=20)
    return _err("uniprot", f"unknown action: {action}")


# ======================================================================
# Batch 1 — high-value skills (2026-04-18)
# ======================================================================


# -------------------------------------------------------------- biorxiv (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/biorxiv-skill/SKILL.md
# API:    https://api.biorxiv.org

async def biorxiv_request(args: dict[str, Any]) -> str:
    """bioRxiv / medRxiv metadata lookup.

    action='details'  — details/<server>/<start>/<end>/<cursor>/json or
                        details/<server>/<doi>/na/json
    action='pubs'     — pubs/<server>/<start>/<end>/<cursor>
    server defaults 'biorxiv'; supply 'doi' OR ('start' + 'end' + cursor).
    """
    action = args.get("action", "details")
    server = args.get("server", "biorxiv")
    cursor = args.get("cursor", 0)
    doi = args.get("doi", "").strip()
    start = args.get("start", "").strip()
    end = args.get("end", "").strip()
    if action not in {"details", "pubs"}:
        return _err("biorxiv", f"unknown action: {action}")
    if doi:
        path = f"{action}/{server}/{doi}/na/json"
    elif start and end:
        path = f"{action}/{server}/{start}/{end}/{cursor}/json"
    else:
        return _err("biorxiv", "need 'doi' or ('start' and 'end')")
    data = await request_json(f"https://api.biorxiv.org/{path}")
    if data is None:
        return _err("biorxiv", "no response")
    collection = data.get("collection", [])
    return _compact({"ok": True, "source": "biorxiv", "action": action,
                     "messages": data.get("messages"),
                     "n": len(collection), "collection": collection},
                    max_items=int(args.get("max_items", 10) or 10))


# ------------------------------------------------------------ cbioportal (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/cbioportal-skill/SKILL.md
# API:    https://www.cbioportal.org/api

async def cbioportal_request(args: dict[str, Any]) -> str:
    """cBioPortal cancer genomics REST.

    path: e.g. 'studies', 'studies/<studyId>/molecular-profiles',
          'molecular-profiles/<profileId>/mutations/fetch'
    method: 'GET' (default) or 'POST' (fetch endpoints need json_body).
    """
    path = args.get("path", "").lstrip("/").strip()
    if not path:
        return _err("cbioportal", "missing 'path'")
    method = args.get("method", "GET").upper()
    url = f"https://www.cbioportal.org/api/{path}"
    kwargs: dict[str, Any] = {
        "method": method,
        "params": args.get("params"),
        "headers": {"Accept": "application/json"},
    }
    if method == "POST":
        kwargs["json_body"] = args.get("json_body") or {}
    data = await request_json(url, **kwargs)
    if data is None:
        return _err("cbioportal", f"no response ({method} {path})")
    return _compact({"ok": True, "source": "cbioportal", "path": path,
                     "method": method, "data": data},
                    max_items=int(args.get("max_items", 10) or 10))


# ------------------------------------------------------------ gnomad (GraphQL)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/gnomad-graphql-skill/SKILL.md
# API:    https://gnomad.broadinstitute.org/api

async def gnomad_graphql(args: dict[str, Any]) -> str:
    query = args.get("query", "").strip()
    if not query:
        return _err("gnomad", "missing 'query'")
    body: dict[str, Any] = {"query": query}
    if "variables" in args:
        body["variables"] = args["variables"]
    data = await request_json(
        "https://gnomad.broadinstitute.org/api",
        method="POST", json_body=body,
    )
    if data is None:
        return _err("gnomad", "no response")
    if "errors" in data:
        return _compact({"ok": False, "source": "gnomad",
                         "errors": data["errors"]}, max_items=5)
    return _compact({"ok": True, "source": "gnomad",
                     "data": data.get("data")},
                    max_items=int(args.get("max_items", 5) or 5))


# ----------------------------------------------------------- ncbi-blast (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/ncbi-blast-skill/SKILL.md
# API:    https://blast.ncbi.nlm.nih.gov/Blast.cgi
#
# NCBI's BLAST Common URL API is async (submit -> RID -> poll -> fetch).
# We expose submit/status/fetch operations; callers drive the polling
# cadence (the upstream rate-limit is >=60s between polls per RID).

async def ncbi_blast(args: dict[str, Any]) -> str:
    import httpx as _httpx
    action = args.get("action", "status")
    url = "https://blast.ncbi.nlm.nih.gov/Blast.cgi"
    hdrs = {"User-Agent": "BioMedArena/openai_ported (+httpx)"}
    try:
        if action == "submit":
            program = args.get("program", "blastp")
            database = args.get("database", "swissprot")
            query_seq = args.get("query", "").strip()
            if not query_seq:
                return _err("ncbi_blast", "action=submit needs 'query' (FASTA)")
            params = {
                "CMD": "Put",
                "PROGRAM": program,
                "DATABASE": database,
                "QUERY": query_seq,
                "HITLIST_SIZE": int(args.get("hitlist_size", 50)),
            }
            async with _httpx.AsyncClient(timeout=60, follow_redirects=True) as c:
                r = await c.post(url, data=params, headers=hdrs)
                r.raise_for_status()
                text = r.text
            # RID and RTOE live in a comment block:
            rid = None
            rtoe = None
            for line in text.splitlines():
                s = line.strip()
                if s.startswith("RID ="):
                    rid = s.split("=", 1)[1].strip()
                elif s.startswith("RTOE ="):
                    rtoe = s.split("=", 1)[1].strip()
            if not rid:
                return _err("ncbi_blast", "submit: no RID returned")
            return _compact({"ok": True, "source": "ncbi_blast",
                             "action": "submit", "rid": rid,
                             "rtoe_seconds": rtoe,
                             "status": "SUBMITTED"}, max_items=5)
        if action == "status":
            rid = args.get("rid", "").strip()
            if not rid:
                return _err("ncbi_blast", "action=status needs 'rid'")
            params = {"CMD": "Get", "RID": rid, "FORMAT_OBJECT": "SearchInfo"}
            async with _httpx.AsyncClient(timeout=60, follow_redirects=True) as c:
                r = await c.get(url, params=params, headers=hdrs)
                r.raise_for_status()
                text = r.text
            status = "UNKNOWN"
            has_hits = None
            for line in text.splitlines():
                s = line.strip()
                if s.startswith("Status="):
                    status = s.split("=", 1)[1].strip()
                elif s.startswith("ThereAreHits="):
                    has_hits = s.split("=", 1)[1].strip().lower() == "yes"
            return _compact({"ok": True, "source": "ncbi_blast",
                             "action": "status", "rid": rid,
                             "status": status, "has_hits": has_hits},
                            max_items=5)
        if action == "fetch":
            rid = args.get("rid", "").strip()
            if not rid:
                return _err("ncbi_blast", "action=fetch needs 'rid'")
            fmt = args.get("result_format", "json2")
            params = {"CMD": "Get", "RID": rid,
                      "FORMAT_TYPE": "JSON2_S" if fmt == "json2" else "Text",
                      "DESCRIPTIONS": int(args.get("descriptions", 5)),
                      "ALIGNMENTS": int(args.get("alignments", 5))}
            async with _httpx.AsyncClient(timeout=120, follow_redirects=True) as c:
                r = await c.get(url, params=params, headers=hdrs)
                r.raise_for_status()
                text = r.text
            return _compact({"ok": True, "source": "ncbi_blast",
                             "action": "fetch", "rid": rid,
                             "format": fmt,
                             "text_head": text[:1200]}, max_items=5)
        return _err("ncbi_blast", f"unknown action: {action}")
    except _httpx.HTTPStatusError as exc:
        return _err("ncbi_blast", f"http {exc.response.status_code}")
    except Exception as exc:
        return _err("ncbi_blast", f"{type(exc).__name__}: {str(exc)[:200]}")


# -------------------------------------------------------- ncbi-datasets (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/ncbi-datasets-skill/SKILL.md
# API:    https://api.ncbi.nlm.nih.gov/datasets/v2

async def ncbi_datasets(args: dict[str, Any]) -> str:
    path = args.get("path", "").lstrip("/").strip()
    if not path:
        return _err("ncbi_datasets", "missing 'path'")
    data = await request_json(
        f"https://api.ncbi.nlm.nih.gov/datasets/v2/{path}",
        params=args.get("params"),
    )
    if data is None:
        return _err("ncbi_datasets", f"no response ({path})")
    return _compact({"ok": True, "source": "ncbi_datasets", "path": path,
                     "data": data},
                    max_items=int(args.get("max_items", 10) or 10))


# ------------------------------------------------------------ string (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/string-skill/SKILL.md
# API:    https://string-db.org/api/json/<method>
#
# STRING uses form-encoded POST bodies, which our shared _http helper
# doesn't cover (json-only). We drop down to httpx directly here.

async def string_request(args: dict[str, Any]) -> str:
    import httpx as _httpx
    path = args.get("path", "network").lstrip("/").strip()
    if path not in {"network", "interaction_partners", "enrichment",
                    "get_string_ids", "homology", "ppi_enrichment"}:
        return _err("string", f"unsupported path: {path}")
    form = dict(args.get("form") or {})
    form.setdefault("caller_identity", "BioMedArena-olsp")
    # Allow top-level convenience keys identifiers/identifier/species
    for k in ("identifiers", "identifier", "species", "limit",
              "required_score", "network_type"):
        if args.get(k) is not None:
            form[k] = args[k]
    url = f"https://string-db.org/api/json/{path}"
    try:
        async with _httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
            r = await c.post(url, data=form,
                              headers={"Accept": "application/json",
                                      "User-Agent": "BioMedArena/olsp"})
            r.raise_for_status()
            data = r.json()
    except _httpx.HTTPStatusError as exc:
        return _err("string", f"http {exc.response.status_code}: "
                                f"{exc.response.text[:200]}")
    except Exception as exc:
        return _err("string", f"{type(exc).__name__}: {str(exc)[:200]}")
    rows = data if isinstance(data, list) else [data]
    return _compact({"ok": True, "source": "string", "path": path,
                     "n": len(rows), "rows": rows},
                    max_items=int(args.get("max_items", 10) or 10))


# ---------------------------------------------------------- gwas-catalog (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/gwas-catalog-skill/SKILL.md
# API:    https://www.ebi.ac.uk/gwas/rest/api/v2

async def gwas_catalog_request(args: dict[str, Any]) -> str:
    path = args.get("path", "").lstrip("/").strip()
    if not path:
        return _err("gwas_catalog", "missing 'path'")
    data = await request_json(
        f"https://www.ebi.ac.uk/gwas/rest/api/v2/{path}",
        params=args.get("params"),
    )
    if data is None:
        return _err("gwas_catalog", f"no response ({path})")
    # HATEOAS: extract _embedded.<resource> when available
    embedded = data.get("_embedded") if isinstance(data, dict) else None
    if isinstance(embedded, dict):
        # Pick the first list inside _embedded
        for k, v in embedded.items():
            if isinstance(v, list):
                return _compact({"ok": True, "source": "gwas_catalog",
                                 "path": path, "resource": k,
                                 "n": len(v), "records": v,
                                 "page": data.get("page")},
                                max_items=int(args.get("max_items", 10) or 10))
    return _compact({"ok": True, "source": "gwas_catalog", "path": path,
                     "data": data},
                    max_items=int(args.get("max_items", 10) or 10))


# ---------------------------------------------------- human-protein-atlas (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/human-protein-atlas-skill/SKILL.md
# API:    https://www.proteinatlas.org

async def human_protein_atlas(args: dict[str, Any]) -> str:
    import httpx as _httpx
    action = args.get("action", "gene")
    base = "https://www.proteinatlas.org"
    if action == "gene":
        ensg = args.get("ensg", "").strip()
        if not ensg:
            return _err("hpa", "action=gene needs 'ensg' (e.g. ENSG00000141510)")
        data = await request_json(f"{base}/{ensg}.json")
        if data is None:
            return _err("hpa", f"not found: {ensg}")
        return _compact({"ok": True, "source": "hpa", "action": "gene",
                         "ensg": ensg, "entry": data}, max_items=20)
    if action == "search_download":
        q = args.get("query", "").strip()
        if not q:
            return _err("hpa", "action=search_download needs 'query'")
        params = {
            "search": q,
            "format": args.get("format", "json"),
            "columns": args.get("columns", "g,gs,chr,up,rnatsm,scm"),
            "compress": "no",
        }
        data = await request_json(f"{base}/api/search_download.php", params=params)
        if data is None:
            return _err("hpa", "search returned no JSON")
        rows = data if isinstance(data, list) else [data]
        return _compact({"ok": True, "source": "hpa",
                         "action": "search_download",
                         "n": len(rows), "records": rows},
                        max_items=int(args.get("max_items", 10) or 10))
    if action == "page_text":
        sub = args.get("subpath", "").strip().lstrip("/")
        if not sub:
            return _err("hpa", "action=page_text needs 'subpath' "
                                "(e.g. search/tissue/TP53)")
        try:
            async with _httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
                r = await c.get(f"{base}/{sub}",
                                 headers={"User-Agent": "BioMedArena/olsp"})
                r.raise_for_status()
                text = r.text
        except Exception as exc:
            return _err("hpa", f"{type(exc).__name__}: {str(exc)[:200]}")
        return _compact({"ok": True, "source": "hpa", "action": "page_text",
                         "subpath": sub, "text_head": text[:1500]},
                        max_items=5)
    return _err("hpa", f"unknown action: {action}")


# ======================================================================
# Batch 2 — medium-value skills (2026-04-18)
# ======================================================================


# --------------------------------------------------------------- quickgo (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/quickgo-skill/SKILL.md
# API:    https://www.ebi.ac.uk/QuickGO/services

async def quickgo_request(args: dict[str, Any]) -> str:
    path = args.get("path", "").lstrip("/").strip()
    if not path:
        return _err("quickgo", "missing 'path'")
    data = await request_json(
        f"https://www.ebi.ac.uk/QuickGO/services/{path}",
        params=args.get("params"),
    )
    if data is None:
        return _err("quickgo", f"no response ({path})")
    results = data.get("results") if isinstance(data, dict) else None
    body: dict[str, Any] = {"ok": True, "source": "quickgo", "path": path}
    if isinstance(results, list):
        body["n"] = len(results)
        body["results"] = results
    else:
        body["data"] = data
    return _compact(body, max_items=int(args.get("max_items", 10) or 10))


# ------------------------------------------------------------- rnacentral (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/rnacentral-skill/SKILL.md
# API:    https://rnacentral.org/api/v1

async def rnacentral_request(args: dict[str, Any]) -> str:
    path = args.get("path", "").strip()
    if not path:
        return _err("rnacentral", "missing 'path'")
    # Keep trailing slash to avoid redirect
    if not path.endswith("/") and "?" not in path:
        path = path + "/"
    data = await request_json(
        f"https://rnacentral.org/api/v1/{path.lstrip('/')}",
        params=args.get("params"),
    )
    if data is None:
        return _err("rnacentral", f"no response ({path})")
    results = data.get("results") if isinstance(data, dict) else None
    body: dict[str, Any] = {"ok": True, "source": "rnacentral", "path": path}
    if isinstance(results, list):
        body["n"] = len(results)
        body["count"] = data.get("count")
        body["results"] = results
    else:
        body["entry"] = data
    return _compact(body, max_items=int(args.get("max_items", 10) or 10))


# ---------------------------------------------------------------- encode (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/encode-skill/SKILL.md
# API:    https://www.encodeproject.org

async def encode_request(args: dict[str, Any]) -> str:
    path = args.get("path", "").lstrip("/").strip()
    if not path:
        return _err("encode", "missing 'path'")
    params = dict(args.get("params") or {})
    params.setdefault("format", "json")
    data = await request_json(
        f"https://www.encodeproject.org/{path}",
        params=params,
        headers={"Accept": "application/json"},
    )
    if data is None:
        return _err("encode", f"no response ({path})")
    graph = data.get("@graph") if isinstance(data, dict) else None
    body: dict[str, Any] = {"ok": True, "source": "encode", "path": path}
    if isinstance(graph, list):
        body["n"] = len(graph)
        body["total"] = (data.get("total") if isinstance(data, dict) else None)
        body["records"] = graph
    else:
        body["entry"] = data
    return _compact(body, max_items=int(args.get("max_items", 10) or 10))


# ------------------------------------------------------------------- rhea (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/rhea-skill/SKILL.md
# API:    https://www.rhea-db.org/rhea

async def rhea_request(args: dict[str, Any]) -> str:
    q = args.get("query", "").strip()
    if not q:
        return _err("rhea", "missing 'query'")
    data = await request_json(
        "https://www.rhea-db.org/rhea",
        params={"query": q, "format": "json"},
    )
    if data is None:
        return _err("rhea", "no response")
    results = data.get("results") if isinstance(data, dict) else data
    rows = results if isinstance(results, list) else [results]
    return _compact({"ok": True, "source": "rhea", "query": q,
                     "n": len(rows), "results": rows},
                    max_items=int(args.get("max_items", 10) or 10))


# ----------------------------------------------- locus-to-gene-mapper (OT L2G)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/locus-to-gene-mapper-skill/SKILL.md
# Upstream is an orchestration skill combining EFO → GWAS → OT L2G → eQTL →
# burden/coding context. We expose a focused wrapper over the single core
# piece — OpenTargets L2G / colocalisation — that the orchestration uses.
# Callers combining traits / variants should chain the olsp_* primitives.

async def locus_to_gene_mapper(args: dict[str, Any]) -> str:
    """Thin Open Targets L2G helper.

    action='credibleSetsForVariant' (needs 'variant_id' e.g. '1_55516888_G_GA')
    action='credibleSetsForStudyLocus' (needs 'study_locus_id')
    action='l2g_for_gene_disease' (needs 'ensembl_id' + 'efo_id')

    For full multi-signal orchestration, chain with olsp_opentargets_graphql,
    olsp_gwas_catalog_request, olsp_gtex_eqtl, and olsp_genebass_gene_burden.
    """
    action = args.get("action", "credibleSetsForVariant")
    if action == "credibleSetsForVariant":
        vid = args.get("variant_id", "").strip()
        if not vid:
            return _err("l2g", "action=credibleSetsForVariant needs 'variant_id'")
        query = (
            "query V($id:String!) {"
            " variant(variantId:$id) {"
            "  credibleSets(page:{index:0,size:10}) {"
            "   studyLocusId study{studyId traitFromSource} pValueExponent"
            "   l2GPredictions{rows{gene{id approvedSymbol} score}}"
            " } } }"
        )
        variables = {"id": vid}
    elif action == "credibleSetsForStudyLocus":
        sl = args.get("study_locus_id", "").strip()
        if not sl:
            return _err("l2g", "action=credibleSetsForStudyLocus needs 'study_locus_id'")
        query = (
            "query SL($id:String!) {"
            " credibleSet(studyLocusId:$id) {"
            "  studyLocusId leadVariant{variantId}"
            "  l2GPredictions{rows{gene{id approvedSymbol} score}}"
            " } }"
        )
        variables = {"id": sl}
    elif action == "l2g_for_gene_disease":
        g = args.get("ensembl_id", "").strip()
        e = args.get("efo_id", "").strip()
        if not g or not e:
            return _err("l2g", "action=l2g_for_gene_disease needs 'ensembl_id' + 'efo_id'")
        query = (
            "query GD($g:String!,$e:String!) {"
            " target(ensemblId:$g) { id approvedSymbol }"
            " disease(efoId:$e) { id name }"
            " }"
        )
        variables = {"g": g, "e": e}
    else:
        return _err("l2g", f"unknown action: {action}")
    data = await request_json(
        "https://api.platform.opentargets.org/api/v4/graphql",
        method="POST", json_body={"query": query, "variables": variables},
    )
    if data is None:
        return _err("l2g", "no response")
    if "errors" in data:
        return _compact({"ok": False, "source": "l2g",
                         "errors": data["errors"]}, max_items=5)
    return _compact({"ok": True, "source": "l2g", "action": action,
                     "data": data.get("data")},
                    max_items=int(args.get("max_items", 10) or 10))


# ----------------------------------------------------------- finngen-phewas (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/finngen-phewas-skill/SKILL.md
# API:    https://r12.finngen.fi/api/variant/<chr:pos-ref-alt> (GRCh38)

async def finngen_phewas(args: dict[str, Any]) -> str:
    from urllib.parse import quote as _q
    v = (args.get("variant") or args.get("grch38") or "").strip()
    if not v:
        return _err("finngen", "needs 'variant' or 'grch38' (chr:pos-ref-alt)")
    # Normalise 'chr:pos:ref:alt' → 'chr:pos-ref-alt' if needed
    if v.count(":") == 3:
        parts = v.split(":")
        v = f"{parts[0]}:{parts[1]}-{parts[2]}-{parts[3]}"
    encoded = _q(v, safe=":-")
    data = await request_json(
        f"https://r12.finngen.fi/api/variant/{encoded}",
    )
    if data is None:
        return _err("finngen", f"not found: {v}")
    assocs = data.get("phenos") if isinstance(data, dict) else None
    body: dict[str, Any] = {"ok": True, "source": "finngen",
                            "query_variant": v,
                            "variant_url": f"https://r12.finngen.fi/variant/{v}"}
    if isinstance(assocs, list):
        mx = int(args.get("max_results", 10) or 10)
        body["association_count_total"] = len(assocs)
        body["truncated"] = len(assocs) > mx
        body["associations"] = assocs[:mx]
    else:
        body["data"] = data
    return _compact(body, max_items=int(args.get("max_results", 10) or 10))


# ----------------------------------------------------- biobankjapan-phewas (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/biobankjapan-phewas-skill/SKILL.md
# API:    https://pheweb.jp/api/variant/<chr:pos-ref-alt> (GRCh37)

async def biobankjapan_phewas(args: dict[str, Any]) -> str:
    from urllib.parse import quote as _q
    v = (args.get("variant") or args.get("grch37") or "").strip()
    if not v:
        return _err("bbj", "needs 'variant' or 'grch37' (chr:pos-ref-alt)")
    if v.count(":") == 3:
        parts = v.split(":")
        v = f"{parts[0]}:{parts[1]}-{parts[2]}-{parts[3]}"
    encoded = _q(v, safe=":-")
    data = await request_json(
        f"https://pheweb.jp/api/variant/{encoded}",
    )
    if data is None:
        return _err("bbj", f"not found: {v}")
    assocs = data.get("phenos") if isinstance(data, dict) else None
    body: dict[str, Any] = {"ok": True, "source": "bbj",
                            "query_variant": v,
                            "variant_url": f"https://pheweb.jp/variant/{v}"}
    if isinstance(assocs, list):
        mx = int(args.get("max_results", 10) or 10)
        body["association_count_total"] = len(assocs)
        body["truncated"] = len(assocs) > mx
        body["associations"] = assocs[:mx]
    else:
        body["data"] = data
    return _compact(body, max_items=int(args.get("max_results", 10) or 10))


# -------------------------------------------------- biostudies-arrayexpress (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/biostudies-arrayexpress-skill/SKILL.md
# API:    https://www.ebi.ac.uk/biostudies/api/v1

async def biostudies_request(args: dict[str, Any]) -> str:
    path = args.get("path", "").lstrip("/").strip()
    if not path:
        return _err("biostudies", "missing 'path'")
    data = await request_json(
        f"https://www.ebi.ac.uk/biostudies/api/v1/{path}",
        params=args.get("params"),
    )
    if data is None:
        return _err("biostudies", f"no response ({path})")
    hits = data.get("hits") if isinstance(data, dict) else None
    body: dict[str, Any] = {"ok": True, "source": "biostudies", "path": path}
    if isinstance(hits, list):
        body["n"] = len(hits)
        body["totalHits"] = data.get("totalHits")
        body["hits"] = hits
    else:
        body["data"] = data
    return _compact(body, max_items=int(args.get("max_items", 10) or 10))


# -------------------------------------------------- genebass-gene-burden (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/genebass-gene-burden-skill/SKILL.md
# API:    https://main.genebass.org/api/phewas/<gene>?burdenSet=<set>

async def genebass_gene_burden(args: dict[str, Any]) -> str:
    from urllib.parse import quote as _q
    gene = (args.get("ensembl_gene_id") or args.get("gene") or "").strip()
    if not gene:
        return _err("genebass", "needs 'ensembl_gene_id' (e.g. ENSG00000173531)")
    burden = args.get("burden_set", "pLoF")
    data = await request_json(
        f"https://main.genebass.org/api/phewas/{_q(gene)}",
        params={"burdenSet": burden},
    )
    if data is None:
        return _err("genebass", f"not found or upstream error: {gene}/{burden}")
    assocs = data if isinstance(data, list) else (
        data.get("results") or data.get("associations") or [])
    mx = int(args.get("max_results", 25) or 25)
    return _compact({"ok": True, "source": "genebass",
                     "ensembl_gene_id": gene, "burden_set": burden,
                     "association_count_total": len(assocs),
                     "truncated": len(assocs) > mx,
                     "associations": assocs[:mx]},
                    max_items=mx)


# ----------------------------------------------------------- epigraphdb (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/epigraphdb-skill/SKILL.md
# API:    https://api.epigraphdb.org

async def epigraphdb_request(args: dict[str, Any]) -> str:
    path = args.get("path", "").lstrip("/").strip()
    if not path:
        return _err("epigraphdb", "missing 'path'")
    data = await request_json(
        f"https://api.epigraphdb.org/{path}",
        params=args.get("params"),
    )
    if data is None:
        return _err("epigraphdb", f"no response ({path})")
    return _compact({"ok": True, "source": "epigraphdb", "path": path,
                     "data": data},
                    max_items=int(args.get("max_items", 10) or 10))


# ------------------------------------------------------- eqtl-catalogue (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/eqtl-catalogue-skill/SKILL.md
# API:    https://www.ebi.ac.uk/eqtl/api
#
# Upstream flags this API as fragile (400/500 on omitted optional params).
# We pass through user params verbatim — caller supplies study/tissue/etc.

async def eqtl_catalogue_request(args: dict[str, Any]) -> str:
    path = args.get("path", "").lstrip("/").strip()
    if not path:
        return _err("eqtl_catalogue", "missing 'path'")
    data = await request_json(
        f"https://www.ebi.ac.uk/eqtl/api/{path}",
        params=args.get("params"),
    )
    if data is None:
        return _err("eqtl_catalogue",
                      "no response (upstream API is fragile; supply required filters)")
    records = data if isinstance(data, list) else (
        data.get("_embedded", {}) if isinstance(data, dict) else {})
    return _compact({"ok": True, "source": "eqtl_catalogue", "path": path,
                     "data": data},
                    max_items=int(args.get("max_items", 10) or 10))


# ======================================================================
# Batch 3 — lower-value skills (2026-04-18)
# ======================================================================


# ---------------------------------------------------- ukb-topmed-phewas (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/ukb-topmed-phewas-skill/SKILL.md
# API:    https://pheweb.org/UKB-TOPMed/api/variant/<variant>

async def ukb_topmed_phewas(args: dict[str, Any]) -> str:
    from urllib.parse import quote as _q
    v = (args.get("variant") or args.get("grch38") or "").strip()
    if not v:
        return _err("ukb_topmed", "needs 'variant' or 'grch38' (chr:pos-ref-alt)")
    if v.count(":") == 3:
        parts = v.split(":")
        v = f"{parts[0]}:{parts[1]}-{parts[2]}-{parts[3]}"
    encoded = _q(v, safe=":-")
    data = await request_json(
        f"https://pheweb.org/UKB-TOPMed/api/variant/{encoded}",
    )
    if data is None:
        return _err("ukb_topmed", f"not found: {v}")
    assocs = data.get("phenos") if isinstance(data, dict) else None
    body: dict[str, Any] = {"ok": True, "source": "ukb_topmed",
                            "query_variant": v,
                            "variant_url": f"https://pheweb.org/UKB-TOPMed/variant/{v}"}
    if isinstance(assocs, list):
        mx = int(args.get("max_results", 10) or 10)
        body["association_count_total"] = len(assocs)
        body["truncated"] = len(assocs) > mx
        body["associations"] = assocs[:mx]
    else:
        body["data"] = data
    return _compact(body, max_items=int(args.get("max_results", 10) or 10))


# -------------------------------------------------------- tpmi-phewas (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/tpmi-phewas-skill/SKILL.md
# API:    https://pheweb.ibms.sinica.edu.tw/api/variant/<variant>

async def tpmi_phewas(args: dict[str, Any]) -> str:
    from urllib.parse import quote as _q
    v = (args.get("variant") or args.get("grch38") or "").strip()
    if not v:
        return _err("tpmi", "needs 'variant' or 'grch38' (chr:pos-ref-alt)")
    if v.count(":") == 3:
        parts = v.split(":")
        v = f"{parts[0]}:{parts[1]}-{parts[2]}-{parts[3]}"
    encoded = _q(v, safe=":-")
    data = await request_json(
        f"https://pheweb.ibms.sinica.edu.tw/api/variant/{encoded}",
    )
    if data is None:
        return _err("tpmi", f"not found: {v}")
    assocs = data.get("phenos") if isinstance(data, dict) else None
    body: dict[str, Any] = {"ok": True, "source": "tpmi",
                            "query_variant": v,
                            "variant_url": f"https://pheweb.ibms.sinica.edu.tw/variant/{v}"}
    if isinstance(assocs, list):
        mx = int(args.get("max_results", 10) or 10)
        body["association_count_total"] = len(assocs)
        body["truncated"] = len(assocs) > mx
        body["associations"] = assocs[:mx]
    else:
        body["data"] = data
    return _compact(body, max_items=int(args.get("max_results", 10) or 10))


# ------------------------------------------------------------------ hmdb (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/hmdb-skill/SKILL.md
# API:    https://hmdb.ca/unearth/q

async def hmdb_request(args: dict[str, Any]) -> str:
    q = args.get("query", "").strip()
    if not q:
        return _err("hmdb", "missing 'query'")
    category = args.get("category", "metabolites")
    params = {"query": q, "category": category, "format": "json",
              "per_page": int(args.get("per_page", 10))}
    data = await request_json("https://hmdb.ca/unearth/q", params=params)
    if data is None:
        return _err("hmdb", "no response")
    records = data.get(category) if isinstance(data, dict) else None
    body: dict[str, Any] = {"ok": True, "source": "hmdb",
                            "query": q, "category": category}
    if isinstance(records, list):
        body["n"] = len(records)
        body["records"] = records
    else:
        body["data"] = data
    return _compact(body, max_items=int(args.get("max_items", 10) or 10))


# -------------------------------------------------------- metabolights (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/metabolights-skill/SKILL.md
# API:    https://www.ebi.ac.uk/metabolights/ws

async def metabolights_request(args: dict[str, Any]) -> str:
    path = args.get("path", "").lstrip("/").strip()
    if not path:
        return _err("metabolights", "missing 'path'")
    data = await request_json(
        f"https://www.ebi.ac.uk/metabolights/ws/{path}",
        params=args.get("params"),
    )
    if data is None:
        return _err("metabolights", f"no response ({path})")
    content = data.get("content") if isinstance(data, dict) else None
    body: dict[str, Any] = {"ok": True, "source": "metabolights", "path": path}
    if isinstance(content, list):
        body["n"] = len(content)
        body["content"] = content
    else:
        body["data"] = data
    return _compact(body, max_items=int(args.get("max_items", 10) or 10))


# ---------------------------------------------------------------- pride (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/pride-skill/SKILL.md
# API:    https://www.ebi.ac.uk/pride/ws/archive/v2

async def pride_request(args: dict[str, Any]) -> str:
    path = args.get("path", "").lstrip("/").strip()
    if not path:
        return _err("pride", "missing 'path'")
    data = await request_json(
        f"https://www.ebi.ac.uk/pride/ws/archive/v2/{path}",
        params=args.get("params"),
    )
    if data is None:
        return _err("pride", f"no response ({path})")
    embedded = data.get("_embedded") if isinstance(data, dict) else None
    body: dict[str, Any] = {"ok": True, "source": "pride", "path": path}
    if isinstance(embedded, dict):
        for k, v in embedded.items():
            if isinstance(v, list):
                body["resource"] = k
                body["n"] = len(v)
                body["records"] = v
                return _compact(body, max_items=int(args.get("max_items", 10) or 10))
    body["data"] = data
    return _compact(body, max_items=int(args.get("max_items", 10) or 10))


# -------------------------------------------------- proteomexchange PROXI (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/proteomexchange-skill/SKILL.md
# API:    https://proteomecentral.proteomexchange.org/api/proxi/v0.1

async def proteomexchange_request(args: dict[str, Any]) -> str:
    path = args.get("path", "").lstrip("/").strip()
    if not path:
        return _err("proteomexchange", "missing 'path'")
    data = await request_json(
        f"https://proteomecentral.proteomexchange.org/api/proxi/v0.1/{path}",
        params=args.get("params"),
    )
    if data is None:
        return _err("proteomexchange", f"no response ({path})")
    rows = data if isinstance(data, list) else None
    body: dict[str, Any] = {"ok": True, "source": "proteomexchange",
                            "path": path}
    if isinstance(rows, list):
        body["n"] = len(rows)
        body["records"] = rows
    else:
        body["data"] = data
    return _compact(body, max_items=int(args.get("max_items", 10) or 10))


# ======================================================================
# Batch 4 — minimal-value skills (2026-04-18)
# ======================================================================


# ---------------------------------------------------------------- mgnify (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/mgnify-skill/SKILL.md
# API:    https://www.ebi.ac.uk/metagenomics/api/v1

async def mgnify_request(args: dict[str, Any]) -> str:
    path = args.get("path", "").lstrip("/").strip()
    if not path:
        return _err("mgnify", "missing 'path'")
    data = await request_json(
        f"https://www.ebi.ac.uk/metagenomics/api/v1/{path}",
        params=args.get("params"),
    )
    if data is None:
        return _err("mgnify", f"no response ({path})")
    # JSON:API — collections in data.data, record in data.data
    rows = data.get("data") if isinstance(data, dict) else None
    body: dict[str, Any] = {"ok": True, "source": "mgnify", "path": path}
    if isinstance(rows, list):
        body["n"] = len(rows)
        body["records"] = rows
    else:
        body["data"] = data
    return _compact(body, max_items=int(args.get("max_items", 10) or 10))


# ------------------------------------------------------------------ eva (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/eva-skill/SKILL.md
# API:    https://www.ebi.ac.uk/eva/webservices/rest/v1

async def eva_request(args: dict[str, Any]) -> str:
    path = args.get("path", "").lstrip("/").strip()
    if not path:
        return _err("eva", "missing 'path'")
    data = await request_json(
        f"https://www.ebi.ac.uk/eva/webservices/rest/v1/{path}",
        params=args.get("params"),
    )
    if data is None:
        return _err("eva", f"no response ({path})")
    # EVA wraps payloads in response[0].result
    result: Any = data
    if isinstance(data, dict):
        resp = data.get("response")
        if isinstance(resp, list) and resp:
            result = resp[0].get("result", resp[0])
    body: dict[str, Any] = {"ok": True, "source": "eva", "path": path}
    if isinstance(result, list):
        body["n"] = len(result)
        body["records"] = result
    else:
        body["data"] = data
    return _compact(body, max_items=int(args.get("max_items", 10) or 10))


# ------------------------------------------------------------------ ipd (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/ipd-skill/SKILL.md
# API:    https://www.ebi.ac.uk/cgi-bin/ipd/api

async def ipd_request(args: dict[str, Any]) -> str:
    path = args.get("path", "allele").lstrip("/").strip()
    params = dict(args.get("params") or {})
    params.setdefault("project", "HLA")
    params.setdefault("limit", int(args.get("limit", 10)))
    data = await request_json(
        f"https://www.ebi.ac.uk/cgi-bin/ipd/api/{path}",
        params=params,
    )
    if data is None:
        return _err("ipd", f"no response ({path})")
    rows = data.get("data") if isinstance(data, dict) else None
    body: dict[str, Any] = {"ok": True, "source": "ipd", "path": path}
    if isinstance(rows, list):
        body["n"] = len(rows)
        body["records"] = rows
    else:
        body["data"] = data
    return _compact(body, max_items=int(args.get("max_items", 10) or 10))


# ------------------------------------------------ ncbi-clinicaltables (REST)
# Source: https://github.com/openai/plugins/blob/main/plugins/life-science-research/skills/ncbi-clinicaltables-skill/SKILL.md
# API:    https://clinicaltables.nlm.nih.gov/api/ncbi_genes/v3/search

async def ncbi_clinicaltables(args: dict[str, Any]) -> str:
    import httpx as _httpx
    terms = args.get("terms", "").strip()
    if not terms:
        return _err("ncbi_clinicaltables", "missing 'terms'")
    params = dict(args.get("params") or {})
    params["terms"] = terms
    params.setdefault("count", int(args.get("count", 10)))
    url = "https://clinicaltables.nlm.nih.gov/api/ncbi_genes/v3/search"
    # ClinicalTables returns a JSON array (not object), which request_json
    # handles but the shape is nonstandard:
    #   [total_hits, codes, extra_fields, display_fields]
    try:
        async with _httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
            r = await c.get(url, params=params,
                              headers={"Accept": "application/json",
                                      "User-Agent": "BioMedArena/olsp"})
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        return _err("ncbi_clinicaltables",
                      f"{type(exc).__name__}: {str(exc)[:200]}")
    if not isinstance(data, list) or len(data) < 2:
        return _compact({"ok": True, "source": "ncbi_clinicaltables",
                         "terms": terms, "data": data}, max_items=5)
    total, codes = data[0], data[1]
    extra = data[2] if len(data) > 2 else None
    display = data[3] if len(data) > 3 else None
    mx = int(args.get("max_items", 10) or 10)
    return _compact({"ok": True, "source": "ncbi_clinicaltables",
                     "terms": terms, "total_hits": total,
                     "codes": codes[:mx],
                     "extra": (extra[:mx] if isinstance(extra, list) else extra),
                     "display": (display[:mx] if isinstance(display, list)
                                 else display)}, max_items=mx)


# ------------------------------------------------------ dispatch registry
HANDLERS: dict[str, Any] = {
    "olsp_bgee_sparql": bgee_sparql,
    "olsp_bindingdb_ligands": bindingdb_ligands,
    "olsp_chebi_lookup": chebi_lookup,
    "olsp_civic_graphql": civic_graphql,
    "olsp_efo_ontology": efo_ontology,
    "olsp_gtex_eqtl": gtex_eqtl,
    "olsp_opentargets_graphql": opentargets_graphql,
    "olsp_pharmgkb_lookup": pharmgkb_lookup,
    "olsp_reactome_query": reactome_query,
    "olsp_uniprot_lookup": uniprot_lookup,
    # Batch 1
    "olsp_biorxiv_request": biorxiv_request,
    "olsp_cbioportal_request": cbioportal_request,
    "olsp_gnomad_graphql": gnomad_graphql,
    "olsp_ncbi_blast": ncbi_blast,
    "olsp_ncbi_datasets": ncbi_datasets,
    "olsp_string_request": string_request,
    "olsp_gwas_catalog_request": gwas_catalog_request,
    "olsp_human_protein_atlas": human_protein_atlas,
    # Batch 2
    "olsp_quickgo_request": quickgo_request,
    "olsp_rnacentral_request": rnacentral_request,
    "olsp_encode_request": encode_request,
    "olsp_rhea_request": rhea_request,
    "olsp_locus_to_gene_mapper": locus_to_gene_mapper,
    "olsp_finngen_phewas": finngen_phewas,
    "olsp_biobankjapan_phewas": biobankjapan_phewas,
    "olsp_biostudies_request": biostudies_request,
    "olsp_genebass_gene_burden": genebass_gene_burden,
    "olsp_epigraphdb_request": epigraphdb_request,
    "olsp_eqtl_catalogue_request": eqtl_catalogue_request,
    # Batch 3
    "olsp_ukb_topmed_phewas": ukb_topmed_phewas,
    "olsp_tpmi_phewas": tpmi_phewas,
    "olsp_hmdb_request": hmdb_request,
    "olsp_metabolights_request": metabolights_request,
    "olsp_pride_request": pride_request,
    "olsp_proteomexchange_request": proteomexchange_request,
    # Batch 4
    "olsp_mgnify_request": mgnify_request,
    "olsp_eva_request": eva_request,
    "olsp_ipd_request": ipd_request,
    "olsp_ncbi_clinicaltables": ncbi_clinicaltables,
}
