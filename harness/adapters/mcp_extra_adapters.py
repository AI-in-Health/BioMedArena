"""Three additional MCP servers registered as preset specs.

Scanning the MCPmed ecosystem and community registries, three MCP
servers ship as installable PyPI packages and use stdio transport:

    - pubmed-search-mcp   (PubMed / NCBI E-utilities)
    - geo-mcp             (NCBI GEO expression datasets)
    - uniprot-mcp         (UniProt protein knowledge base)

Each is spawned via `uvx --from <pkg> <pkg>` so our main venv stays
clean. Several require per-deployment env vars (e.g. NCBI_EMAIL for
pubmed-search-mcp). The adapters below register specs and probe
availability; live smoke is marked @slow and skipped unless the
corresponding *_ENABLED env var is set, because cold-start of the
less-popular servers can take minutes and some mandate API keys.
"""

from __future__ import annotations

import os
from typing import Any

from harness.adapters.mcp_base import (
    MCPAdapter, MCPServerSpec, register_mcp_server,
)


# ---------------------------------------------------------------------------
# PubMed MCP (pubmed-search-mcp)
# ---------------------------------------------------------------------------


PUBMED_MCP_SPEC = register_mcp_server(MCPServerSpec(
    name="pubmed_mcp",
    command="uvx",
    args=["--python", "3.11", "--from", "pubmed-search-mcp", "pubmed-search-mcp"],
    env={"NCBI_EMAIL": os.environ.get("NCBI_EMAIL", "anonymous@example.com")},
    description=(
        "pubmed-search-mcp — NCBI E-utilities via MCP. Exposes PubMed "
        "search, fetch, MeSH term explore, and related-article lookup. "
        "Complements BioMCP's article tools with deeper MeSH queries."
    ),
    modality="literature",
    tool_prefix="mcp_pubmed_",
))


class PubMedMCPAdapter(MCPAdapter):
    default_spec = PUBMED_MCP_SPEC

    def __init__(self, config: dict | None = None, **kwargs: Any):
        super().__init__(
            spec=PUBMED_MCP_SPEC, config=config,
            startup_timeout=kwargs.pop("startup_timeout", 240),
            per_tool_timeout=kwargs.pop("per_tool_timeout", 60.0),
            **kwargs,
        )

    def capabilities(self) -> list[str]:
        return ["pubmed_advanced_search", "mesh_explore",
                  "pubmed_citation_gen", "mcp_protocol"]


# ---------------------------------------------------------------------------
# GEO MCP (geo-mcp)
# ---------------------------------------------------------------------------


GEO_MCP_SPEC = register_mcp_server(MCPServerSpec(
    name="geo_mcp",
    command="uvx",
    args=["--python", "3.11", "--from", "geo-mcp", "geo-mcp"],
    env={},
    description=(
        "geo-mcp — NCBI GEO gene-expression datasets via MCP. Search "
        "series, platforms, samples; download SOFT/GSE files. "
        "Complements single-cell + genomics workflows."
    ),
    modality="genomics",
    tool_prefix="mcp_geo_",
))


class GEOMCPAdapter(MCPAdapter):
    default_spec = GEO_MCP_SPEC

    def __init__(self, config: dict | None = None, **kwargs: Any):
        super().__init__(
            spec=GEO_MCP_SPEC, config=config,
            startup_timeout=kwargs.pop("startup_timeout", 240),
            per_tool_timeout=kwargs.pop("per_tool_timeout", 60.0),
            **kwargs,
        )

    def capabilities(self) -> list[str]:
        return ["geo_dataset_search", "geo_series_download",
                  "gene_expression_query", "mcp_protocol"]


# ---------------------------------------------------------------------------
# UniProt MCP (uniprot-mcp)
# ---------------------------------------------------------------------------


UNIPROT_MCP_SPEC = register_mcp_server(MCPServerSpec(
    name="uniprot_mcp",
    command="uvx",
    args=["--python", "3.11", "--from", "uniprot-mcp", "uniprot-mcp"],
    env={},
    description=(
        "uniprot-mcp — UniProt protein knowledge base via MCP. Search "
        "by accession / gene / species; fetch sequence, annotations, "
        "cross-references. Complements our ESM / AlphaFold-DB tools."
    ),
    modality="protein",
    tool_prefix="mcp_uniprot_",
))


class UniProtMCPAdapter(MCPAdapter):
    default_spec = UNIPROT_MCP_SPEC

    def __init__(self, config: dict | None = None, **kwargs: Any):
        super().__init__(
            spec=UNIPROT_MCP_SPEC, config=config,
            startup_timeout=kwargs.pop("startup_timeout", 240),
            per_tool_timeout=kwargs.pop("per_tool_timeout", 60.0),
            **kwargs,
        )

    def capabilities(self) -> list[str]:
        return ["uniprot_lookup", "protein_sequence_fetch",
                  "protein_xref_resolution", "mcp_protocol"]
