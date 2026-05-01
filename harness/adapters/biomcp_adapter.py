"""BioMCP adapter — runs BlueCorner/Microsoft BioMCP via uvx.

BioMCP exposes ~36 biomedical tools: article_searcher, variant_searcher,
gene_getter, drug_searcher, trial_searcher, disease_getter, organization
queries, NCI CTS biomarker search, FDA openFDA endpoints, and a general
`search` + `fetch` pair backed by PubTator3 / MyVariant / MyGene / MyChem.

Routing: `harness/adapters/mcp_base.py` handles the protocol plumbing
(persistent stdio subprocess, JSON-schema conversion, two-tier errors).
This module just declares the server spec and registers it.

Startup: `uvx --from biomcp-python biomcp run --mode stdio`. The first
invocation downloads the package (~80 packages) into uv's tool cache
and can take ~60–120s. Subsequent sessions reuse the cache (<2s).
"""

from __future__ import annotations

from harness.adapters.mcp_base import (
    MCPAdapter, MCPServerSpec, register_mcp_server,
)


BIOMCP_SPEC = register_mcp_server(MCPServerSpec(
    name="biomcp",
    command="uvx",
    args=["--from", "biomcp-python", "biomcp", "run", "--mode", "stdio"],
    env={},
    description=(
        "BioMCP — biomedical MCP server with 36 tools covering PubMed "
        "literature search, clinical trials (ClinicalTrials.gov & NCI CTS), "
        "variant/gene/drug/disease queries (MyVariant/MyGene/MyChem), and "
        "openFDA endpoints. Use for multi-source biomedical evidence gathering."
    ),
    modality="reasoning",
    tool_prefix="mcp_biomcp_",
    # Do NOT whitelist — all 36 tools should be exposed. ToolUniverse
    # uses `allowed_tools` because it has 200+.
    allowed_tools=set(),
))


class BioMCPAdapter(MCPAdapter):
    """Adapter bound to the BioMCP server."""

    default_spec = BIOMCP_SPEC

    def __init__(self, config: dict | None = None, **kwargs):
        # BioMCP cold start via uvx can be slow — give it plenty of time
        super().__init__(
            spec=BIOMCP_SPEC,
            config=config,
            startup_timeout=kwargs.pop("startup_timeout", 180),
            per_tool_timeout=kwargs.pop("per_tool_timeout", 60.0),
            **kwargs,
        )

    def capabilities(self) -> list[str]:
        return [
            "pubmed_search",
            "clinical_trial_search",
            "variant_lookup",
            "gene_info",
            "drug_info",
            "disease_info",
            "openfda_query",
            "biomarker_search",
            "mcp_protocol",
        ]
