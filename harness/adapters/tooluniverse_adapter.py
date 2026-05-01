"""ToolUniverse adapter — Harvard Zitnik Lab's 2214-tool
biomedical MCP server, exposed via its compact mode.

Compact mode (ToolUniverse's default stdio entry point) is designed
specifically to avoid the LLM-context blow-up from listing thousands of
tools upfront. Instead of 2214 function specs, the LLM sees 5
meta-tools:

    - list_tools       — paginated enumeration
    - grep_tools       — text/regex search over tool descriptions
    - find_tools       — embedding / LLM-powered semantic search
    - get_tool_info    — schema of a specific underlying tool
    - execute_tool     — dispatch by name with arbitrary args

This matches the Inspect-AI "tool retrieval" pattern and follows the
``allowed_tools`` whitelisting style without having to hand-curate a
list of 30-50 tool names.

Routing: `harness/adapters/mcp_base.py` does the protocol plumbing.
This module just declares the server spec and subclasses MCPAdapter.

Startup: `uvx --from tooluniverse tooluniverse` (first invocation
~60-180s to download deps; cache makes subsequent ~3-5s).
"""

from __future__ import annotations

from harness.adapters.mcp_base import (
    MCPAdapter, MCPServerSpec, register_mcp_server,
)


TOOLUNIVERSE_SPEC = register_mcp_server(MCPServerSpec(
    name="tooluniverse",
    command="uvx",
    args=["--from", "tooluniverse", "tooluniverse"],
    env={},
    description=(
        "ToolUniverse SMCP (Harvard Zitnik Lab) — 2214 biomedical tools "
        "exposed via compact-mode meta-dispatch: list_tools / grep_tools / "
        "find_tools / get_tool_info / execute_tool. Covers UniProt, PubChem, "
        "Reactome, ChEMBL, DrugBank, NCBI, WHO GHO, openFDA, ClinicalTrials, "
        "UCSC Genome, UMLS, PharmGKB, Wikipathways and many more, without "
        "flooding the LLM's context with thousands of function specs."
    ),
    modality="reasoning",
    tool_prefix="mcp_tu_",
    # DO NOT whitelist here — compact mode itself is the whitelist.
    # If future work wants to bypass compact mode (via
    # `tooluniverse-smcp-stdio --no-compact`), set an allowed_tools set
    # of 30-50 names to avoid context overflow.
    allowed_tools=set(),
))


class ToolUniverseAdapter(MCPAdapter):
    """Adapter bound to the ToolUniverse compact-mode SMCP server."""

    default_spec = TOOLUNIVERSE_SPEC

    def __init__(self, config: dict | None = None, **kwargs):
        # ToolUniverse cold start is slow — first uvx install needs a
        # big timeout; subsequent sessions reuse the cache.
        super().__init__(
            spec=TOOLUNIVERSE_SPEC,
            config=config,
            startup_timeout=kwargs.pop("startup_timeout", 300),
            per_tool_timeout=kwargs.pop("per_tool_timeout", 90.0),
            **kwargs,
        )

    def capabilities(self) -> list[str]:
        return [
            "tool_meta_dispatch",
            "tool_semantic_search",
            "pubchem_query",
            "uniprot_query",
            "chembl_query",
            "reactome_query",
            "clinvar_query",
            "umls_query",
            "pharmgkb_query",
            "who_gho_query",
            "mcp_protocol",
        ]
