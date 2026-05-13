"""Function-calling multi-hop runner.

Replaces the static triage→evidence→reason pipeline with dynamic LLM-driven
tool calling. The LLM decides which tools to invoke based on prior results
and iterates up to max_iterations times.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from harness.adapters.ncbi_tools_adapter import NCBIToolsAdapter
from harness.adapters.retrieval import (
    RxNavClient, OpenFDAClient, DailyMedClient,
    OMIMClient, OrphanetClient, MedlinePlusClient,
)
from harness.llm_client import LLMClient
from harness.tools.clinical_calculators import CALCULATORS


# OpenAI/Gemini function-calling schema for all tools
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "pubmed_search",
            "description": "Search PubMed for biomedical research papers. Use for finding literature supporting a clinical or genetic claim.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "PubMed search query (natural language or MeSH terms)"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "gene_lookup",
            "description": "Look up a human gene in NCBI Gene database. Returns official symbol, location, function, and summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "gene_symbol": {"type": "string", "description": "Gene symbol (e.g., BRCA1, TP53)"},
                },
                "required": ["gene_symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clinvar_lookup",
            "description": "Look up a genetic variant's clinical significance in ClinVar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "variant": {"type": "string", "description": "Variant (e.g., 'BRCA1 c.5266dupC' or rsID)"},
                },
                "required": ["variant"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rxnav_drug",
            "description": "Look up drug info and interactions from NIH RxNav/RxNorm.",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {"type": "string"},
                },
                "required": ["drug_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "openfda_adverse",
            "description": "Query FDA adverse events and drug label (indications, warnings) for a drug.",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {"type": "string"},
                },
                "required": ["drug_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dailymed_label",
            "description": "Retrieve FDA Structured Product Label (SPL) from DailyMed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_name": {"type": "string"},
                },
                "required": ["drug_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "omim_lookup",
            "description": "Look up a genetic disorder or gene in OMIM (requires API key; gracefully skips if unavailable).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Disease name, gene, or MIM number"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "orphanet_lookup",
            "description": "Look up a rare disease in Orphanet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "disease": {"type": "string"},
                },
                "required": ["disease"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "medlineplus_topic",
            "description": "Search NIH MedlinePlus for patient-friendly clinical topic info (replaces UpToDate).",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                },
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_calculator",
            "description": (
                "Run a named clinical calculator. Available: "
                "cha2ds2_vasc, heart_score, wells_dvt, wells_pe, curb65, qsofa, "
                "meld, child_pugh, ckd_epi_egfr, apache_ii, glasgow_coma_scale, "
                "bmi, bsa, framingham_10yr_cvd, ascvd_pooled_cohort, has_bled, "
                "abcd2_stroke, nih_stroke_scale, bishop_score, apgar_score."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "calculator_name": {"type": "string"},
                    "params": {"type": "object", "description": "Keyword arguments for the calculator"},
                },
                "required": ["calculator_name", "params"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator_eval",
            "description": (
                "Evaluate a mathematical formula. Use when the formula is known "
                "but no built-in calculator matches. Example: '(140-87)*48*1/(1.4*72)'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "formula": {"type": "string", "description": "Python-syntax arithmetic expression"},
                },
                "required": ["formula"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_exec",
            "description": (
                "Execute arbitrary Python code in an isolated subprocess (5s timeout, no network). "
                "Use for scientific algorithmic problems, numerical analysis, "
                "data manipulation, or any computation that needs real Python execution. "
                "Available libs: math, statistics, json, re, itertools, functools, collections, "
                "fractions, decimal, numpy (if installed), scipy (if installed). "
                "Print results — only stdout is returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source code. End with print(result)."},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bixbench_sandbox_exec",
            "description": (
                "BixBench only. Execute Python, R, or bash analysis commands against "
                "the task's mounted official data capsule. In Docker mode the capsule "
                "is read-only at /capsule and /work is writable. Use this for "
                "capsule-backed BixBench data analysis instead of guessing from memory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run, e.g. `python3 - <<'PY' ... PY`, `Rscript script.R`, or `ls -R /capsule | head`.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_search",
            "description": (
                "Search a code corpus for symbol/string matches. Useful for "
                "questions about specific functions or classes in known repositories. "
                "Returns matching file paths and surrounding context. "
                "Currently performs a web/PubMed-style fallback search since no local repo index exists."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol_or_string": {"type": "string"},
                    "repo": {"type": "string", "description": "Repo name like 'astropy/astropy' (optional context)"},
                },
                "required": ["symbol_or_string"],
            },
        },
    },
]


# Chemistry tools (rdkit / datamol)
try:
    from harness.tools.chemistry_tools import (
        CHEMISTRY_TOOL_SPECS,
        CHEMISTRY_TOOL_NAMES,
        handle_mol_tool,
    )
    TOOL_SPECS.extend(CHEMISTRY_TOOL_SPECS)
except ImportError:
    CHEMISTRY_TOOL_NAMES = set()
    handle_mol_tool = None  # type: ignore


# Native ADMET tools — optional, requires admet-ai
try:
    from harness.tools.chemistry_tools import (
        ADMET_TOOL_SPECS,
        ADMET_TOOL_NAMES,
        handle_admet_tool,
    )
    # Only expose the tools to the LLM if admet_ai itself imports — the
    # tool specs exist at module level even without the dep, but calling
    # them would fall back to the "not installed" path.
    try:
        import admet_ai  # noqa: F401
        TOOL_SPECS.extend(ADMET_TOOL_SPECS)
    except ImportError:
        ADMET_TOOL_NAMES = set()
        handle_admet_tool = None  # type: ignore
except ImportError:
    ADMET_TOOL_NAMES = set()
    handle_admet_tool = None  # type: ignore


# PyTDC tools — vendor-isolated via subprocess
PYTDC_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "tdc_admet_lookup",
            "description": (
                "Query Therapeutics Data Commons (TDC) ADMET datasets for a SMILES. "
                "Returns the measured endpoint value if the compound is in the dataset, "
                "else null. Common endpoints: Caco2_Wang, Lipophilicity_AstraZeneca, "
                "HIA_Hou, PAMPA_NCATS, Bioavailability_Ma, Solubility_AqSolDB, PPBR_AZ, "
                "VDss_Lombardo, CYP2D6_Veith, CYP3A4_Veith, CYP2C9_Veith, "
                "Half_Life_Obach, Clearance_Hepatocyte_AZ, hERG, AMES, DILI, LD50_Zhu."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {"type": "string"},
                    "endpoints": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of TDC ADMET endpoint names",
                    },
                },
                "required": ["smiles"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tdc_load_dataset_sample",
            "description": (
                "Return the first N rows of a TDC dataset for exploration. Useful "
                "to see typical values and structure before running a full analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "n": {"type": "integer", "default": 5},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tdc_molecule_generation_sample",
            "description": (
                "Return N example drug-like SMILES from TDC MolGen (ZINC). Useful "
                "for negative controls or baseline distributions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "default": 5},
                },
                "required": [],
            },
        },
    },
]
TOOL_SPECS.extend(PYTDC_TOOL_SPECS)
PYTDC_TOOL_NAMES = {"tdc_admet_lookup", "tdc_load_dataset_sample", "tdc_molecule_generation_sample"}


# gget + mygene tools — stateless HTTP wrappers
try:
    from harness.tools.gget_tools import (
        GGET_TOOL_SPECS,
        GGET_TOOL_NAMES,
        handle_gget_tool,
    )
    TOOL_SPECS.extend(GGET_TOOL_SPECS)
except ImportError:
    GGET_TOOL_NAMES = set()
    handle_gget_tool = None  # type: ignore


# DICOM tools — pydicom metadata + pixel stats
try:
    from harness.tools.dicom_tools import (
        DICOM_TOOL_SPECS,
        DICOM_TOOL_NAMES,
        handle_dicom_tool,
    )
    TOOL_SPECS.extend(DICOM_TOOL_SPECS)
except ImportError:
    DICOM_TOOL_NAMES = set()
    handle_dicom_tool = None  # type: ignore


# Protein tools — AlphaFold-DB HTTP lookup
try:
    from harness.tools.protein_tools import (
        PROTEIN_TOOL_SPECS,
        PROTEIN_TOOL_NAMES,
        handle_protein_tool,
    )
    TOOL_SPECS.extend(PROTEIN_TOOL_SPECS)
except ImportError:
    PROTEIN_TOOL_NAMES = set()
    handle_protein_tool = None  # type: ignore


# Biopython tools — FASTA/GenBank/sequence ops (Step 16 P1)
try:
    from harness.tools.biopython_tools import (
        BIOPYTHON_TOOL_SPECS,
        BIOPYTHON_TOOL_NAMES,
        handle_biopython_tool,
    )
    TOOL_SPECS.extend(BIOPYTHON_TOOL_SPECS)
except ImportError:
    BIOPYTHON_TOOL_NAMES = set()
    handle_biopython_tool = None  # type: ignore


# Survival analysis tools — lifelines (Step 16 P1)
try:
    from harness.tools.survival_tools import (
        SURVIVAL_TOOL_SPECS,
        SURVIVAL_TOOL_NAMES,
        handle_survival_tool,
    )
    TOOL_SPECS.extend(SURVIVAL_TOOL_SPECS)
except ImportError:
    SURVIVAL_TOOL_NAMES = set()
    handle_survival_tool = None  # type: ignore


# Molfeat — molecular featurizers (Step 17 P1)
try:
    from harness.tools.molfeat_tools import (
        MOLFEAT_TOOL_SPECS,
        MOLFEAT_TOOL_NAMES,
        handle_molfeat_tool,
    )
    # molfeat imports lazily inside each tool; spec-level import needs
    # the library present, so double-guard.
    try:
        import molfeat  # noqa: F401
        TOOL_SPECS.extend(MOLFEAT_TOOL_SPECS)
    except ImportError:
        MOLFEAT_TOOL_NAMES = set()
        handle_molfeat_tool = None  # type: ignore
except ImportError:
    MOLFEAT_TOOL_NAMES = set()
    handle_molfeat_tool = None  # type: ignore


# MONAI — medical imaging transforms (Step 17 P1)
try:
    from harness.tools.monai_tools import (
        MONAI_TOOL_SPECS,
        MONAI_TOOL_NAMES,
        handle_monai_tool,
    )
    try:
        import monai  # noqa: F401
        TOOL_SPECS.extend(MONAI_TOOL_SPECS)
    except ImportError:
        MONAI_TOOL_NAMES = set()
        handle_monai_tool = None  # type: ignore
except ImportError:
    MONAI_TOOL_NAMES = set()
    handle_monai_tool = None  # type: ignore


# Web search tools (Serper + Jina) — always registered in the pool;
# filter_tools() gates whether they are advertised to the model based on
# the BIOMEDARENA_WEB_TOOLS env var ("off" / "only" / "combined").
try:
    from harness.tools.web_search import (
        WEB_TOOL_SPECS,
        WEB_TOOL_NAMES,
        handle_web_tool,
    )
    TOOL_SPECS.extend(WEB_TOOL_SPECS)
except ImportError:
    WEB_TOOL_NAMES = set()  # type: ignore
    handle_web_tool = None  # type: ignore


# MCP servers — enumerated lazily on first use.
# We can't call `await adapter.list_tool_specs()` at import time, so we
# keep a singleton MCP registry keyed by tool-name prefix and populate it
# inside `_ensure_mcp_loaded()` the first time a FunctionCallingRunner
# instance needs a tool list. `register_mcp_adapters()` is also exposed
# for explicit eager loading (e.g. at benchmark startup).
MCP_ADAPTERS: dict[str, Any] = {}  # prefix -> MCPAdapter instance
_MCP_SPECS_LOADED = False


def _mcp_registered_adapters() -> list[tuple[str, Any]]:
    """Return [(prefix, adapter), …] for adapters whose specs we should
    expose. Import is guarded so absence of the `mcp` SDK is non-fatal.
    Order = registration order, which becomes spec-append order (stable
    across runs).
    """
    import os as _os
    out: list[tuple[str, Any]] = []
    # BioMCP (Step 10)
    try:
        from harness.adapters.biomcp_adapter import BioMCPAdapter
        bmcp = BioMCPAdapter()
        if bmcp.available:
            out.append((bmcp.tool_prefix(), bmcp))
    except Exception:
        pass
    # ToolUniverse (Step 11) — compact mode only exposes 5 meta-tools
    try:
        from harness.adapters.tooluniverse_adapter import ToolUniverseAdapter
        tu = ToolUniverseAdapter()
        if tu.available:
            out.append((tu.tool_prefix(), tu))
    except Exception:
        pass
    # dicom-mcp (Step 12) — PACS access via DIMSE. Skipped unless a
    # live PACS is reachable (HARNESS_DICOM_PACS_READY=1) because
    # otherwise every DIMSE call returns 'connection refused' and
    # wastes LLM iterations during benchmarks without an Orthanc
    # instance. Specs for imaging-specific benchmarks should set the
    # env var explicitly.
    if _os.environ.get("HARNESS_DICOM_PACS_READY") == "1":
        try:
            from harness.adapters.dicom_mcp_adapter import DicomMCPAdapter
            dmcp = DicomMCPAdapter()
            if dmcp.available:
                out.append((dmcp.tool_prefix(), dmcp))
        except Exception:
            pass
    # Step 19 optional extras — PubMed / GEO / UniProt MCP. Only
    # registered at lazy-load time when the opt-in env var is set,
    # because cold-start of these less-popular servers is slow and
    # several require site-specific config.
    if _os.environ.get("HARNESS_ENABLE_EXTRA_MCP"):
        try:
            from harness.adapters.mcp_extra_adapters import (
                PubMedMCPAdapter, GEOMCPAdapter, UniProtMCPAdapter,
            )
            for cls in (PubMedMCPAdapter, GEOMCPAdapter, UniProtMCPAdapter):
                try:
                    a = cls()
                    if a.available:
                        out.append((a.tool_prefix(), a))
                except Exception:
                    pass
        except Exception:
            pass
    return out


async def register_mcp_adapters(timeout: float = 30.0) -> int:
    """Eagerly initialise MCP adapters and register their tool specs.

    Safe to call multiple times. Returns number of tool specs added.
    """
    global _MCP_SPECS_LOADED
    if _MCP_SPECS_LOADED:
        return 0
    added = 0
    for prefix, adapter in _mcp_registered_adapters():
        try:
            specs = await asyncio.wait_for(adapter.list_tool_specs(), timeout=timeout)
        except Exception:
            continue
        if not specs:
            continue
        TOOL_SPECS.extend(specs)
        MCP_ADAPTERS[prefix] = adapter
        added += len(specs)
    _MCP_SPECS_LOADED = True
    return added


def _mcp_adapter_for(name: str) -> Any | None:
    """Look up the MCP adapter that owns a given tool name (by prefix)."""
    for prefix, adapter in MCP_ADAPTERS.items():
        if name.startswith(prefix):
            return adapter
    return None


# Per-benchmark iteration budgets. Default is 50 for all benchmarks,
# letting the model drive depth of tool-calling iterations.
# The budget can be overridden per-run via --max-iterations CLI arg.
DEFAULT_MAX_ITERATIONS: dict[str, int] = {
    "_default": 50,
}


# Uniform per-tool timeout (seconds). All tools share the same budget.
TOOL_TIMEOUTS: dict[str, int] = {
    "_default": 60,
}


def default_max_iterations_for(benchmark_key: str | None) -> int:
    """Return the default iteration budget for a benchmark key.

    Unknown keys fall through to `_default`. The key is a short, stable
    string like 'labbench_litqa2' — not a display name. Callers that
    know the exact subtask should pass it through, otherwise the
    per-benchmark runner constructor will default to 5.
    """
    if benchmark_key and benchmark_key in DEFAULT_MAX_ITERATIONS:
        return DEFAULT_MAX_ITERATIONS[benchmark_key]
    return DEFAULT_MAX_ITERATIONS["_default"]


SYSTEM_PROMPT = (
    "You are an expert physician-scientist with access to medical databases, "
    "clinical calculators, and web search via tool calls. Your goal is to find "
    "the CORRECT answer through thorough, multi-step research.\n\n"
    "IMPORTANT: NEVER answer from memory alone. Always use tools to search and "
    "verify before answering. Even if you think you know the answer, you MUST "
    "confirm it with evidence from your available tools.\n\n"
    "Strategy:\n"
    "1. Start by analyzing the question and identifying what information you need.\n"
    "2. Use 2-3 diverse tool calls to gather initial evidence from different angles.\n"
    "3. Based on the results, refine your understanding and make follow-up tool calls "
    "to fill gaps or verify findings.\n"
    "4. Continue iterating — each round of tool calls should build on what you learned "
    "from previous rounds. Try at least 2-3 rounds of search-then-verify.\n"
    "5. Only give your final answer when you have converging evidence from multiple "
    "independent sources. If sources conflict, dig deeper.\n\n"
    "Guidelines:\n"
    "- Prefer depth over breadth: carefully reading results from 3-4 high-quality "
    "sources beats skimming 10.\n"
    "- When a search returns no useful results, vary your strategy: try different "
    "keywords, use more specific queries, or search for sub-problems.\n"
    "- For numerical, formula, or factual answers, always verify the exact value from "
    "a primary source rather than relying on secondary summaries.\n"
    "- For multiple choice, end with: The answer is [X].\n"
    "- For exact answer questions, end with: The answer is [your answer].\n"
    "- For numeric calculation questions, end with: The answer is [number]."
)

CONTINUATION_PROMPT = (
    "You have not done enough research yet. Continue investigating using your tools. "
    "Do NOT give a final answer until you have gathered sufficient evidence from "
    "multiple sources. Search for additional information, verify your findings, "
    "and cross-reference across different databases or web sources."
)


class FunctionCallingRunner:
    """LLM-driven multi-hop tool calling."""

    def __init__(self, llm: LLMClient, max_iterations: int | None = None,
                  min_iterations: int = 0,
                  per_tool_timeout: int = 60, truncate_chars: int = 0,
                  truncate_tokens: int = 16000,
                  enable_thinking: bool = False,
                  thinking_budget: int = 8192,
                  enable_mcp: bool = False,
                  mcp_timeout: float = 60.0,
                  enable_retrieval: bool = False,
                  retrieval_top_k: int = 15,
                  retrieval_embedder: Any = None,
                  enable_budget_tracking: bool = False,
                  budget_mode: str = "light",
                  benchmark_key: str | None = None,
                  _web_tools_override: str | None = None,
                  _system_prompt_override: str | None = None):
        """If `max_iterations` is None, the budget is resolved from
        `benchmark_key` via `DEFAULT_MAX_ITERATIONS` (falling back to
        `_default` = 50). Callers that want a hard override pass an
        explicit integer; it wins over the per-benchmark default.

        `truncate_chars` = 0 means no character-level truncation.
        `truncate_tokens` = 16000 applies approximate token-based
        truncation (1 token ≈ 4 chars) when truncate_chars is 0.
        `enable_thinking` enables extended thinking in the ReAct loop
        (Anthropic only; requires temperature=1.0)."""
        self.llm = llm
        self.benchmark_key = benchmark_key
        if max_iterations is None:
            self.max_iterations = default_max_iterations_for(benchmark_key)
        else:
            self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        self.per_tool_timeout = per_tool_timeout
        self.truncate_chars = truncate_chars
        self.truncate_tokens = truncate_tokens
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.enable_mcp = enable_mcp
        self.mcp_timeout = mcp_timeout
        # Retrieval (Step 13 / Section 3.5)
        self.enable_retrieval = enable_retrieval
        self.retrieval_top_k = retrieval_top_k
        self.retrieval_embedder = retrieval_embedder
        self._retriever = None  # built lazily in run()
        # Budget tracking (Step 13)
        self.enable_budget_tracking = enable_budget_tracking
        self.budget_mode = budget_mode
        # Last-run ablation log (populated when enable_retrieval=True)
        self._web_tools_override = _web_tools_override
        self._system_prompt_override = _system_prompt_override
        self.last_retrieval_log: dict[str, Any] | None = None
        # Exposed after run() for summarizer access (multi-agent mode)
        self._last_messages: list[dict[str, Any]] | None = None
        self._current_task: dict[str, Any] | None = None
        # Context management (CM_* env vars)
        self._context_manager = None
        try:
            from harness.context_managers import build_context_manager
            self._context_manager = build_context_manager()
            # Wire up LLM for summarization strategies
            if self._context_manager is not None:
                for strategy in self._context_manager.strategies:
                    if hasattr(strategy, "_llm"):
                        strategy._llm = llm
        except Exception as exc:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "Context manager init failed (CM disabled for this runner): %s", exc,
            )
        # Lazy-init clients
        self._ncbi: NCBIToolsAdapter | None = None
        self._rxnav: RxNavClient | None = None
        self._openfda: OpenFDAClient | None = None
        self._dailymed: DailyMedClient | None = None
        self._omim: OMIMClient | None = None
        self._orphanet: OrphanetClient | None = None
        self._medlineplus: MedlinePlusClient | None = None

    def _get_ncbi(self) -> NCBIToolsAdapter:
        if self._ncbi is None:
            self._ncbi = NCBIToolsAdapter()
        return self._ncbi

    async def _dispatch_tool(self, name: str, args: dict[str, Any]) -> str:
        """Execute a single tool call and return a string result.

        Uniform 60s per-tool timeout resolved from TOOL_TIMEOUTS.
        Token-based truncation (truncate_tokens × 4 chars) applied when > 0;
        character-based truncation (truncate_chars) used as a secondary
        compatibility path.
        """
        timeout = TOOL_TIMEOUTS.get(name, TOOL_TIMEOUTS.get("_default", self.per_tool_timeout))
        try:
            result = await asyncio.wait_for(
                self._call(name, args), timeout=timeout
            )
            text = json.dumps(result) if not isinstance(result, str) else result
            if self.truncate_chars > 0:
                text = text[:self.truncate_chars]
            elif self.truncate_tokens > 0:
                max_chars = self.truncate_tokens * 4
                if len(text) > max_chars:
                    text = text[:max_chars]
            return text
        except asyncio.TimeoutError:
            return f"[{name} timed out after {timeout}s]"
        except Exception as exc:
            return f"[{name} error: {exc}]"

    async def _call(self, name: str, args: dict[str, Any]) -> Any:
        if name == "pubmed_search":
            r = await self._get_ncbi().pubmed_search(args["query"], args.get("max_results", 5))
            return r.get("summary", "")
        if name == "gene_lookup":
            r = await self._get_ncbi().gene_info(args["gene_symbol"])
            return r.get("summary", "")
        if name == "clinvar_lookup":
            r = await self._get_ncbi().clinvar_lookup(args["variant"])
            return r.get("summary", "")
        if name == "rxnav_drug":
            if self._rxnav is None:
                self._rxnav = RxNavClient()
            r = await self._rxnav.summary(args["drug_name"])
            return r.get("summary", r.get("reason", ""))
        if name == "openfda_adverse":
            if self._openfda is None:
                self._openfda = OpenFDAClient()
            r = await self._openfda.summary(args["drug_name"])
            return r.get("summary", r.get("reason", ""))
        if name == "dailymed_label":
            if self._dailymed is None:
                self._dailymed = DailyMedClient()
            r = await self._dailymed.summary(args["drug_name"])
            return r.get("summary", r.get("reason", ""))
        if name == "omim_lookup":
            if self._omim is None:
                self._omim = OMIMClient()
            r = await self._omim.summary(args["query"])
            return r.get("summary") or r.get("reason", "")
        if name == "orphanet_lookup":
            if self._orphanet is None:
                self._orphanet = OrphanetClient()
            r = await self._orphanet.summary(args["disease"])
            return r.get("summary", r.get("reason", ""))
        if name == "medlineplus_topic":
            if self._medlineplus is None:
                self._medlineplus = MedlinePlusClient()
            r = await self._medlineplus.summary(args["topic"])
            return r.get("summary", r.get("reason", ""))
        if name == "compute_calculator":
            calc = args["calculator_name"]
            params = args.get("params", {})
            if calc in CALCULATORS:
                try:
                    result = CALCULATORS[calc](**params)
                    return f"Score={result['score']}, Category={result['category']}, Rec={result['recommendation']}"
                except Exception as exc:
                    return f"[calculator error: {exc}]"
            return f"[unknown calculator: {calc}]"
        if name == "calculator_eval":
            formula = args.get("formula", "")
            # Safe eval — only arithmetic operators
            try:
                # Disallow anything but numbers and basic operators
                import re as _re
                if not _re.match(r"^[\d\s\+\-\*\/\(\)\.\,eE]+$", formula):
                    return f"[unsafe formula: {formula}]"
                value = eval(formula, {"__builtins__": {}}, {})
                return f"= {value}"
            except Exception as exc:
                return f"[eval error: {exc}]"
        if name == "python_exec":
            return await self._run_python_exec(args.get("code", ""))
        if name == "bixbench_sandbox_exec":
            return await self._run_bixbench_sandbox(args.get("command", ""))
        if name == "code_search":
            return await self._run_code_search(
                args.get("symbol_or_string", ""), args.get("repo", "")
            )
        # Chemistry tools — dispatch to synchronous RDKit worker in thread
        if name in CHEMISTRY_TOOL_NAMES and handle_mol_tool is not None:
            return await asyncio.to_thread(handle_mol_tool, name, args)
        # Native ADMET tools — chemprop ensemble
        if name in ADMET_TOOL_NAMES and handle_admet_tool is not None:
            return await asyncio.to_thread(handle_admet_tool, name, args)
        # PyTDC tools — dispatch to vendor-isolated subprocess
        if name in PYTDC_TOOL_NAMES:
            return await self._dispatch_pytdc(name, args)
        # gget / mygene tools — stateless HTTP, run in thread
        if name in GGET_TOOL_NAMES and handle_gget_tool is not None:
            return await asyncio.to_thread(handle_gget_tool, name, args)
        # DICOM tools — file I/O, run in thread
        if name in DICOM_TOOL_NAMES and handle_dicom_tool is not None:
            return await asyncio.to_thread(handle_dicom_tool, name, args)
        # Protein tools — HTTP lookup, run in thread
        if name in PROTEIN_TOOL_NAMES and handle_protein_tool is not None:
            return await asyncio.to_thread(handle_protein_tool, name, args)
        # Biopython tools — pure CPU, run in thread
        if name in BIOPYTHON_TOOL_NAMES and handle_biopython_tool is not None:
            return await asyncio.to_thread(handle_biopython_tool, name, args)
        # Survival analysis (lifelines) — pure CPU, run in thread
        if name in SURVIVAL_TOOL_NAMES and handle_survival_tool is not None:
            return await asyncio.to_thread(handle_survival_tool, name, args)
        # Molfeat — run in thread (RDKit + molfeat graph ops)
        if name in MOLFEAT_TOOL_NAMES and handle_molfeat_tool is not None:
            return await asyncio.to_thread(handle_molfeat_tool, name, args)
        # MONAI — medical image I/O, run in thread
        if name in MONAI_TOOL_NAMES and handle_monai_tool is not None:
            return await asyncio.to_thread(handle_monai_tool, name, args)
        # Web search tools (Serper + Jina) — async HTTP
        if name in WEB_TOOL_NAMES and handle_web_tool is not None:
            # Lazily init web clients with summarizer LLM on first call
            from harness.tools.web_search import _jina_client, init_web_clients
            if _jina_client is None:
                init_web_clients(summarizer_llm=self.llm)
            return await handle_web_tool(name, args)
        # MCP tools — dispatch through the owning adapter (lazy-started).
        mcp_adapter = _mcp_adapter_for(name)
        if mcp_adapter is not None:
            try:
                return await mcp_adapter.call_tool(name, args)
            except Exception as exc:  # MCPTransportError or others
                return f"[{name} transport error: {exc}]"
        return f"[unknown tool: {name}]"

    async def _dispatch_pytdc(self, name: str, args: dict[str, Any]) -> str:
        """Route PyTDC tool calls to vendor-isolated subprocess."""
        try:
            from harness.adapters.pytdc_adapter import PyTDCAdapter
        except ImportError as exc:
            return f"[PyTDC adapter missing: {exc}]"

        adapter = PyTDCAdapter()
        if not adapter.available:
            return f"[PyTDC unavailable: {adapter.unavailable_reason}]"

        try:
            if name == "tdc_admet_lookup":
                resp = await adapter.admet_predict(
                    args["smiles"], args.get("endpoints") or ["Caco2_Wang"],
                )
            elif name == "tdc_load_dataset_sample":
                resp = await adapter.load_dataset_sample(
                    args.get("name", "Caco2_Wang"),
                    n=int(args.get("n", 5)),
                )
            elif name == "tdc_molecule_generation_sample":
                resp = await adapter.mol_generation_sample(n=int(args.get("n", 5)))
            else:
                return f"[unknown PyTDC tool: {name}]"
        except Exception as exc:
            return f"[{name} error: {exc}]"

        if not resp.get("ok"):
            return f"[{name} error: {resp.get('error', 'unknown')}]"
        return json.dumps({k: v for k, v in resp.items() if k not in ("id", "ok")})

    async def _run_python_exec(self, code: str, timeout: int = 5) -> str:
        """Execute Python code in an isolated subprocess. Returns stdout (truncated)."""
        import asyncio as _asyncio
        import sys as _sys

        if not code.strip():
            return "[python_exec: empty code]"

        # Crude safety filter — no network, no file deletion, no os.system
        forbidden = ["socket", "urllib", "requests", "subprocess", "os.system",
                      "os.remove", "os.unlink", "shutil.rmtree", "__import__('os')"]
        for pattern in forbidden:
            if pattern in code:
                return f"[python_exec: forbidden pattern '{pattern}']"

        try:
            proc = await _asyncio.create_subprocess_exec(
                _sys.executable, "-c", code,
                stdout=_asyncio.subprocess.PIPE,
                stderr=_asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await _asyncio.wait_for(proc.communicate(), timeout=timeout)
            except _asyncio.TimeoutError:
                proc.kill()
                return f"[python_exec: timed out after {timeout}s]"
            out = stdout.decode("utf-8", errors="replace")
            err = stderr.decode("utf-8", errors="replace")
            if proc.returncode != 0:
                return f"[python_exec error]\n{err[:2000]}"
            return out[:10000] if out else "[python_exec: no output]"
        except Exception as exc:
            return f"[python_exec dispatch error: {exc}]"

    async def _run_bixbench_sandbox(self, command: str) -> str:
        """Run a BixBench capsule command through the configured sandbox."""
        import os as _os

        if not str(command or "").strip():
            return "[tool error: bixbench_sandbox_exec empty command]"
        task = self._current_task or {}
        ctx = task.get("context") or {}
        capsule_path = ctx.get("capsule_path")
        if not capsule_path:
            return (
                "[tool error: bixbench_sandbox_exec missing capsule_path. "
                "Run `biomedarena prepare-bixbench --extract` and pass "
                "`--bixbench-capsules`.]"
            )
        backend = _os.environ.get("BIOMEDARENA_BIXBENCH_SANDBOX", "docker").strip() or "docker"
        try:
            from harness.eval.bixbench_official import (
                BixBenchSandbox,
                BixBenchSandboxConfig,
                resolve_docker_command,
            )
            docker_cmd = resolve_docker_command() if backend == "docker" else None
            if backend == "docker" and docker_cmd is None:
                return (
                    "[tool error: bixbench_sandbox_exec docker command not found. "
                    "Install Docker and build `biomedarena/bixbench-sandbox:latest`, "
                    "or explicitly opt into the unsafe local backend only for trusted debugging.]"
                )
            cfg = BixBenchSandboxConfig(
                backend=backend,
                image=_os.environ.get(
                    "BIOMEDARENA_BIXBENCH_DOCKER_IMAGE",
                    "biomedarena/bixbench-sandbox:latest",
                ),
                docker_cmd=docker_cmd,
                timeout_s=int(_os.environ.get("BIOMEDARENA_BIXBENCH_TIMEOUT_S", "900")),
                allow_local=_os.environ.get("BIOMEDARENA_BIXBENCH_ALLOW_LOCAL", "0") == "1",
            )
            result = await asyncio.to_thread(
                BixBenchSandbox(cfg).run,
                capsule_path,
                str(command),
            )
            payload = {
                "returncode": result.returncode,
                "backend": result.backend,
                "stdout": result.stdout[:8000],
                "stderr": result.stderr[:2000],
            }
            return json.dumps(payload)
        except Exception as exc:
            return f"[bixbench_sandbox_exec error: {type(exc).__name__}: {exc}]"

    async def _run_code_search(self, query: str, repo: str = "") -> str:
        """Code/symbol search. Without a local repo index, falls back to PubMed-like
        web search via NCBI as a placeholder. Returns descriptive text."""
        if not query.strip():
            return "[code_search: empty query]"

        # Best-effort: search NCBI (works for general scientific queries) and
        # tag the result with the repo for context. A future improvement could
        # use GitHub code search API or a local repo index.
        try:
            r = await self._get_ncbi().pubmed_search(query, max_results=3)
            if r.get("summary"):
                return (f"[code_search fallback for '{query}' in repo '{repo or 'unknown'}']\n"
                        f"{r['summary']}")
        except Exception:
            pass
        return (f"[code_search] No local repo index for '{repo}'. "
                f"Asked about: '{query}'. Use general knowledge to answer.")

    async def run(self, task: dict[str, Any]) -> tuple[str, list[str]]:
        """Run multi-hop tool calling for a task.

        Returns: (final_text_answer, list_of_tool_names_called)
        """
        self._current_task = task
        # H1 fix: rebuild context manager per task to avoid cross-task state
        # leak (ScratchpadStrategy._scratchpad, IncrementalSummaryStrategy
        # counters, etc.) when the runner is cached across tasks.
        try:
            from harness.context_managers import build_context_manager
            self._context_manager = build_context_manager()
            if self._context_manager is not None:
                for strategy in self._context_manager.strategies:
                    if hasattr(strategy, "_llm"):
                        strategy._llm = self.llm
        except Exception as exc:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "Context manager rebuild failed at run() start: %s", exc,
            )
            self._context_manager = None

        if self.enable_mcp:
            try:
                await register_mcp_adapters(timeout=self.mcp_timeout)
            except Exception as exc:  # noqa: BLE001
                # MCP registration should never abort a run; the LLM can
                # still use the non-MCP tools.
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "MCP registration failed: %s", exc,
                )

        # Benchmark-aware tool filtering. When a benchmark_key
        # is known AND the benchmark has a registered whitelist/category
        # config, advertise only the relevant subset. Falls through to
        # full TOOL_SPECS when no config is registered. Still backward
        # compatible: unconfigured benchmarks see the full list.
        import os as _os
        from harness.benchmark_config import filter_tools, get_config
        _bench_key = getattr(self, "benchmark_key", None) or task.get("_benchmark_key")
        _bench_cfg = get_config(_bench_key)
        # Sub-agent web_tools override: temporarily set env var for filter_tools
        _prev_web = None
        if self._web_tools_override is not None:
            _prev_web = _os.environ.get("BIOMEDARENA_WEB_TOOLS")
            _os.environ["BIOMEDARENA_WEB_TOOLS"] = self._web_tools_override
        _tool_pool = filter_tools(_bench_key, TOOL_SPECS)
        # Restore env var
        if _prev_web is not None:
            _os.environ["BIOMEDARENA_WEB_TOOLS"] = _prev_web
        elif self._web_tools_override is not None:
            _os.environ.pop("BIOMEDARENA_WEB_TOOLS", None)
        if not _tool_pool:
            # Empty whitelist → tools disabled for this benchmark.
            _tool_pool = []

        # Build the per-call tool set — either the benchmark-filtered pool
        # or a retrieval-filtered subset (Step 13 / Section 3.5).
        call_tools: list[dict[str, Any]] = _tool_pool
        # Skip retrieval when the benchmark either asked for it off or
        # already has a tight whitelist / empty tool set.
        _run_retrieval = (
            self.enable_retrieval
            and _bench_cfg.enable_retrieval
            and len(_tool_pool) > self.retrieval_top_k
        )
        if _run_retrieval:
            try:
                from harness.context.tool_retrieval import ToolRetriever
                if self._retriever is None:
                    self._retriever = ToolRetriever(
                        _tool_pool,
                        embed_fn=self.retrieval_embedder,
                    )
                call_tools = self._retriever.retrieve(
                    query=task.get("question", ""),
                    context=task,
                    top_k=self.retrieval_top_k,
                )
                retrieved_names = [t["function"]["name"] for t in call_tools]
                ignored = [
                    t["function"]["name"] for t in _tool_pool
                    if t["function"]["name"] not in set(retrieved_names)
                ]
                self.last_retrieval_log = {
                    "total_pool": len(_tool_pool),
                    "retrieved": retrieved_names,
                    "ignored_count": len(ignored),
                    "fallback_used": bool(self._retriever.fallback_used),
                }
            except Exception as exc:  # noqa: BLE001
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "Retrieval failed (%s); falling back to filtered pool", exc,
                )
                call_tools = _tool_pool
                self.last_retrieval_log = {"error": str(exc)}

        # Optional token-budget accounting (Step 13)
        tracker = None
        if self.enable_budget_tracking:
            from harness.context.budget import TokenBudgetTracker
            tracker = TokenBudgetTracker(mode=self.budget_mode)

        from harness.benchmark_config import build_system_prompt, build_user_prompt
        _base_prompt = self._system_prompt_override or SYSTEM_PROMPT
        _system_prompt = build_system_prompt(_bench_key, _base_prompt)

        # Append scratchpad instruction when CM_SCRATCHPAD is enabled
        import os as _os
        if _os.environ.get("CM_SCRATCHPAD", "").lower() in ("1", "true", "yes"):
            from harness.context_managers.external_memory import ScratchpadStrategy
            _system_prompt += ScratchpadStrategy.SYSTEM_PROMPT_ADDITION

        _user_prompt = build_user_prompt(_bench_key, task)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _system_prompt},
            {"role": "user", "content": _user_prompt},
        ]
        # Expose full conversation for summarizer (multi-agent mode).
        # This is a reference — it stays up-to-date as the loop appends.
        self._last_messages = messages
        tools_called: list[str] = []
        seen_calls: set[str] = set()  # dedupe (name, args) pairs

        # Import here to avoid circular-import risk at module load.
        from harness.trace import get_active_trace

        for iteration in range(self.max_iterations):
            # Budget check before each LLM call
            if tracker is not None:
                tracker.tick_iteration()
                tracker.observe_input(messages=messages)
                action = tracker.degrade_action()
                if action == "truncate":
                    messages = tracker.truncate(messages)
                elif action == "force_answer":
                    # Give up mid-loop; return whatever we last generated
                    return (content if "content" in dir() and content else
                             "[budget exceeded]"), tools_called

            # Count this ReAct iteration in the per-task trace.
            _tr = get_active_trace()
            if _tr is not None:
                _tr.increment_iteration()

            _temperature = 1.0 if self.enable_thinking else 0.1
            _thinking_kw = dict(
                enable_thinking=True,
                thinking_budget=self.thinking_budget,
            ) if self.enable_thinking else {}
            try:
                resp = await self.llm.chat_with_tools(
                    messages=messages, tools=call_tools,
                    temperature=_temperature, max_tokens=16384,
                    **_thinking_kw,
                )
            except Exception as exc:
                # Context overflow: try to reduce context and retry once
                exc_str = str(exc).lower()
                if self._context_manager is not None and (
                    "context" in exc_str or "token" in exc_str
                    or "too long" in exc_str or "max_tokens" in exc_str
                ):
                    try:
                        self._context_manager.reduce_context(messages, exc)
                        resp = await self.llm.chat_with_tools(
                            messages=messages, tools=call_tools,
                            temperature=_temperature, max_tokens=16384,
                            **_thinking_kw,
                        )
                    except Exception as retry_exc:
                        return f"[LLM error: {retry_exc}]", tools_called
                else:
                    return f"[LLM error: {exc}]", tools_called

            content = resp.get("content")
            tool_calls = resp.get("tool_calls", [])

            if tracker is not None:
                tracker.observe_output(text=content or "")

            if not tool_calls:
                # Continuation mechanism: if min_iterations not met, nudge model
                if self.min_iterations and iteration < self.min_iterations - 1:
                    _cont_msg: dict[str, Any] = {
                        "role": "assistant",
                        "content": content or "",
                    }
                    # Preserve thinking blocks for Anthropic fidelity
                    if resp.get("_raw_blocks"):
                        _cont_msg["_raw_blocks"] = resp["_raw_blocks"]
                    messages.append(_cont_msg)
                    messages.append({
                        "role": "user",
                        "content": CONTINUATION_PROMPT,
                    })
                    continue
                return content or "", tools_called

            # Append assistant message with tool calls
            _assistant_tool_calls: list[dict[str, Any]] = []
            for tc in tool_calls:
                tc_msg = {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                if tc.get("extra_content"):
                    tc_msg["extra_content"] = tc["extra_content"]
                _assistant_tool_calls.append(tc_msg)
            _assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": content,
                "tool_calls": _assistant_tool_calls,
            }
            # Preserve raw blocks (including thinking) for Anthropic fidelity
            if resp.get("_raw_blocks"):
                _assistant_msg["_raw_blocks"] = resp["_raw_blocks"]
            messages.append(_assistant_msg)

            # Execute all tool calls in parallel (deduped)
            async def _run_one(tc):
                import time as _time
                name = tc["name"]
                try:
                    args = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
                except json.JSONDecodeError:
                    args = {}
                key = f"{name}:{json.dumps(args, sort_keys=True)}"
                if key in seen_calls:
                    result_text = "[duplicate call — refusing]"
                    _tr_inner = get_active_trace()
                    if _tr_inner is not None:
                        _tr_inner.record_tool_call(
                            name=name, arguments=args,
                            result=result_text, success=False,
                            error="duplicate call", latency_ms=0,
                        )
                    return tc["id"], name, result_text
                seen_calls.add(key)
                tools_called.append(name)
                t0 = _time.monotonic()
                try:
                    result_text = await self._dispatch_tool(name, args)
                    latency_ms = int((_time.monotonic() - t0) * 1000)
                    # Heuristic: dispatched tools return ``[error: ...]`` or
                    # ``[tool error: ...]`` on failure rather than raising.
                    lower_rt = str(result_text or "").lower()
                    is_err = lower_rt.startswith(("[error", "[tool error"))
                    _tr_inner = get_active_trace()
                    if _tr_inner is not None:
                        _tr_inner.record_tool_call(
                            name=name, arguments=args,
                            result=result_text, success=(not is_err),
                            error=(result_text if is_err else None),
                            latency_ms=latency_ms,
                        )
                    return tc["id"], name, result_text
                except Exception as exc:
                    latency_ms = int((_time.monotonic() - t0) * 1000)
                    err = f"{type(exc).__name__}: {exc}"
                    _tr_inner = get_active_trace()
                    if _tr_inner is not None:
                        _tr_inner.record_tool_call(
                            name=name, arguments=args,
                            result=None, success=False, error=err,
                            latency_ms=latency_ms,
                        )
                    return tc["id"], name, f"[tool error: {err}]"

            results = await asyncio.gather(*[_run_one(tc) for tc in tool_calls])

            # Append tool results
            for tc_id, name, result_text in results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result_text,
                })

            # Context management: apply after each turn
            if self._context_manager is not None:
                try:
                    self._context_manager.apply_management(messages)
                except Exception:
                    pass

        # Max iterations reached — ask LLM for final answer with FULL context
        messages.append({
            "role": "user",
            "content": (
                "You have reached the maximum number of tool-call iterations. "
                "Based on ALL the evidence you have gathered above, provide your "
                "final answer now. Synthesize your findings carefully."
            ),
        })
        try:
            # Keep the full message history (including tool results) so the
            # model can synthesize everything it has gathered.
            final = await self.llm.chat_with_tools(
                messages=messages, tools=[],
                temperature=0.0, max_tokens=4096,
            )
            return final.get("content", "") if isinstance(final, dict) else str(final), tools_called
        except Exception:
            # Fallback: try plain chat with full history
            try:
                final = await self.llm.chat(
                    messages=messages,
                    temperature=0.0, max_tokens=4096,
                )
                return final, tools_called
            except Exception as exc:
                return f"[LLM final error: {exc}]", tools_called
