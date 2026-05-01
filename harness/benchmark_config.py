"""Benchmark-specific harness configuration.

Each benchmark has its own:
  - system_prompt_hint: additional guidance appended to the generic system prompt
  - tool_whitelist: explicit subset of TOOL_SPECS the model sees for this benchmark
  - tool_categories: category tags; tools matching any of these are advertised
  - context_formatter: optional function to inject task context into prompt
  - expected_answer_format: short instruction appended to the system prompt
  - enable_retrieval: whether to run tool retrieval before advertising tools

Benchmarks not registered here fall back to defaults (all tools, generic
prompt, default context injection) — preserving existing behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class BenchmarkHarnessConfig:
    """Per-benchmark harness tuning."""

    name: str
    """Benchmark registry key (matches benchmark_key used by the runner)."""

    system_prompt_hint: str = ""
    """Appended to the generic system prompt. Brief (1-3 sentences)."""

    tool_whitelist: Optional[list[str]] = None
    """If set (including empty list), only these tool names are advertised.
    An empty list disables tools entirely. None = no whitelist, all tools."""

    tool_categories: list[str] = field(default_factory=list)
    """Category tags for grouped tool whitelisting. Mutually complementary
    with tool_whitelist: if BOTH are set, whitelist wins (it is explicit)."""

    context_formatter: Optional[Callable[[dict], str]] = None
    """Function that takes a task dict and returns formatted prompt text.
    If None, harness.context_injection.format_task_prompt is used (which
    handles key_passage / sources injection by default)."""

    expected_answer_format: str = ""
    """One-line description of expected answer format, appended to prompt."""

    enable_retrieval: bool = True
    """Whether the retrieval subsystem should be enabled for this benchmark.
    When a tight whitelist is set, retrieval adds little value."""


# Registry of per-benchmark configs. Populated by
# `harness.benchmark_configs.registry` at import time.
BENCHMARK_CONFIGS: dict[str, BenchmarkHarnessConfig] = {}


def register_config(cfg: BenchmarkHarnessConfig) -> None:
    """Register a benchmark's harness config. Overwrites on duplicate."""
    BENCHMARK_CONFIGS[cfg.name] = cfg


def get_config(benchmark_name: str | None) -> BenchmarkHarnessConfig:
    """Get config for a benchmark. Returns a safe default if unconfigured."""
    if benchmark_name and benchmark_name in BENCHMARK_CONFIGS:
        return BENCHMARK_CONFIGS[benchmark_name]
    return BenchmarkHarnessConfig(name=benchmark_name or "")


# ---------------------------------------------------------------------------
# Web search prompts (per-benchmark task descriptions for web search mode)
# ---------------------------------------------------------------------------

# Per-benchmark task descriptions for web-search prompts.
# Each entry: (task_description, difficulty_warning)
_BENCHMARK_WEB_DESC: dict[str, tuple[str, str]] = {
    "hle_gold": (
        "expert-level biology, medicine, and chemistry questions",
        "These questions are designed to be extremely difficult — your initial "
        "intuition is often wrong. Thorough research is essential.",
    ),
    "gpqa_bio": (
        "PhD-level biology questions",
        "These are graduate-level questions where non-expert accuracy is very low. "
        "Your initial intuition is often wrong. Thorough research is essential.",
    ),
    "hle_bio_med_chem": (
        "PhD-level biology questions",
        "These are graduate-level questions where non-expert accuracy is very low. "
        "Your initial intuition is often wrong. Thorough research is essential.",
    ),
    "medcalc": (
        "clinical calculator questions covering medical formulas and computations",
        "These questions require precise numerical computation. Double-check "
        "all formulas and values before computing the final answer.",
    ),
    "medxpertqa": (
        "expert-level medical questions spanning multiple specialties",
        "These are expert-level medical questions with multiple options. "
        "Surface-level reasoning is often insufficient — verify with evidence.",
    ),
    "medxpertqa_mm": (
        "multimodal medical questions with clinical images",
        "These are expert-level medical questions requiring image interpretation. "
        "Surface-level reasoning is often insufficient — verify with evidence.",
    ),
    "labbench": (
        "biomedical lab agent tasks covering literature QA, "
        "database queries, sequence analysis, and cloning scenarios",
        "These tasks require precise domain knowledge. Verify all facts "
        "against primary sources rather than relying on general knowledge.",
    ),
    "labbench2": (
        "biomedical lab tasks with open-ended answers requiring "
        "evidence retrieval",
        "These tasks require precise domain knowledge. Verify all facts "
        "against primary sources rather than relying on general knowledge.",
    ),
    "bioasq": (
        "biomedical questions requiring PubMed literature retrieval",
        "These questions require evidence-based answers from biomedical literature. "
        "Always search for and cite primary sources.",
    ),
    "pathvqa": (
        "pathology visual question-answering tasks on histopathology images",
        "These require careful visual reasoning and domain expertise.",
    ),
    "bixbench": (
        "bioinformatics analysis tasks covering computational biology",
        "These tasks require precise bioinformatics knowledge. Verify results "
        "against primary computational biology sources.",
    ),
    "bixbench_closed_book": (
        "bioinformatics analysis tasks (closed-book format)",
        "These tasks require precise bioinformatics knowledge. Verify results "
        "against primary computational biology sources.",
    ),
    "medagentbench": (
        "clinical EHR workflow tasks covering FHIR operations",
        "These tasks require precise clinical workflow knowledge. Verify "
        "FHIR standards and clinical procedures against official documentation.",
    ),
    "agentclinic": (
        "clinical diagnostic simulation tasks across multiple specialties",
        "These are diagnostic reasoning tasks requiring careful differential diagnosis. "
        "Verify clinical criteria against authoritative medical references.",
    ),
    "medical_qa": (
        "medical knowledge questions covering clinical reasoning and diagnostics",
        "These are medical exam-level questions. Verify answers against "
        "established medical references.",
    ),
    "rag_essential": (
        "retrieval-augmented generation tasks covering recent developments "
        "and structured database lookups",
        "These tasks are specifically designed to require up-to-date information "
        "beyond LLM training data. Web search is critical for accurate answers.",
    ),
    "supergpqa": (
        "graduate-level biology and pharmacy questions "
        "covering genetics, microbiology, pharmacology, and more",
        "These are PhD-level questions requiring precise domain knowledge. "
        "Verify answers against primary sources and databases.",
    ),
    "superchem": (
        "expert-level chemistry reasoning problems covering "
        "organic, inorganic, physical, and analytical chemistry",
        "These are reasoning-intensive chemistry problems requiring step-by-step "
        "analysis of reactions, mechanisms, and chemical principles.",
    ),
    "genotex": (
        "gene expression and genomic trait association tasks",
        "These tasks require precise genomics knowledge. Verify gene-trait "
        "associations against curated databases.",
    ),
}

# Fallback for benchmarks not in the table
_DEFAULT_WEB_DESC = (
    "expert-level scientific and medical questions",
    "These questions require careful reasoning and verification. "
    "Your initial intuition may be wrong — always search and verify.",
)


def _get_benchmark_desc(benchmark_name: str | None) -> tuple[str, str]:
    """Return (task_description, difficulty_warning) for a benchmark."""
    if benchmark_name:
        # Try exact match, then prefix match for sub-benchmarks like labbench_litqa2
        if benchmark_name in _BENCHMARK_WEB_DESC:
            return _BENCHMARK_WEB_DESC[benchmark_name]
        for key in _BENCHMARK_WEB_DESC:
            if benchmark_name.startswith(key):
                return _BENCHMARK_WEB_DESC[key]
    return _DEFAULT_WEB_DESC


def _build_web_only_prompt(benchmark_name: str | None) -> str:
    """Build the full system prompt for web-only mode (replaces base prompt)."""
    desc, warning = _get_benchmark_desc(benchmark_name)
    return (
        f"You are a tool-augmented QA agent answering {desc}. "
        f"{warning}\n\n"
        "IMPORTANT: NEVER answer from memory alone. Always search and verify before "
        "answering. Even if you think you know the answer, you must confirm it with "
        "evidence from the web.\n\n"
        "Strategy:\n"
        "1. Start with 2-3 diverse search queries to scope the problem from different angles.\n"
        "2. Visit at least 3-4 of the most promising results to read their full content. "
        "Cross-reference findings across multiple sources.\n"
        "3. If initial searches are insufficient, reformulate your queries using terminology "
        "or details discovered in earlier results. Try at least 2-3 rounds of search-then-visit.\n"
        "4. Only give your final answer when you have converging evidence from multiple "
        "independent sources. If sources conflict, dig deeper.\n\n"
        "Guidelines:\n"
        "- Prefer depth over breadth: visiting and carefully reading 3-4 high-quality pages "
        "beats skimming 10.\n"
        "- When a search returns no useful results, vary your query strategy: try different "
        "keywords, use quoted phrases for exact matches, or search for the specific subproblem "
        "rather than the full question.\n"
        "- For numerical, formula, or factual answers, always verify the exact value from a "
        "primary source (paper, textbook, official documentation) rather than relying on "
        "secondary summaries.\n\n"
        "Answer format:\n"
        "- For multiple choice, end with: The answer is [X].\n"
        "- For exact answer questions, end with: The answer is [your answer].\n"
        "- For numeric calculation questions, end with: The answer is [number]."
    )


def _build_web_combined_hint(benchmark_name: str | None) -> str:
    """Build the web-search hint appended in combined mode."""
    desc, warning = _get_benchmark_desc(benchmark_name)
    return (
        "\n\n## Web search strategy\n"
        "You also have access to web search (serper_search) and page reading "
        "(jina_read_page) tools.\n"
        f"Note: You are answering {desc}. {warning}\n"
        "When you choose to use web search, follow this strategy:\n\n"
        "IMPORTANT: NEVER answer from memory alone for factual questions. "
        "Always search and verify.\n\n"
        "Strategy:\n"
        "1. Start with 2-3 diverse search queries to scope the problem from different angles.\n"
        "2. Visit at least 3-4 of the most promising results to read their full content. "
        "Cross-reference findings across multiple sources.\n"
        "3. If initial searches are insufficient, reformulate your queries using terminology "
        "or details discovered in earlier results. Try at least 2-3 rounds of search-then-visit.\n"
        "4. Only give your final answer when you have converging evidence from multiple "
        "independent sources. If sources conflict, dig deeper.\n\n"
        "Guidelines:\n"
        "- Prefer depth over breadth: visiting and carefully reading 3-4 high-quality pages "
        "beats skimming 10.\n"
        "- When a search returns no useful results, vary your query strategy: try different "
        "keywords, use quoted phrases for exact matches, or search for the specific subproblem "
        "rather than the full question.\n"
        "- For numerical, formula, or factual answers, always verify the exact value from a "
        "primary source (paper, textbook, official documentation) rather than relying on "
        "secondary summaries."
    )


def build_system_prompt(
    benchmark_name: str | None,
    base_prompt: str,
) -> str:
    """Return the assembled system prompt.

    Web-mode behaviour:
      - "only":     replace base_prompt with benchmark-specific web-only prompt
      - "combined": keep base_prompt + append benchmark-specific web hint
      - "off":      keep base_prompt unchanged
    """
    import os
    web_mode = os.environ.get("BIOAGENT_WEB_TOOLS", "off")

    cfg = get_config(benchmark_name)

    # For web-only mode, use the dedicated web-search prompt instead of base
    if web_mode == "only":
        parts = [_build_web_only_prompt(benchmark_name)]
    else:
        parts = [base_prompt]

    if cfg.system_prompt_hint:
        parts.append("\n\n## Task-specific guidance\n" + cfg.system_prompt_hint)
    if cfg.expected_answer_format:
        parts.append("\n\n## Expected answer format\n" + cfg.expected_answer_format)

    # For combined mode, append web-search guidance with benchmark context
    if web_mode == "combined":
        parts.append(_build_web_combined_hint(benchmark_name))

    # Always add deep-reasoning guidance for research-heavy benchmarks
    _research_heavy = {
        "hle_gold", "hle_bio_med_chem", "labbench_litqa2",
        "labbench_protocolqa_open", "labbench2", "bixbench_notebook",
        "genotex", "rag_essential",
    }
    if cfg.name in _research_heavy and web_mode == "off":
        parts.append(
            "\n\n## Research strategy\n"
            "These questions require thorough investigation. Do NOT answer from "
            "memory alone. Use your available tools iteratively:\n"
            "1. Start with broad searches to scope the problem.\n"
            "2. Based on initial results, make targeted follow-up queries.\n"
            "3. Cross-reference findings across multiple tool results.\n"
            "4. Only finalize your answer when you have converging evidence."
        )

    return "".join(parts)


def build_user_prompt(benchmark_name: str | None, task: dict) -> str:
    """Return the user-visible prompt. Uses config.context_formatter when
    present; otherwise falls back to harness.context_injection.format_task_prompt.
    """
    cfg = get_config(benchmark_name)
    if cfg.context_formatter is not None:
        return cfg.context_formatter(task)
    # Lazy import to avoid circular dependency.
    from harness.context_injection import format_task_prompt
    return format_task_prompt(task)


def filter_tools(
    benchmark_name: str | None,
    all_tool_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the subset of TOOL_SPECS advertised for this benchmark.

    Precedence:
      1. If config.tool_whitelist is not None → use whitelist (may be []).
      2. Else if config.tool_categories is non-empty → category-filter.
      3. Else → return all_tool_specs unchanged.

    The ``BIOAGENT_WEB_TOOLS`` env var controls web tool inclusion:
      - "off" (default): exclude web tools from the pool.
      - "only": return ONLY the web tools (serper_search + jina_read_page).
      - "combined": include web tools alongside the benchmark's category tools.
    """
    import os
    web_mode = os.environ.get("BIOAGENT_WEB_TOOLS", "off")

    # Lazy import to avoid circular dependency.
    from harness.tools.tool_categories import (
        get_tools_by_whitelist,
        get_tools_by_category,
    )

    # "only" mode: return exclusively web tools regardless of benchmark config
    if web_mode == "only":
        return get_tools_by_category(["web"], all_tool_specs)

    cfg = get_config(benchmark_name)

    if cfg.tool_whitelist is not None:
        base = get_tools_by_whitelist(cfg.tool_whitelist, all_tool_specs)
    elif cfg.tool_categories:
        base = get_tools_by_category(cfg.tool_categories, all_tool_specs)
    else:
        base = list(all_tool_specs)

    # "combined" mode: merge web tools into the benchmark's tool set
    if web_mode == "combined":
        web_tools = get_tools_by_category(["web"], all_tool_specs)
        # Avoid duplicates
        base_names = {_tool_name(s) for s in base}
        for wt in web_tools:
            if _tool_name(wt) not in base_names:
                base.append(wt)

    # "off" mode: strip web tools if they somehow got in
    if web_mode == "off":
        web_names = {"serper_search", "jina_read_page"}
        base = [s for s in base if _tool_name(s) not in web_names]

    return base


def _tool_name(spec: dict[str, Any]) -> str:
    """Extract tool name from a TOOL_SPECS entry."""
    if "function" in spec and isinstance(spec["function"], dict):
        return str(spec["function"].get("name") or "")
    return str(spec.get("name") or "")


# Import the registry module when the package is imported. Deferred to the
# bottom so register_config is defined first. Failures are tolerated so that
# unit tests of this module do not require the full registry.
try:
    from harness import benchmark_configs as _  # noqa: F401
except Exception:  # pragma: no cover — safety net for partial imports
    pass
