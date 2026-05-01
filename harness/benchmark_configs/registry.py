"""Register per-benchmark harness configs.

Side-effect module: imported once, calls `register_config` for each
benchmark. Add new benchmarks here, not in `benchmark_config.py`.

The benchmark name keys MUST match the `benchmark_key` stamped onto
tasks by `BenchmarkSuite.eval_tasks`. Keys shown in parentheses below
are the runner-side iteration-budget keys where they differ from the
loader name.
"""
from __future__ import annotations

from harness.benchmark_config import BenchmarkHarnessConfig, register_config
from harness.eval.bench_aa_lcr import format_aa_lcr_prompt


# ---------------------------------------------------------------- medcalc
register_config(BenchmarkHarnessConfig(
    name="medcalc",
    system_prompt_hint=(
        "You are solving a medical calculation problem. "
        "Use the compute_calculator / calculator_eval tools for arithmetic "
        "and named clinical formulas (BMI, eGFR, Cockcroft-Gault, etc.). "
        "Prefer direct local calculator calls over tool-discovery helpers."
    ),
    tool_categories=["calculation", "clinical"],
    expected_answer_format=(
        "Final answer should be a single number with units where applicable, "
        "e.g., 'The answer is 52.3 mL/min/1.73m^2.'"
    ),
))


# ---------------------------------------------------------- medxpertqa (text)
register_config(BenchmarkHarnessConfig(
    name="medxpertqa",
    system_prompt_hint=(
        "You are solving an expert-level medical multiple-choice question. "
        "Consider each option carefully. When uncertain, search clinical "
        "literature (pubmed_search) or look up variants / drugs."
    ),
    tool_categories=["literature", "search", "clinical", "variant", "drug"],
    expected_answer_format=(
        "Final answer should be a single letter: A, B, C, D, or E."
    ),
))


# -------------------------------------------------------- medxpertqa_mm (VQA)
register_config(BenchmarkHarnessConfig(
    name="medxpertqa_mm",
    system_prompt_hint=(
        "You are solving a multimodal medical question. Examine the images "
        "carefully. Use clinical reference tools to confirm terminology or "
        "disease concepts."
    ),
    tool_categories=["clinical", "literature", "search"],
    expected_answer_format="Final answer should be a single letter.",
))


# ---------------------------------------------------------------- labbench v1
register_config(BenchmarkHarnessConfig(
    name="labbench",
    system_prompt_hint=(
        "You are solving a biomedical research question. "
        "Use bioinformatics tools liberally: gene / protein lookup, "
        "literature search, sequence utilities, chemistry tools."
    ),
    tool_categories=[
        "literature", "search", "protein", "genetics",
        "sequence", "chemistry", "drug",
    ],
    expected_answer_format=(
        "Final answer should be a single letter: A, B, C, D, or E."
    ),
))

# LAB-Bench subtask keys used by the iteration-budget map — alias to the
# same config so they pick up the identical specialisation.
for _sub in ("labbench_litqa2", "labbench_protocolqa_open",
              "labbench_cloning_scenarios"):
    register_config(BenchmarkHarnessConfig(
        name=_sub,
        system_prompt_hint=(
            "You are solving a biomedical research question. "
            "Use bioinformatics tools liberally: gene / protein lookup, "
            "literature search, sequence utilities, chemistry tools."
        ),
        tool_categories=[
            "literature", "search", "protein", "genetics",
            "sequence", "chemistry", "drug",
        ],
        expected_answer_format=(
            "Final answer should be a single letter: A, B, C, D, or E."
        ),
    ))


# ---------------------------------------------------------------- labbench2
register_config(BenchmarkHarnessConfig(
    name="labbench2",
    system_prompt_hint=(
        "You are solving a biomedical research question with retrieval context. "
        "The user message includes relevant sources and a key passage — "
        "read the key passage carefully, it contains the exact facts "
        "needed for the answer. Match the phrasing of the question."
    ),
    # Retrieval-first benchmark: context already injected. Keep a small
    # literature-oriented tool set for occasional follow-up searches.
    tool_categories=["literature", "search"],
    expected_answer_format=(
        "Answer directly with the specific fact from the key passage. "
        "Be concise — a short phrase or exact value."
    ),
))


# ---------------------------------------------------------------- bioasq
register_config(BenchmarkHarnessConfig(
    name="bioasq",
    system_prompt_hint=(
        "You are solving a biomedical question-answering task. Use pubmed_search "
        "to ground your answer in published literature when helpful."
    ),
    tool_categories=["literature", "search", "clinical"],
))


# --------------------------------------------------------- gpqa_bio (hle_bio)
# The runner's iteration budget uses key 'hle_bio_med_chem'; the matrix
# config passes that through as benchmark_key. Register both so either
# key resolves.
_gpqa_hint = (
    "You are solving a graduate-level biology / medicine / chemistry "
    "question. Think step by step. Use reference tools for domain "
    "facts you need to verify."
)
_gpqa_cats = ["literature", "search", "protein", "genetics",
               "chemistry", "clinical"]
_gpqa_fmt = "Final answer should be a single letter: A, B, C, or D."
for _nm in ("gpqa_bio", "hle_bio_med_chem"):
    register_config(BenchmarkHarnessConfig(
        name=_nm,
        system_prompt_hint=_gpqa_hint,
        tool_categories=_gpqa_cats,
        expected_answer_format=_gpqa_fmt,
    ))


# ---------------------------------------------------------------- pathvqa
register_config(BenchmarkHarnessConfig(
    name="pathvqa",
    system_prompt_hint=(
        "You are answering a pathology visual question. Examine the image "
        "carefully and answer concisely."
    ),
    # Pure vision — no tools.
    tool_whitelist=[],
    enable_retrieval=False,
))


# ---------------------------------------------------------------- bixbench
for _nm in ("bixbench", "bixbench_closed_book"):
    register_config(BenchmarkHarnessConfig(
        name=_nm,
        system_prompt_hint=(
            "You are solving a bioinformatics task. Use sequence tools, "
            "gene / protein lookup, and literature search as needed."
        ),
        tool_categories=[
            "protein", "genetics", "literature", "search", "sequence",
            "gene_expression",
        ],
    ))


# ---------------------------------------------------------------- medagentbench
register_config(BenchmarkHarnessConfig(
    name="medagentbench",
    system_prompt_hint=(
        "You are a medical AI agent handling an EHR / clinical workflow. "
        "Use clinical search, variant lookup, and calculation tools."
    ),
    tool_categories=["clinical", "calculation", "search", "variant", "drug"],
))


# ---------------------------------------------------------------- agentclinic
register_config(BenchmarkHarnessConfig(
    name="agentclinic",
    system_prompt_hint=(
        "You are a physician in a clinical simulation. Ask clarifying "
        "questions, order tests, and reach a diagnosis based on available "
        "history and findings."
    ),
    tool_categories=["clinical", "drug"],
))


for _nm in ("medqa", "medmcqa", "pubmedqa"):
    register_config(BenchmarkHarnessConfig(
        name=_nm,
        system_prompt_hint=(
            "You are solving a medical knowledge multiple-choice question. "
            "Use the provided options and clinical reasoning. Use literature "
            "search only when necessary."
        ),
        tool_categories=["literature", "search", "clinical"],
        expected_answer_format="Final answer should be a single letter.",
    ))


# ---------------------------------------------------------------- mmlu
register_config(BenchmarkHarnessConfig(
    name="mmlu",
    system_prompt_hint=(
        "You are solving a multiple-choice knowledge question. Choose the best "
        "answer from the provided options."
    ),
    tool_whitelist=[],
    expected_answer_format="Final answer should be a single letter: A, B, C, or D.",
    enable_retrieval=False,
))


# ---------------------------------------------------------------- healthbench
register_config(BenchmarkHarnessConfig(
    name="healthbench",
    system_prompt_hint=(
        "You are responding to a realistic health conversation. Provide a "
        "clinically cautious, helpful, and context-aware answer to the final "
        "user message. Do not claim certainty where the prompt is underspecified."
    ),
    tool_categories=["clinical", "literature", "search"],
    expected_answer_format=(
        "Answer naturally and concisely. Include emergency or clinician-seeking "
        "guidance when medically appropriate."
    ),
))


# ---------------------------------------------------------------- bioprobench
register_config(BenchmarkHarnessConfig(
    name="bioprobench",
    system_prompt_hint=(
        "You are solving a biological protocol analysis task. Pay close "
        "attention to order, reagent identity, concentrations, timing, and "
        "experimental purpose."
    ),
    tool_categories=["literature", "search", "protein", "genetics", "chemistry"],
    expected_answer_format=(
        "For multiple choice, return a single letter. For protocol tasks, "
        "return the exact corrected step, ordered steps, or requested protocol."
    ),
))


# ---------------------------------------------------------------- medhelm
register_config(BenchmarkHarnessConfig(
    name="medhelm",
    system_prompt_hint=(
        "You are solving a medical reasoning task. "
        "Use clinically grounded reasoning and follow the requested answer "
        "format exactly."
    ),
    tool_categories=["clinical", "literature", "search", "drug"],
))


# ------------------------------------------------------------- generic HF set
try:
    from harness.eval.hf_benchmark_registry import HF_BENCHMARK_SPECS

    _HF_DOMAIN_CATEGORIES = {
        "medical": ["clinical", "literature", "search", "drug"],
        "clinical": ["clinical", "literature", "search"],
        "healthcare": ["clinical", "literature", "search"],
        "biomedical": ["literature", "search", "clinical"],
        "chemistry": ["chemistry", "drug", "literature", "search"],
        "protein": ["protein", "sequence", "literature", "search"],
        "genomics": ["genetics", "gene_expression", "sequence", "search"],
        "dna": ["genetics", "sequence", "search"],
        "rna": ["genetics", "sequence", "gene_expression", "search"],
    }
    for _hf_key, _hf_spec in HF_BENCHMARK_SPECS.items():
        register_config(BenchmarkHarnessConfig(
            name=_hf_key,
            system_prompt_hint=(
                f"You are solving a {_hf_spec.domain} domain question. Follow the "
                "requested answer format and use domain tools when helpful."
            ),
            tool_categories=_HF_DOMAIN_CATEGORIES.get(
                _hf_spec.domain,
                ["literature", "search"],
            ),
            expected_answer_format=(
                "For multiple choice, return a single letter. For other tasks, "
                "return a concise answer matching the gold output."
            ),
        ))
except Exception:
    HF_BENCHMARK_SPECS = {}


# ---------------------------------------------------------------- rag_essential
register_config(BenchmarkHarnessConfig(
    name="rag_essential",
    system_prompt_hint=(
        "You are solving a retrieval-augmented question that rewards "
        "evidence grounding. Consult literature and reference tools."
    ),
    tool_categories=["literature", "search", "clinical"],
))


# ---------------------------------------------------------------- genotex
register_config(BenchmarkHarnessConfig(
    name="genotex",
    system_prompt_hint=(
        "You are solving a gene-expression / transcriptomics task. "
        "Use gene lookup, expression atlas, and eQTL tools "
        "(Bgee, GTEx, ArrayExpress, Human Protein Atlas, ENCODE)."
    ),
    tool_categories=[
        "gene_expression", "genetics", "protein", "search",
        "ontology",
    ],
))


# ---------------------------------------------------------------- hle_gold
register_config(BenchmarkHarnessConfig(
    name="hle_gold",
    system_prompt_hint=(
        "You are solving a PhD-level biology, medicine, or chemistry "
        "question. Think carefully step by step. Use literature search "
        "and domain tools (gene/protein lookup, chemistry databases, "
        "ontology) as needed. These are graduate/expert questions — "
        "precision matters."
    ),
    tool_categories=[
        "literature", "search", "protein", "genetics", "chemistry",
        "clinical", "variant", "drug", "ontology",
    ],
    expected_answer_format=(
        "For MCQ: a single letter (A-E). "
        "For open-ended: a concise, precise answer (1-2 sentences max). "
        "End with 'The answer is: <your answer>'."
    ),
))


# ---------------------------------------------------------------- super_chemistry
register_config(BenchmarkHarnessConfig(
    name="super_chemistry",
    system_prompt_hint=(
        "You are solving a multiple-choice chemistry question. "
        "Each question has between 4 and 26 options (labelled A-Z); exactly "
        "one option is correct. Many questions include molecular structure "
        "images — reason about the structures when present. Work through "
        "the mechanism, pKa, stereochemistry, or thermodynamics step by step."
    ),
    tool_categories=[
        "chemistry", "search", "literature", "ontology",
    ],
    expected_answer_format=(
        "For MCQ: a single letter (A-Z). "
        "End with 'The answer is: <letter>'."
    ),
))


# ---------------------------------------------------------------- aa_lcr
register_config(BenchmarkHarnessConfig(
    name="aa_lcr",
    system_prompt_hint=(
        "You are solving a long-context reasoning question. The prompt "
        "includes the relevant document set. Use only those documents, "
        "synthesize evidence across them, and avoid outside knowledge "
        "unless explicitly requested."
    ),
    tool_whitelist=[],
    context_formatter=format_aa_lcr_prompt,
    expected_answer_format=(
        "Return a concise final answer matching the requested format. "
        "Do not include citations unless the question asks for them."
    ),
    enable_retrieval=False,
))


# --------------------------------------------------------------- quick_suite
register_config(BenchmarkHarnessConfig(
    name="quick_suite",
    system_prompt_hint=(
        "You are solving a short biomedical knowledge task. "
        "Answer in the requested format; do not use external tools unless "
        "a calculation is explicitly needed."
    ),
    tool_categories=["calculation"],
    expected_answer_format=(
        "For multiple choice, return a single letter. For numeric tasks, "
        "return a number. For text tasks, return a concise phrase."
    ),
))


# Convenience: which benchmark names this module registered — useful in
# tests and the BRIEF_B_DONE report.
REGISTERED_BENCHMARKS: tuple[str, ...] = (
    "medcalc", "medxpertqa", "medxpertqa_mm", "labbench",
    "labbench_litqa2", "labbench_protocolqa_open",
    "labbench_cloning_scenarios", "labbench2", "bioasq",
    "gpqa_bio", "hle_bio_med_chem", "pathvqa", "bixbench",
    "bixbench_closed_book", "medagentbench", "agentclinic",
    "medqa", "medmcqa", "pubmedqa", "mmlu",
    "healthbench", "bioprobench", "medhelm",
    "rag_essential", "genotex", "hle_gold",
    "super_chemistry", "aa_lcr", "quick_suite",
) + tuple(HF_BENCHMARK_SPECS)
