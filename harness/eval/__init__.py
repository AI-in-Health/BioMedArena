from harness.eval.scoring import extract_answer_from_response, score_open_text
from harness.eval.metrics import BenchmarkMetrics, QuestionMetric, build_metrics
from harness.eval.benchmark_suite import BenchmarkSuite
from harness.eval.bench_medagentbench import load_medagentbench_tasks
from harness.eval.bench_agentclinic import load_agentclinic_tasks
from harness.eval.bench_genotex import load_genotex_tasks
from harness.eval.bench_medcalc import load_medcalc_tasks
from harness.eval.bench_medxpertqa import load_medxpertqa_tasks
from harness.eval.bench_medxpertqa_mm import load_medxpertqa_mm_tasks
from harness.eval.bench_rag_essential import load_rag_essential_tasks
from harness.eval.bench_labbench import load_labbench_tasks
from harness.eval.bench_labbench2 import (
    load_labbench2_tasks,
    register_tool_specs as _register_labbench2_tool_specs,
)
from harness.eval.labbench2_scorer import score_labbench2_regex
from harness.eval.bench_bioasq import load_bioasq_tasks
from harness.eval.bench_gpqa_bio import load_gpqa_bio_tasks
from harness.eval.bench_pathvqa import load_pathvqa_tasks
from harness.eval.bench_hle_gold import load_hle_gold_tasks
from harness.eval.bench_bixbench import load_bixbench_tasks
from harness.eval.bench_super_chemistry import load_super_chemistry_tasks
from harness.eval.bench_aa_lcr import load_aa_lcr_tasks
from harness.eval.bench_healthbench import load_healthbench_tasks
from harness.eval.bench_mmlu import load_mmlu_tasks
from harness.eval.bench_bioprobench import load_bioprobench_tasks
from harness.eval.bench_medhelm import load_medhelm_tasks
from harness.eval.bench_hf_benchmark import load_hf_benchmark_tasks
from harness.eval.bench_quick_suite import load_quick_suite_tasks
from harness.eval.bench_superchem import load_superchem_tasks
from harness.eval.bench_supergpqa import load_supergpqa_tasks
# Combined Medical-QA loader (MedQA + MedMCQA + PubMedQA)
from harness.eval.bench_medical_qa import load_medical_qa_tasks
from harness.eval.self_consistency import self_consistent_run

# OpenAI-plugins life-science skills (merge-safe registration: handlers
# live in harness/tools/openai_ported/, registered via extend() here —
# no edits to function_calling_runner.py).
try:
    from harness.tools.openai_ported import register_openai_ported as _register_olsp
    _register_olsp()
except Exception:
    pass

# MERGE-SAFETY: register benchmark-specific tool specs via explicit
# hooks rather than by editing function_calling_runner.py directly.
_register_labbench2_tool_specs()
