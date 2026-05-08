"""Explicit scorer registry used by benchmark metadata and validation.

The harness still routes per-question scoring through ``harness.eval.scoring``
for lightweight built-ins. This registry gives benchmark configs a stable
public scorer name and records which scorers require optional packages or
external official runners.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScorerSpec:
    name: str
    family: str
    requires_external_env: bool = False
    requires_optional_package: str | None = None
    notes: str = ""


SCORER_REGISTRY: dict[str, ScorerSpec] = {
    # exact match family
    "mcq_exact_match": ScorerSpec("mcq_exact_match", "exact"),
    "yesno_macro_f1": ScorerSpec("yesno_macro_f1", "exact"),
    "classification_accuracy": ScorerSpec("classification_accuracy", "exact"),

    # generation family
    "rouge_bertscore": ScorerSpec("rouge_bertscore", "generation", requires_optional_package="evaluate"),
    "llm_as_judge": ScorerSpec("llm_as_judge", "generation", notes="Requires configured judge model for reproducibility."),

    # IE family
    "ner_entity_f1": ScorerSpec("ner_entity_f1", "ie"),
    "pio_span_f1": ScorerSpec("pio_span_f1", "ie"),
    "relation_classification_f1": ScorerSpec("relation_classification_f1", "ie"),

    # chemistry family
    "smiles_topk_canonical_match": ScorerSpec("smiles_topk_canonical_match", "chemistry", requires_optional_package="rdkit"),
    "smiles_validity_plus_tanimoto": ScorerSpec("smiles_validity_plus_tanimoto", "chemistry", requires_optional_package="rdkit"),

    # regression family
    "regression_spearman_auc": ScorerSpec("regression_spearman_auc", "regression"),

    # rubric / official runner family
    "openai_simple_evals_rubric_judge": ScorerSpec(
        "openai_simple_evals_rubric_judge", "rubric", requires_external_env=True,
        notes="Use OpenAI simple-evals HealthBench grader; grader model must be configured.",
    ),
    "agentclinic_dialogue_loop": ScorerSpec("agentclinic_dialogue_loop", "agent", requires_external_env=True),
    "bixbench_agent_runner": ScorerSpec("bixbench_agent_runner", "agent", requires_external_env=True),
    "labbench_official_evaluator": ScorerSpec(
        "labbench_official_evaluator", "official_runner", requires_external_env=True,
        requires_optional_package="labbench",
    ),

    # benchmark-specific
    "bioasq_official": ScorerSpec("bioasq_official", "benchmark_specific"),
    "delta_ei": ScorerSpec("delta_ei", "benchmark_specific", requires_optional_package="evidence_inference"),
    "chembench_official": ScorerSpec("chembench_official", "benchmark_specific", requires_optional_package="chembench"),
    "custom_re_with_silver_warning": ScorerSpec(
        "custom_re_with_silver_warning", "benchmark_specific",
        notes="Silver-standard relation extraction; do not compare as a fully canonical leaderboard metric.",
    ),
}


def validate_benchmark_config(bench_id: str, config: dict[str, Any]) -> None:
    """Validate scorer/readiness consistency before a benchmark is exposed."""
    scorer = str(config.get("scorer") or "")
    if not scorer:
        return
    if scorer not in SCORER_REGISTRY:
        raise ValueError(f"{bench_id}: unknown scorer {scorer!r}")

    if scorer == "mcq_exact_match":
        has_options = bool(config.get("options_field") or config.get("choice_fields"))
        has_answer = bool(config.get("answer_field") or config.get("answer_fields"))
        if not (has_options and has_answer):
            raise ValueError(f"{bench_id}: MCQ scorer needs options_field/choice_fields and answer_field/answer_fields")

    if "smiles" in scorer:
        try:
            from rdkit import Chem  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(f"{bench_id}: {scorer} requires rdkit; install rdkit-pypi or rdkit") from exc

    if "rubric_judge" in scorer and not config.get("grader_model"):
        raise ValueError(f"{bench_id}: rubric judge requires grader_model, e.g. gpt-4.1")

    if SCORER_REGISTRY[scorer].family == "agent" and config.get("readiness") != "needs_external_env":
        raise ValueError(f"{bench_id}: agent benchmark must declare readiness='needs_external_env'")


__all__ = ["SCORER_REGISTRY", "ScorerSpec", "validate_benchmark_config"]
