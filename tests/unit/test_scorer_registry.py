import pytest

from harness.scorers import SCORER_REGISTRY, validate_benchmark_config


def test_scorer_registry_contains_required_families():
    for name in [
        "mcq_exact_match",
        "pio_span_f1",
        "smiles_topk_canonical_match",
        "openai_simple_evals_rubric_judge",
        "agentclinic_dialogue_loop",
        "bixbench_agent_runner",
        "labbench_official_evaluator",
        "bioasq_official",
        "delta_ei",
        "chembench_official",
    ]:
        assert name in SCORER_REGISTRY


def test_validate_benchmark_config_catches_mcq_mapping():
    with pytest.raises(ValueError):
        validate_benchmark_config("bad_mcq", {"scorer": "mcq_exact_match", "answer_fields": ("answer",)})

    validate_benchmark_config(
        "good_mcq",
        {"scorer": "mcq_exact_match", "choice_fields": ("options",), "answer_fields": ("answer",)},
    )


def test_validate_agent_scorer_requires_external_readiness():
    with pytest.raises(ValueError):
        validate_benchmark_config("agentclinic", {"scorer": "agentclinic_dialogue_loop", "readiness": "official_ready"})

    validate_benchmark_config(
        "agentclinic",
        {"scorer": "agentclinic_dialogue_loop", "readiness": "needs_external_env"},
    )
