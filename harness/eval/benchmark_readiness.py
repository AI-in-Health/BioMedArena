"""Benchmark split/count/readiness audit helpers."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from harness.eval.hf_benchmark_registry import HF_BENCHMARK_SPECS, HF_DEPRECATED_ALIASES, HF_REMOVED_NONBENCHMARK_KEYS


DATASET_LEVEL_METRIC_KEYS = {
    "hf_mteb_medical_qa",
    "hf_mteb_medical_retrieval",
    "hf_proteingym_v01",
    "hf_proteingym_v1",
    "hf_icml2022_proteingym",
    "hf_genbio_proteingym_dms",
    "hf_rna_downstream_tasks",
    "hf_rna_expression_hek",
    "hf_rna_expression_muscle",
    "hf_rna_expression_pc3",
    "hf_rna_mean_ribosome_load",
    "hf_rna_modification_site",
    "hf_rna_ncrna_family_bnoise0",
    "hf_rna_splice_site_acceptor",
    "hf_rna_splice_site_donor",
    "hf_bacbench_antibiotic_resistance_dna",
    "hf_bacbench_phenotypic_traits_dna",
}

EXTERNAL_ENV_KEYS = {
    "medagentbench": "offline text mode loads; official workflow scoring needs a FHIR mock server",
    "agentclinic": "official scoring requires the multi-turn AgentClinic doctor-patient simulator",
    "bixbench": "official agent benchmark requires notebook/data-file execution sandbox",
}

VISION_OR_MULTIMODAL_KEYS = {
    "pathvqa",
    "medxpertqa_mm",
}

OFFICIAL_RUNNER_GAP_KEYS: dict[str, str] = {}

JUDGE_OR_RUBRIC_KEYS = {
    "aa_lcr": "open-ended long-context QA uses harness LLM judge",
    "labbench2": "text-only subsets load; multimodal/file subtasks are skipped and open answers use configured scorer/judge",
    "rag_essential": "open-ended RAG questions use harness LLM judge",
}

OFFICIAL_EVALUATOR_KEYS = {
    "healthbench": "requires OpenAI simple-evals rubric grader configuration",
    "labbench": "requires Future-House labbench official evaluator; vision subtasks need a vision backend",
    "bioasq": "uses BioASQ per-type official metrics: yes/no macro-F1, factoid MRR, list F1, ideal ROUGE-L",
    "hf_ebm_nlp": "PIO span extraction with token-level F1",
    "hf_bigbio_pubmed_qa": "PubMedQA final_decision exact accuracy",
    "hf_chembench": "ChemBench multiple-choice evaluator-compatible scoring",
    "hf_uspto_reaction_prediction": "SMILES canonical/top-k product matching",
    "hf_lpm24_eval_molgen": "SMILES validity/exact/tanimoto metrics",
    "hf_protein_binding_sequences": "protein binding aggregate regression/classification metrics",
}

GATED_OR_TOKEN_KEYS = {
    "gpqa_bio",
    "hle_gold",
}


def readiness_for_benchmark(key: str, meta: dict[str, Any] | None = None) -> tuple[str, str]:
    """Return (readiness_label, reason) for the current codebase."""
    meta = meta or {}
    spec = HF_BENCHMARK_SPECS.get(key)
    if key in HF_REMOVED_NONBENCHMARK_KEYS:
        return "not_user_selectable", "removed from benchmark registry: training-only, duplicate, or non-public canonical benchmark"
    if key in HF_DEPRECATED_ALIASES:
        return "deprecated_alias", f"deprecated alias; use {HF_DEPRECATED_ALIASES[key]}"
    if key in GATED_OR_TOKEN_KEYS or (spec and spec.extra.get("gated")):
        return "needs_token_or_external_env", "gated dataset or explicit HF access token required"
    if key in EXTERNAL_ENV_KEYS:
        return "needs_token_or_external_env", EXTERNAL_ENV_KEYS[key]
    if key in OFFICIAL_RUNNER_GAP_KEYS:
        return "loader_ready_metric_custom", OFFICIAL_RUNNER_GAP_KEYS[key]
    if key in VISION_OR_MULTIMODAL_KEYS or (spec and spec.extra.get("multimodal")):
        return "vision_ready_needs_vision_backend", "image inputs load and require a vision-capable backend for real evaluation"
    if key in OFFICIAL_EVALUATOR_KEYS:
        return "official_ready", OFFICIAL_EVALUATOR_KEYS[key]
    if key in JUDGE_OR_RUBRIC_KEYS:
        return "loader_ready_metric_custom", JUDGE_OR_RUBRIC_KEYS[key]
    if key in {"hf_mteb_medical_qa", "hf_mteb_medical_retrieval"}:
        return "official_ready", "candidate reranking tasks emit ranked doc ids and report nDCG/MRR/Recall"
    if key in {"hf_bacbench_antibiotic_resistance_dna", "hf_bacbench_phenotypic_traits_dna"}:
        return "official_ready", "official label CSV join and aggregate metrics are wired; whole-genome prompts are length-capped"
    if key in DATASET_LEVEL_METRIC_KEYS:
        return "official_ready", "dataset-level predictive metric support is registered"
    if spec and spec.extra.get("scorer") in {
        "mcq_exact_match", "smiles_topk_canonical_match", "smiles_validity_plus_tanimoto",
        "regression_spearman_auc", "pio_span_f1", "delta_ei", "custom_re_with_silver_warning",
    }:
        if spec.extra.get("scorer") in {"delta_ei", "custom_re_with_silver_warning"}:
            return "loader_ready_metric_custom", f"requires benchmark-specific scorer {spec.extra['scorer']}"
        return "official_ready", f"explicit scorer configured: {spec.extra['scorer']}"
    if spec and spec.task_type in {"molecule_property", "protein_fitness", "regression"}:
        return "official_ready", "dataset-level predictive metric support is registered"
    if spec and spec.task_type in {"qa", "summarization", "text", "sequence"}:
        return "loader_ready_metric_custom", "generation/open-text scoring is harness-defined unless official judge is wired"
    if meta.get("answer_type") == "openText":
        return "loader_ready_metric_custom", "open-text scoring uses harness judge or heuristic"
    return "official_ready", "loader and scorer match a standard task-level metric"


def registered_benchmark_metadata(key: str, cli_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = HF_BENCHMARK_SPECS.get(key)
    if spec:
        data = asdict(spec)
        data["source"] = spec.repo
        data["source_url"] = f"https://huggingface.co/datasets/{spec.repo}"
        data["official_split"] = (
            spec.extra.get("parquet_split")
            or spec.split
            or "auto(test/validation/dev/train)"
        )
        data["needs_hf_token"] = bool(spec.extra.get("gated", False))
        data["needs_external_env"] = bool(
            spec.extra.get("raw_urls")
            or spec.extra.get("archive")
            or spec.extra.get("remote_files")
        )
        data["input_modality"] = "multimodal" if spec.extra.get("multimodal") else "text/structured"
        label, reason = readiness_for_benchmark(key, {"answer_type": data.get("answer_type")})
        data["readiness"] = label
        data["readiness_reason"] = reason
        return data

    cli_meta = cli_meta or {}
    label, reason = readiness_for_benchmark(key, cli_meta)
    return {
        "key": key,
        "source": cli_meta.get("benchmark_key", key),
        "source_url": "",
        "domain": "core",
        "task_type": cli_meta.get("loader", "").replace("load_", "").replace("_tasks", ""),
        "answer_type": "",
        "scorer": cli_meta.get("scorer", ""),
        "official_split": "custom loader",
        "needs_hf_token": key in GATED_OR_TOKEN_KEYS,
        "needs_external_env": key in EXTERNAL_ENV_KEYS,
        "input_modality": "multimodal" if key in VISION_OR_MULTIMODAL_KEYS else "text/structured",
        "readiness": label,
        "readiness_reason": reason,
    }


def probe_hf_count(key: str, *, cache_dir: str = "data/cache/huggingface") -> tuple[int | None, str]:
    """Best-effort split count without materialising full datasets."""
    spec = HF_BENCHMARK_SPECS.get(key)
    if spec is None:
        return None, "not_hf"
    try:
        from datasets import load_dataset_builder
    except ImportError:
        return None, "datasets_not_installed"

    repo = spec.extra.get("loader_repo") or spec.repo
    config = spec.config
    if spec.extra.get("parquet_glob"):
        return None, "direct_file_count_requires_load"
    if spec.extra.get("raw_urls") or spec.extra.get("archive") or spec.extra.get("files"):
        return None, "custom_file_loader"
    try:
        configs = spec.extra.get("configs") or ()
        if configs:
            total = 0
            counted = 0
            split_names: list[str] = []
            for cfg in configs:
                builder = load_dataset_builder(repo, cfg, cache_dir=cache_dir)
                splits = getattr(builder.info, "splits", None) or {}
                preferred = spec.split or "test"
                chosen = None
                if preferred in splits:
                    chosen = preferred
                else:
                    for candidate in ("test", "validation", "valid", "dev", "eval", "train"):
                        if candidate in splits:
                            chosen = candidate
                            break
                if chosen:
                    total += int(splits[chosen].num_examples)
                    counted += 1
                    split_names.append(f"{cfg}:{chosen}")
            if counted:
                return total, "+".join(split_names)
            return None, "no_split_metadata"
        builder = load_dataset_builder(repo, config, cache_dir=cache_dir) if config else load_dataset_builder(repo, cache_dir=cache_dir)
        splits = getattr(builder.info, "splits", None) or {}
        preferred = spec.split or "test"
        if preferred in splits:
            return int(splits[preferred].num_examples), preferred
        for candidate in ("test", "validation", "valid", "dev", "eval", "train"):
            if candidate in splits:
                return int(splits[candidate].num_examples), candidate
        if splits:
            name, split_info = next(iter(splits.items()))
            return int(split_info.num_examples), str(name)
        return None, "no_split_metadata"
    except Exception as exc:
        return None, f"count_probe_failed: {exc}"
