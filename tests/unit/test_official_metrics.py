import math

from harness.eval.official_metrics import compute_official_metrics


def test_moleculenet_classification_metrics_from_hard_labels():
    records = [
        {"expected": "0", "predicted": "0", "context": {"dataset_key": "hf_moleculenet_bbbp"}},
        {"expected": "0", "predicted": "1", "context": {"dataset_key": "hf_moleculenet_bbbp"}},
        {"expected": "1", "predicted": "1", "context": {"dataset_key": "hf_moleculenet_bbbp"}},
        {"expected": "1", "predicted": "1", "context": {"dataset_key": "hf_moleculenet_bbbp"}},
    ]

    metrics = compute_official_metrics("hf_moleculenet_bbbp", records)

    assert metrics["status"] == "ok"
    assert metrics["metric_type"] == "classification"
    assert metrics["accuracy"] == 0.75
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["prc_auc"] <= 1.0


def test_moleculenet_regression_metrics():
    records = [
        {"expected": "1.0", "predicted": "1.5", "context": {"dataset_key": "hf_moleculenet_esol"}},
        {"expected": "3.0", "predicted": "2.5", "context": {"dataset_key": "hf_moleculenet_esol"}},
    ]

    metrics = compute_official_metrics("hf_moleculenet_esol", records)

    assert metrics["status"] == "ok"
    assert metrics["metric_type"] == "regression"
    assert metrics["mae"] == 0.5
    assert metrics["rmse"] == 0.5
    assert math.isclose(metrics["spearman"], 1.0)


def test_mteb_retrieval_metrics_require_rankings_and_score_when_present():
    unavailable = compute_official_metrics(
        "hf_mteb_medical_retrieval",
        [{"expected": "doc1", "predicted": "some generated text", "context": {"query_id": "q1", "corpus_id": "doc1"}}],
    )
    assert unavailable["status"] == "unavailable"

    metrics = compute_official_metrics(
        "hf_mteb_medical_retrieval",
        [
            {
                "expected": '["doc1"]',
                "predicted": '["doc2", "doc1"]',
                "context": {"query_id": "q1", "relevant_doc_ids": ["doc1"]},
            }
        ],
    )

    assert metrics["status"] == "ok"
    assert metrics["recall@10"] == 1.0
    assert metrics["mrr@10"] == 0.5


def test_proteingym_uses_spearman_for_numeric_predictions():
    records = [
        {"expected": "0.1", "predicted": "0.2", "context": {"dataset_key": "hf_proteingym_v1", "assay_id": "a", "DMS_score_bin": 0}},
        {"expected": "0.4", "predicted": "0.3", "context": {"dataset_key": "hf_proteingym_v1", "assay_id": "a", "DMS_score_bin": 0}},
        {"expected": "0.9", "predicted": "0.7", "context": {"dataset_key": "hf_proteingym_v1", "assay_id": "a", "DMS_score_bin": 1, "uniprot_id": "u1"}},
        {"expected": "0.2", "predicted": "0.1", "context": {"dataset_key": "hf_proteingym_v1", "assay_id": "b", "DMS_score_bin": 0}},
        {"expected": "1.0", "predicted": "0.8", "context": {"dataset_key": "hf_proteingym_v1", "assay_id": "b", "DMS_score_bin": 1, "uniprot_id": "u1"}},
    ]

    metrics = compute_official_metrics("hf_proteingym_v1", records)

    assert metrics["status"] == "ok"
    assert metrics["metric_type"] == "protein_fitness_dms"
    assert math.isclose(metrics["spearman"], 1.0)
    assert math.isclose(metrics["ndcg@10%"], 1.0)
    assert math.isclose(metrics["top_recall@10%"], 1.0)
    assert math.isclose(metrics["auc"], 1.0)
    assert math.isclose(metrics["mcc"], 1.0)
    assert metrics["n_assays"] == 2
    assert math.isclose(metrics["uniprot_id_mean_spearman"], 1.0)


def test_rna_regression_reports_correlations_and_error():
    records = [
        {"expected": "1.0", "predicted": "1.1", "context": {"dataset_key": "hf_rna_mean_ribosome_load", "official_metric": "rna_regression_spearman_pearson"}},
        {"expected": "2.0", "predicted": "1.9", "context": {"dataset_key": "hf_rna_mean_ribosome_load", "official_metric": "rna_regression_spearman_pearson"}},
        {"expected": "3.0", "predicted": "3.2", "context": {"dataset_key": "hf_rna_mean_ribosome_load", "official_metric": "rna_regression_spearman_pearson"}},
    ]

    metrics = compute_official_metrics("hf_rna_mean_ribosome_load", records)

    assert metrics["status"] == "ok"
    assert metrics["metric_type"] == "regression"
    assert math.isclose(metrics["spearman"], 1.0)
    assert "pearson" in metrics
    assert "mae" in metrics


def test_rna_binary_multiclass_and_multilabel_metrics():
    binary = compute_official_metrics(
        "hf_rna_splice_site_acceptor",
        [
            {"expected": "0", "predicted": "0", "context": {"dataset_key": "hf_rna_splice_site_acceptor", "official_metric": "rna_binary_accuracy_f1_auroc"}},
            {"expected": "1", "predicted": "1", "context": {"dataset_key": "hf_rna_splice_site_acceptor", "official_metric": "rna_binary_accuracy_f1_auroc"}},
            {"expected": "1", "predicted": "0", "context": {"dataset_key": "hf_rna_splice_site_acceptor", "official_metric": "rna_binary_accuracy_f1_auroc"}},
        ],
    )
    multiclass = compute_official_metrics(
        "hf_rna_ncrna_family_bnoise0",
        [
            {"expected": "0", "predicted": "0", "context": {"dataset_key": "hf_rna_ncrna_family_bnoise0", "official_metric": "rna_multiclass_accuracy_macro_f1"}},
            {"expected": "2", "predicted": "1", "context": {"dataset_key": "hf_rna_ncrna_family_bnoise0", "official_metric": "rna_multiclass_accuracy_macro_f1"}},
        ],
    )
    multilabel = compute_official_metrics(
        "hf_rna_modification_site",
        [
            {"expected": "[1, 0, 1]", "predicted": "[1, 0, 0]", "context": {"dataset_key": "hf_rna_modification_site", "official_metric": "rna_multilabel_macro_micro_f1"}},
            {"expected": "[0, 1, 0]", "predicted": "[0, 1, 0]", "context": {"dataset_key": "hf_rna_modification_site", "official_metric": "rna_multilabel_macro_micro_f1"}},
        ],
    )

    assert binary["metric_type"] == "binary_classification"
    assert binary["accuracy"] == 2 / 3
    assert multiclass["metric_type"] == "multiclass_classification"
    assert multiclass["accuracy"] == 0.5
    assert multilabel["metric_type"] == "multilabel_classification"
    assert multilabel["n_labels"] == 3


def test_bacbench_dispatches_binary_and_regression_metrics():
    binary = compute_official_metrics(
        "hf_bacbench_antibiotic_resistance_dna",
        [
            {"expected": "0", "predicted": "0", "context": {"dataset_key": "hf_bacbench_antibiotic_resistance_dna", "official_metric": "bacbench_binary_auprc"}},
            {"expected": "1", "predicted": "1", "context": {"dataset_key": "hf_bacbench_antibiotic_resistance_dna", "official_metric": "bacbench_binary_auprc"}},
        ],
    )
    regression = compute_official_metrics(
        "hf_bacbench_phenotypic_traits_dna",
        [
            {"expected": "1.0", "predicted": "1.0", "context": {"dataset_key": "hf_bacbench_phenotypic_traits_dna", "official_metric": "bacbench_regression_r2"}},
            {"expected": "2.0", "predicted": "2.0", "context": {"dataset_key": "hf_bacbench_phenotypic_traits_dna", "official_metric": "bacbench_regression_r2"}},
        ],
    )

    assert binary["metric_type"] == "binary_classification"
    assert binary["prc_auc"] == 1.0
    assert regression["metric_type"] == "regression"
    assert regression["r2"] == 1.0


def test_bacbench_categorical_string_metrics():
    metrics = compute_official_metrics(
        "hf_bacbench_phenotypic_traits_dna",
        [
            {"expected": "positive", "predicted": "positive", "context": {"dataset_key": "hf_bacbench_phenotypic_traits_dna", "official_metric": "bacbench_multiclass_accuracy_macro_f1"}},
            {"expected": "aerobic", "predicted": "anaerobic", "context": {"dataset_key": "hf_bacbench_phenotypic_traits_dna", "official_metric": "bacbench_multiclass_accuracy_macro_f1"}},
        ],
    )

    assert metrics["status"] == "ok"
    assert metrics["metric_type"] == "categorical_classification"
    assert metrics["accuracy"] == 0.5
