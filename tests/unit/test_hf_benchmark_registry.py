from types import SimpleNamespace
import json


def test_hf_registry_adds_verified_cli_benchmarks():
    from harness.cli import BENCHMARKS
    from harness.eval.hf_benchmark_registry import (
        HF_DEPRECATED_ALIASES,
        HF_BENCHMARK_SPECS,
        HF_VERIFIED_BENCHMARK_KEYS,
    )

    hf_cli_keys = {key for key in BENCHMARKS if key.startswith("hf_")}

    assert len(HF_BENCHMARK_SPECS) >= len(HF_VERIFIED_BENCHMARK_KEYS)
    assert len(HF_VERIFIED_BENCHMARK_KEYS) >= 100
    assert HF_VERIFIED_BENCHMARK_KEYS <= hf_cli_keys
    assert set(HF_DEPRECATED_ALIASES) & hf_cli_keys
    assert BENCHMARKS["hf_medquad"]["loader"] == "load_hf_benchmark_tasks"


def test_registry_removes_training_only_and_keeps_deprecated_aliases():
    from harness.cli import BENCHMARKS
    from harness.eval.hf_benchmark_registry import (
        HF_BENCHMARK_SPECS,
        HF_DEPRECATED_ALIASES,
        HF_VERIFIED_BENCHMARK_KEYS,
        TRAINING_DATA_SPECS,
    )

    for key in ("hf_asclepius_clinical_notes", "hf_augmented_clinical_notes", "hf_icliniq_10k", "hf_healthcare_data"):
        assert key not in HF_BENCHMARK_SPECS
        assert key not in HF_VERIFIED_BENCHMARK_KEYS
        assert key not in BENCHMARKS
        assert TRAINING_DATA_SPECS[key].extra["is_training_only"] is True

    assert HF_DEPRECATED_ALIASES["hf_chinese_medbench"] == "hf_cmb"
    assert BENCHMARKS["hf_chinese_medbench"]["deprecated_alias_for"] == "hf_cmb"
    assert "hf_medsts" not in HF_VERIFIED_BENCHMARK_KEYS
    assert "hf_clinical_sts" not in HF_VERIFIED_BENCHMARK_KEYS


def test_hf_loader_can_use_explicit_streaming(monkeypatch, tmp_path):
    calls = []
    rows = [
        {
            "id": "q1",
            "question": "Which option is correct?",
            "options": ["alpha", "beta", "gamma", "delta"],
            "answer": "B",
        }
    ]

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return rows

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )

    from harness.eval.bench_hf_benchmark import load_hf_benchmark_tasks

    tasks = load_hf_benchmark_tasks(
        dataset_key="hf_medmcqa_explanations",
        split="train",
        limit=1,
        streaming=True,
        cache_dir=tmp_path,
    )

    assert calls[0][1]["streaming"] is True
    assert tasks[0]["answer_type"] == "multipleChoice"
    assert tasks[0]["context"]["answer_type"] == "multipleChoice"
    assert tasks[0]["context"]["scorer_kind"] == "mcq"
    assert tasks[0]["answer"] == "B"


def test_hf_loader_normalises_qa_and_classification(monkeypatch, tmp_path):
    rows_by_repo = {
        "keivalya/MedQuad-MedicalQnADataset": [
            {
                "Question": "What is hypertension?",
                "Answer": "High blood pressure.",
            }
        ],
        "pietrolesci/pubmed-200k-rct": [
            {
                "abstract": "We tested a treatment in a randomized trial.",
                "label": "METHODS",
            }
        ],
    }

    def fake_load_dataset(repo, *args, **kwargs):
        return rows_by_repo[repo]

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )

    from harness.eval.bench_hf_benchmark import load_hf_benchmark_tasks

    qa_tasks = load_hf_benchmark_tasks(dataset_key="hf_medquad", limit=1, cache_dir=tmp_path)
    cls_tasks = load_hf_benchmark_tasks(
        dataset_key="hf_pubmed_200k_rct",
        limit=1,
        cache_dir=tmp_path,
    )

    assert qa_tasks[0]["answer_type"] == "openText"
    assert qa_tasks[0]["scorer_kind"] == "llm_judge"
    assert qa_tasks[0]["context"]["scorer_kind"] == "llm_judge"
    assert cls_tasks[0]["answer_type"] == "exactMatch"
    assert cls_tasks[0]["context"]["answer_type"] == "exactMatch"
    assert cls_tasks[0]["answer"] == "METHODS"


def test_hf_loader_parses_json_options_and_one_based_answers(monkeypatch, tmp_path):
    rows = [
        {
            "question": "Which option is best?",
            "options": '["alpha", "beta", "gamma", "delta"]',
            "answer": "2",
        }
    ]

    def fake_load_dataset(*args, **kwargs):
        return rows

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )

    from harness.eval.bench_hf_benchmark import load_hf_benchmark_tasks

    tasks = load_hf_benchmark_tasks(dataset_key="hf_medmcqa_explanations", limit=1, cache_dir=tmp_path)

    assert tasks[0]["choices"] == ["alpha", "beta", "gamma", "delta"]
    assert tasks[0]["answer"] == "B"


def test_blurb_file_loader_uses_token_f1(monkeypatch, tmp_path):
    sample = tmp_path / "ner.tsv"
    sample.write_text("alpha\tO\nbeta\tB-GENE\n\n", encoding="utf-8")

    import harness.eval.bench_hf_benchmark as bench

    monkeypatch.setattr(bench, "_download_url", lambda url, cache_path: sample)

    tasks = bench.load_hf_benchmark_tasks(dataset_key="hf_bc5cdr", limit=1, cache_dir=tmp_path)

    assert tasks[0]["answer_type"] == "tokenSequence"
    assert tasks[0]["scorer_kind"] == "token_f1"
    assert json.loads(tasks[0]["answer"]) == ["O", "B-GENE"]


def test_bioinfer_file_loader_uses_relation_f1(monkeypatch, tmp_path):
    sample = tmp_path / "bioinfer.xml"
    sample.write_text(
        '<corpus><document><sentence id="s1" text="A binds B.">'
        '<entity id="e1" type="Individual_protein" text="A" charOffset="0-1" />'
        '<entity id="e2" type="Individual_protein" text="B" charOffset="8-9" />'
        '<interaction id="i1" type="PPI" e1="e1" e2="e2" />'
        "</sentence></document></corpus>",
        encoding="utf-8",
    )

    import harness.eval.bench_hf_benchmark as bench

    monkeypatch.setattr(bench, "_download_url", lambda url, cache_path: sample)

    tasks = bench.load_hf_benchmark_tasks(dataset_key="hf_ppi_benchmark", limit=1, cache_dir=tmp_path)

    assert tasks[0]["answer_type"] == "relationSet"
    assert tasks[0]["scorer_kind"] == "relation_f1"
    assert json.loads(tasks[0]["answer"])[0]["type"] == "PPI"


def test_generic_ner_loader_uses_token_f1_for_sequence_labels(monkeypatch, tmp_path):
    rows = [
        {
            "id": "s1",
            "tokens": ["A", "gene", "appears"],
            "ner_tags": [0, 1, 1],
        }
    ]

    def fake_load_dataset(*args, **kwargs):
        return rows

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )

    from harness.eval.bench_hf_benchmark import load_hf_benchmark_tasks

    tasks = load_hf_benchmark_tasks(dataset_key="hf_bc2gm", limit=1, cache_dir=tmp_path)

    assert tasks[0]["answer_type"] == "tokenSequence"
    assert tasks[0]["scorer_kind"] == "token_f1"
    assert json.loads(tasks[0]["answer"]) == ["O", "B-GENE", "I-GENE"]


def test_litcovid_loader_uses_multilabel_f1(monkeypatch, tmp_path):
    rows = [
        {
            "pmid": "p1",
            "text": "COVID-19 treatment article",
            "label": [1, 0, 1, 0, 0, 0, 0],
        }
    ]

    def fake_load_dataset(*args, **kwargs):
        return rows

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )

    from harness.eval.bench_hf_benchmark import load_hf_benchmark_tasks

    tasks = load_hf_benchmark_tasks(dataset_key="hf_litcovid", limit=1, cache_dir=tmp_path)

    assert tasks[0]["answer_type"] == "multiLabel"
    assert tasks[0]["scorer_kind"] == "multilabel_f1"
    assert json.loads(tasks[0]["answer"]) == ["1", "0", "1", "0", "0", "0", "0"]


def test_mteb_loader_builds_ranked_doc_id_tasks(monkeypatch, tmp_path):
    def fake_load_dataset(repo, config, *args, **kwargs):
        if config == "queries":
            return [{"_id": "q1", "text": "heart attack treatment"}]
        if config == "corpus":
            return [
                {"_id": "d1", "title": "Relevant", "text": "Treatment for myocardial infarction."},
                {"_id": "d2", "title": "Distractor", "text": "Unrelated dermatology passage."},
                {"_id": "d3", "title": "Distractor 2", "text": "General wellness passage."},
            ]
        if config == "default":
            return [{"query-id": "q1", "corpus-id": "d1", "score": 1}]
        raise AssertionError(config)

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )

    from harness.eval.bench_hf_benchmark import load_hf_benchmark_tasks

    tasks = load_hf_benchmark_tasks(dataset_key="hf_mteb_medical_qa", limit=1, cache_dir=tmp_path)

    assert tasks[0]["answer_type"] == "ranking"
    assert tasks[0]["scorer_kind"] == "retrieval_hit"
    assert json.loads(tasks[0]["answer"]) == ["d1"]
    assert tasks[0]["context"]["relevant_doc_ids"] == ["d1"]
    assert set(tasks[0]["context"]["candidate_doc_ids"]) == {"d1", "d2", "d3"}
    assert "Return a JSON list of document ids" in tasks[0]["question"]


def test_proteingym_loader_preserves_assay_metadata(monkeypatch, tmp_path):
    rows = [
        {
            "DMS_score": 0.5,
            "DMS_score_bin": 1,
            "mutated_sequence": "ACD",
            "target_seq": "AAD",
            "mutant": "A2C",
            "DMS_id": "ASSAY1",
        }
    ]

    def fake_load_dataset(*args, **kwargs):
        return rows

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )

    from harness.eval.bench_hf_benchmark import load_hf_benchmark_tasks

    tasks = load_hf_benchmark_tasks(dataset_key="hf_proteingym_v1", limit=1, cache_dir=tmp_path)

    assert tasks[0]["id"] == "ASSAY1:0"
    assert tasks[0]["answer"] == "0.5"
    assert tasks[0]["context"]["assay_id"] == "ASSAY1"
    assert tasks[0]["context"]["DMS_score_bin"] == "1"
    assert tasks[0]["context"]["official_metric"] == "proteingym_dms_zero_shot"


def test_proteingym_csv_loader_adds_reference_metadata(monkeypatch, tmp_path):
    csv_path = tmp_path / "ASSAY1.csv"
    csv_path.write_text(
        "mutant,DMS_score,DMS_score_bin\nA2C,0.5,1\n",
        encoding="utf-8",
    )
    ref_path = tmp_path / "ProteinGym_reference_file_substitutions.csv"
    ref_path.write_text(
        "DMS_id,UniProt_ID,taxon,selection_type,target_seq\n"
        "ASSAY1,P12345,Bacteria,binding,AAD\n",
        encoding="utf-8",
    )

    class FakeApi:
        def list_repo_files(self, *args, **kwargs):
            return ["ProteinGym_substitutions/ASSAY1.csv", "ProteinGym_reference_file_substitutions.csv"]

    def fake_hf_hub_download(repo, filename, *args, **kwargs):
        if filename == "ProteinGym_reference_file_substitutions.csv":
            return str(ref_path)
        return str(csv_path)

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *args, **kwargs: []),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeApi, hf_hub_download=fake_hf_hub_download),
    )

    from harness.eval.bench_hf_benchmark import load_hf_benchmark_tasks

    tasks = load_hf_benchmark_tasks(dataset_key="hf_proteingym_v01", limit=1, cache_dir=tmp_path)

    assert tasks[0]["context"]["uniprot_id"] == "P12345"
    assert tasks[0]["context"]["taxon"] == "Bacteria"
    assert tasks[0]["context"]["selection_type"] == "binding"


def test_rna_loader_uses_official_target_fields(monkeypatch, tmp_path):
    rows = {
        "test": [
            {
                "index": 103498,
                "Unnamed: 0": 103498,
                "utr": "ACGU",
                "rl": 5.38245493113,
                "total": 0.1,
            }
        ]
    }

    def fake_load_dataset(*args, **kwargs):
        return rows

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )

    from harness.eval.bench_hf_benchmark import load_hf_benchmark_tasks

    tasks = load_hf_benchmark_tasks(dataset_key="hf_rna_mean_ribosome_load", limit=1, cache_dir=tmp_path)

    assert tasks[0]["answer"] == "5.38245493113"
    assert tasks[0]["answer"] != "103498"
    assert tasks[0]["context"]["target_field"] == "rl"
    assert tasks[0]["context"]["official_metric"] == "rna_regression_spearman_pearson"


def test_rna_loader_combines_splice_test_species(monkeypatch, tmp_path):
    rows = {
        "train": [{"sequences": "AAAA", "labels": 1}],
        "test_fly": [{"sequences": "CCCC", "labels": 0}],
        "test_worm": [{"sequences": "GGGG", "labels": 1}],
    }

    def fake_load_dataset(*args, **kwargs):
        return rows

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )

    from harness.eval.bench_hf_benchmark import load_hf_benchmark_tasks

    tasks = load_hf_benchmark_tasks(dataset_key="hf_rna_splice_site_acceptor", limit=5, cache_dir=tmp_path)

    assert [task["context"]["species_split"] for task in tasks] == ["test_fly", "test_worm"]
    assert all(task["context"]["official_metric"] == "rna_binary_accuracy_f1_auroc" for task in tasks)


def test_bacbench_loader_joins_genomes_to_label_csv(monkeypatch, tmp_path):
    label_path = tmp_path / "binary_labels.csv"
    label_path.write_text(
        "genome_name,ampicillin,gentamicin\n"
        "G1,1.0,\n"
        "G2,,0.0\n",
        encoding="utf-8",
    )
    rows = [
        {"genome_name": "G1", "dna_sequence": "ACGT" * 4000, "taxid": "562"},
        {"genome_name": "G2", "dna_sequence": "TGCA", "taxid": "562"},
    ]

    def fake_load_dataset(*args, **kwargs):
        return rows

    def fake_hf_hub_download(*args, **kwargs):
        return str(label_path)

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=fake_hf_hub_download),
    )

    from harness.eval.bench_hf_benchmark import load_hf_benchmark_tasks

    tasks = load_hf_benchmark_tasks(dataset_key="hf_bacbench_antibiotic_resistance_dna", limit=2, cache_dir=tmp_path)

    assert [task["context"]["label_name"] for task in tasks] == ["ampicillin", "gentamicin"]
    assert [task["answer"] for task in tasks] == ["1", "0"]
    assert tasks[0]["context"]["official_metric"] == "bacbench_binary_auprc"
    assert tasks[0]["context"]["prompt_sequence_truncated"] is True
