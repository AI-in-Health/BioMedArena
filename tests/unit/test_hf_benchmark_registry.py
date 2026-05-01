from types import SimpleNamespace


def test_hf_registry_adds_more_than_100_cli_benchmarks():
    from harness.cli import BENCHMARKS
    from harness.eval.hf_benchmark_registry import (
        HF_BENCHMARK_SPECS,
        HF_VERIFIED_BENCHMARK_KEYS,
    )

    hf_cli_keys = {key for key in BENCHMARKS if key.startswith("hf_")}

    assert len(HF_BENCHMARK_SPECS) >= 100
    assert len(HF_VERIFIED_BENCHMARK_KEYS) >= 100
    assert hf_cli_keys == HF_VERIFIED_BENCHMARK_KEYS
    assert BENCHMARKS["hf_medqa_usmle_4_options"]["loader"] == "load_hf_benchmark_tasks"


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
        dataset_key="hf_medqa_usmle_4_options",
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
        "armanc/pubmed-rct20k": [
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
        dataset_key="hf_pubmed_rct20k",
        limit=1,
        cache_dir=tmp_path,
    )

    assert qa_tasks[0]["answer_type"] == "openText"
    assert qa_tasks[0]["scorer_kind"] == "llm_judge"
    assert qa_tasks[0]["context"]["scorer_kind"] == "llm_judge"
    assert cls_tasks[0]["answer_type"] == "exactMatch"
    assert cls_tasks[0]["context"]["answer_type"] == "exactMatch"
    assert cls_tasks[0]["answer"] == "METHODS"
