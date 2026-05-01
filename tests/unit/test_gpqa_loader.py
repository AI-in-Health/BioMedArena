from types import SimpleNamespace


def test_gpqa_loader_defaults_to_main_without_domain_filter(monkeypatch):
    calls = []

    rows = [
        {
            "Record ID": "physics-1",
            "Question": "Question?",
            "Correct Answer": "Correct",
            "Incorrect Answer 1": "Wrong 1",
            "Incorrect Answer 2": "Wrong 2",
            "Incorrect Answer 3": "Wrong 3",
            "High-level domain": "Physics",
            "Subdomain": "Mechanics",
        },
        {
            "Record ID": "biology-1",
            "Question": "Question?",
            "Correct Answer": "Correct",
            "Incorrect Answer 1": "Wrong 1",
            "Incorrect Answer 2": "Wrong 2",
            "Incorrect Answer 3": "Wrong 3",
            "High-level domain": "Biology",
            "Subdomain": "Genetics",
        },
    ]

    def fake_load_dataset(name, variant, split):
        calls.append((name, variant, split))
        return rows

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )

    from harness.eval.bench_gpqa_bio import load_gpqa_bio_tasks

    tasks = load_gpqa_bio_tasks()

    assert calls == [("Idavidrein/gpqa", "gpqa_main", "train")]
    assert len(tasks) == 2
    assert {task["context"]["high_level_domain"] for task in tasks} == {
        "Physics",
        "Biology",
    }


def test_gpqa_loader_can_still_filter_by_domain(monkeypatch):
    rows = [
        {
            "Record ID": "physics-1",
            "Question": "Question?",
            "Correct Answer": "Correct",
            "Incorrect Answer 1": "Wrong 1",
            "Incorrect Answer 2": "Wrong 2",
            "Incorrect Answer 3": "Wrong 3",
            "High-level domain": "Physics",
            "Subdomain": "Mechanics",
        },
        {
            "Record ID": "biology-1",
            "Question": "Question?",
            "Correct Answer": "Correct",
            "Incorrect Answer 1": "Wrong 1",
            "Incorrect Answer 2": "Wrong 2",
            "Incorrect Answer 3": "Wrong 3",
            "High-level domain": "Biology",
            "Subdomain": "Genetics",
        },
    ]

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *args, **kwargs: rows),
    )

    from harness.eval.bench_gpqa_bio import load_gpqa_bio_tasks

    tasks = load_gpqa_bio_tasks(domain_filter="biology")

    assert len(tasks) == 1
    assert tasks[0]["id"] == "biology-1"


def test_gpqa_loader_falls_back_from_unavailable_extended_config(monkeypatch):
    calls = []
    rows = [
        {
            "Record ID": "biology-1",
            "Question": "Question?",
            "Correct Answer": "Correct",
            "Incorrect Answer 1": "Wrong 1",
            "Incorrect Answer 2": "Wrong 2",
            "Incorrect Answer 3": "Wrong 3",
            "High-level domain": "Biology",
            "Subdomain": "Genetics",
        },
    ]

    def fake_load_dataset(name, variant, split):
        calls.append((name, variant, split))
        if variant == "gpqa_extended":
            raise ValueError("missing config")
        return rows

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )

    from harness.eval.bench_gpqa_bio import load_gpqa_bio_tasks

    tasks = load_gpqa_bio_tasks(variant="gpqa_extended")

    assert calls == [
        ("Idavidrein/gpqa", "gpqa_extended", "train"),
        ("Idavidrein/gpqa", "gpqa_main", "train"),
    ]
    assert len(tasks) == 1
    assert tasks[0]["context"]["variant"] == "gpqa_main"
