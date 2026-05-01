from harness.cli import BENCHMARKS
from harness.eval.bench_quick_suite import load_quick_suite_tasks
from harness.eval.scoring import score_question


def test_quick_suite_is_registered_and_offline():
    assert "quick_suite" in BENCHMARKS
    tasks = load_quick_suite_tasks()
    assert len(tasks) == 20
    assert all(t["context"]["requires_network"] is False for t in tasks)
    assert all(t["context"]["requires_token"] is False for t in tasks)


def test_quick_suite_canonical_answers_score_correct():
    for task in load_quick_suite_tasks():
        answer = str(task["answer"])
        if task["answer_type"] == "openText":
            predicted = answer
        else:
            predicted = f"The answer is {answer}."
        assert score_question(
            predicted,
            answer,
            task["answer_type"],
            task.get("context") or {},
        )
