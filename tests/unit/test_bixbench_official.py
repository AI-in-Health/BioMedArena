from __future__ import annotations

import zipfile

import pytest

from harness.eval.bixbench_official import (
    BixBenchCapsuleManager,
    BixBenchMCQConfig,
    attach_capsule_paths,
    score_bixbench_mcq,
    score_bixbench_open_answer,
)


def test_capsule_filename_normalisation():
    assert (
        BixBenchCapsuleManager.filename_for("abc")
        == "CapsuleFolder-abc.zip"
    )
    assert (
        BixBenchCapsuleManager.filename_for("CapsuleFolder-abc")
        == "CapsuleFolder-abc.zip"
    )
    assert (
        BixBenchCapsuleManager.filename_for("CapsuleFolder-abc.zip")
        == "CapsuleFolder-abc.zip"
    )


def test_safe_extract_rejects_zip_slip(tmp_path):
    manager = BixBenchCapsuleManager(cache_dir=tmp_path)
    archive = manager.archive_dir / "CapsuleFolder-bad.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.txt", "nope")

    with pytest.raises(RuntimeError, match="unsafe path"):
        manager.extract_capsule("bad")


def test_attach_capsule_paths_from_existing_cache(tmp_path):
    manager = BixBenchCapsuleManager(cache_dir=tmp_path, revision="main")
    info = manager.capsule_info("abc")
    info.extract_dir.mkdir(parents=True)
    (info.extract_dir / "data.csv").write_text("x,y\n1,2\n")
    tasks = [{
        "id": "q1",
        "metadata": {"data_folder": "CapsuleFolder-abc.zip"},
        "context": {},
    }]

    attach_capsule_paths(tasks, cache_dir=tmp_path, revision="main", require=True)

    assert tasks[0]["metadata"]["capsule_path"] == str(info.extract_dir)
    assert tasks[0]["context"]["capsule_path"] == str(info.extract_dir)


def test_bixbench_mcq_majority_and_opt_out():
    result = score_bixbench_mcq(
        ["I do not know", "Answer: B", "The answer is B"],
        "B",
        config=BixBenchMCQConfig(majority_vote_n=3, allow_opt_out=True),
    )
    assert result["correct"] is True
    assert result["selected"] == "B"
    assert result["opted_out"] == 1


@pytest.mark.asyncio
async def test_bixbench_range_verifier():
    task = {
        "question": "q",
        "answer": "(0.024,0.026)",
        "context": {"eval_mode": "range_verifier"},
    }
    result = await score_bixbench_open_answer(task, "The answer is 0.025.")
    assert result["correct"] is True
    assert result["method"] == "bixbench_range_verifier"


@pytest.mark.asyncio
async def test_bixbench_range_verifier_preserves_decimal_comma_separator():
    task = {
        "question": "q",
        "answer": "(1.50,1.54)",
        "context": {"eval_mode": "range_verifier"},
    }

    wrong = await score_bixbench_open_answer(task, "The answer is 1.05.")
    right = await score_bixbench_open_answer(task, "The answer is 1.52.")

    assert wrong["correct"] is False
    assert wrong["details"]["expected_range"] == [1.5, 1.54]
    assert right["correct"] is True


@pytest.mark.asyncio
async def test_bixbench_str_verifier_extracts_final_answer():
    task = {
        "question": "q",
        "answer": "0.0002",
        "context": {"eval_mode": "str_verifier"},
    }
    result = await score_bixbench_open_answer(
        task,
        'Long analysis text. The adjusted p-value is 0.0001.\n\nThe answer is 0.0002',
    )
    assert result["correct"] is True
    assert result["method"] == "bixbench_str_verifier"


@pytest.mark.asyncio
async def test_bixbench_llm_verifier_uses_judge():
    async def fake_judge(question, expected, predicted):
        return {"correct": True, "reasoning": "ok"}

    task = {
        "question": "q",
        "answer": "expected",
        "context": {"eval_mode": "llm_verifier"},
    }
    result = await score_bixbench_open_answer(
        task,
        "semantically matching prediction",
        judge_fn=fake_judge,
    )

    assert result["correct"] is True
    assert result["method"] == "bixbench_llm_verifier"
