from pathlib import Path

import pytest

Image = pytest.importorskip("PIL.Image")

from harness.eval.bench_medxpertqa_mm import load_medxpertqa_mm_tasks
from harness.eval.bench_pathvqa import _save_pil_image


def test_pathvqa_saved_image_is_real_file(tmp_path):
    image = Image.new("RGB", (4, 4), color="white")
    path = _save_pil_image(image, tmp_path, "demo")
    assert path is not None
    assert Path(path).exists()
    with Image.open(path) as opened:
        assert opened.size == (4, 4)


def test_medxpertqa_mm_skips_rows_when_required_images_missing(monkeypatch, tmp_path):
    rows = [{
        "id": "mm_1",
        "question": "Q?",
        "options": {"A": "yes", "B": "no"},
        "label": "A",
        "images": ["missing.png"],
        "medical_task": "demo",
        "body_system": "demo",
        "question_type": "mcq",
    }]

    def fake_load_dataset(*args, **kwargs):
        return rows

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    assert load_medxpertqa_mm_tasks(image_dir=tmp_path, require_images=True) == []

    tasks = load_medxpertqa_mm_tasks(image_dir=tmp_path, require_images=False)
    assert len(tasks) == 1
    assert tasks[0]["context"]["multimodal"] is True
    assert tasks[0]["context"]["input_type"] == "image+text"
