"""MedHELM public-scenario loader.

Official project: https://medhelm.org/
Official code: https://github.com/Pacific-AI-Corp/medhelm

MedHELM is a benchmark framework rather than one monolithic dataset. This
loader mirrors two public MedHELM scenarios from the official implementation:

* ``medhallu`` from ``UTAustin-AIHealth/MedHallu`` at the pinned revision used
  by MedHELM.
* ``medication_qa`` from the official MedicationQA MedInfo 2019 spreadsheet.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

_MEDHALLU_REPO = "UTAustin-AIHealth/MedHallu"
_MEDHALLU_REVISION = "515060458a945c633debc6fd5baac7764416b724"
_MEDICATION_QA_URL = (
    "https://github.com/abachaa/Medication_QA_MedInfo2019/raw/master/"
    "MedInfo2019-QA-Medications.xlsx"
)


def _medhallu_tasks(
    limit: int | None = None,
    cache_dir: str | Path = "data/cache/huggingface",
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    try:
        ds = load_dataset(
            _MEDHALLU_REPO,
            "pqa_labeled",
            split="train",
            revision=_MEDHALLU_REVISION,
            cache_dir=str(cache_dir),
        )
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    for idx, row in enumerate(ds):
        knowledge = row.get("Knowledge") or []
        if isinstance(knowledge, list):
            knowledge_text = "\n".join(str(x) for x in knowledge)
        else:
            knowledge_text = str(knowledge)
        question = str(row.get("Question") or "")
        for kind, answer_text, label in (
            ("ground_truth", row.get("Ground Truth"), "0"),
            ("hallucinated", row.get("Hallucinated Answer"), "1"),
        ):
            prompt = (
                "Classify whether the answer is factual or hallucinated. "
                "Return 0 for factual and 1 for hallucinated.\n\n"
                f"World Knowledge: {knowledge_text}\n\n"
                f"Question: {question}\n\n"
                f"Answer: {answer_text or ''}"
            )
            out.append({
                "id": f"medhallu_{idx:06d}_{kind}",
                "question": prompt,
                "answer": label,
                "answer_type": "exact",
                "category": "MedHELM/MedHallu",
                "raw_subject": row.get("Difficulty Level") or "medhallu",
                "context": {
                    "source": _MEDHALLU_REPO,
                    "revision": _MEDHALLU_REVISION,
                    "scenario": "medhallu",
                    "kind": kind,
                    "hallucination_category": row.get("Category of Hallucination"),
                },
            })
            if limit and len(out) >= limit:
                return out
    return out


def _medication_qa_tasks(
    limit: int | None = None,
    cache_dir: str | Path = "data/cache/medhelm",
) -> list[dict[str, Any]]:
    try:
        import urllib.request
    except ImportError:
        return []

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    xlsx_path = cache_path / "MedInfo2019-QA-Medications.xlsx"
    if not xlsx_path.exists():
        try:
            urllib.request.urlretrieve(_MEDICATION_QA_URL, xlsx_path)
        except Exception:
            return []

    out: list[dict[str, Any]] = []
    for idx, row in enumerate(_read_xlsx_rows(xlsx_path)):
        question = row.get("Question")
        answer = row.get("Answer")
        if not question or not answer:
            continue
        out.append({
            "id": f"medication_qa_{idx:06d}",
            "question": str(question),
            "answer": str(answer),
            "answer_type": "openText",
            "category": "MedHELM/MedicationQA",
            "raw_subject": "medication_qa",
            "context": {
                "source": _MEDICATION_QA_URL,
                "scenario": "medication_qa",
            },
        })
        if limit and len(out) >= limit:
            break
    return out


def _read_xlsx_rows(path: str | Path) -> list[dict[str, str]]:
    """Read the first worksheet from a simple xlsx file using stdlib only."""
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for item in root.findall("x:si", ns):
                texts = [node.text or "" for node in item.findall(".//x:t", ns)]
                shared.append("".join(texts))
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))

    rows: list[list[str]] = []
    for row_el in sheet.findall(".//x:sheetData/x:row", ns):
        values: list[str] = []
        for cell in row_el.findall("x:c", ns):
            ref = cell.attrib.get("r", "")
            col_idx = _xlsx_col_to_index(re.sub(r"\d", "", ref))
            while len(values) < col_idx:
                values.append("")
            value_el = cell.find("x:v", ns)
            value = value_el.text if value_el is not None else ""
            if cell.attrib.get("t") == "s" and value:
                value = shared[int(value)]
            values.append(value or "")
        rows.append(values)

    if not rows:
        return []
    headers = [h.strip() for h in rows[0]]
    out: list[dict[str, str]] = []
    for row in rows[1:]:
        item = {headers[i]: row[i] for i in range(min(len(headers), len(row))) if headers[i]}
        out.append(item)
    return out


def _xlsx_col_to_index(col: str) -> int:
    total = 0
    for ch in col:
        total = total * 26 + (ord(ch.upper()) - ord("A") + 1)
    return max(total - 1, 0)


def load_medhelm_tasks(
    limit: int | None = None,
    scenarios: list[str] | None = None,
    cache_dir: str | Path = "data/cache/medhelm",
) -> list[dict[str, Any]]:
    """Load selected public MedHELM scenario tasks.

    Args:
        limit: optional total cap across scenarios.
        scenarios: subset of ``medhallu`` and ``medication_qa``. Defaults to
            both public scenarios implemented here.
        cache_dir: local cache for MedicationQA spreadsheet.
    """
    selected = scenarios or ["medhallu", "medication_qa"]
    out: list[dict[str, Any]] = []
    per_scenario_limit = None
    if limit is not None and len(selected) > 1:
        per_scenario_limit = max(1, (limit + len(selected) - 1) // len(selected))
    for scenario in selected:
        remaining = per_scenario_limit
        if remaining is None:
            remaining = None if limit is None else max(limit - len(out), 0)
        if remaining == 0:
            break
        if scenario == "medhallu":
            out.extend(_medhallu_tasks(remaining, cache_dir=Path(cache_dir).parent / "huggingface"))
        elif scenario == "medication_qa":
            out.extend(_medication_qa_tasks(remaining, cache_dir=cache_dir))
        else:
            raise ValueError("unknown MedHELM scenario; choose ['medhallu', 'medication_qa']")
    return out[:limit] if limit else out
