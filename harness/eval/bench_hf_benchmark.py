"""Generic Hugging Face dataset loader for text/table biomedical benchmarks."""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

from harness.eval.hf_benchmark_registry import HF_BENCHMARK_SPECS, HFDatasetSpec

logger = logging.getLogger(__name__)

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

_QUESTION_FIELDS = (
    "question", "Question", "query", "prompt", "input", "sent1",
    "Open-ended Verifiable Question", "Description", "Patient", "human",
    "problem", "text", "sentence", "abstract", "article", "title",
    "instruction",
)
_ANSWER_FIELDS = (
    "answer_idx", "gold_index", "cop", "Correct Option", "Correct Answer",
    "answer", "Answer", "Ground-True Answer", "Response", "output",
    "response", "response (content)", "Doctor", "gpt", "completion", "target",
    "label", "labels", "Stay (in days)",
    "final_decision", "summary", "abstract", "caption",
)
_CHOICE_FIELDS = (
    "choices", "options", "Options", "answers", "candidates",
)
_CONTEXT_FIELDS = (
    "context", "contexts", "passage", "document", "body", "article",
    "knowledge", "Knowledge", "Patient",
)
_TEXT_FIELDS = (
    "text", "sentence", "abstract", "article", "content", "document",
    "dialogue", "section_text", "smiles", "sequence", "protein_sequence", "dna", "rna",
)
_SMILES_FIELDS = ("smiles", "SMILES", "canonical_smiles", "mol", "molecule")
_SEQUENCE_FIELDS = (
    "sequence", "sequences", "protein_sequence", "seq", "primary", "dna", "rna", "nucleotide_sequence",
)
_PROPERTY_VALUE_FIELDS = (
    "target", "label", "labels", "y", "activity_value", "log_fluorescence",
    "exp_mean [nM]", "fitness", "score", "value",
)


def load_hf_benchmark_tasks(
    *,
    dataset_key: str,
    limit: int | None = None,
    split: str | None = None,
    cache_dir: str | Path = "data/cache/huggingface",
    streaming: bool | None = None,
) -> list[dict[str, Any]]:
    """Load one configured generic HF benchmark.

    This loader is intentionally permissive: it normalizes common HF schemas
    into the task shape used by the harness and fails closed with ``[]`` when a
    dataset/config is unavailable.
    """
    spec = HF_BENCHMARK_SPECS.get(dataset_key)
    if spec is None:
        raise ValueError(f"unknown HF benchmark dataset_key={dataset_key!r}")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        logger.warning("datasets is required for %s: %s", dataset_key, exc)
        return []

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    split_name = split or spec.split

    try:
        ds = _load_dataset(
            load_dataset,
            spec,
            split_name,
            cache_path,
            prefer_streaming=bool(streaming if streaming is not None else spec.extra.get("streaming")),
        )
    except Exception as exc:
        logger.warning("HF benchmark %s failed to load: %s", dataset_key, exc)
        return []

    if isinstance(ds, Mapping):
        split_name = _choose_split(ds, split_name)
        ds = ds[split_name]

    tasks: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        task = _normalise_row(spec, row, i, split_name or "default")
        if task is None:
            continue
        tasks.append(task)
        if limit and len(tasks) >= limit:
            break
    return tasks


def _load_dataset(
    load_dataset: Any,
    spec: HFDatasetSpec,
    split: str | None,
    cache_path: Path,
    *,
    prefer_streaming: bool = False,
) -> Any:
    if prefer_streaming:
        try:
            return _load_dataset_once(
                load_dataset,
                spec,
                split,
                cache_path,
                streaming=True,
            )
        except Exception as exc:
            logger.info(
                "HF benchmark %s streaming load failed; falling back to local cache load: %s",
                spec.key,
                exc,
            )
    return _load_dataset_once(load_dataset, spec, split, cache_path, streaming=False)


def _load_dataset_once(
    load_dataset: Any,
    spec: HFDatasetSpec,
    split: str | None,
    cache_path: Path,
    *,
    streaming: bool,
) -> Any:
    kwargs = {"cache_dir": str(cache_path)}
    if streaming:
        kwargs["streaming"] = True
    if split:
        kwargs["split"] = split
    if spec.config:
        return load_dataset(spec.repo, spec.config, **kwargs)
    try:
        return load_dataset(spec.repo, **kwargs)
    except Exception:
        # Some HF repos need split omitted first because they expose custom
        # split names or no split metadata until the builder is initialized.
        if split:
            kwargs.pop("split", None)
            return load_dataset(spec.repo, **kwargs)
        raise


def _choose_split(ds: dict[str, Any], preferred: str | None) -> str:
    if preferred and preferred in ds:
        return preferred
    for candidate in ("test", "validation", "valid", "dev", "eval", "train"):
        if candidate in ds:
            return candidate
    return next(iter(ds))


def _normalise_row(
    spec: HFDatasetSpec,
    row: dict[str, Any],
    idx: int,
    split: str,
) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    row = _expand_row(row)
    task_type = spec.task_type
    if task_type == "mcq":
        return _make_mcq(spec, row, idx, split)
    if task_type in {"qa", "retrieval"}:
        return _make_qa(spec, row, idx, split)
    if task_type == "summarization":
        return _make_summarization(spec, row, idx, split)
    if task_type in {"classification", "pair_classification"}:
        return _make_classification(spec, row, idx, split, pair=task_type == "pair_classification")
    if task_type in {"molecule_property", "protein_fitness", "regression"}:
        return _make_structured_prediction(spec, row, idx, split)
    if task_type in {"sequence", "text"}:
        return _make_text_completion(spec, row, idx, split)
    return _make_qa(spec, row, idx, split)


def _make_base(spec: HFDatasetSpec, row: dict[str, Any], idx: int, split: str) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or row.get("idx") or row.get("pmid") or f"{spec.key}_{idx}"),
        "category": f"HF/{spec.domain}/{spec.key}",
        "raw_subject": spec.domain,
        "context": {
            "source": spec.repo,
            "config": spec.config,
            "split": split,
            "task_type": spec.task_type,
            "dataset_key": spec.key,
        },
        "metadata": {
            "source": "hf_generic",
            "repo": spec.repo,
            "dataset_key": spec.key,
            "split": split,
        },
    }


def _set_scoring(
    task: dict[str, Any],
    *,
    answer_type: str,
    scorer_kind: str,
    scorer_params: dict[str, Any] | None = None,
) -> None:
    params = scorer_params or {}
    task["answer_type"] = answer_type
    task["scorer_kind"] = scorer_kind
    if params:
        task["scorer_params"] = params
    context = task.setdefault("context", {})
    if isinstance(context, dict):
        context["answer_type"] = answer_type
        context["scorer_kind"] = scorer_kind
        if params:
            context["scorer_params"] = params


def _make_mcq(spec: HFDatasetSpec, row: dict[str, Any], idx: int, split: str) -> dict[str, Any] | None:
    question = _first_str(row, spec.question_fields or _QUESTION_FIELDS)
    choices = _extract_choices(row, spec.choice_fields or _CHOICE_FIELDS)
    answer = _extract_answer(row, spec.answer_fields or _ANSWER_FIELDS)
    if not question:
        question = _row_preview(row)
    if not choices:
        # Fall back to QA when a supposed MCQ mirror exposes answer text only.
        return _make_qa(spec, row, idx, split)
    answer_letter = _answer_to_letter(answer, choices)
    if not answer_letter or answer_letter not in _LETTERS[:len(choices)]:
        return None
    opts = "\n".join(f"{_LETTERS[i]}. {choice}" for i, choice in enumerate(choices[:26]))
    task = _make_base(spec, row, idx, split)
    task.update({
        "question": f"{question}\n\nOptions:\n{opts}",
        "choices": choices,
        "answer": answer_letter,
    })
    _set_scoring(task, answer_type="multipleChoice", scorer_kind="mcq")
    return task


def _make_qa(spec: HFDatasetSpec, row: dict[str, Any], idx: int, split: str) -> dict[str, Any] | None:
    question = _first_str(row, spec.question_fields or _QUESTION_FIELDS)
    answer = _extract_answer(row, spec.answer_fields or _ANSWER_FIELDS)
    context = _context_text(row, spec.context_fields or _CONTEXT_FIELDS)
    if not question:
        question = _row_preview(row)
    if not answer:
        answer = _first_str(row, ("target", "label", "output", "completion"))
    if not question or not answer:
        return None
    prompt = f"Context:\n{context}\n\nQuestion: {question}" if context else question
    task = _make_base(spec, row, idx, split)
    task.update({
        "question": prompt,
        "answer": answer,
    })
    _set_scoring(
        task,
        answer_type="openText",
        scorer_kind="llm_judge",
        scorer_params={"ground_truth": answer},
    )
    return task


def _make_summarization(spec: HFDatasetSpec, row: dict[str, Any], idx: int, split: str) -> dict[str, Any] | None:
    text = _first_str(row, spec.text_fields or ("article", "document", "text", "dialogue", "input"))
    answer = _extract_answer(row, spec.answer_fields or ("summary", "abstract", "section_text", "output", "target"))
    if not text or not answer:
        return _make_qa(spec, row, idx, split)
    task = _make_base(spec, row, idx, split)
    task.update({
        "question": f"Summarize the following biomedical text.\n\n{text}",
        "answer": answer,
    })
    _set_scoring(
        task,
        answer_type="openText",
        scorer_kind="llm_judge",
        scorer_params={"ground_truth": answer},
    )
    return task


def _make_classification(
    spec: HFDatasetSpec,
    row: dict[str, Any],
    idx: int,
    split: str,
    *,
    pair: bool = False,
) -> dict[str, Any] | None:
    answer = _extract_answer(row, spec.label_fields or spec.answer_fields or _ANSWER_FIELDS)
    if not answer:
        answer = _label_vector(row) or _first_property_value(row)
    if not answer:
        return _make_qa(spec, row, idx, split)
    if pair:
        texts = [
            _first_str(row, ("question1", "sentence1", "text1", "query", "question")),
            _first_str(row, ("question2", "sentence2", "text2", "document", "answer")),
        ]
        question = f"Classify the relationship between these two medical texts.\n\nText A: {texts[0]}\n\nText B: {texts[1]}"
    else:
        text = _first_str(row, spec.text_fields or _TEXT_FIELDS) or _row_preview(row)
        question = f"Classify this {spec.domain} example.\n\n{text}"
    task = _make_base(spec, row, idx, split)
    task.update({
        "question": question,
        "answer": answer,
    })
    _set_scoring(task, answer_type="exactMatch", scorer_kind="exact")
    return task


def _make_structured_prediction(spec: HFDatasetSpec, row: dict[str, Any], idx: int, split: str) -> dict[str, Any] | None:
    structure = _first_str(row, _SMILES_FIELDS + _SEQUENCE_FIELDS + _TEXT_FIELDS) or _row_preview(row)
    answer = _extract_answer(row, spec.answer_fields or spec.label_fields or _ANSWER_FIELDS)
    if not answer:
        answer = _first_property_value(row)
    if not structure or not answer:
        return None
    task = _make_base(spec, row, idx, split)
    task.update({
        "question": (
            f"Predict the target property or label for this {spec.domain} record.\n\n"
            f"Input: {structure}"
        ),
        "answer": answer,
    })
    answer_type = "exactNumeric" if _looks_numeric(answer) else "exactMatch"
    _set_scoring(task, answer_type=answer_type, scorer_kind="exact")
    return task


def _first_property_value(row: dict[str, Any]) -> str:
    skip = set(_SMILES_FIELDS + _SEQUENCE_FIELDS + _TEXT_FIELDS)
    skip.update({"id", "idx", "name", "split", "data"})
    for key in _PROPERTY_VALUE_FIELDS:
        if key in row:
            text = _stringify(row[key])
            if text:
                return text
    for key, value in row.items():
        if key in skip or str(key).startswith("_"):
            continue
        text = _stringify(value)
        if text:
            return text
    return ""


def _looks_numeric(value: str) -> bool:
    text = str(value or "").strip().replace(",", "")
    try:
        float(text)
        return True
    except ValueError:
        return False


def _label_vector(row: dict[str, Any]) -> str:
    label_keys = sorted(
        (key for key in row if str(key).startswith("labels_")),
        key=lambda key: int(str(key).split("_", 1)[1]) if str(key).split("_", 1)[1].isdigit() else str(key),
    )
    if not label_keys:
        return ""
    return json.dumps([row[key] for key in label_keys])


def _make_text_completion(spec: HFDatasetSpec, row: dict[str, Any], idx: int, split: str) -> dict[str, Any] | None:
    text = _first_str(row, spec.text_fields or _TEXT_FIELDS) or _row_preview(row)
    answer = _extract_answer(row, spec.answer_fields or ("target", "label", "output"))
    if not text:
        return None
    if not answer:
        answer = text[:500]
        question = f"Continue or characterize this {spec.domain} text.\n\n{text[:2000]}"
    else:
        question = f"Given this {spec.domain} input, produce the expected output.\n\n{text}"
    task = _make_base(spec, row, idx, split)
    task.update({
        "question": question,
        "answer": answer,
    })
    _set_scoring(
        task,
        answer_type="openText",
        scorer_kind="llm_judge",
        scorer_params={"ground_truth": answer},
    )
    return task


def _first_str(row: dict[str, Any], fields: Iterable[str]) -> str:
    for field in fields:
        if field in row:
            value = row[field]
            text = _stringify(value)
            if text:
                return text
    return ""


def _expand_row(row: dict[str, Any]) -> dict[str, Any]:
    expanded = dict(row)
    data = _parse_json_like(expanded.get("data"))
    if isinstance(data, dict):
        for key, value in data.items():
            expanded.setdefault(key, value)

    messages = _parse_json_like(expanded.get("messages"))
    if isinstance(messages, list):
        _merge_dialogue_turns(expanded, messages, role_key="role", user_roles={"user", "human"}, assistant_roles={"assistant", "gpt"})

    conversations = _parse_json_like(expanded.get("conversations"))
    if isinstance(conversations, list):
        _merge_dialogue_turns(expanded, conversations, role_key="from", user_roles={"human", "user"}, assistant_roles={"gpt", "assistant"})

    return expanded


def _merge_dialogue_turns(
    row: dict[str, Any],
    turns: list[Any],
    *,
    role_key: str,
    user_roles: set[str],
    assistant_roles: set[str],
) -> None:
    users = []
    assistants = []
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get(role_key) or "").lower()
        content = _stringify(turn.get("content") or turn.get("value") or turn.get("text"))
        if not content:
            continue
        if role in user_roles:
            users.append(content)
        elif role in assistant_roles:
            assistants.append(content)
    if users:
        row.setdefault("human", users[0])
        row.setdefault("question", users[0])
    if assistants:
        row.setdefault("gpt", assistants[-1])
        row.setdefault("answer", assistants[-1])


def _parse_json_like(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text[0] not in "[{":
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_answer(row: dict[str, Any], fields: Iterable[str]) -> str:
    for field in fields:
        if field not in row:
            continue
        value = row[field]
        if isinstance(value, list) and value and isinstance(value[0], dict):
            for key in ("text", "answer", "content"):
                text = _stringify(value[0].get(key))
                if text:
                    return text
        text = _stringify(value)
        if text:
            return text
    return ""


def _extract_choices(row: dict[str, Any], fields: Iterable[str]) -> list[str]:
    for field in fields:
        if field not in row:
            continue
        value = row[field]
        if isinstance(value, dict):
            return [_stringify(value[k]) for k in sorted(value) if _stringify(value[k])]
        if isinstance(value, list):
            if value and isinstance(value[0], dict):
                out = []
                for item in value:
                    out.append(_stringify(item.get("text") or item.get("label") or item.get("value") or item))
                return [x for x in out if x]
            return [_stringify(v) for v in value if _stringify(v)]
    # Common MedMCQA-style fields.
    lettered = []
    for idx in range(26):
        key = f"ending{idx}"
        if key in row and _stringify(row[key]):
            lettered.append(_stringify(row[key]))
    if lettered:
        return lettered
    lettered = []
    for key in ("opa", "opb", "opc", "opd", "ope"):
        if key in row and _stringify(row[key]):
            lettered.append(_stringify(row[key]))
    if lettered:
        return lettered
    lettered = []
    for key in ("A", "B", "C", "D", "E"):
        if key in row and _stringify(row[key]):
            lettered.append(_stringify(row[key]))
    return lettered


def _answer_to_letter(answer: str, choices: list[str]) -> str:
    ans = str(answer).strip()
    if len(ans) == 1 and ans.upper() in _LETTERS:
        return ans.upper()
    if ans.isdigit():
        idx = int(ans)
        # Support both 0-based and 1-based integer labels.
        if 0 <= idx < len(choices):
            return _LETTERS[idx]
        if 1 <= idx <= len(choices):
            return _LETTERS[idx - 1]
    normalised = ans.lower().strip()
    for i, choice in enumerate(choices):
        if normalised == choice.lower().strip():
            return _LETTERS[i]
    return ans


def _context_text(row: dict[str, Any], fields: Iterable[str]) -> str:
    chunks = []
    for field in fields:
        if field in row:
            text = _stringify(row[field])
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def _row_preview(row: dict[str, Any]) -> str:
    parts = []
    for key, value in row.items():
        if key.startswith("_"):
            continue
        text = _stringify(value)
        if text:
            parts.append(f"{key}: {text}")
        if len(parts) >= 6:
            break
    return "\n".join(parts)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(_stringify(v) for v in value if _stringify(v))
    if isinstance(value, dict):
        if "text" in value:
            return _stringify(value["text"])
        if "answer" in value:
            return _stringify(value["answer"])
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    return str(value).strip()
