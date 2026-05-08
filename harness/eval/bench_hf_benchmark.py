"""Generic Hugging Face dataset loader for text/table biomedical benchmarks."""
from __future__ import annotations

import json
import logging
import re
import warnings
import urllib.request
import ast
import csv
import hashlib
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import replace
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable

from harness.eval.hf_benchmark_registry import HF_BENCHMARK_SPECS, HF_DEPRECATED_ALIASES, HFDatasetSpec

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
    "choices", "options", "Options", "option", "answers", "candidates",
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
_BACBENCH_SEQUENCE_PROMPT_LIMIT = 12000


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
    if dataset_key in HF_DEPRECATED_ALIASES:
        canonical = HF_DEPRECATED_ALIASES[dataset_key]
        warnings.warn(
            f"{dataset_key} is deprecated; redirecting to {canonical}",
            DeprecationWarning,
            stacklevel=2,
        )
        if canonical not in HF_BENCHMARK_SPECS:
            raise ValueError(f"{dataset_key!r} redirects to non-HF benchmark {canonical!r}; use the CLI alias")
        dataset_key = canonical
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
    if dataset_key == "hf_cmb":
        return _load_cmb_exam_tasks(load_dataset, spec, limit, cache_path)
    if dataset_key == "hf_chinese_medbench":
        return _load_cmb_exam_tasks(load_dataset, spec, limit, cache_path)
    if dataset_key == "hf_discoverybench_biomedical":
        return _load_discoverybench_tasks(load_dataset, spec, limit, split, cache_path)
    if dataset_key == "hf_meds_bench":
        return _load_meds_bench_mcqa_tasks(spec, limit, cache_path)
    if dataset_key == "hf_medexqa":
        return _load_medexqa_tasks(spec, limit, cache_path)
    if dataset_key == "hf_chemllmbench":
        return _load_json_file_tasks(spec, limit, cache_path, split_name="test")
    if dataset_key in {"hf_mol_instructions_pubchemqa", "hf_smiles_caption_mol2text"}:
        return _load_mol_instructions_tasks(spec, limit, cache_path)
    if dataset_key in {"hf_bacbench_antibiotic_resistance_dna", "hf_bacbench_phenotypic_traits_dna"}:
        return _load_bacbench_tasks(spec, limit, cache_path)
    if dataset_key in {"hf_mteb_medical_qa", "hf_mteb_medical_retrieval"}:
        return _load_mteb_retrieval_tasks(load_dataset, spec, limit, cache_path)
    if dataset_key in {"hf_blurb", "hf_blue_benchmark", "hf_bc5cdr", "hf_jnlpba", "hf_ncbi_disease"}:
        return _load_blurb_ner_tasks(spec, limit, cache_path)
    if dataset_key == "hf_ppi_benchmark":
        return _load_bioinfer_relation_tasks(spec, limit, cache_path, split or spec.split or "test")
    if dataset_key == "hf_pgr":
        return _load_pgr_relation_tasks(spec, limit, cache_path, split or spec.split or "test")
    if dataset_key in {"hf_proteingym_v1", "hf_proteingym_v01", "hf_icml2022_proteingym"}:
        return _load_proteingym_tasks(spec, limit, cache_path)
    if spec.repo == "genbio-ai/rna-downstream-tasks":
        return _load_rna_downstream_tasks(load_dataset, spec, limit, split, cache_path)
    tasks: list[dict[str, Any]] = []
    configs = spec.extra.get("configs") or ()
    load_specs = [replace(spec, config=config) for config in configs] if configs else [spec]
    for load_spec in load_specs:
        split_name = split or load_spec.split
        try:
            ds = _load_dataset(
                load_dataset,
                load_spec,
                split_name,
                cache_path,
                prefer_streaming=bool(streaming if streaming is not None else load_spec.extra.get("streaming")),
            )
        except Exception as exc:
            logger.warning("HF benchmark %s failed to load: %s", dataset_key, exc)
            continue

        if _is_split_mapping(ds):
            split_name = _choose_split(ds, split_name)
            ds = ds[split_name]

        for row in ds:
            task = _normalise_row(load_spec, row, len(tasks), split_name or "default")
            if task is None:
                continue
            tasks.append(task)
            if limit and len(tasks) >= limit:
                return tasks
    return tasks


def _load_discoverybench_tasks(
    load_dataset: Any,
    spec: HFDatasetSpec,
    limit: int | None,
    split: str | None,
    cache_path: Path,
) -> list[dict[str, Any]]:
    """Expand DiscoveryBench rows into query-level biology tasks."""
    try:
        ds = _load_dataset(load_dataset, spec, split or spec.split, cache_path)
    except Exception as exc:
        logger.warning("HF benchmark %s failed to load: %s", spec.key, exc)
        return []
    if _is_split_mapping(ds):
        split_name = _choose_split(ds, split or spec.split)
        ds = ds[split_name]
    else:
        split_name = split or spec.split or "default"

    tasks: list[dict[str, Any]] = []
    for row_idx, row in enumerate(ds):
        if not isinstance(row, dict):
            continue
        domain = _stringify(row.get("domain")).lower()
        if domain and domain != "biology":
            continue
        datasets = _parse_json_like(row.get("datasets"))
        queries = _parse_json_like(row.get("queries"))
        if not isinstance(queries, list):
            continue
        dataset_context = _format_discoverybench_datasets(datasets)
        domain_knowledge = _stringify(row.get("domain_knowledge"))
        workflow_tags = _stringify(row.get("workflow_tags"))
        for query_idx, query in enumerate(queries):
            if not isinstance(query, dict):
                continue
            question = _stringify(query.get("question"))
            answer = _stringify(query.get("true_hypothesis"))
            if not question or not answer:
                continue
            prompt_parts = []
            if domain_knowledge:
                prompt_parts.append(f"Domain knowledge:\n{domain_knowledge}")
            if dataset_context:
                prompt_parts.append(f"Available datasets:\n{dataset_context}")
            prompt_parts.append(f"Question: {question}")
            task = _make_base(spec, row, len(tasks), split_name)
            task.update({
                "id": f"{spec.key}_{row_idx}_{query_idx}",
                "question": "\n\n".join(prompt_parts),
                "answer": answer,
            })
            task["context"].update({
                "workflow_tags": workflow_tags,
                "question_type": query.get("question_type"),
                "source_domain": row.get("domain"),
            })
            _set_scoring(
                task,
                answer_type="openText",
                scorer_kind="llm_judge",
                scorer_params={"ground_truth": answer},
            )
            tasks.append(task)
            if limit and len(tasks) >= limit:
                return tasks
    return tasks


def _load_meds_bench_mcqa_tasks(
    spec: HFDatasetSpec,
    limit: int | None,
    cache_path: Path,
) -> list[dict[str, Any]]:
    """Load the MedS-Bench MCQA files without invoking the mixed-schema builder."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        logger.warning("huggingface_hub is required for %s: %s", spec.key, exc)
        return []

    files = spec.extra.get("files") or (
        "MCQA/task122_medmcqa_test_set.json",
        "MCQA/task123_pubmedqa_test_set.json",
        "MCQA/task129_headqa_question_answering.jsonl",
        "MCQA/task57_medqa_question_answering_en.jsonl",
        "MCQA/task58_medqa_question_answering_zh.jsonl",
        "MCQA/task59_igakuqa_question_answering.jsonl",
        "MCQA/task60_frenchmedmcqa_question_answering.jsonl",
        "MCQA/task61_rumedbench_question_answering.jsonl",
    )
    tasks: list[dict[str, Any]] = []
    for file_name in files:
        try:
            path = Path(
                hf_hub_download(
                    spec.repo,
                    file_name,
                    repo_type="dataset",
                    cache_dir=str(cache_path),
                )
            )
            payload = json.loads(path.read_text())
        except Exception as exc:
            logger.warning("HF benchmark %s failed to load %s: %s", spec.key, file_name, exc)
            continue
        instances = payload.get("Instances") if isinstance(payload, dict) else None
        if not isinstance(instances, list):
            continue
        for inst_idx, instance in enumerate(instances):
            if not isinstance(instance, dict):
                continue
            question = _stringify(instance.get("input"))
            output = _stringify(instance.get("output"))
            answer = _extract_meds_bench_answer(output)
            if not question or not answer:
                continue
            task = _make_base(spec, {"id": f"{file_name}:{inst_idx}"}, len(tasks), "test")
            choices = _extract_lettered_options_from_text(question)
            task.update({
                "id": f"{spec.key}_{Path(file_name).stem}_{inst_idx}",
                "question": question,
                "answer": answer,
            })
            if choices:
                task["choices"] = choices
            task["context"].update({
                "file": file_name,
                "source": payload.get("Source") if isinstance(payload, dict) else spec.repo,
            })
            _set_scoring(task, answer_type="multipleChoice", scorer_kind="mcq")
            tasks.append(task)
            if limit and len(tasks) >= limit:
                return tasks
    return tasks


def _load_medexqa_tasks(
    spec: HFDatasetSpec,
    limit: int | None,
    cache_path: Path,
) -> list[dict[str, Any]]:
    """Load MedExQA TSV files; HF's CSV builder treats the first row as headers."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        logger.warning("huggingface_hub is required for %s: %s", spec.key, exc)
        return []
    configs = spec.extra.get("configs") or (
        "biomedical_engineer",
        "clinical_laboratory_scientist",
        "clinical_psychologist",
        "occupational_therapist",
        "speech_pathologist",
    )
    tasks: list[dict[str, Any]] = []
    for config in configs:
        file_name = f"test/{config}_test.tsv"
        try:
            path = Path(hf_hub_download(spec.repo, file_name, repo_type="dataset", cache_dir=str(cache_path)))
        except Exception as exc:
            logger.warning("HF benchmark %s failed to load %s: %s", spec.key, file_name, exc)
            continue
        with path.open(newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for row_idx, cols in enumerate(reader):
                if len(cols) < 8:
                    continue
                question = _stringify(cols[0])
                choices = [_stringify(value) for value in cols[1:5]]
                answer = _answer_to_letter(_stringify(cols[-1]), choices)
                if not question or not choices or not answer:
                    continue
                opts = "\n".join(f"{_LETTERS[i]}. {choice}" for i, choice in enumerate(choices))
                task = _make_base(spec, {"id": f"{file_name}:{row_idx}"}, len(tasks), "test")
                task.update({
                    "id": f"{spec.key}_{config}_{row_idx}",
                    "question": f"{question}\n\nOptions:\n{opts}",
                    "choices": choices,
                    "answer": answer,
                })
                task["context"].update({"config": config, "file": file_name})
                _set_scoring(task, answer_type="multipleChoice", scorer_kind="mcq")
                tasks.append(task)
                if limit and len(tasks) >= limit:
                    return tasks
    return tasks


def _load_json_file_tasks(
    spec: HFDatasetSpec,
    limit: int | None,
    cache_path: Path,
    *,
    split_name: str,
) -> list[dict[str, Any]]:
    """Load benchmark JSON files that fail through the generic HF builder."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        logger.warning("huggingface_hub is required for %s: %s", spec.key, exc)
        return []
    files = spec.extra.get("files") or ()
    tasks: list[dict[str, Any]] = []
    for file_name in files:
        try:
            path = Path(hf_hub_download(spec.repo, file_name, repo_type="dataset", cache_dir=str(cache_path)))
            payload = json.loads(path.read_text())
        except Exception as exc:
            logger.warning("HF benchmark %s failed to load %s: %s", spec.key, file_name, exc)
            continue
        records = payload if isinstance(payload, list) else payload.get("Instances") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            continue
        for row_idx, row in enumerate(records):
            if not isinstance(row, dict):
                continue
            row = _expand_row(row)
            question = _first_str(row, spec.question_fields or ("query", "input", "question", "prompt"))
            answer = _extract_answer(row, spec.answer_fields or ("gt", "output", "answer", "target"))
            if not question or not answer:
                continue
            task = _make_base(spec, {"id": f"{file_name}:{row_idx}"}, len(tasks), split_name)
            task.update({
                "id": f"{spec.key}_{Path(file_name).stem}_{row_idx}",
                "question": question,
                "answer": answer,
            })
            task["context"].update({
                "file": file_name,
                "task": row.get("task"),
                "subtask": row.get("subtask"),
            })
            _set_scoring(
                task,
                answer_type="openText",
                scorer_kind="llm_judge",
                scorer_params={"ground_truth": answer},
            )
            tasks.append(task)
            if limit and len(tasks) >= limit:
                return tasks
    return tasks


def _load_mol_instructions_tasks(
    spec: HFDatasetSpec,
    limit: int | None,
    cache_path: Path,
) -> list[dict[str, Any]]:
    """Load Mol-Instructions zip archives without executing its HF dataset script."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        logger.warning("huggingface_hub is required for %s: %s", spec.key, exc)
        return []
    archive = spec.extra.get("archive")
    files = spec.extra.get("files") or ()
    if not archive or not files:
        return []
    try:
        path = Path(hf_hub_download(spec.repo, archive, repo_type="dataset", cache_dir=str(cache_path)))
    except Exception as exc:
        logger.warning("HF benchmark %s failed to load %s: %s", spec.key, archive, exc)
        return []
    tasks: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as zf:
        for file_name in files:
            try:
                records = json.loads(zf.read(file_name).decode("utf-8"))
            except Exception as exc:
                logger.warning("HF benchmark %s failed to read %s: %s", spec.key, file_name, exc)
                continue
            if not isinstance(records, list):
                continue
            for row_idx, row in enumerate(records):
                if not isinstance(row, dict):
                    continue
                instruction = _stringify(row.get("instruction"))
                input_text = _stringify(row.get("input"))
                answer = _stringify(row.get("output"))
                if not instruction or not answer:
                    continue
                question = instruction if not input_text else f"{instruction}\n\nInput: {input_text}"
                task = _make_base(spec, {"id": f"{file_name}:{row_idx}"}, len(tasks), "train")
                task.update({
                    "id": f"{spec.key}_{Path(file_name).stem}_{row_idx}",
                    "question": question,
                    "answer": answer,
                })
                metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                task["context"].update({
                    "file": file_name,
                    "mol_instruction_task": metadata.get("task"),
                    "split": metadata.get("split") or "train",
                })
                _set_scoring(
                    task,
                    answer_type="openText",
                    scorer_kind="llm_judge",
                    scorer_params={"ground_truth": answer},
                )
                tasks.append(task)
                if limit and len(tasks) >= limit:
                    return tasks
    return tasks


def _download_url(url: str, cache_path: Path) -> Path:
    """Download a raw benchmark file into the local HF cache."""
    raw_dir = cache_path / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(url.split("?", 1)[0]).suffix
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    path = raw_dir / f"{digest}{suffix}"
    if path.exists() and path.stat().st_size:
        return path
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())
    return path


def _load_blurb_ner_tasks(
    spec: HFDatasetSpec,
    limit: int | None,
    cache_path: Path,
) -> list[dict[str, Any]]:
    """Load BLURB/BLUE NER TSV files directly from the upstream GitHub mirror.

    This avoids invoking the EMBO/BLURB legacy dataset script, which modern
    ``datasets`` refuses to execute.
    """
    base_url = spec.extra.get(
        "raw_base_url",
        "https://github.com/cambridgeltl/MTL-Bioinformatics-2016/raw/master/data",
    ).rstrip("/")
    configs = spec.extra.get("blurb_ner_configs") or (
        "BC5CDR-chem-IOB",
        "BC5CDR-disease-IOB",
        "BC2GM-IOB",
        "NCBI-disease-IOB",
        "JNLPBA",
    )
    split_files = {
        "train": "train.tsv",
        "validation": "devel.tsv",
        "test": "test.tsv",
    }
    split_name = spec.split or "test"
    file_name = split_files.get(split_name, split_name)
    tasks: list[dict[str, Any]] = []
    for config in configs:
        url = f"{base_url}/{config}/{file_name}"
        try:
            path = _download_url(url, cache_path)
        except Exception as exc:
            logger.warning("HF benchmark %s failed to download %s: %s", spec.key, url, exc)
            continue
        for row_idx, row in enumerate(_iter_iob_tsv(path)):
            if not row["tokens"] or not row["tags"]:
                continue
            task = _make_base(spec, {"id": f"{config}:{split_name}:{row_idx}"}, len(tasks), split_name)
            task.update({
                "id": f"{spec.key}_{config}_{split_name}_{row_idx}",
                "question": (
                    "Tag each token with its biomedical IOB named-entity label. "
                    "Return a JSON list of labels in token order.\n\n"
                    f"Tokens: {json.dumps(row['tokens'], ensure_ascii=False)}"
                ),
                "answer": json.dumps(row["tags"], ensure_ascii=False),
            })
            task["context"].update({
                "source": spec.repo,
                "source_url": url,
                "blurb_config": config,
                "tokens": row["tokens"],
            })
            _set_scoring(task, answer_type="tokenSequence", scorer_kind="token_f1")
            tasks.append(task)
            if limit and len(tasks) >= limit:
                return tasks
    return tasks


def _iter_iob_tsv(path: Path) -> Iterable[dict[str, list[str]]]:
    tokens: list[str] = []
    tags: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                if tokens:
                    yield {"tokens": tokens, "tags": tags}
                    tokens, tags = [], []
                continue
            parts = stripped.split("\t")
            if len(parts) < 2:
                parts = stripped.split()
            if len(parts) < 2:
                continue
            tokens.append(parts[0])
            tags.append(parts[-1])
    if tokens:
        yield {"tokens": tokens, "tags": tags}


def _load_bioinfer_relation_tasks(
    spec: HFDatasetSpec,
    limit: int | None,
    cache_path: Path,
    split_name: str,
) -> list[dict[str, Any]]:
    """Load BioInfer XML directly from the official ppi-dataset mirror."""
    urls = spec.extra.get("raw_urls") or {
        "train": "https://github.com/metalrt/ppi-dataset/raw/master/csv_output/BioInfer-train.xml",
        "test": "https://github.com/metalrt/ppi-dataset/raw/master/csv_output/BioInfer-test.xml",
    }
    url = urls.get(split_name) or urls.get("test")
    if not url:
        return []
    try:
        path = _download_url(url, cache_path)
    except Exception as exc:
        logger.warning("HF benchmark %s failed to download %s: %s", spec.key, url, exc)
        return []
    tasks: list[dict[str, Any]] = []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        logger.warning("HF benchmark %s failed to parse %s: %s", spec.key, url, exc)
        return []
    for sent_idx, sentence in enumerate(root.iter("sentence")):
        text = _stringify(sentence.attrib.get("text"))
        if not text:
            continue
        entities = [_bioinfer_entity(child) for child in sentence if child.tag == "entity"]
        entities = [entity for entity in entities if entity]
        relations = [_bioinfer_relation(child) for child in sentence if child.tag == "interaction"]
        relations = [relation for relation in relations if relation]
        if not relations:
            continue
        task = _make_base(spec, {"id": sentence.attrib.get("id") or sent_idx}, len(tasks), split_name)
        task.update({
            "id": f"{spec.key}_{split_name}_{sent_idx}",
            "question": (
                "Extract all protein/gene/RNA relations from this sentence. "
                "Return a JSON list of objects with type, arg1_id, and arg2_id.\n\n"
                f"Sentence: {text}\n\nEntities: {json.dumps(entities, ensure_ascii=False)}"
            ),
            "answer": json.dumps(relations, ensure_ascii=False),
        })
        task["context"].update({
            "source_url": url,
            "entities": entities,
            "relations": relations,
        })
        _set_scoring(task, answer_type="relationSet", scorer_kind="relation_f1")
        tasks.append(task)
        if limit and len(tasks) >= limit:
            return tasks
    return tasks


def _bioinfer_entity(node: ET.Element) -> dict[str, Any]:
    offsets = []
    for offset in _stringify(node.attrib.get("charOffset")).split(","):
        if "-" not in offset:
            continue
        start, end = offset.split("-", 1)
        try:
            offsets.append([int(start), int(end)])
        except ValueError:
            continue
    return {
        "id": _stringify(node.attrib.get("id")),
        "type": _stringify(node.attrib.get("type")),
        "text": _stringify(node.attrib.get("text")),
        "offsets": offsets,
    }


def _bioinfer_relation(node: ET.Element) -> dict[str, str]:
    return {
        "type": _stringify(node.attrib.get("type")),
        "arg1_id": _stringify(node.attrib.get("e1")),
        "arg2_id": _stringify(node.attrib.get("e2")),
    }


def _load_pgr_relation_tasks(
    spec: HFDatasetSpec,
    limit: int | None,
    cache_path: Path,
    split_name: str,
) -> list[dict[str, Any]]:
    """Load the PGR relation TSV directly from the official GitHub repo."""
    urls = spec.extra.get("raw_urls") or {
        "train": "https://raw.githubusercontent.com/lasigeBioTM/PGR/master/corpora/10_12_2018_corpus/train.tsv",
        "test": "https://raw.githubusercontent.com/lasigeBioTM/PGR/master/corpora/10_12_2018_corpus/test.tsv",
    }
    url = urls.get(split_name) or urls.get("test")
    if not url:
        return []
    try:
        path = _download_url(url, cache_path)
    except Exception as exc:
        logger.warning("HF benchmark %s failed to download %s: %s", spec.key, url, exc)
        return []
    tasks: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row_idx, row in enumerate(reader):
            sentence = _stringify(row.get("SENTENCE"))
            relation_value = _stringify(row.get("RELATION")).upper()
            if not sentence or not relation_value:
                continue
            relation = {
                "type": "phenotype_gene" if relation_value == "TRUE" else "no_relation",
                "arg1": _stringify(row.get("GENE")),
                "arg2": _stringify(row.get("PHENOTYPE")),
            }
            task = _make_base(spec, {"id": f"{row.get('FILE_ID')}:{row_idx}"}, len(tasks), split_name)
            expected_relations = [relation] if relation_value == "TRUE" else []
            task.update({
                "id": f"{spec.key}_{split_name}_{row_idx}",
                "question": (
                    "Extract the phenotype-gene relation from this sentence. "
                    "Return a JSON list with one relation object when the gene and "
                    "phenotype are related, otherwise return an empty JSON list.\n\n"
                    f"Sentence: {sentence}\nGene: {relation['arg1']}\nPhenotype: {relation['arg2']}"
                ),
                "answer": json.dumps(expected_relations, ensure_ascii=False),
            })
            task["context"].update({
                "source_url": url,
                "gene": relation["arg1"],
                "phenotype": relation["arg2"],
                "relation": relation,
                "confirmation": row.get("CONFIRMATION (CORRECT(C) | INCORRECT(I) | UNCERTAIN(U))"),
            })
            _set_scoring(task, answer_type="relationSet", scorer_kind="relation_f1")
            tasks.append(task)
            if limit and len(tasks) >= limit:
                return tasks
    return tasks


def _load_proteingym_tasks(
    spec: HFDatasetSpec,
    limit: int | None,
    cache_path: Path,
) -> list[dict[str, Any]]:
    """Load a small ProteinGym mutation-effect slice from official files."""
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        logger.warning("datasets and huggingface_hub are required for %s: %s", spec.key, exc)
        return []

    rows: list[dict[str, Any]] = []
    reference_by_dms: dict[str, dict[str, str]] = {}
    if spec.extra.get("parquet_glob"):
        try:
            ds = load_dataset(
                "parquet",
                data_files=spec.extra["parquet_glob"],
                split="train",
                cache_dir=str(cache_path),
            )
            for row in ds:
                rows.append(dict(row))
                if limit and len(rows) >= limit:
                    break
        except Exception as exc:
            logger.warning("HF benchmark %s failed to load parquet: %s", spec.key, exc)
            return []
    else:
        try:
            files = HfApi().list_repo_files(spec.repo, repo_type="dataset")
            reference_by_dms = _load_proteingym_reference(hf_hub_download, spec, cache_path)
            candidate = next(
                f for f in files
                if f.startswith("ProteinGym_substitutions/") and f.endswith(".csv")
            )
            path = Path(hf_hub_download(spec.repo, candidate, repo_type="dataset", cache_dir=str(cache_path)))
        except Exception as exc:
            logger.warning("HF benchmark %s failed to locate ProteinGym CSV: %s", spec.key, exc)
            return []
        dms_id = Path(candidate).stem
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["DMS_id"] = dms_id
                row.update({f"reference_{key}": value for key, value in reference_by_dms.get(dms_id, {}).items()})
                rows.append(row)
                if limit and len(rows) >= limit:
                    break

    tasks: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        mutation = _first_str(row, ("mutant", "mutated_sequence", "mutant_sequence", "mutation"))
        sequence = _first_str(row, ("sequence", "mutated_sequence", "mutant_sequence"))
        score = _extract_answer(row, ("DMS_score", "DMS_score_bin", "score", "target", "fitness"))
        if not score:
            score = _first_property_value(row)
        if not mutation and not sequence:
            continue
        task = _make_base(spec, row, len(tasks), spec.split or "train")
        assay_id = _first_str(row, ("DMS_id", "assay_id", "dataset", "protein_id")) or spec.key
        target_seq = _first_str(row, ("target_seq", "wildtype_sequence", "wt_sequence"))
        score_bin = _extract_answer(row, ("DMS_score_bin", "label_bin", "fitness_bin"))
        task.update({
            "id": f"{assay_id}:{len(tasks)}",
            "question": (
                "Predict the measured mutation effect or fitness score for this ProteinGym record.\n\n"
                f"Assay: {assay_id}\nMutation: {mutation or 'unknown'}\n"
                f"Reference sequence: {target_seq[:1000] if target_seq else 'unknown'}\n"
                f"Mutated sequence: {sequence[:1000]}"
            ),
            "answer": score,
        })
        task["context"].update({
            "assay_id": assay_id,
            "DMS_id": assay_id,
            "DMS_score": score,
            "DMS_score_bin": score_bin,
            "mutant": mutation,
            "target_seq": target_seq,
            "uniprot_id": _first_str(row, ("reference_UniProt_ID", "UniProt_ID")),
            "taxon": _first_str(row, ("reference_taxon", "taxon")),
            "selection_type": _first_str(row, ("reference_selection_type", "selection_type")),
            "official_metric": "proteingym_dms_zero_shot",
        })
        answer_type = "exactNumeric" if _looks_numeric(score) else "exactMatch"
        _set_scoring(task, answer_type=answer_type, scorer_kind="exact")
        tasks.append(task)
        if limit and len(tasks) >= limit:
            return tasks
    return tasks


def _load_proteingym_reference(
    hf_hub_download: Any,
    spec: HFDatasetSpec,
    cache_path: Path,
) -> dict[str, dict[str, str]]:
    try:
        path = Path(hf_hub_download(
            spec.repo,
            "ProteinGym_reference_file_substitutions.csv",
            repo_type="dataset",
            cache_dir=str(cache_path),
        ))
    except Exception:
        return {}
    out: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            dms_id = str(row.get("DMS_id") or Path(str(row.get("DMS_filename") or "")).stem)
            if dms_id:
                out[dms_id] = {
                    key: str(row.get(key) or "")
                    for key in ("UniProt_ID", "taxon", "selection_type", "target_seq")
                }
    return out


def _load_rna_downstream_tasks(
    load_dataset: Any,
    spec: HFDatasetSpec,
    limit: int | None,
    split: str | None,
    cache_path: Path,
) -> list[dict[str, Any]]:
    """Load AIDO.RNA downstream tasks with official target fields and splits."""
    try:
        ds = load_dataset(spec.repo, spec.config, cache_dir=str(cache_path))
    except Exception as exc:
        logger.warning("HF benchmark %s failed to load RNA dataset: %s", spec.key, exc)
        return []

    if _is_split_mapping(ds):
        split_names = _rna_splits(ds, spec, split)
    else:
        split_names = [split or spec.split or "train"]

    tasks: list[dict[str, Any]] = []
    for split_name in split_names:
        rows = ds[split_name] if _is_split_mapping(ds) else ds
        for row in rows:
            if not isinstance(row, dict):
                continue
            task = _make_rna_downstream_task(spec, _expand_row(dict(row)), len(tasks), split_name)
            if task is None:
                continue
            tasks.append(task)
            if limit and len(tasks) >= limit:
                return tasks
    return tasks


def _rna_splits(ds: Mapping[str, Any], spec: HFDatasetSpec, split: str | None) -> list[str]:
    if split and split in ds:
        return [split]
    config = str(spec.config or "")
    if config.startswith("splice_site_"):
        test_splits = sorted(name for name in ds.keys() if str(name).startswith("test_"))
        if test_splits:
            return test_splits
    return [_choose_split(ds, spec.split or split)]


def _make_rna_downstream_task(
    spec: HFDatasetSpec,
    row: dict[str, Any],
    idx: int,
    split: str,
) -> dict[str, Any] | None:
    config = str(spec.config or "")
    sequence = _first_str(row, ("sequences", "utr", "sequence", "rna"))
    if not sequence:
        return None

    task = _make_base(spec, row, idx, split)
    task["id"] = f"{spec.key}:{split}:{idx}"
    task["context"].update({
        "config": config,
        "official_source": "genbio-ai/rna-downstream-tasks",
    })

    if config == "mean_ribosome_load":
        answer = _extract_answer(row, ("rl",))
        if not answer:
            return None
        task.update({
            "question": (
                "Predict the mean ribosome load for this 5' UTR sequence. "
                "Return a single numeric value.\n\n"
                f"Sequence: {sequence}"
            ),
            "answer": answer,
        })
        task["context"].update({
            "target_field": "rl",
            "official_metric": "rna_regression_spearman_pearson",
        })
        _set_scoring(task, answer_type="exactNumeric", scorer_kind="exact")
        return task

    if config.startswith("expression_"):
        answer = _extract_answer(row, ("labels",))
        if not answer:
            return None
        task.update({
            "question": (
                "Predict the mRNA expression level for this 5' UTR sequence. "
                "Return a single numeric value.\n\n"
                f"Sequence: {sequence}"
            ),
            "answer": answer,
        })
        task["context"].update({
            "target_field": "labels",
            "fold_id": _extract_answer(row, ("fold_id",)),
            "official_metric": "rna_regression_spearman_pearson",
        })
        _set_scoring(task, answer_type="exactNumeric", scorer_kind="exact")
        return task

    if config == "modification_site":
        answer = _label_vector(row)
        if not answer:
            return None
        label_names = ["Am", "Cm", "Gm", "Tm", "m1A", "m5C", "m5U", "m6A", "m6Am", "m7G", "Psi", "I"]
        task.update({
            "question": (
                "Predict the 12 RNA modification-site labels for this sequence. "
                "Return a JSON array of 12 binary values in this order: "
                f"{', '.join(label_names)}.\n\nSequence: {sequence}"
            ),
            "answer": answer,
        })
        task["context"].update({
            "label_names": label_names,
            "official_metric": "rna_multilabel_macro_micro_f1",
        })
        _set_scoring(task, answer_type="multiLabel", scorer_kind="multilabel_f1")
        return task

    if config.startswith("ncrna_family"):
        answer = _extract_answer(row, ("labels",))
        if not answer:
            return None
        task.update({
            "question": (
                "Classify this small noncoding RNA sequence into its official ncRNA family label. "
                "Return the integer class id only.\n\n"
                f"Sequence: {sequence}"
            ),
            "answer": answer,
        })
        task["context"]["official_metric"] = "rna_multiclass_accuracy_macro_f1"
        _set_scoring(task, answer_type="exactMatch", scorer_kind="exact")
        return task

    if config.startswith("splice_site_"):
        answer = _extract_answer(row, ("labels",))
        if not answer:
            return None
        site = "acceptor" if "acceptor" in config else "donor"
        task.update({
            "question": (
                f"Predict whether this pre-mRNA fragment contains a {site} splice site. "
                "Return 1 for positive and 0 for negative.\n\n"
                f"Sequence: {sequence}"
            ),
            "answer": answer,
        })
        task["context"].update({
            "species_split": split,
            "official_metric": "rna_binary_accuracy_f1_auroc",
        })
        _set_scoring(task, answer_type="exactMatch", scorer_kind="exact")
        return task

    return _make_structured_prediction(spec, row, idx, split)


def _load_bacbench_tasks(
    spec: HFDatasetSpec,
    limit: int | None,
    cache_path: Path,
) -> list[dict[str, Any]]:
    """Load BacBench genomes and join them to the official label CSV files."""
    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        logger.warning("datasets and huggingface_hub are required for %s: %s", spec.key, exc)
        return []

    label_file = "binary_labels.csv" if "antibiotic_resistance" in spec.key else "labels.csv"
    try:
        label_path = Path(hf_hub_download(
            spec.repo,
            label_file,
            repo_type="dataset",
            cache_dir=str(cache_path),
        ))
        labels_by_genome = _load_bacbench_label_table(label_path)
        try:
            shard_path = hf_hub_download(
                spec.repo,
                spec.extra.get("file") or "data/train-00000-of-00050.parquet",
                repo_type="dataset",
                cache_dir=str(cache_path),
            )
            ds = load_dataset("parquet", data_files=shard_path, split="train", cache_dir=str(cache_path))
        except Exception:
            ds = load_dataset(spec.repo, split="train", streaming=True, cache_dir=str(cache_path))
    except Exception as exc:
        logger.warning("HF benchmark %s failed to load BacBench source files: %s", spec.key, exc)
        return []

    tasks: list[dict[str, Any]] = []
    for row in ds:
        if not isinstance(row, dict):
            continue
        label_row = labels_by_genome.get(str(row.get("genome_name") or ""))
        if not label_row:
            continue
        for label_name, label_value in _iter_bacbench_labels(label_row):
            task = _make_bacbench_task(spec, row, label_name, label_value, len(tasks))
            if task is None:
                continue
            tasks.append(task)
            if limit and len(tasks) >= limit:
                return tasks
    return tasks


def _load_bacbench_label_table(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {
            str(row.get("genome_name") or ""): row
            for row in reader
            if row.get("genome_name")
        }


def _iter_bacbench_labels(row: dict[str, str]) -> Iterable[tuple[str, str]]:
    skip = {"genome_name", "species", "genus", "family"}
    for key, value in row.items():
        if key in skip or _is_missing_value(value):
            continue
        yield key, str(value)


def _make_bacbench_task(
    spec: HFDatasetSpec,
    row: dict[str, Any],
    label_name: str,
    label_value: str,
    idx: int,
) -> dict[str, Any] | None:
    genome_name = _first_str(row, ("genome_name",))
    sequence = _first_str(row, ("dna_sequence", "sequence", "dna"))
    if not genome_name or not sequence:
        return None
    label_kind = _bacbench_label_kind(label_name, label_value)
    task = _make_base(spec, row, idx, "train")
    prompt_sequence = sequence[:_BACBENCH_SEQUENCE_PROMPT_LIMIT]
    task["id"] = f"{spec.key}:{genome_name}:{label_name}:{idx}"
    task["context"].update({
        "genome_name": genome_name,
        "label_name": label_name,
        "label_kind": label_kind,
        "taxid": _first_str(row, ("taxid",)),
        "official_split": "5-fold cross-validation recommended by dataset card",
        "sequence_length": len(sequence),
        "prompt_sequence_length": len(prompt_sequence),
        "prompt_sequence_truncated": len(prompt_sequence) < len(sequence),
    })

    if label_kind == "regression":
        task.update({
            "question": (
                "Predict this BacBench quantitative phenotype or MIC value from the bacterial genome DNA. "
                "Return a single numeric value.\n\n"
                f"Genome: {genome_name}\nTarget: {label_name}\nDNA sequence: {prompt_sequence}"
            ),
            "answer": label_value,
        })
        task["context"]["official_metric"] = "bacbench_regression_r2"
        _set_scoring(task, answer_type="exactNumeric", scorer_kind="exact")
        return task

    if label_kind == "binary":
        task.update({
            "question": (
                "Predict this BacBench binary bacterial trait from the genome DNA. "
                "Return 1 for positive/resistant and 0 for negative/susceptible.\n\n"
                f"Genome: {genome_name}\nTarget: {label_name}\nDNA sequence: {prompt_sequence}"
            ),
            "answer": str(int(float(label_value))),
        })
        task["context"]["official_metric"] = "bacbench_binary_auprc"
        _set_scoring(task, answer_type="exactMatch", scorer_kind="exact")
        return task

    task.update({
        "question": (
            "Predict this BacBench categorical bacterial phenotype from the genome DNA. "
            "Return the exact category label.\n\n"
            f"Genome: {genome_name}\nTarget: {label_name}\nDNA sequence: {prompt_sequence}"
        ),
        "answer": label_value,
    })
    task["context"]["official_metric"] = "bacbench_multiclass_accuracy_macro_f1"
    _set_scoring(task, answer_type="exactMatch", scorer_kind="exact")
    return task


def _bacbench_label_kind(label_name: str, label_value: str) -> str:
    if label_name.startswith("madin_quantitative_"):
        return "regression"
    try:
        numeric = float(label_value)
    except ValueError:
        return "multiclass"
    if numeric in (0.0, 1.0):
        return "binary"
    return "regression"


def _is_missing_value(value: Any) -> bool:
    text = str(value or "").strip()
    return not text or text.lower() in {"nan", "none", "null", "na", "n/a"}


def _load_mteb_retrieval_tasks(
    load_dataset: Any,
    spec: HFDatasetSpec,
    limit: int | None,
    cache_path: Path,
) -> list[dict[str, Any]]:
    """Turn MTEB query/corpus/qrels triples into candidate reranking tasks."""
    split = "queries" if spec.repo == "mteb/medical_qa" else "dev"
    qrels_split = "test" if spec.repo == "mteb/medical_qa" else "dev"
    try:
        queries = load_dataset(spec.repo, "queries", split=split, cache_dir=str(cache_path))
        corpus = load_dataset(spec.repo, "corpus", split="corpus" if spec.repo == "mteb/medical_qa" else "dev", cache_dir=str(cache_path))
        qrels = load_dataset(spec.repo, "default", split=qrels_split, cache_dir=str(cache_path))
    except Exception as exc:
        logger.warning("HF benchmark %s failed to load MTEB retrieval triples: %s", spec.key, exc)
        return []
    query_by_id = {str(row.get("_id")): _stringify(row.get("text")) for row in queries if isinstance(row, dict)}
    corpus_by_id: dict[str, str] = {}
    for row in corpus:
        if not isinstance(row, dict):
            continue
        doc_id = str(row.get("_id"))
        text = " ".join(part for part in (
            _stringify(row.get("title")),
            _stringify(row.get("text")),
        ) if part)
        if doc_id and text:
            corpus_by_id[doc_id] = text
    qrels_by_query: dict[str, dict[str, float]] = {}
    for row in qrels:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("query-id"))
        cid = str(row.get("corpus-id"))
        if not qid or not cid or cid not in corpus_by_id:
            continue
        try:
            score = float(row.get("score", 1))
        except (TypeError, ValueError):
            score = 1.0
        if score <= 0:
            continue
        qrels_by_query.setdefault(qid, {})[cid] = score
    corpus_ids = list(corpus_by_id)
    tasks: list[dict[str, Any]] = []
    for qid, relevant_scores in qrels_by_query.items():
        question = query_by_id.get(qid)
        if not question:
            continue
        relevant_ids = list(relevant_scores)
        candidate_ids = _mteb_candidate_doc_ids(qid, relevant_ids, corpus_ids, limit=20)
        candidate_text = "\n\n".join(
            f"[{doc_id}] {corpus_by_id[doc_id][:1200]}"
            for doc_id in candidate_ids
            if doc_id in corpus_by_id
        )
        task = _make_base(spec, {"id": qid}, len(tasks), qrels_split)
        task.update({
            "id": qid,
            "question": (
                "Rank the candidate medical documents for relevance to the query. "
                "Return a JSON list of document ids ordered from most to least relevant.\n\n"
                f"Query: {question}\n\nCandidates:\n{candidate_text}"
            ),
            "answer": json.dumps(relevant_ids, ensure_ascii=False),
        })
        task["context"].update({
            "query_id": qid,
            "relevant_doc_ids": relevant_ids,
            "candidate_doc_ids": candidate_ids,
            "qrel_scores": relevant_scores,
            "official_metric": "candidate_ndcg_mrr_recall",
        })
        _set_scoring(
            task,
            answer_type="ranking",
            scorer_kind="retrieval_hit",
            scorer_params={"relevant_doc_ids": relevant_ids},
        )
        tasks.append(task)
        if limit and len(tasks) >= limit:
            return tasks
    return tasks


def _mteb_candidate_doc_ids(
    query_id: str,
    relevant_ids: list[str],
    corpus_ids: list[str],
    *,
    limit: int,
) -> list[str]:
    relevant = [doc_id for doc_id in relevant_ids if doc_id in corpus_ids]
    seed = int(hashlib.sha256(query_id.encode("utf-8")).hexdigest()[:8], 16)
    start = seed % max(len(corpus_ids), 1)
    distractors: list[str] = []
    relevant_set = set(relevant)
    for offset in range(len(corpus_ids)):
        doc_id = corpus_ids[(start + offset) % len(corpus_ids)]
        if doc_id not in relevant_set:
            distractors.append(doc_id)
        if len(relevant) + len(distractors) >= limit:
            break
    combined = relevant + distractors
    # Deterministic shuffle so the relevant document is not always first.
    return sorted(combined, key=lambda doc_id: hashlib.sha256(f"{query_id}:{doc_id}".encode("utf-8")).hexdigest())


def _extract_meds_bench_answer(output: str) -> str:
    m = re.search(r"\b(?:right\s+answer|answer)\s+is\s+([A-Z])\b", output, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"^\s*([A-Z])\s*[:.]", output)
    if m:
        return m.group(1).upper()
    return ""


def _extract_lettered_options_from_text(text: str) -> list[str]:
    if "Options:" not in text:
        return []
    options_text = text.split("Options:", 1)[1]
    pattern = re.compile(r"(?:^|\s|\t)([A-Z])[\.:]\s*(.*?)(?=(?:\s|\t)[A-Z][\.:]\s*|$)", re.DOTALL)
    matches = pattern.findall(options_text)
    if not matches:
        return []
    ordered = []
    for letter, value in matches:
        expected = _LETTERS[len(ordered)]
        if letter.upper() != expected:
            break
        cleaned = re.sub(r"\s+", " ", value).strip(" \t\n.")
        if not cleaned:
            break
        ordered.append(cleaned)
    return ordered


def _format_discoverybench_datasets(datasets: Any) -> str:
    if not isinstance(datasets, list):
        return ""
    chunks = []
    for item in datasets:
        if not isinstance(item, dict):
            continue
        name = _stringify(item.get("name"))
        description = _stringify(item.get("description"))
        columns = []
        column_groups = item.get("columns")
        if isinstance(column_groups, dict):
            for group_items in column_groups.values():
                if not isinstance(group_items, list):
                    continue
                for column in group_items:
                    if isinstance(column, dict):
                        col_name = _stringify(column.get("name"))
                        col_desc = _stringify(column.get("description"))
                        if col_name:
                            columns.append(f"{col_name}: {col_desc}" if col_desc else col_name)
        header = name or "dataset"
        body = description
        if columns:
            body = f"{body}\nColumns: " + "; ".join(columns[:30])
        chunks.append(f"- {header}: {body}".strip())
    return "\n".join(chunks)


def _load_cmb_exam_tasks(
    load_dataset: Any,
    spec: HFDatasetSpec,
    limit: int | None,
    cache_path: Path,
) -> list[dict[str, Any]]:
    """Load CMB-Exam test questions and the official public answer file."""
    question_file = spec.extra.get(
        "data_file",
        "hf://datasets/FreedomIntelligence/CMB/CMB-Exam/CMB-test/CMB-test-choice-question-merge.json",
    )
    answer_url = spec.extra.get(
        "answer_url",
        "https://raw.githubusercontent.com/FreedomIntelligence/CMB/main/data/CMB-test-choice-answer.json",
    )
    try:
        question_path = _resolve_hf_or_url_file(question_file, cache_path)
        ds = load_dataset("json", data_files=str(question_path), split="train", cache_dir=str(cache_path))
        answers = _load_json_url_cached(answer_url, cache_path)
    except Exception as exc:
        logger.warning("HF benchmark %s failed to load: %s", spec.key, exc)
        return []
    answer_by_id = {str(row.get("id")): _stringify(row.get("answer")) for row in answers if isinstance(row, dict)}
    tasks: list[dict[str, Any]] = []
    for row in ds:
        if not isinstance(row, dict):
            continue
        answer = answer_by_id.get(str(row.get("id")))
        if not answer:
            continue
        merged = dict(row)
        merged["answer"] = answer
        task = _make_mcq(spec, merged, len(tasks), "test")
        if task is None:
            continue
        task["context"]["question_type"] = row.get("question_type")
        task["context"]["exam_type"] = row.get("exam_type")
        task["context"]["exam_class"] = row.get("exam_class")
        task["context"]["exam_subject"] = row.get("exam_subject")
        tasks.append(task)
        if limit and len(tasks) >= limit:
            break
    return tasks


def _resolve_hf_or_url_file(path_or_url: str, cache_path: Path) -> Path:
    if not path_or_url.startswith("hf://"):
        return Path(path_or_url)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for hf:// files") from exc
    marker = "hf://datasets/"
    if not path_or_url.startswith(marker):
        raise ValueError(f"Unsupported HF file URL: {path_or_url}")
    repo_and_file = path_or_url[len(marker):]
    repo_parts = repo_and_file.split("/", 2)
    if len(repo_parts) != 3:
        raise ValueError(f"Unsupported HF dataset URL: {path_or_url}")
    repo = f"{repo_parts[0]}/{repo_parts[1]}"
    filename = repo_parts[2]
    try:
        return Path(hf_hub_download(repo, filename, repo_type="dataset", cache_dir=str(cache_path)))
    except Exception:
        return Path(
            hf_hub_download(
                repo,
                filename,
                repo_type="dataset",
                cache_dir=str(cache_path),
                local_files_only=True,
            )
        )


def _load_json_url_cached(url: str, cache_path: Path) -> Any:
    cache_path.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    local_path = cache_path / f"url-{digest}.json"
    if local_path.exists():
        return json.loads(local_path.read_text(encoding="utf-8"))
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = response.read()
    local_path.write_bytes(payload)
    return json.loads(payload.decode("utf-8"))


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
    if spec.key in {"hf_bacbench_antibiotic_resistance_dna", "hf_bacbench_phenotypic_traits_dna"}:
        from huggingface_hub import hf_hub_download
        shard = spec.extra.get("file") or "default/partial-train/0000.parquet"
        path = hf_hub_download(
            spec.repo,
            shard,
            repo_type="dataset",
            revision=spec.extra.get("file_revision", "refs/convert/parquet"),
            cache_dir=str(cache_path),
        )
        return load_dataset("parquet", data_files=path, split="train", cache_dir=str(cache_path))
    if spec.extra.get("parquet_glob"):
        kwargs = {"cache_dir": str(cache_path)}
        data_format = spec.extra.get("data_format")
        if not data_format:
            data_format = "csv" if str(spec.extra["parquet_glob"]).lower().endswith(".csv") else "parquet"
        return load_dataset(data_format, data_files=spec.extra["parquet_glob"], split="train", **kwargs)
    if spec.extra.get("parquet_config"):
        parquet_split = spec.extra.get("parquet_split") or split or spec.split or "train"
        parquet_revision = spec.extra.get("parquet_revision", "refs/convert/parquet")
        parquet_path = (
            f"hf://datasets/{spec.repo}@{parquet_revision}/"
            f"{spec.extra['parquet_config']}/{parquet_split}/*.parquet"
        )
        kwargs = {"cache_dir": str(cache_path)}
        return load_dataset("parquet", data_files=parquet_path, split="train", **kwargs)
    kwargs = {"cache_dir": str(cache_path)}
    if streaming:
        kwargs["streaming"] = True
    load_split = spec.extra.get("loader_split", split)
    if load_split:
        kwargs["split"] = load_split
    repo = spec.extra.get("loader_repo", spec.repo)
    config = spec.extra.get("loader_config", spec.config)
    if config:
        return load_dataset(repo, config, **kwargs)
    try:
        return load_dataset(repo, **kwargs)
    except Exception:
        # Some HF repos need split omitted first because they expose custom
        # split names or no split metadata until the builder is initialized.
        if load_split:
            kwargs.pop("split", None)
            return load_dataset(repo, **kwargs)
        raise


def _choose_split(ds: dict[str, Any], preferred: str | None) -> str:
    if preferred and preferred in ds:
        return preferred
    for candidate in ("test", "validation", "valid", "dev", "eval", "train"):
        if candidate in ds:
            return candidate
    return next(iter(ds))


def _is_split_mapping(ds: Any) -> bool:
    return isinstance(ds, Mapping) or (hasattr(ds, "keys") and hasattr(ds, "__getitem__"))


def _normalise_row(
    spec: HFDatasetSpec,
    row: dict[str, Any],
    idx: int,
    split: str,
) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    row = _expand_row(row)
    if spec.key == "hf_chembench":
        return _make_chembench(spec, row, idx, split)
    if spec.key == "hf_mollangbench":
        return _make_mollangbench(spec, row, idx, split)
    if spec.key == "hf_nlmchem":
        return _make_nlmchem(spec, row, idx, split)
    if spec.key == "hf_longhealth":
        return _make_longhealth(spec, row, idx, split)
    if spec.key == "hf_biosses":
        return _make_pair_regression(spec, row, idx, split)
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
    answer_letter = _answer_to_letter(answer, choices, index_base=spec.extra.get("answer_index_base"))
    valid_letters = set(_LETTERS[:len(choices)])
    if not answer_letter or any(letter not in valid_letters for letter in answer_letter):
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


def _make_chembench(spec: HFDatasetSpec, row: dict[str, Any], idx: int, split: str) -> dict[str, Any] | None:
    examples = row.get("examples")
    if not isinstance(examples, list) or not examples or not isinstance(examples[0], dict):
        return _make_qa(spec, row, idx, split)
    example = examples[0]
    question = _stringify(example.get("input"))
    target_scores = _parse_json_like(example.get("target_scores"))
    if not question or not isinstance(target_scores, dict):
        return None
    choices = [str(choice) for choice in target_scores]
    correct = ""
    for choice, score in target_scores.items():
        try:
            if float(score) > 0:
                correct = str(choice)
                break
        except (TypeError, ValueError):
            continue
    answer_letter = _answer_to_letter(correct, choices)
    if not answer_letter or answer_letter not in _LETTERS[:len(choices)]:
        return None
    opts = "\n".join(f"{_LETTERS[i]}. {choice}" for i, choice in enumerate(choices[:26]))
    task = _make_base(spec, row, idx, split)
    task.update({
        "id": str(row.get("uuid") or task["id"]),
        "question": f"{question}\n\nOptions:\n{opts}",
        "choices": choices,
        "answer": answer_letter,
    })
    task["context"].update({
        "chembench_metric": row.get("preferred_score") or "multiple_choice_grade",
        "subfield": row.get("subfield"),
        "keywords": row.get("keywords"),
    })
    _set_scoring(task, answer_type="multipleChoice", scorer_kind="mcq")
    return task


def _make_mollangbench(spec: HFDatasetSpec, row: dict[str, Any], idx: int, split: str) -> dict[str, Any] | None:
    config = spec.config or ""
    if config == "edit":
        original = _first_str(row, ("original_smiles", "smiles"))
        instruction = _first_str(row, ("edit_instructions", "instruction"))
        answer = _first_str(row, ("edited_smiles", "target"))
        if not original or not instruction or not answer:
            return None
        question = (
            "Apply the molecular edit instruction to the input SMILES and return "
            f"the edited SMILES.\n\nInput SMILES: {original}\nInstruction: {instruction}"
        )
    elif config == "generation":
        description = _first_str(row, ("structure_description", "description", "input"))
        answer = _first_str(row, ("smiles", "target"))
        if not description or not answer:
            return None
        question = f"Generate a SMILES string matching this molecular description.\n\n{description}"
    elif config == "recognition":
        smiles = _first_str(row, ("smiles",))
        task = _first_str(row, ("task",))
        note = _first_str(row, ("note",))
        answer = _first_str(row, ("result_1", "result_2", "target"))
        if not smiles or not task or not answer:
            return None
        question = f"Analyze this molecule for the requested functional group task.\n\nSMILES: {smiles}\nTask: {task}\n{note}"
    else:
        return _make_qa(spec, row, idx, split)
    task = _make_base(spec, row, idx, split)
    task.update({
        "question": question,
        "answer": answer,
    })
    _set_scoring(task, answer_type="exactMatch", scorer_kind="exact")
    return task


def _make_nlmchem(spec: HFDatasetSpec, row: dict[str, Any], idx: int, split: str) -> dict[str, Any] | None:
    text = _first_str(row, ("text", "input"))
    answer = _first_str(row, ("output",))
    if not answer and "Result:" in text:
        prompt, answer = text.rsplit("Result:", 1)
        question = prompt.strip()
        answer = answer.strip()
    else:
        question = text
    if not question or not answer:
        return None
    task = _make_base(spec, row, idx, split)
    task.update({
        "question": question,
        "answer": answer,
    })
    _set_scoring(task, answer_type="exactMatch", scorer_kind="exact")
    return task


def _make_longhealth(spec: HFDatasetSpec, row: dict[str, Any], idx: int, split: str) -> dict[str, Any] | None:
    prompt = _first_str(row, ("prompt", "query", "question"))
    extra_info = _parse_json_like(row.get("extra_info"))
    ground_truth = extra_info.get("ground_truth") if isinstance(extra_info, dict) else None
    answer = ""
    if isinstance(ground_truth, dict):
        answer = _stringify(ground_truth.get("answer_letter") or ground_truth.get("answer_text"))
    if not prompt or not answer:
        return _make_qa(spec, row, idx, split)
    task = _make_base(spec, row, idx, split)
    task.update({
        "question": prompt,
        "answer": answer,
    })
    if isinstance(extra_info, dict):
        task["context"].update({
            "patient_id": extra_info.get("patient_id"),
            "question_no": extra_info.get("question_no"),
            "variant": extra_info.get("variant"),
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
    scorer = spec.extra.get("scorer")
    if scorer in {"smiles_topk_canonical_match", "smiles_validity_plus_tanimoto", "regression_spearman_auc", "pio_span_f1"}:
        _set_scoring(task, answer_type="exactMatch", scorer_kind=scorer)
        return task
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
    if spec.key in {"hf_anatem", "hf_bc2gm"}:
        return _make_token_sequence_classification(spec, row, idx, split)
    if spec.key == "hf_litcovid":
        return _make_multilabel_classification(spec, row, idx, split)

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


def _make_token_sequence_classification(
    spec: HFDatasetSpec,
    row: dict[str, Any],
    idx: int,
    split: str,
) -> dict[str, Any] | None:
    tokens = _token_list(row, spec.text_fields or ("tokens",))
    labels = _sequence_label_list(row, spec.label_fields or spec.answer_fields or ("ner_tags",))
    if not tokens or not labels:
        return _make_classification_fallback(spec, row, idx, split)
    if len(tokens) != len(labels):
        labels = labels[:len(tokens)]
    task = _make_base(spec, row, idx, split)
    task.update({
        "question": (
            "Label each token in this biomedical sentence with BIO tags. "
            "Return a JSON list with one tag per token.\n\n"
            f"Tokens: {json.dumps(tokens, ensure_ascii=False)}"
        ),
        "answer": json.dumps(labels, ensure_ascii=False),
    })
    _set_scoring(task, answer_type="tokenSequence", scorer_kind="token_f1")
    task["context"].update({"tokens": tokens, "official_metric": "entity_level_f1"})
    return task


def _make_multilabel_classification(
    spec: HFDatasetSpec,
    row: dict[str, Any],
    idx: int,
    split: str,
) -> dict[str, Any] | None:
    answer = _extract_answer(row, spec.label_fields or spec.answer_fields or _ANSWER_FIELDS)
    if not answer:
        answer = _label_vector(row) or _first_property_value(row)
    if not answer:
        return _make_qa(spec, row, idx, split)
    text = _first_str(row, spec.text_fields or _TEXT_FIELDS) or _row_preview(row)
    task = _make_base(spec, row, idx, split)
    task.update({
        "question": f"Assign all applicable biomedical labels for this example.\n\n{text}",
        "answer": _json_label_answer(answer),
    })
    _set_scoring(task, answer_type="multiLabel", scorer_kind="multilabel_f1")
    task["context"]["official_metric"] = "macro_micro_instance_f1"
    return task


def _make_classification_fallback(
    spec: HFDatasetSpec,
    row: dict[str, Any],
    idx: int,
    split: str,
) -> dict[str, Any] | None:
    answer = _extract_answer(row, spec.label_fields or spec.answer_fields or _ANSWER_FIELDS)
    if not answer:
        answer = _label_vector(row) or _first_property_value(row)
    if not answer:
        return _make_qa(spec, row, idx, split)
    text = _first_str(row, spec.text_fields or _TEXT_FIELDS) or _row_preview(row)
    task = _make_base(spec, row, idx, split)
    task.update({"question": f"Classify this {spec.domain} example.\n\n{text}", "answer": answer})
    _set_scoring(task, answer_type="exactMatch", scorer_kind="exact")
    return task


def _make_pair_regression(spec: HFDatasetSpec, row: dict[str, Any], idx: int, split: str) -> dict[str, Any] | None:
    text_a = _first_str(row, ("sentence1", "text1", "premise"))
    text_b = _first_str(row, ("sentence2", "text2", "hypothesis"))
    answer = _extract_answer(row, spec.answer_fields or ("score", "label", "similarity_score"))
    if not text_a or not text_b or not answer:
        return _make_structured_prediction(spec, row, idx, split)
    task = _make_base(spec, row, idx, split)
    task.update({
        "question": (
            "Rate the semantic similarity of these biomedical sentence pairs "
            "using the dataset's numeric scale.\n\n"
            f"Sentence A: {text_a}\n\nSentence B: {text_b}"
        ),
        "answer": answer,
    })
    _set_scoring(task, answer_type="exactNumeric", scorer_kind="exact")
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
    if spec.task_type == "protein_fitness":
        assay_id = _first_str(row, ("DMS_id", "assay_id", "dataset", "protein_id", "fold_id")) or spec.key
        task["context"].update({
            "assay_id": assay_id,
            "DMS_id": assay_id,
            "DMS_score": answer,
            "DMS_score_bin": _extract_answer(row, ("DMS_score_bin", "label_bin", "fitness_bin")),
            "official_metric": "proteingym_dms_zero_shot",
        })
    if spec.task_type == "molecule_property":
        task["context"]["official_metric"] = _moleculenet_official_metric(spec, answer)
        if _is_binary_label(answer):
            answer_type = "exactMatch"
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


def _token_list(row: dict[str, Any], fields: Iterable[str]) -> list[str]:
    for field in fields:
        if field not in row:
            continue
        value = row[field]
        parsed = _parse_json_like(value)
        if parsed is not None:
            value = parsed
        if isinstance(value, list):
            tokens = [_stringify(item) for item in value]
            return [token for token in tokens if token]
        text = _stringify(value)
        if text:
            return [token for token in re.split(r"\s+", text) if token]
    return []


def _sequence_label_list(row: dict[str, Any], fields: Iterable[str]) -> list[str]:
    for field in fields:
        if field not in row:
            continue
        value = row[field]
        parsed = _parse_json_like(value)
        if parsed is not None:
            value = parsed
        if isinstance(value, list):
            raw_labels = [_stringify(item) for item in value]
        else:
            raw_labels = [part for part in re.split(r"[\s,]+", _stringify(value)) if part]
        if not raw_labels:
            continue
        if all(label in {"0", "1", "0.0", "1.0", "False", "True", "false", "true"} for label in raw_labels):
            return _binary_sequence_to_bio(raw_labels, entity_type="GENE")
        return raw_labels
    return []


def _binary_sequence_to_bio(labels: list[str], *, entity_type: str) -> list[str]:
    bio: list[str] = []
    inside = False
    for raw_label in labels:
        active = str(raw_label).strip().lower() in {"1", "1.0", "true"}
        if active:
            prefix = "I" if inside else "B"
            bio.append(f"{prefix}-{entity_type}")
            inside = True
        else:
            bio.append("O")
            inside = False
    return bio


def _json_label_answer(answer: str) -> str:
    parsed = _parse_json_like(answer)
    if isinstance(parsed, list):
        return json.dumps(parsed, ensure_ascii=False)
    parts = [part.strip() for part in re.split(r"[\n,;|]+", str(answer)) if part.strip()]
    if len(parts) > 1:
        return json.dumps(parts, ensure_ascii=False)
    return json.dumps([answer], ensure_ascii=False)


def _is_binary_label(answer: str) -> bool:
    return str(answer).strip() in {"0", "1", "0.0", "1.0", "False", "True", "false", "true"}


def _moleculenet_official_metric(spec: HFDatasetSpec, answer: str) -> str:
    repo = spec.repo.lower()
    key = spec.key.lower()
    if any(name in repo or name in key for name in ("esol", "freesolv", "lipo", "lipophilicity")):
        return "mae_rmse"
    if _is_binary_label(answer):
        return "roc_auc"
    return "dataset_level_metric"


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
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
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
        parsed = _parse_json_like(value)
        if isinstance(parsed, (dict, list)):
            value = parsed
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
    for idx in range(1, 27):
        key = f"option{idx}"
        if key in row and _stringify(row[key]):
            lettered.append(_stringify(row[key]))
    if lettered:
        return lettered
    lettered = []
    for idx in range(1, 27):
        key = f"op{idx}"
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


def _answer_to_letter(answer: str, choices: list[str], *, index_base: int | None = None) -> str:
    ans = str(answer).strip()
    option_match = re.fullmatch(r"option\s+(\d+)", ans, flags=re.IGNORECASE)
    if option_match:
        ans = option_match.group(1)
    if len(ans) == 1 and ans.upper() in _LETTERS:
        return ans.upper()
    if ans.isdigit():
        idx = int(ans)
        if index_base == 0 and 0 <= idx < len(choices):
            return _LETTERS[idx]
        if index_base == 1 and 1 <= idx <= len(choices):
            return _LETTERS[idx - 1]
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
