"""Dataset-level metrics for official benchmark reproduction.

The per-task scorer answers "did this one generated answer match?". Several
biomedical benchmarks instead report aggregate ranking or predictive metrics
such as ROC-AUC, PRC-AUC, Spearman, MAE/RMSE, nDCG, MRR, or Recall@k. This
module computes those metrics from saved per-question records when the record
format contains enough information, and returns an explicit unavailable reason
when it does not.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable

from harness.eval.scoring import extract_numeric_answer


def compute_official_metrics(benchmark: str, records: Iterable[Any]) -> dict[str, Any]:
    """Compute official-style aggregate metrics for a benchmark result set."""
    rows = [_normalise_record(record) for record in records]
    rows = [row for row in rows if row]
    if not rows:
        return {"status": "unavailable", "reason": "no_records"}

    key = str(benchmark or "").lower()
    contexts = [row.get("context") or {} for row in rows]
    dataset_keys = {
        str(ctx.get("dataset_key", "")).lower()
        for ctx in contexts
        if isinstance(ctx, dict)
    }
    joined = " ".join([key, *sorted(dataset_keys)])

    if "moleculenet" in joined or "moleculeace" in joined:
        return _compute_predictive_metrics(rows, benchmark_family="moleculenet")
    if "mteb_medical" in joined or "mteb/medical" in joined or "medicalretrieval" in joined:
        return _compute_retrieval_metrics(rows)
    if "proteingym" in joined or "protein_gym" in joined:
        return _compute_proteingym_metrics(rows)
    if "rna_downstream" in joined or "rna-" in joined or "rna_" in joined:
        return _compute_rna_metrics(rows)
    if "bacbench" in joined:
        return _compute_bacbench_metrics(rows)

    return {"status": "not_applicable", "reason": "no_official_dataset_level_metric_registered"}


def _normalise_record(record: Any) -> dict[str, Any]:
    if is_dataclass(record):
        record = asdict(record)
    elif not isinstance(record, dict) and hasattr(record, "__dict__"):
        record = dict(record.__dict__)
    if not isinstance(record, dict):
        return {}
    return {
        "id": record.get("id") or record.get("question_id"),
        "expected": record.get("expected", ""),
        "predicted": record.get("predicted", ""),
        "predicted_raw": record.get("predicted_raw", record.get("predicted", "")),
        "context": record.get("context") or {},
        "task_success": record.get("task_success"),
    }


def _compute_predictive_metrics(rows: list[dict[str, Any]], *, benchmark_family: str) -> dict[str, Any]:
    pairs: list[tuple[list[float], list[float]]] = []
    skipped = 0
    for row in rows:
        y_true = _parse_numeric_vector(row.get("expected", ""))
        y_pred = _parse_numeric_vector(row.get("predicted", "")) or _parse_numeric_vector(row.get("predicted_raw", ""))
        if not y_true or not y_pred:
            skipped += 1
            continue
        width = min(len(y_true), len(y_pred))
        if width <= 0:
            skipped += 1
            continue
        pairs.append((y_true[:width], y_pred[:width]))

    if not pairs:
        return {
            "status": "unavailable",
            "benchmark_family": benchmark_family,
            "reason": "no_numeric_or_label_predictions",
            "n_records": len(rows),
            "n_skipped": skipped,
        }

    max_width = max(len(true) for true, _ in pairs)
    is_binary = True
    for true, _ in pairs:
        if any(value not in (0.0, 1.0) for value in true):
            is_binary = False
            break

    if is_binary:
        by_task = []
        for idx in range(max_width):
            y_true = [true[idx] for true, pred in pairs if idx < len(true) and idx < len(pred)]
            y_pred = [pred[idx] for true, pred in pairs if idx < len(true) and idx < len(pred)]
            if len(set(y_true)) < 2:
                continue
            by_task.append({
                "task_index": idx,
                "roc_auc": _roc_auc(y_true, y_pred),
                "prc_auc": _average_precision(y_true, y_pred),
                "accuracy": _accuracy(y_true, [_threshold_label(v) for v in y_pred]),
                "f1": _binary_f1(y_true, [_threshold_label(v) for v in y_pred]),
                "n": len(y_true),
            })
        if not by_task:
            return {
                "status": "unavailable",
                "benchmark_family": benchmark_family,
                "reason": "classification_labels_have_single_class",
                "n_records": len(rows),
                "n_scored": len(pairs),
                "n_skipped": skipped,
            }
        return {
            "status": "ok",
            "benchmark_family": benchmark_family,
            "metric_type": "classification",
            "roc_auc": _mean(item["roc_auc"] for item in by_task),
            "prc_auc": _mean(item["prc_auc"] for item in by_task),
            "accuracy": _mean(item["accuracy"] for item in by_task),
            "f1": _mean(item["f1"] for item in by_task),
            "n_records": len(rows),
            "n_scored": len(pairs),
            "n_skipped": skipped,
            "n_tasks_scored": len(by_task),
        }

    y_true = [true[0] for true, pred in pairs if true and pred]
    y_pred = [pred[0] for true, pred in pairs if true and pred]
    return {
        "status": "ok",
        "benchmark_family": benchmark_family,
        "metric_type": "regression",
        "mae": _mae(y_true, y_pred),
        "rmse": _rmse(y_true, y_pred),
        "spearman": _spearman(y_true, y_pred),
        "n_records": len(rows),
        "n_scored": len(y_true),
        "n_skipped": skipped,
    }


def _compute_proteingym_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    skipped = 0
    for row in rows:
        ctx = row.get("context") or {}
        true_score = _first_numeric(row.get("expected", ""))
        pred_score = _first_available_numeric(row.get("predicted", ""), row.get("predicted_raw", ""))
        if true_score is None or pred_score is None:
            skipped += 1
            continue
        assay_id = str(ctx.get("assay_id") or ctx.get("DMS_id") or ctx.get("fold_id") or "default_assay")
        score_bin = _first_numeric(ctx.get("DMS_score_bin"))
        groups.setdefault(assay_id, []).append({
            "true": true_score,
            "pred": pred_score,
            "bin": score_bin,
        })

    assay_metrics = []
    for assay_id, items in groups.items():
        if len(items) < 2:
            continue
        y_true = [item["true"] for item in items]
        y_pred = [item["pred"] for item in items]
        y_bin = [item["bin"] for item in items]
        metrics = {
            "assay_id": assay_id,
            "n": len(items),
            "spearman": _spearman(y_true, y_pred),
            "ndcg@10%": _continuous_ndcg_at_fraction(y_true, y_pred, 0.10),
            "top_recall@10%": _top_recall_at_fraction(y_true, y_pred, 0.10),
        }
        if all(value in (0.0, 1.0) for value in y_bin) and len(set(y_bin)) == 2:
            metrics["auc"] = _roc_auc(y_bin, y_pred)
            metrics["mcc"] = _best_mcc(y_bin, y_pred)
        assay_metrics.append(metrics)

    if not assay_metrics:
        return {
            "status": "unavailable",
            "benchmark_family": "proteingym",
            "reason": "need_at_least_two_scored_variants_per_assay",
            "n_records": len(rows),
            "n_skipped": skipped,
            "n_assays": len(groups),
        }

    result = {
        "status": "ok",
        "benchmark_family": "proteingym",
        "metric_type": "protein_fitness_dms",
        "spearman": _mean(item["spearman"] for item in assay_metrics),
        "ndcg@10%": _mean(item["ndcg@10%"] for item in assay_metrics),
        "top_recall@10%": _mean(item["top_recall@10%"] for item in assay_metrics),
        "n_records": len(rows),
        "n_scored": sum(item["n"] for item in assay_metrics),
        "n_skipped": skipped,
        "n_assays": len(assay_metrics),
        "aggregation": "mean_across_DMS_assays",
    }
    auc_values = [item["auc"] for item in assay_metrics if "auc" in item]
    mcc_values = [item["mcc"] for item in assay_metrics if "mcc" in item]
    if auc_values:
        result["auc"] = _mean(auc_values)
        result["n_auc_assays"] = len(auc_values)
    if mcc_values:
        result["mcc"] = _mean(mcc_values)
        result["n_mcc_assays"] = len(mcc_values)
    uniprot_metrics = _compute_proteingym_grouped_metrics(rows, "uniprot_id")
    if uniprot_metrics:
        result.update(uniprot_metrics)
    return result


def _compute_proteingym_grouped_metrics(rows: list[dict[str, Any]], group_key: str) -> dict[str, Any]:
    groups: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        ctx = row.get("context") or {}
        group_id = str(ctx.get(group_key) or "")
        true_score = _first_numeric(row.get("expected", ""))
        pred_score = _first_available_numeric(row.get("predicted", ""), row.get("predicted_raw", ""))
        if not group_id or true_score is None or pred_score is None:
            continue
        groups.setdefault(group_id, []).append((true_score, pred_score))
    scored = []
    for group_id, items in groups.items():
        if len(items) < 2:
            continue
        y_true = [item[0] for item in items]
        y_pred = [item[1] for item in items]
        scored.append({"group_id": group_id, "spearman": _spearman(y_true, y_pred), "n": len(items)})
    if not scored:
        return {}
    return {
        f"{group_key}_mean_spearman": _mean(item["spearman"] for item in scored),
        f"n_{group_key}_groups_scored": len(scored),
    }


def _compute_rna_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = {
        str((row.get("context") or {}).get("official_metric") or "")
        for row in rows
    }
    metric_hint = next((name for name in metric_names if name), "")
    if "multilabel" in metric_hint:
        return _compute_multilabel_metrics(rows, benchmark_family="rna_downstream")
    if "multiclass" in metric_hint:
        return _compute_multiclass_metrics(rows, benchmark_family="rna_downstream")
    if "binary" in metric_hint:
        return _compute_binary_metrics(rows, benchmark_family="rna_downstream")
    if "regression" in metric_hint:
        return _compute_regression_metrics(rows, benchmark_family="rna_downstream")
    return _compute_predictive_metrics(rows, benchmark_family="rna_downstream")


def _compute_bacbench_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = {
        str((row.get("context") or {}).get("official_metric") or "")
        for row in rows
    }
    metric_hint = next((name for name in metric_names if name), "")
    if "regression" in metric_hint:
        return _compute_regression_metrics(rows, benchmark_family="bacbench")
    if "multiclass" in metric_hint:
        return _compute_categorical_metrics(rows, benchmark_family="bacbench")
    if "binary" in metric_hint:
        return _compute_binary_metrics(rows, benchmark_family="bacbench")
    return _compute_predictive_metrics(rows, benchmark_family="bacbench")


def _compute_regression_metrics(rows: list[dict[str, Any]], *, benchmark_family: str) -> dict[str, Any]:
    pairs: list[tuple[float, float, dict[str, Any]]] = []
    skipped = 0
    for row in rows:
        y_true = _first_numeric(row.get("expected", ""))
        y_pred = _first_available_numeric(row.get("predicted", ""), row.get("predicted_raw", ""))
        if y_true is None or y_pred is None:
            skipped += 1
            continue
        pairs.append((y_true, y_pred, row.get("context") or {}))
    if not pairs:
        return {
            "status": "unavailable",
            "benchmark_family": benchmark_family,
            "reason": "no_numeric_predictions",
            "n_records": len(rows),
            "n_skipped": skipped,
        }
    y_true = [item[0] for item in pairs]
    y_pred = [item[1] for item in pairs]
    result = {
        "status": "ok",
        "benchmark_family": benchmark_family,
        "metric_type": "regression",
        "pearson": _pearson(y_true, y_pred),
        "spearman": _spearman(y_true, y_pred),
        "mae": _mae(y_true, y_pred),
        "rmse": _rmse(y_true, y_pred),
        "r2": _r2(y_true, y_pred),
        "n_records": len(rows),
        "n_scored": len(pairs),
        "n_skipped": skipped,
    }
    fold_values = sorted({str(ctx.get("fold_id")) for _, _, ctx in pairs if ctx.get("fold_id") not in (None, "")})
    if fold_values:
        fold_metrics = []
        for fold_id in fold_values:
            fold_pairs = [(true, pred) for true, pred, ctx in pairs if str(ctx.get("fold_id")) == fold_id]
            if len(fold_pairs) < 2:
                continue
            fold_true = [item[0] for item in fold_pairs]
            fold_pred = [item[1] for item in fold_pairs]
            fold_metrics.append({
                "fold_id": fold_id,
                "n": len(fold_pairs),
                "pearson": _pearson(fold_true, fold_pred),
                "spearman": _spearman(fold_true, fold_pred),
            })
        if fold_metrics:
            result["fold_mean_pearson"] = _mean(item["pearson"] for item in fold_metrics)
            result["fold_mean_spearman"] = _mean(item["spearman"] for item in fold_metrics)
            result["n_folds_scored"] = len(fold_metrics)
    return result


def _compute_binary_metrics(rows: list[dict[str, Any]], *, benchmark_family: str) -> dict[str, Any]:
    y_true: list[float] = []
    y_score: list[float] = []
    skipped = 0
    for row in rows:
        true = _first_numeric(row.get("expected", ""))
        pred_values = _parse_numeric_vector(row.get("predicted", "")) or _parse_numeric_vector(row.get("predicted_raw", ""))
        if true is None or not pred_values:
            skipped += 1
            continue
        score = pred_values[1] if len(pred_values) > 1 else pred_values[0]
        y_true.append(1.0 if true >= 0.5 else 0.0)
        y_score.append(score)
    if not y_true:
        return {
            "status": "unavailable",
            "benchmark_family": benchmark_family,
            "reason": "no_binary_predictions",
            "n_records": len(rows),
            "n_skipped": skipped,
        }
    y_pred = [_threshold_label(value) for value in y_score]
    return {
        "status": "ok",
        "benchmark_family": benchmark_family,
        "metric_type": "binary_classification",
        "accuracy": _accuracy(y_true, y_pred),
        "f1": _binary_f1(y_true, y_pred),
        "roc_auc": _roc_auc(y_true, y_score) if len(set(y_true)) == 2 else float("nan"),
        "prc_auc": _average_precision(y_true, y_score) if len(set(y_true)) == 2 else float("nan"),
        "mcc": _mcc(y_true, y_pred),
        "n_records": len(rows),
        "n_scored": len(y_true),
        "n_skipped": skipped,
    }


def _compute_multiclass_metrics(rows: list[dict[str, Any]], *, benchmark_family: str) -> dict[str, Any]:
    y_true: list[int] = []
    y_pred: list[int] = []
    skipped = 0
    for row in rows:
        true_values = _parse_numeric_vector(row.get("expected", ""))
        pred_values = _parse_numeric_vector(row.get("predicted", "")) or _parse_numeric_vector(row.get("predicted_raw", ""))
        if not true_values or not pred_values:
            skipped += 1
            continue
        true_label = int(round(true_values[0]))
        pred_label = _argmax(pred_values) if len(pred_values) > 1 else int(round(pred_values[0]))
        y_true.append(true_label)
        y_pred.append(pred_label)
    if not y_true:
        return {
            "status": "unavailable",
            "benchmark_family": benchmark_family,
            "reason": "no_multiclass_predictions",
            "n_records": len(rows),
            "n_skipped": skipped,
        }
    return {
        "status": "ok",
        "benchmark_family": benchmark_family,
        "metric_type": "multiclass_classification",
        "accuracy": sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true),
        "macro_f1": _macro_f1(y_true, y_pred),
        "micro_f1": _micro_f1(y_true, y_pred),
        "n_records": len(rows),
        "n_scored": len(y_true),
        "n_skipped": skipped,
        "n_classes": len(set(y_true) | set(y_pred)),
    }


def _compute_categorical_metrics(rows: list[dict[str, Any]], *, benchmark_family: str) -> dict[str, Any]:
    y_true: list[str] = []
    y_pred: list[str] = []
    skipped = 0
    for row in rows:
        true = _normalise_category(row.get("expected", ""))
        pred = _normalise_category(row.get("predicted", "")) or _normalise_category(row.get("predicted_raw", ""))
        if not true or not pred:
            skipped += 1
            continue
        y_true.append(true)
        y_pred.append(pred)
    if not y_true:
        return {
            "status": "unavailable",
            "benchmark_family": benchmark_family,
            "reason": "no_categorical_predictions",
            "n_records": len(rows),
            "n_skipped": skipped,
        }
    labels = sorted(set(y_true) | set(y_pred))
    as_int = {label: idx for idx, label in enumerate(labels)}
    y_true_int = [as_int[value] for value in y_true]
    y_pred_int = [as_int[value] for value in y_pred]
    return {
        "status": "ok",
        "benchmark_family": benchmark_family,
        "metric_type": "categorical_classification",
        "accuracy": sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true),
        "macro_f1": _macro_f1(y_true_int, y_pred_int),
        "micro_f1": _micro_f1(y_true_int, y_pred_int),
        "n_records": len(rows),
        "n_scored": len(y_true),
        "n_skipped": skipped,
        "n_classes": len(labels),
    }


def _compute_multilabel_metrics(rows: list[dict[str, Any]], *, benchmark_family: str) -> dict[str, Any]:
    true_vectors: list[list[float]] = []
    pred_vectors: list[list[float]] = []
    skipped = 0
    for row in rows:
        true = _parse_numeric_vector(row.get("expected", ""))
        pred = _parse_numeric_vector(row.get("predicted", "")) or _parse_numeric_vector(row.get("predicted_raw", ""))
        if not true or not pred:
            skipped += 1
            continue
        width = min(len(true), len(pred))
        true_vectors.append([1.0 if value >= 0.5 else 0.0 for value in true[:width]])
        pred_vectors.append(pred[:width])
    if not true_vectors:
        return {
            "status": "unavailable",
            "benchmark_family": benchmark_family,
            "reason": "no_multilabel_predictions",
            "n_records": len(rows),
            "n_skipped": skipped,
        }
    width = max(len(vec) for vec in true_vectors)
    pred_binary = [[_threshold_label(value) for value in vec] for vec in pred_vectors]
    label_f1 = []
    label_auc = []
    label_ap = []
    for idx in range(width):
        true = [vec[idx] for vec in true_vectors if idx < len(vec)]
        scores = [vec[idx] for vec in pred_vectors if idx < len(vec)]
        pred = [vec[idx] for vec in pred_binary if idx < len(vec)]
        if true:
            label_f1.append(_binary_f1(true, pred))
        if len(set(true)) == 2:
            label_auc.append(_roc_auc(true, scores))
            label_ap.append(_average_precision(true, scores))
    flat_true = [value for vec in true_vectors for value in vec]
    flat_pred = [value for vec in pred_binary for value in vec]
    subset_accuracy = sum(1 for true, pred in zip(true_vectors, pred_binary) if true == pred) / len(true_vectors)
    return {
        "status": "ok",
        "benchmark_family": benchmark_family,
        "metric_type": "multilabel_classification",
        "macro_f1": _mean(label_f1),
        "micro_f1": _binary_f1(flat_true, flat_pred),
        "subset_accuracy": subset_accuracy,
        "macro_roc_auc": _mean(label_auc) if label_auc else float("nan"),
        "macro_prc_auc": _mean(label_ap) if label_ap else float("nan"),
        "n_records": len(rows),
        "n_scored": len(true_vectors),
        "n_skipped": skipped,
        "n_labels": width,
    }


def _compute_retrieval_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        ctx = row.get("context") or {}
        qid = str(ctx.get("query_id") or row.get("id") or "")
        relevant_ids = _as_id_list(
            ctx.get("relevant_doc_ids")
            or ctx.get("corpus_id")
            or ctx.get("expected_doc_id")
            or row.get("expected")
            or ""
        )
        ranking = _parse_ranking(row)
        if not qid or not relevant_ids:
            continue
        slot = grouped.setdefault(qid, {"relevant": set(), "ranking": []})
        slot["relevant"].update(relevant_ids)
        if ranking and not slot["ranking"]:
            slot["ranking"] = ranking

    scored = [item for item in grouped.values() if item["ranking"]]
    if not scored:
        return {
            "status": "unavailable",
            "benchmark_family": "mteb_retrieval",
            "reason": "ranked_doc_ids_required_for_ndcg_mrr_recall",
            "n_records": len(rows),
            "n_queries": len(grouped),
        }

    metrics = {k: [] for k in ("ndcg@10", "mrr@10", "recall@10", "recall@100")}
    for item in scored:
        relevant = item["relevant"]
        ranking = item["ranking"]
        metrics["ndcg@10"].append(_ndcg_at_k(ranking, relevant, 10))
        metrics["mrr@10"].append(_mrr_at_k(ranking, relevant, 10))
        metrics["recall@10"].append(_recall_at_k(ranking, relevant, 10))
        metrics["recall@100"].append(_recall_at_k(ranking, relevant, 100))
    return {
        "status": "ok",
        "benchmark_family": "mteb_retrieval",
        "metric_type": "retrieval",
        **{key: _mean(values) for key, values in metrics.items()},
        "n_records": len(rows),
        "n_queries": len(grouped),
        "n_queries_scored": len(scored),
    }


def _parse_numeric_vector(value: Any) -> list[float]:
    if isinstance(value, (int, float, bool)):
        return [float(value)]
    parsed = _json_or_literal(value)
    if isinstance(parsed, dict):
        for key in ("labels", "label", "scores", "score", "prediction", "predictions"):
            if key in parsed:
                return _parse_numeric_vector(parsed[key])
    if isinstance(parsed, list):
        out: list[float] = []
        for item in parsed:
            sub = _parse_numeric_vector(item)
            if len(sub) == 1:
                out.append(sub[0])
        return out
    text = str(value or "").strip()
    if not text:
        return []
    numeric, reason = extract_numeric_answer(text)
    if numeric is not None and reason in {"direct_scalar", "primary_trailer", "answer_prefix", "boxed"}:
        return [numeric]
    tokens = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    return [float(token) for token in tokens]


def _first_numeric(value: Any) -> float | None:
    values = _parse_numeric_vector(value)
    return values[0] if values else None


def _first_available_numeric(*values: Any) -> float | None:
    for value in values:
        numeric = _first_numeric(value)
        if numeric is not None:
            return numeric
    return None


def _json_or_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _normalise_category(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"(?:answer|label)\s*[:=]\s*(.+)", text, flags=re.I)
    if match:
        text = match.group(1).strip()
    return text.strip().strip("\"'").lower()


def _parse_ranking(row: dict[str, Any]) -> list[str]:
    ctx = row.get("context") or {}
    for key in ("predicted_doc_ids", "ranking", "ranked_doc_ids"):
        value = row.get(key) or ctx.get(key)
        parsed = _json_or_literal(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item)]
        if isinstance(parsed, dict):
            for nested_key in ("ranked_doc_ids", "doc_ids", "ids"):
                nested = parsed.get(nested_key)
                if isinstance(nested, list):
                    return [str(item) for item in nested if str(item)]
    text = str(row.get("predicted_raw") or row.get("predicted") or "")
    parsed = _json_or_literal(text)
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item)]
    if isinstance(parsed, dict):
        for key in ("ranked_doc_ids", "doc_ids", "ids"):
            value = parsed.get(key)
            if isinstance(value, list):
                return [str(item) for item in value if str(item)]
    ids = re.findall(r"(?:doc(?:ument)?[_\s-]?id|corpus[_\s-]?id)\s*[:=]\s*([A-Za-z0-9_.:-]+)", text, re.I)
    if ids:
        return ids
    return []


def _as_id_list(value: Any) -> list[str]:
    parsed = _json_or_literal(value)
    if isinstance(parsed, dict):
        for key in ("relevant_doc_ids", "doc_ids", "ids"):
            if key in parsed:
                return _as_id_list(parsed[key])
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item)]
    text = str(parsed or value or "").strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"[\s,;]+", text) if part.strip()]


def _rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(order):
        end = idx + 1
        while end < len(order) and values[order[end]] == values[order[idx]]:
            end += 1
        rank = (idx + 1 + end) / 2.0
        for j in range(idx, end):
            ranks[order[j]] = rank
        idx = end
    return ranks


def _roc_auc(y_true: list[float], y_score: list[float]) -> float:
    positives = sum(1 for y in y_true if y == 1.0)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _rankdata(y_score)
    rank_sum_pos = sum(rank for rank, y in zip(ranks, y_true) if y == 1.0)
    return (rank_sum_pos - positives * (positives + 1) / 2) / (positives * negatives)


def _average_precision(y_true: list[float], y_score: list[float]) -> float:
    positives = sum(1 for y in y_true if y == 1.0)
    if positives == 0:
        return float("nan")
    order = sorted(range(len(y_score)), key=lambda i: y_score[i], reverse=True)
    hits = 0
    precision_sum = 0.0
    for rank, idx in enumerate(order, start=1):
        if y_true[idx] == 1.0:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / positives


def _threshold_label(value: float) -> float:
    return 1.0 if value >= 0.5 else 0.0


def _accuracy(y_true: list[float], y_pred: list[float]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def _binary_f1(y_true: list[float], y_pred: list[float]) -> float:
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1.0 and b == 1.0)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0.0 and b == 1.0)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1.0 and b == 0.0)
    if tp == 0 and (fp or fn):
        return 0.0
    if tp == fp == fn == 0:
        return 1.0
    return 2 * tp / (2 * tp + fp + fn)


def _argmax(values: list[float]) -> int:
    return max(range(len(values)), key=lambda idx: values[idx])


def _macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    scores = []
    for label in labels:
        true_binary = [1.0 if value == label else 0.0 for value in y_true]
        pred_binary = [1.0 if value == label else 0.0 for value in y_pred]
        scores.append(_binary_f1(true_binary, pred_binary))
    return _mean(scores)


def _micro_f1(y_true: list[int], y_pred: list[int]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    true_flat: list[float] = []
    pred_flat: list[float] = []
    for true, pred in zip(y_true, y_pred):
        for label in labels:
            true_flat.append(1.0 if true == label else 0.0)
            pred_flat.append(1.0 if pred == label else 0.0)
    return _binary_f1(true_flat, pred_flat)


def _mcc(y_true: list[float], y_pred: list[float]) -> float:
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1.0 and b == 1.0)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0.0 and b == 0.0)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0.0 and b == 1.0)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1.0 and b == 0.0)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return ((tp * tn) - (fp * fn)) / denom if denom else 0.0


def _best_mcc(y_true: list[float], y_score: list[float]) -> float:
    thresholds = sorted(set(y_score))
    if not thresholds:
        return float("nan")
    return max(_mcc(y_true, [1.0 if score >= threshold else 0.0 for score in y_score]) for threshold in thresholds)


def _continuous_ndcg_at_fraction(y_true: list[float], y_score: list[float], fraction: float) -> float:
    if not y_true:
        return float("nan")
    offset = min(y_true)
    relevance = [value - offset for value in y_true]
    k = max(1, math.ceil(len(y_true) * fraction))
    order = sorted(range(len(y_score)), key=lambda idx: y_score[idx], reverse=True)
    ideal = sorted(range(len(y_true)), key=lambda idx: y_true[idx], reverse=True)
    dcg = sum(relevance[idx] / math.log2(rank + 1) for rank, idx in enumerate(order[:k], start=1))
    idcg = sum(relevance[idx] / math.log2(rank + 1) for rank, idx in enumerate(ideal[:k], start=1))
    return dcg / idcg if idcg else float("nan")


def _top_recall_at_fraction(y_true: list[float], y_score: list[float], fraction: float) -> float:
    if not y_true:
        return float("nan")
    k = max(1, math.ceil(len(y_true) * fraction))
    top_true = set(sorted(range(len(y_true)), key=lambda idx: y_true[idx], reverse=True)[:k])
    top_pred = set(sorted(range(len(y_score)), key=lambda idx: y_score[idx], reverse=True)[:k])
    return len(top_true & top_pred) / len(top_true) if top_true else float("nan")


def _mae(y_true: list[float], y_pred: list[float]) -> float:
    return _mean(abs(a - b) for a, b in zip(y_true, y_pred))


def _rmse(y_true: list[float], y_pred: list[float]) -> float:
    return math.sqrt(_mean((a - b) ** 2 for a, b in zip(y_true, y_pred)))


def _r2(y_true: list[float], y_pred: list[float]) -> float:
    if not y_true:
        return float("nan")
    mean_true = _mean(y_true)
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred))
    ss_tot = sum((a - mean_true) ** 2 for a in y_true)
    return 1 - (ss_res / ss_tot) if ss_tot else float("nan")


def _spearman(y_true: list[float], y_pred: list[float]) -> float:
    if len(y_true) < 2:
        return float("nan")
    return _pearson(_rankdata(y_true), _rankdata(y_pred))


def _pearson(x: list[float], y: list[float]) -> float:
    mx = _mean(x)
    my = _mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in x))
    den_y = math.sqrt(sum((b - my) ** 2 for b in y))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def _ndcg_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for idx, doc_id in enumerate(ranking[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def _mrr_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    for idx, doc_id in enumerate(ranking[:k], start=1):
        if doc_id in relevant:
            return 1.0 / idx
    return 0.0


def _recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranking[:k]) & relevant) / len(relevant)


def _mean(values: Iterable[float]) -> float:
    vals = [value for value in values if not math.isnan(value)]
    return sum(vals) / len(vals) if vals else float("nan")
