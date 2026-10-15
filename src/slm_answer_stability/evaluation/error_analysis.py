"""Inspect MLP residuals, leakage clues, calibration, and systematic groups."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from slm_answer_stability.dataset_cli import WorkflowError, normalize, read_jsonl, write_jsonl
from slm_answer_stability.dataset_prep import validate_final
from slm_answer_stability.models.embeddings import sha256


def _number(value: Any, name: str, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise WorkflowError(f"{where}: {name} must be a finite number")
    return float(value)


def _bucket_length(length: int) -> str:
    if length <= 80:
        return "0-80"
    if length <= 240:
        return "81-240"
    return "241+"


def _bucket_n(count: int) -> str:
    return "1-3" if count <= 3 else "4-10" if count <= 10 else "11+"


def _bucket_target(value: float) -> str:
    return "0-0.25" if value < 0.25 else "0.25-0.5" if value < 0.5 else "0.5-0.75" if value < 0.75 else "0.75-1"


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    errors = [row["residual"] for row in rows]
    return {
        "mae": sum(abs(error) for error in errors) / len(errors),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "bias": sum(errors) / len(errors),
    }


def _load_lengths(features_dir: Path | None, split_dir: Path, split: str, ids: set[str]) -> dict[str, dict]:
    if features_dir is None:
        return {}
    try:
        manifest = json.loads((features_dir / "manifest.json").read_text(encoding="utf-8"))
        details = manifest["splits"][split]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise WorkflowError(f"Cannot read feature manifest: {exc}") from exc
    lengths_file = features_dir / f"{split}_lengths.jsonl"
    if details.get("source_sha256") != sha256(split_dir / f"{split}.jsonl"):
        raise WorkflowError("Feature lengths belong to a different source JSONL")
    if details.get("lengths_sha256") != sha256(lengths_file):
        raise WorkflowError("Feature lengths changed since embedding")
    rows = read_jsonl(lengths_file)
    by_id = {row.get("question_id"): row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != ids:
        raise WorkflowError("Feature lengths do not match prediction question_id")
    for question_id, row in by_id.items():
        if type(row.get("question_truncated")) is not bool or type(row.get("answer_truncated")) is not bool:
            raise WorkflowError(f"{question_id}: invalid truncation flags")
    return by_id


def analyze(predictions_path: Path, splits_dir: Path, features_dir: Path | None, near_zero_mae: float, bias_threshold: float, min_group_size: int, bins: int, top: int, final: bool) -> tuple[dict, list[dict], list[dict], list[dict]]:
    if not math.isfinite(near_zero_mae) or not math.isfinite(bias_threshold) or near_zero_mae < 0 or bias_threshold < 0 or min_group_size < 1 or bins < 2 or top < 1:
        raise WorkflowError("Invalid analysis thresholds")
    predictions = read_jsonl(predictions_path)
    splits = {row.get("split") for row in predictions}
    if len(splits) != 1 or next(iter(splits)) not in ("val", "test"):
        raise WorkflowError("Predictions must contain one split: val or test")
    split = next(iter(splits))
    if split == "test" and not final:
        raise WorkflowError("Use --final to analyze the test split")
    source_rows = read_jsonl(splits_dir / f"{split}.jsonl")
    train_rows = read_jsonl(splits_dir / "train.jsonl")
    for name, rows in ((split, source_rows), ("train", train_rows)):
        problems = validate_final(rows)
        if problems:
            raise WorkflowError(f"Invalid {name} data: " + "; ".join(problems))
    source = {row["question_id"]: row for row in source_rows}
    train_ids = {row["question_id"] for row in train_rows}
    if train_ids & set(source):
        raise WorkflowError("question_id overlap between train and evaluation split")
    ids = [row.get("question_id") for row in predictions]
    if len(ids) != len(set(ids)) or set(ids) != set(source):
        raise WorkflowError("Prediction question_id must match the evaluation split exactly")
    lengths = _load_lengths(features_dir, splits_dir, split, set(source))
    constant = sum(row["target"] for row in train_rows) / len(train_rows)
    train_questions = {normalize(row["question"]) for row in train_rows}
    records = []
    for row in predictions:
        question_id = row["question_id"]
        original = source[question_id]
        if row.get("question") != original["question"] or row.get("answer") != original["answer"]:
            raise WorkflowError(f"{question_id}: original pair differs from split JSONL")
        if row.get("k") != original["k"] or row.get("N") != original["N"]:
            raise WorkflowError(f"{question_id}: k/N differs from split JSONL")
        target = _number(row.get("target"), "target", question_id)
        prediction = _number(row.get("prediction"), "prediction", question_id)
        baseline = _number(row.get("constant_prediction"), "constant_prediction", question_id)
        if abs(target - original["target"]) > 1e-9 or abs(baseline - constant) > 1e-8:
            raise WorkflowError(f"{question_id}: target or constant baseline differs from source")
        if not 0 <= prediction <= 1:
            raise WorkflowError(f"{question_id}: prediction must be in [0, 1]")
        residual = prediction - target
        if "absolute_error" in row and abs(_number(row["absolute_error"], "absolute_error", question_id) - abs(residual)) > 1e-8:
            raise WorkflowError(f"{question_id}: incorrect absolute_error")
        token_row = lengths.get(question_id)
        records.append({
            **row,
            "residual": residual,
            "absolute_error": abs(residual),
            "source": original.get("source") or "(missing)",
            "topic": original.get("topic") or "(missing)",
            "language": original.get("language") or "(missing)",
            "question_length": len(original["question"]),
            "answer_length": len(original["answer"]),
            "truncated": (token_row["question_truncated"] or token_row["answer_truncated"]) if token_row else None,
            "exact_train_question_duplicate": normalize(original["question"]) in train_questions,
            "variants": original["variants"],
        })
    mlp = _metrics(records)
    baseline_errors = [constant - row["target"] for row in records]
    baseline_metrics = {
        "mae": sum(abs(error) for error in baseline_errors) / len(records),
        "rmse": math.sqrt(sum(error * error for error in baseline_errors) / len(records)),
    }
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        dimensions = {
            "source": str(record["source"]),
            "topic": str(record["topic"]),
            "language": str(record["language"]),
            "question_length": _bucket_length(record["question_length"]),
            "answer_length": _bucket_length(record["answer_length"]),
            "N": _bucket_n(record["N"]),
            "target": _bucket_target(record["target"]),
        }
        if record["truncated"] is not None:
            dimensions["truncated"] = str(record["truncated"]).lower()
        for transformation in {variant.get("transformation_type") for variant in record["variants"] if isinstance(variant, dict)}:
            if transformation:
                groups[("transformation_type", str(transformation))].append(record)
        for dimension, value in dimensions.items():
            groups[(dimension, value)].append(record)
    group_rows = []
    for (dimension, value), members in sorted(groups.items()):
        stats = _metrics(members)
        group_rows.append({
            "dimension": dimension, "value": value, "questions": len(members),
            **stats, "sufficient_sample": len(members) >= min_group_size,
            "large_bias": len(members) >= min_group_size and abs(stats["bias"]) >= bias_threshold,
        })
    calibration = []
    for index in range(bins):
        members = [record for record in records if min(int(record["prediction"] * bins), bins - 1) == index]
        calibration.append({
            "bin": index, "lower": index / bins, "upper": (index + 1) / bins,
            "questions": len(members),
            "mean_prediction": sum(row["prediction"] for row in members) / len(members) if members else None,
            "mean_target": sum(row["target"] for row in members) / len(members) if members else None,
        })
    result = {
        "split": split, "questions": len(records), "mlp": mlp, "constant": baseline_metrics,
        "mae_improvement_over_constant": baseline_metrics["mae"] - mlp["mae"],
        "near_zero_mae": mlp["mae"] <= near_zero_mae,
        "exact_train_question_duplicates": [row["question_id"] for row in records if row["exact_train_question_duplicate"]],
        "large_bias_groups": [{"dimension": row["dimension"], "value": row["value"], "questions": row["questions"], "bias": row["bias"]} for row in group_rows if row["large_bias"]],
        "metadata_gaps": [name for name in ("source", "topic", "language") if all(row[name] == "(missing)" for row in records)],
        "truncation_available": bool(lengths),
        "thresholds": {"near_zero_mae": near_zero_mae, "bias_threshold": bias_threshold, "min_group_size": min_group_size},
        "calibration": calibration,
    }
    top_rows = sorted(records, key=lambda row: (-row["absolute_error"], row["question_id"]))[:top]
    return result, group_rows, calibration, top_rows


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(output_dir: Path, result: dict, groups: list[dict], calibration: list[dict], top_rows: list[dict]) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise WorkflowError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(output_dir / "group_metrics.csv", groups, ["dimension", "value", "questions", "mae", "rmse", "bias", "sufficient_sample", "large_bias"])
    _write_csv(output_dir / "calibration.csv", calibration, ["bin", "lower", "upper", "questions", "mean_prediction", "mean_target"])
    write_jsonl(output_dir / "top_errors.jsonl", top_rows)
    suspicious = ", ".join(result["exact_train_question_duplicates"][:10]) or "нет точных совпадений"
    flagged = ", ".join(f"{item['dimension']}={item['value']} (n={item['questions']})" for item in result["large_bias_groups"]) or "нет групп выше заданного порога"
    lines = [
        f"# Анализ ошибок: {result['split']}", "",
        f"Задач: {result['questions']}; MAE MLP: {result['mlp']['mae']:.4f}; MAE постоянного прогноза: {result['constant']['mae']:.4f}.",
        f"Среднее смещение прогноза: {result['mlp']['bias']:.4f}.",
        f"Почти нулевая ошибка по заданному порогу: {'да' if result['near_zero_mae'] else 'нет'}.",
        f"Точные дубликаты вопросов из train: {suspicious}.",
        f"Группы с заметным смещением: {flagged}.",
        "",
        "Проверьте верхние ошибки вручную; отсутствие обнаруженных групп не доказывает случайность ошибки.",
        "Смысловые дубликаты, надёжность решений оценщика и разброс по нескольким seed требуют отдельной проверки.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="predictions.jsonl")
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--near-zero-mae", type=float, default=0.02)
    parser.add_argument("--bias-threshold", type=float, default=0.15)
    parser.add_argument("--min-group-size", type=int, default=5)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args(argv)
    try:
        result, groups, calibration, top_rows = analyze(
            args.input, args.splits_dir, args.features_dir,
            args.near_zero_mae, args.bias_threshold, args.min_group_size,
            args.calibration_bins, args.top, args.final,
        )
        write_report(args.output_dir, result, groups, calibration, top_rows)
    except (WorkflowError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output_dir": str(args.output_dir), "summary": result}, ensure_ascii=False, indent=2))
    return 0
