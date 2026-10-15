"""Import questions, split reviewed data, and check dataset readiness."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

from .dataset_cli import WorkflowError, normalize, read_jsonl, unique_ids, write_jsonl


def import_questions(args: argparse.Namespace) -> list[dict[str, Any]]:
    delimiter = args.delimiter or ("\t" if args.input.suffix.lower() == ".tsv" else ",")
    try:
        with args.input.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source, delimiter=delimiter)
            if not reader.fieldnames or args.question_column not in reader.fieldnames:
                raise WorkflowError(f"Missing CSV column {args.question_column!r}")
            if args.id_column and args.id_column not in reader.fieldnames:
                raise WorkflowError(f"Missing CSV column {args.id_column!r}")
            rows = []
            seen_questions: set[str] = set()
            for number, item in enumerate(reader, 2):
                if None in item:
                    raise WorkflowError(f"CSV row {number}: more fields than header")
                question = (item.get(args.question_column) or "").strip()
                if not question:
                    raise WorkflowError(f"CSV row {number}: empty question")
                key = normalize(question)
                if key in seen_questions:
                    raise WorkflowError(f"CSV row {number}: duplicate question")
                seen_questions.add(key)
                question_id = (
                    (item.get(args.id_column) or "").strip() if args.id_column else
                    "imported-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
                )
                if not question_id:
                    raise WorkflowError(f"CSV row {number}: empty question ID")
                row = {
                    "question_id": question_id,
                    "question": question,
                    "source": args.source,
                    "duplicate_status": "unchecked",
                    "duplicate_of": None,
                    "question_review": "pending",
                }
                for field in ("topic", "language"):
                    if item.get(field):
                        row[field] = item[field].strip()
                rows.append(row)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise WorkflowError(f"Cannot import {args.input}: {exc}") from exc
    if not rows:
        raise WorkflowError("CSV has no questions")
    unique_ids(rows, "question_id")
    return rows


def validate_final(rows: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    ids: set[str] = set()
    questions: set[str] = set()
    all_variant_ids: set[str] = set()
    for index, row in enumerate(rows, 1):
        question_id = row.get("question_id")
        where = str(question_id) if isinstance(question_id, str) and question_id else f"row {index}"
        if not isinstance(question_id, str) or not question_id.strip():
            problems.append(f"{where}: missing question_id")
        elif question_id in ids:
            problems.append(f"{where}: duplicate question_id")
        else:
            ids.add(question_id)
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            problems.append(f"{where}: missing question")
        elif normalize(question) in questions:
            problems.append(f"{where}: duplicate question text")
        else:
            questions.add(normalize(question))
        answer = row.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            problems.append(f"{where}: missing original answer")
        if row.get("review_status") != "approved":
            problems.append(f"{where}: review_status must be approved")
        variants = row.get("variants")
        if not isinstance(variants, list) or not variants:
            problems.append(f"{where}: variants must be a nonempty list")
            continue
        variant_ids: set[str] = set()
        statuses = []
        for variant in variants:
            if not isinstance(variant, dict):
                problems.append(f"{where}: invalid variant")
                continue
            variant_id = variant.get("variant_id")
            if not isinstance(variant_id, str) or not variant_id.strip() or variant_id in variant_ids or variant_id in all_variant_ids:
                problems.append(f"{where}: missing or duplicate variant_id")
            else:
                variant_ids.add(variant_id)
                all_variant_ids.add(variant_id)
            if not isinstance(variant.get("question"), str) or not variant["question"].strip():
                problems.append(f"{where}: missing variant question")
            if not isinstance(variant.get("answer"), str) or not variant["answer"].strip():
                problems.append(f"{where}: missing variant answer")
            status = variant.get("answer_equivalence")
            if status not in ("equivalent", "non_equivalent"):
                problems.append(f"{where}: unresolved answer_equivalence")
            statuses.append(status)
        n, k = len(variants), statuses.count("equivalent")
        if row.get("N") != n or isinstance(row.get("N"), bool):
            problems.append(f"{where}: N does not match variants")
        if row.get("k") != k or isinstance(row.get("k"), bool):
            problems.append(f"{where}: k does not match variants")
        if row.get("unresolved") != 0:
            problems.append(f"{where}: unresolved must be 0")
        target = row.get("target")
        if not isinstance(target, (int, float)) or isinstance(target, bool) or abs(target - k / n) > 1e-9:
            problems.append(f"{where}: target does not equal k/N")
    return problems


def split_dataset(rows: list[dict[str, Any]], ratios: tuple[float, float, float], seed: int) -> dict[str, list[dict[str, Any]]]:
    problems = validate_final(rows)
    if problems:
        raise WorkflowError("Invalid final dataset: " + "; ".join(problems))
    if any(not 0 < ratio < 1 for ratio in ratios) or abs(sum(ratios) - 1) > 1e-9:
        raise WorkflowError("Split ratios must be positive fractions summing to 1")
    if len(rows) < 3:
        raise WorkflowError("At least three questions are required for three nonempty splits")
    shuffled = sorted(rows, key=lambda row: row["question_id"])
    random.Random(seed).shuffle(shuffled)
    # Reserve one question for each split, then distribute the remainder by largest remainder.
    remainder = len(rows) - 3
    quotas = [remainder * ratio for ratio in ratios]
    counts = [1 + int(quota) for quota in quotas]
    for index in sorted(range(3), key=lambda i: (quotas[i] % 1, -i), reverse=True)[:remainder - sum(map(int, quotas))]:
        counts[index] += 1
    names = ("train", "val", "test")
    output: dict[str, list[dict[str, Any]]] = {}
    offset = 0
    for name, count in zip(names, counts):
        output[name] = [{**row, "split": name} for row in shuffled[offset:offset + count]]
        offset += count
    return output


def check_dataset(rows: list[dict[str, Any]], min_questions: int, min_variants: int, min_total_variants: int) -> dict[str, Any]:
    problems = validate_final(rows)
    total_variants = sum(len(row.get("variants", [])) for row in rows if isinstance(row.get("variants"), list))
    if len(rows) < min_questions:
        problems.append(f"questions: {len(rows)} < {min_questions}")
    if total_variants < min_total_variants:
        problems.append(f"total variants: {total_variants} < {min_total_variants}")
    for row in rows:
        variants = row.get("variants")
        count = len(variants) if isinstance(variants, list) else 0
        if count < min_variants:
            problems.append(f"{row.get('question_id', '?')}: variants {count} < {min_variants}")
    return {"passed": not problems, "questions": len(rows), "total_variants": total_variants, "problems": problems}


def main(command: str, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    if command == "import_questions":
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--question-column", default="question")
        parser.add_argument("--id-column")
        parser.add_argument("--delimiter")
        parser.add_argument("--source", default="imported")
    elif command == "split_dataset":
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--train", type=float, required=True)
        parser.add_argument("--val", type=float, required=True)
        parser.add_argument("--test", type=float, required=True)
        parser.add_argument("--seed", type=int, required=True)
    elif command == "check_dataset":
        parser.add_argument("--min-questions", type=int, required=True)
        parser.add_argument("--min-variants-per-question", type=int, required=True)
        parser.add_argument("--min-total-variants", type=int, default=0)
        parser.add_argument("--report", type=Path)
    else:
        raise ValueError(command)
    args = parser.parse_args(argv)
    try:
        if command == "import_questions":
            if args.delimiter is not None and len(args.delimiter) != 1:
                raise WorkflowError("--delimiter must be one character")
            rows = import_questions(args)
            write_jsonl(args.output, rows)
            print(f"Wrote {len(rows)} questions to {args.output}")
        elif command == "split_dataset":
            rows = read_jsonl(args.input)
            splits = split_dataset(rows, (args.train, args.val, args.test), args.seed)
            if args.output_dir.exists() and any(args.output_dir.iterdir()):
                raise WorkflowError(f"Output directory is not empty: {args.output_dir}")
            for name, split_rows in splits.items():
                write_jsonl(args.output_dir / f"{name}.jsonl", split_rows)
            manifest = {"input": str(args.input), "seed": args.seed, "ratios": {"train": args.train, "val": args.val, "test": args.test}, "counts": {name: len(value) for name, value in splits.items()}}
            (args.output_dir / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(manifest, ensure_ascii=False))
        else:
            if min(args.min_questions, args.min_variants_per_question, args.min_total_variants) < 0:
                raise WorkflowError("Minimum counts cannot be negative")
            report = check_dataset(read_jsonl(args.input), args.min_questions, args.min_variants_per_question, args.min_total_variants)
            if args.report:
                if args.report.exists():
                    raise WorkflowError(f"Output already exists: {args.report}")
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["passed"] else 1
    except (WorkflowError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
