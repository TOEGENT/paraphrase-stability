"""File-based, human-reviewed dataset workflow built on model_tools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from .model_tools import ModelToolError, load_config, run as run_model


class WorkflowError(Exception):
    """Invalid input or an incomplete human review."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise WorkflowError(f"Cannot read {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WorkflowError(f"Invalid JSON in {path}:{number}: {exc}") from exc
        if not isinstance(row, dict):
            raise WorkflowError(f"Expected a JSON object in {path}:{number}")
        rows.append(row)
    if not rows:
        raise WorkflowError(f"No records in {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise WorkflowError(f"Output already exists: {path}; choose a new file")
    if not rows:
        raise WorkflowError("No records to write")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    except OSError as exc:
        raise WorkflowError(f"Cannot write {path}: {exc}") from exc


def required_text(row: dict[str, Any], field: str, where: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise WorkflowError(f"{where}: missing {field}")
    return value.strip()


def reviewed(row: dict[str, Any], field: str, where: str) -> bool:
    value = row.get(field)
    if value not in ("approved", "rejected"):
        raise WorkflowError(f"{where}: set {field} to approved or rejected before continuing")
    return value == "approved"


def unique_ids(rows: list[dict[str, Any]], field: str) -> None:
    values = [required_text(row, field, f"row {i}") for i, row in enumerate(rows, 1)]
    if len(values) != len(set(values)):
        raise WorkflowError(f"Duplicate {field}")


def normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def call_model(config: dict[str, Any], command: str, **kwargs: Any) -> dict[str, Any]:
    try:
        return run_model(argparse.Namespace(command=command, **kwargs), config)
    except ModelToolError as exc:
        raise WorkflowError(f"{command}: {exc}") from exc


def generate_questions(args: argparse.Namespace, config: dict[str, Any]) -> list[dict[str, Any]]:
    existing = read_jsonl(args.existing) if args.existing else []
    seen = [required_text(row, "question", "existing question") for row in existing]
    seen_exact = {normalize(question): question for question in seen}
    batch_size = getattr(args, "batch_size", 20)
    duplicate_check = getattr(args, "duplicate_check", "exact")
    output = []
    while len(output) < args.count:
        count = min(batch_size, args.count - len(output))
        try:
            result = call_model(
                config, "questions", topic=args.topic, count=count, existing_questions=seen.copy()
            )
            if not isinstance(result.get("questions"), list) or len(result["questions"]) != count:
                raise WorkflowError(f"Generator returned an incomplete batch: expected {count} questions")
        except WorkflowError as exc:
            recoverable = any(message in str(exc) for message in (
                "generator returned invalid JSON",
                "generator returned the wrong number of questions",
                "Generator returned an incomplete batch",
            ))
            if not recoverable or count == 1:
                raise
            batch_size = max(1, count // 2)
            if getattr(args, "progress", False):
                print(f"Question batch of {count} failed ({exc}); retrying with {batch_size}", file=sys.stderr)
            continue
        for item in result["questions"]:
            question = item["question"].strip()
            exact_prior = seen_exact.get(normalize(question))
            duplicate_status = "exact" if exact_prior else "unchecked" if duplicate_check == "exact" else "none"
            duplicate_of = exact_prior
            if not exact_prior and duplicate_check == "semantic":
                for prior in seen:
                    decision = call_model(
                        config, "judge-question", question=prior, rewritten_question=question
                    )["status"]
                    if decision in ("equivalent", "unclear", "insufficient_context"):
                        duplicate_status, duplicate_of = decision, prior
                        break
            output.append({
                "question_id": "synthetic-" + uuid4().hex[:12],
                "question": question,
                "source": "synthetic",
                "topic": args.topic,
                "generator_model": result["model"],
                "duplicate_status": duplicate_status,
                "duplicate_of": duplicate_of,
                "question_review": "pending",
            })
            seen.append(question)
            seen_exact.setdefault(normalize(question), question)
        if getattr(args, "progress", False):
            print(f"Generated {len(output)}/{args.count} question candidates", file=sys.stderr)
    return output


def generate_paraphrases(args: argparse.Namespace, config: dict[str, Any]) -> list[dict[str, Any]]:
    questions = read_jsonl(args.input)
    unique_ids(questions, "question_id")
    output = []
    for row in questions:
        question_id = row["question_id"]
        if not reviewed(row, "question_review", question_id):
            continue
        question = required_text(row, "question", question_id)
        result = call_model(config, "paraphrases", question=question, count=args.count)
        for index, item in enumerate(result["variants"], 1):
            output.append({
                "question_id": question_id,
                "variant_id": f"{question_id}-v{index}",
                "question_original": question,
                "question_transformed": item["question"].strip(),
                "transformation_type": item["transformation_type"],
                "source": row.get("source", "unknown"),
                "topic": row.get("topic"),
                "language": row.get("language"),
                "generator_model": result["model"],
                "paraphrase_review": "pending",
            })
    return output


def evaluate_paraphrases(args: argparse.Namespace, config: dict[str, Any]) -> list[dict[str, Any]]:
    variants = read_jsonl(args.input)
    unique_ids(variants, "variant_id")
    output = []
    for row in variants:
        variant_id = row["variant_id"]
        if not reviewed(row, "paraphrase_review", variant_id):
            continue
        decision = call_model(
            config,
            "judge-question",
            question=required_text(row, "question_original", variant_id),
            rewritten_question=required_text(row, "question_transformed", variant_id),
        )
        output.append({
            **row,
            "question_equivalence": decision["status"],
            "judge_status": decision["status"],
            "judge_rationale": decision["rationale_short"],
            "judge_model": decision["model"],
            "equivalence_review": "pending",
        })
    return output


def generate_answers(args: argparse.Namespace, config: dict[str, Any]) -> list[dict[str, Any]]:
    variants = read_jsonl(args.input)
    unique_ids(variants, "variant_id")
    original_answers: dict[str, str] = {}
    original_questions: dict[str, str] = {}
    output = []
    for row in variants:
        variant_id = row["variant_id"]
        if not reviewed(row, "equivalence_review", variant_id):
            continue
        if row.get("question_equivalence") != "equivalent":
            continue
        question_id = required_text(row, "question_id", variant_id)
        question = required_text(row, "question_original", variant_id)
        transformed = required_text(row, "question_transformed", variant_id)
        if question_id in original_questions and original_questions[question_id] != question:
            raise WorkflowError(f"Conflicting original question for {question_id}")
        original_questions[question_id] = question
        if question_id not in original_answers:
            original_answers[question_id] = call_model(config, "answer", question=question)["answer"]
        transformed_answer = call_model(config, "answer", question=transformed)["answer"]
        output.append({
            "question_id": question_id,
            "variant_id": variant_id,
            "question_original": question,
            "question_transformed": transformed,
            "answer_original": original_answers[question_id],
            "answer_transformed": transformed_answer,
            "source": row.get("source", "unknown"),
            "topic": row.get("topic"),
            "language": row.get("language"),
            "transformation_type": row.get("transformation_type"),
            "generator_model": row.get("generator_model"),
            "question_equivalence": row["question_equivalence"],
            "question_judge_status": row.get("judge_status"),
            "question_judge_rationale": row.get("judge_rationale"),
            "slm_model": config["slm"]["model"],
        })
    return output


def compute_stability(args: argparse.Namespace, config: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows = read_jsonl(args.input)
    output: dict[str, dict[str, Any]] = {}
    seen_variants: set[str] = set()
    for row in rows:
        variant_id = required_text(row, "variant_id", "answer row")
        if variant_id in seen_variants:
            raise WorkflowError(f"Duplicate variant_id {variant_id}")
        seen_variants.add(variant_id)
        question_id = required_text(row, "question_id", variant_id)
        question = required_text(row, "question_original", variant_id)
        original_answer = required_text(row, "answer_original", variant_id)
        transformed_answer = required_text(row, "answer_transformed", variant_id)
        if question_id not in output:
            output[question_id] = {
                "question_id": question_id,
                "question": question,
                "answer": original_answer,
                "source": row.get("source", "unknown"),
                "topic": row.get("topic"),
                "language": row.get("language"),
                "slm_model": row.get("slm_model"),
                "variants": [],
                "review_status": "pending",
            }
        group = output[question_id]
        if (group["question"] != question or group["answer"] != original_answer
                or group["slm_model"] != row.get("slm_model")
                or any(group[field] != row.get(field) for field in ("topic", "language"))
                or group["source"] != row.get("source", "unknown")):
            raise WorkflowError(f"Conflicting original pair for {question_id}")
        status = row.get("answer_equivalence")
        rationale = row.get("answer_judge_rationale")
        if status not in ("equivalent", "non_equivalent", "unclear", "insufficient_context"):
            if config is None:
                raise WorkflowError(f"{variant_id}: missing answer_equivalence for --recompute")
            decision = call_model(
                config,
                "judge-answer",
                question=question,
                answer=original_answer,
                rewritten_answer=transformed_answer,
            )
            status, rationale = decision["status"], decision["rationale_short"]
        group["variants"].append({
            "variant_id": variant_id,
            "question": required_text(row, "question_transformed", variant_id),
            "answer": transformed_answer,
            "transformation_type": row.get("transformation_type"),
            "question_equivalence": row.get("question_equivalence"),
            "question_judge_status": row.get("question_judge_status"),
            "question_judge_rationale": row.get("question_judge_rationale"),
            "generator_model": row.get("generator_model"),
            "answer_equivalence": status,
            "answer_judge_rationale": rationale,
            "answer_judge_model": config["judge"]["model"] if config else row.get("answer_judge_model"),
        })
    results = []
    for group in output.values():
        variants = group["variants"]
        k = sum(row["answer_equivalence"] == "equivalent" for row in variants)
        unresolved = sum(row["answer_equivalence"] in ("unclear", "insufficient_context") for row in variants)
        n = len(variants)
        group.update({"k": k, "N": n, "target": None if unresolved else k / n, "unresolved": unresolved})
        results.append(group)
    return results


def recompute(args: argparse.Namespace) -> list[dict[str, Any]]:
    groups = read_jsonl(args.input)
    unique_ids(groups, "question_id")
    results = []
    for group in groups:
        question_id = group["question_id"]
        variants = group.get("variants")
        if not isinstance(variants, list) or not variants:
            raise WorkflowError(f"{question_id}: missing variants")
        statuses = [variant.get("answer_equivalence") for variant in variants if isinstance(variant, dict)]
        if len(statuses) != len(variants) or any(status not in (
            "equivalent", "non_equivalent", "unclear", "insufficient_context"
        ) for status in statuses):
            raise WorkflowError(f"{question_id}: invalid answer_equivalence")
        n = len(variants)
        k = statuses.count("equivalent")
        unresolved = statuses.count("unclear") + statuses.count("insufficient_context")
        results.append({**group, "k": k, "N": n, "target": None if unresolved else k / n, "unresolved": unresolved})
    return results


def finalize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    approved = []
    for row in rows:
        question_id = row["question_id"]
        if not reviewed(row, "review_status", question_id):
            continue
        if row["target"] is None:
            raise WorkflowError(f"{question_id}: unresolved answer equivalence")
        approved.append(row)
    if not approved:
        raise WorkflowError("No approved records to finalize")
    return approved


def main(command: str, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/model_tools.example.json"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    if command == "generate_questions":
        parser.add_argument("--topic", required=True)
        parser.add_argument("--count", required=True, type=int)
        parser.add_argument("--batch-size", type=int, default=20)
        parser.add_argument("--duplicate-check", choices=("exact", "semantic"), default="exact")
        parser.add_argument("--existing", type=Path)
    elif command == "generate_paraphrases":
        parser.add_argument("--count", required=True, type=int)
    elif command == "compute_stability":
        parser.add_argument("--recompute", action="store_true")
        parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args(argv)
    if command == "generate_questions" and not 1 <= args.count <= 10000:
        parser.error("--count must be between 1 and 10000")
    if command == "generate_questions" and not 1 <= args.batch_size <= 100:
        parser.error("--batch-size must be between 1 and 100")
    if command == "generate_paraphrases" and not 1 <= args.count <= 100:
        parser.error("--count must be between 1 and 100")
    if command != "generate_questions" and args.input is None:
        parser.error("--input is required")
    if command == "compute_stability" and args.finalize and not args.recompute:
        parser.error("--finalize requires --recompute")
    try:
        config = None if command == "compute_stability" and args.recompute else load_config(args.config)
        if command == "generate_questions":
            args.progress = True
            rows = generate_questions(args, config)
        elif command == "generate_paraphrases":
            rows = generate_paraphrases(args, config)
        elif command == "evaluate_paraphrases":
            rows = evaluate_paraphrases(args, config)
        elif command == "generate_answers":
            rows = generate_answers(args, config)
        elif command == "compute_stability":
            rows = recompute(args) if args.recompute else compute_stability(args, config)
            if args.finalize:
                rows = finalize(rows)
        else:
            raise WorkflowError(f"Unknown command {command}")
        write_jsonl(args.output, rows)
    except (WorkflowError, ModelToolError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {len(rows)} records to {args.output}")
    return 0
