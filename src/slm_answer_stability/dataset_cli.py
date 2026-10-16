"""File-based, human-reviewed dataset workflow built on model_tools."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .model_tools import (PARAPHRASE_MODES, ModelToolError,
                          load_config, run as run_model)


class WorkflowError(Exception):
    """Invalid input or an incomplete human review."""


TARGET_FILES = {
    "translation": Path("configs/paraphrase_languages.txt"),
    "structure": Path("configs/paraphrase_structures.txt"),
    "style": Path("configs/paraphrase_styles.txt"),
}


def read_targets(path: Path) -> list[str]:
    try:
        values = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()
                  if line.strip() and not line.lstrip().startswith("#")]
    except OSError as exc:
        raise WorkflowError(f"Cannot read targets from {path}: {exc}") from exc
    if not values:
        raise WorkflowError(f"No targets in {path}")
    if len({value.casefold() for value in values}) != len(values):
        raise WorkflowError(f"Duplicate targets in {path}")
    return values


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


def append_log(path: Path | None, event: dict[str, Any]) -> None:
    if path is None:
        return
    event = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **event}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError as exc:
        raise WorkflowError(f"Cannot write log {path}: {exc}") from exc


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


def _combined_targets(args: argparse.Namespace) -> dict[str, list[str]]:
    choices = {
        "structure": getattr(args, "structures", None) or read_targets(TARGET_FILES["structure"]),
        "style": getattr(args, "styles", None) or read_targets(TARGET_FILES["style"]),
        "translation": getattr(args, "languages", None) or read_targets(TARGET_FILES["translation"]),
    }
    for mode, values in choices.items():
        if len(values) < args.count:
            raise WorkflowError(
                f"combined mode needs at least {args.count} {mode} targets"
            )
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise WorkflowError(f"combined mode has an empty {mode} target")
        if len({value.casefold() for value in values[:args.count]}) != args.count:
            raise WorkflowError(f"combined mode needs distinct {mode} targets")
    return {mode: list(values[:args.count]) for mode, values in choices.items()}


def _generate_combined_question(
    row: dict[str, Any], question: str, args: argparse.Namespace,
    config: dict[str, Any], targets: dict[str, list[str]],
    previous: list[str], used_ids: set[str],
) -> list[dict[str, Any]]:
    question_id = row["question_id"]
    seen = previous.copy()
    seen_exact = {normalize(value) for value in [question, *seen]}
    log_path = getattr(args, "log_file", None)
    run_id = getattr(args, "run_id", None)

    def generate_one(parent: str, mode: str, target: str, path: tuple[int, ...]) -> str:
        for attempt in range(5):
            request_id = uuid4().hex

            def model_trace(event: dict[str, Any]) -> None:
                append_log(log_path, {
                    "run_id": run_id, "question_id": question_id,
                    "request_id": request_id, "event": f"llm_{event['event']}",
                    "mode": mode, "target": target, "path": path,
                    "variation_attempt": attempt,
                    **{key: value for key, value in event.items() if key != "event"},
                })

            error = None
            try:
                result = call_model(
                    config, "paraphrases", question=parent, count=1,
                    mode=mode, target=target, reference_question=question,
                    preserve_structure=mode != "structure",
                    existing_paraphrases=seen.copy(), allow_duplicates=True,
                    allow_partial=True, variation_attempt=attempt,
                    model_trace=model_trace if log_path is not None else None,
                )
            except WorkflowError as exc:
                if "generator returned invalid JSON" not in str(exc):
                    append_log(log_path, {"run_id": run_id, "question_id": question_id,
                                          "event": "request_error", "mode": mode,
                                          "target": target, "path": path, "error": str(exc)})
                    raise
                error = str(exc)
                result = {"variants": []}
            variants = result.get("variants")
            if not isinstance(variants, list):
                raise WorkflowError(f"{question_id}: generator returned an invalid paraphrase batch")
            duplicate_examples = []
            for item in variants:
                transformed = item["question"].strip()
                if normalize(transformed) in seen_exact:
                    duplicate_examples.append(transformed[:300])
                    continue
                seen.append(transformed)
                seen_exact.add(normalize(transformed))
                append_log(log_path, {
                    "run_id": run_id, "question_id": question_id,
                    "request_id": request_id, "event": "batch_result",
                    "mode": mode, "target": target, "path": path,
                    "accepted": 1, "duplicate_examples": duplicate_examples[:5],
                    "variation_attempt": attempt, "error": error,
                })
                return transformed
            append_log(log_path, {
                "run_id": run_id, "question_id": question_id,
                "request_id": request_id, "event": "batch_result",
                "mode": mode, "target": target, "path": path,
                "accepted": 0, "duplicate_examples": duplicate_examples[:5],
                "variation_attempt": attempt, "error": error,
            })
        raise WorkflowError(
            f"{question_id}: no usable {mode}:{target} variant after 5 attempts at {path}"
        )

    output = []
    next_index = 1
    for structure_index, structure in enumerate(targets["structure"]):
        structured = generate_one(question, "structure", structure, (structure_index,))
        for style_index, style in enumerate(targets["style"]):
            styled = generate_one(structured, "style", style,
                                  (structure_index, style_index))
            for language_index, language in enumerate(targets["translation"]):
                transformed = generate_one(
                    styled, "translation", language,
                    (structure_index, style_index, language_index),
                )
                while f"{question_id}-v{next_index}" in used_ids:
                    next_index += 1
                variant_id = f"{question_id}-v{next_index}"
                used_ids.add(variant_id)
                next_index += 1
                output.append({
                    "question_id": question_id,
                    "variant_id": variant_id,
                    "question_original": question,
                    "question_transformed": transformed,
                    "transformation_type": "combined",
                    "transformation_mode": "combined",
                    "transformation_target": language,
                    "transformation_chain": [
                        {"mode": "structure", "target": structure, "question": structured},
                        {"mode": "style", "target": style, "question": styled},
                        {"mode": "translation", "target": language, "question": transformed},
                    ],
                    "source": row.get("source", "unknown"),
                    "topic": row.get("topic"),
                    "language": row.get("language"),
                    "generator_model": config["generator"]["model"],
                    "paraphrase_review": "pending",
                })
    return output


def generate_paraphrases(args: argparse.Namespace, config: dict[str, Any]) -> list[dict[str, Any]]:
    mode = getattr(args, "mode", "general")
    if mode not in (*PARAPHRASE_MODES, "combined"):
        raise WorkflowError(f"Unknown paraphrase mode: {mode}")
    targets = getattr(args, "targets", None)
    if mode == "combined" and targets:
        raise WorkflowError("combined mode uses --structures, --styles, and --languages")
    if mode == "general" and targets:
        raise WorkflowError("--targets requires translation, structure, or style mode")
    targets = targets or (read_targets(TARGET_FILES[mode]) if mode in TARGET_FILES else ())
    if any(not isinstance(target, str) or not target.strip() for target in targets):
        raise WorkflowError("--targets must contain non-empty names")
    combined_targets = _combined_targets(args) if mode == "combined" else None
    questions = read_jsonl(args.input)
    unique_ids(questions, "question_id")
    existing = read_jsonl(args.existing) if getattr(args, "existing", None) else []
    if existing:
        unique_ids(existing, "variant_id")
    previous: dict[str, list[str]] = {}
    original_by_id: dict[str, str] = {}
    used_ids = {row["variant_id"] for row in existing}
    for row in existing:
        question_id = required_text(row, "question_id", "existing paraphrase")
        original = required_text(row, "question_original", row["variant_id"])
        if question_id in original_by_id and original_by_id[question_id] != original:
            raise WorkflowError(f"{question_id}: conflicting original questions in --existing")
        original_by_id[question_id] = original
        previous.setdefault(question_id, []).append(
            required_text(row, "question_transformed", row["variant_id"])
        )
    output = []
    log_path = getattr(args, "log_file", None)
    run_id = getattr(args, "run_id", None)
    for row in questions:
        question_id = row["question_id"]
        if not reviewed(row, "question_review", question_id):
            continue
        question = required_text(row, "question", question_id)
        if question_id in original_by_id and original_by_id[question_id] != question:
            raise WorkflowError(f"{question_id}: original question differs from --existing")
        if mode == "combined":
            output.extend(_generate_combined_question(
                row, question, args, config, combined_targets,
                previous.get(question_id, []), used_ids,
            ))
            if getattr(args, "progress", False):
                print(f"Generated {args.count ** 3} combined paraphrases for {question_id}",
                      file=sys.stderr)
            continue
        seen = previous.get(question_id, []).copy()
        seen_exact = {normalize(value) for value in [question, *seen]}
        next_index = 1
        generated = 0
        stalled_attempts = 0
        batch_size = getattr(args, "batch_size", 5)
        while generated < args.count:
            count = 1 if mode != "general" else min(batch_size, args.count - generated)
            target = targets[generated % len(targets)] if targets else None
            error = None
            request_id = uuid4().hex

            def model_trace(event: dict[str, Any]) -> None:
                append_log(log_path, {"run_id": run_id, "question_id": question_id,
                                      "request_id": request_id, "event": f"llm_{event['event']}",
                                      "requested": count, "variation_attempt": stalled_attempts,
                                      "mode": mode, "target": target,
                                      **{key: value for key, value in event.items() if key != "event"}})

            try:
                model_kwargs = dict(question=question, count=count,
                                    existing_paraphrases=seen.copy(), allow_duplicates=True,
                                    allow_partial=True, variation_attempt=stalled_attempts,
                                    model_trace=model_trace if log_path is not None else None)
                if mode != "general":
                    model_kwargs.update(mode=mode, target=target)
                result = call_model(config, "paraphrases", **model_kwargs)
            except WorkflowError as exc:
                if "generator returned invalid JSON" not in str(exc):
                    append_log(log_path, {"run_id": run_id, "question_id": question_id,
                                          "event": "request_error", "requested": count,
                                          "accepted_before": generated, "error": str(exc)})
                    raise
                error = str(exc)
                result = {"variants": [], "model": config["generator"]["model"]}
            if not isinstance(result.get("variants"), list):
                append_log(log_path, {"run_id": run_id, "question_id": question_id,
                                      "event": "invalid_batch", "requested": count,
                                      "accepted_before": generated,
                                      "error": "generator returned a non-list variants field"})
                raise WorkflowError(f"{question_id}: generator returned an invalid paraphrase batch")
            accepted = 0
            duplicate_examples = []
            for item in result["variants"]:
                if generated >= args.count or accepted >= count:
                    break
                transformed = item["question"].strip()
                if normalize(transformed) in seen_exact:
                    duplicate_examples.append({"text": transformed[:300],
                                               "reason": "original" if normalize(transformed) == normalize(question)
                                               else "previous_paraphrase"})
                    continue
                seen_exact.add(normalize(transformed))
                seen.append(transformed)
                while f"{question_id}-v{next_index}" in used_ids:
                    next_index += 1
                variant_id = f"{question_id}-v{next_index}"
                used_ids.add(variant_id)
                next_index += 1
                output.append({
                    "question_id": question_id,
                    "variant_id": variant_id,
                    "question_original": question,
                    "question_transformed": transformed,
                    "transformation_type": (f"{mode}:{target}" if target else
                                            item["transformation_type"]),
                    "transformation_mode": mode,
                    "transformation_target": target,
                    "source": row.get("source", "unknown"),
                    "topic": row.get("topic"),
                    "language": row.get("language"),
                    "generator_model": result["model"],
                    "paraphrase_review": "pending",
                })
                generated += 1
                accepted += 1
            append_log(log_path, {
                "run_id": run_id, "question_id": question_id, "request_id": request_id,
                "event": "batch_result",
                "requested": count, "raw_variants": result.get("raw_variant_count"),
                "valid_variants": len(result["variants"]),
                "invalid_variants": result.get("invalid_variant_count"),
                "accepted": accepted, "accepted_total": generated,
                "duplicate_count": len(duplicate_examples),
                "duplicate_examples": duplicate_examples[:5],
                "variation_attempt": stalled_attempts, "error": error,
                "mode": mode, "target": target,
            })
            if accepted:
                stalled_attempts = 0
            else:
                stalled_attempts += 1
                batch_size = max(1, batch_size // 2)
                if stalled_attempts >= 5:
                    append_log(log_path, {"run_id": run_id, "question_id": question_id,
                                          "event": "stopped", "accepted_total": generated,
                                          "target": args.count, "reason": "five_empty_attempts"})
                    raise WorkflowError(
                        f"{question_id}: generator produced no usable new paraphrase in 5 attempts "
                        f"({generated}/{args.count} completed)"
                    )
        if getattr(args, "progress", False):
            print(f"Generated {args.count} paraphrases for {question_id}", file=sys.stderr)
    return output


def generate_answers(args: argparse.Namespace, config: dict[str, Any]) -> list[dict[str, Any]]:
    variants = read_jsonl(args.input)
    unique_ids(variants, "variant_id")
    original_answers: dict[str, str] = {}
    original_questions: dict[str, str] = {}
    output = []
    for row in variants:
        variant_id = row["variant_id"]
        if not reviewed(row, "paraphrase_review", variant_id):
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
            "transformation_mode": row.get("transformation_mode"),
            "transformation_target": row.get("transformation_target"),
            "transformation_chain": row.get("transformation_chain"),
            "generator_model": row.get("generator_model"),
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
            "transformation_mode": row.get("transformation_mode"),
            "transformation_target": row.get("transformation_target"),
            "transformation_chain": row.get("transformation_chain"),
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
        parser.add_argument("--batch-size", type=int, default=5)
        parser.add_argument("--mode", choices=(*PARAPHRASE_MODES, "combined"), default="general")
        parser.add_argument("--targets", nargs="+")
        parser.add_argument("--structures", nargs="+")
        parser.add_argument("--styles", nargs="+")
        parser.add_argument("--languages", nargs="+")
        parser.add_argument("--existing", type=Path)
        parser.add_argument("--log-file", type=Path)
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
    if command == "generate_paraphrases" and args.mode == "combined" and args.count > 10:
        parser.error("--mode combined requires --count at most 10 (count cubed outputs)")
    if command == "generate_paraphrases" and not 1 <= args.batch_size <= 100:
        parser.error("--batch-size must be between 1 and 100")
    if command == "generate_paraphrases" and args.mode == "general" and args.targets:
        parser.error("--targets requires --mode translation, structure, or style")
    if command == "generate_paraphrases" and args.mode == "combined" and args.targets:
        parser.error("--mode combined uses --structures, --styles, and --languages")
    if command == "generate_paraphrases" and args.mode != "combined" and any(
        (args.structures, args.styles, args.languages)
    ):
        parser.error("--structures, --styles, and --languages require --mode combined")
    if command == "generate_paraphrases" and args.mode == "combined":
        try:
            _combined_targets(args)
        except WorkflowError as exc:
            parser.error(str(exc))
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
            args.progress = True
            args.log_file = args.log_file or args.output.with_suffix(".log.jsonl")
            args.run_id = uuid4().hex
            print(f"Paraphrase log: {args.log_file}", file=sys.stderr)
            rows = generate_paraphrases(args, config)
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
