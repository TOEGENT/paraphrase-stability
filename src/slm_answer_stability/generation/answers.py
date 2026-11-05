"""Ask the SLM for answers to every approved question variant."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from time import monotonic
from typing import Any

from ..data.io import WorkflowError, ensure_new_output, read_jsonl, required_text, reviewed, unique_ids
from . import client


def generate_answers(args: argparse.Namespace, config: dict[str, Any]) -> list[dict[str, Any]]:
    variants = read_jsonl(args.input)
    unique_ids(variants, "variant_id")
    output_path = getattr(args, "output", None)
    if output_path is not None:
        ensure_new_output(output_path)
    total = sum(row.get("paraphrase_review") == "approved" for row in variants)
    original_answers: dict[str, str] = {}
    original_questions: dict[str, str] = {}
    output = []
    output_file = None
    started_at = monotonic()
    try:
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
                original_answers[question_id] = client.call_model(config, "answer", question=question)["answer"]
            transformed_answer = client.call_model(config, "answer", question=transformed)["answer"]
            answer_row = {
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
            }
            if output_path is not None:
                if output_file is None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_file = output_path.open("x", encoding="utf-8")
                output_file.write(json.dumps(answer_row, ensure_ascii=False) + "\n")
                output_file.flush()
            output.append(answer_row)
            if getattr(args, "progress", False):
                filled = 20 * len(output) // total
                elapsed = max(0, monotonic() - started_at)
                remaining = elapsed / len(output) * (total - len(output))
                print(
                    f"Answers [{('#' * filled).ljust(20, '-')}] {len(output)}/{total} "
                    f"elapsed {timedelta(seconds=round(elapsed))} "
                    f"remaining ~{timedelta(seconds=round(remaining))}",
                    file=sys.stderr,
                )
    except OSError as exc:
        raise WorkflowError(f"Cannot write {output_path}: {exc}") from exc
    finally:
        if output_file is not None:
            output_file.close()
    if output_path is not None and not output:
        raise WorkflowError("No records to write")
    return output
