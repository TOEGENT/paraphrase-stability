"""Judge answer equivalence with the judge model."""

from __future__ import annotations

import argparse
from typing import Any

from ..data.io import WorkflowError, read_jsonl, required_text, unique_ids
from ..generation import client


def generate_answer_equivalence_ai(args: argparse.Namespace, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = read_jsonl(args.input)
    unique_ids(rows, "variant_id")
    output = []
    for row in rows:
        variant_id = row["variant_id"]
        question = required_text(row, "question_original", variant_id)
        original_answer = required_text(row, "answer_original", variant_id)
        required_text(row, "question_transformed", variant_id)
        transformed_answer = required_text(row, "answer_transformed", variant_id)
        if "answer_equivalence" in row:
            raise WorkflowError(f"{variant_id}: answer_equivalence already exists")
        decision = client.call_model(
            config, "judge-answer", question=question,
            answer=original_answer, rewritten_answer=transformed_answer,
        )
        output.append({
            **row,
            "answer_equivalence": decision["status"],
            "answer_judge_rationale": decision["rationale_short"],
            "answer_judge_model": config["judge"]["model"],
        })
    return output
