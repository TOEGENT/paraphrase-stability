"""Generate synthetic source questions with the generator model."""

from __future__ import annotations

import argparse
import sys
from typing import Any
from uuid import uuid4

from ..constants import POSSIBLE_MATCH_STATUSES
from ..data.io import normalize, read_jsonl, required_text
from . import client


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
            result = client.call_model(
                config, "questions", topic=args.topic, count=count, existing_questions=seen.copy()
            )
            if not isinstance(result.get("questions"), list) or len(result["questions"]) != count:
                raise client.IncompleteBatch(
                    f"Generator returned an incomplete batch: expected {count} questions")
        except (client.InvalidGeneratorJSON, client.IncompleteBatch) as exc:
            if count == 1:
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
                    decision = client.call_model(
                        config, "judge-question", question=prior, rewritten_question=question
                    )["status"]
                    if decision in POSSIBLE_MATCH_STATUSES:
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
