"""Generate reviewed-question paraphrases (general, translation, structure, style, combined)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..data.io import (WorkflowError, append_log, normalize, read_jsonl, required_text, reviewed,
                       unique_ids)
from ..model_tools import PARAPHRASE_MODES
from . import client


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


class VariantIdAllocator:
    """Hands out unused ``<question_id>-v<N>`` ids, counting up from 1."""

    def __init__(self, question_id: str, used_ids: set[str]):
        self.question_id = question_id
        self.used_ids = used_ids
        self.index = 1

    def next(self) -> str:
        while f"{self.question_id}-v{self.index}" in self.used_ids:
            self.index += 1
        variant_id = f"{self.question_id}-v{self.index}"
        self.used_ids.add(variant_id)
        self.index += 1
        return variant_id


class DedupTracker:
    """Wordings already used for one question: the original, earlier runs and accepted variants."""

    def __init__(self, original: str, previous: list[str]):
        self.original = original
        self.history = list(previous)  # shown to the model; the original is not part of it
        self._keys = {normalize(value) for value in [original, *self.history]}

    def is_duplicate(self, text: str) -> bool:
        return normalize(text) in self._keys

    def reason(self, text: str) -> str:
        return "original" if normalize(text) == normalize(self.original) else "previous_paraphrase"

    def add(self, text: str) -> None:
        self._keys.add(normalize(text))
        self.history.append(text)


class ParaphraseLogger:
    """Writes the run log for one question; a no-op when no log file was requested."""

    def __init__(self, path: Path | None, run_id: str | None, question_id: str):
        self.path = path
        self.run_id = run_id
        self.question_id = question_id

    def log(self, **fields: Any) -> None:
        append_log(self.path, {"run_id": self.run_id, "question_id": self.question_id, **fields})

    def trace(self, request_id: str, **context: Any):
        """Callback recording the exact request and raw response of one model call."""
        if self.path is None:
            return None

        def record(event: dict[str, Any]) -> None:
            self.log(request_id=request_id, event=f"llm_{event['event']}", **context,
                     **{key: value for key, value in event.items() if key != "event"})
        return record


def _variant_row(row: dict[str, Any], variant_id: str, question: str, transformed: str,
                 transformation_type: str, mode: str, target: str | None, generator_model: str,
                 chain: list[dict[str, str]] | None = None) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "question_id": row["question_id"],
        "variant_id": variant_id,
        "question_original": question,
        "question_transformed": transformed,
        "transformation_type": transformation_type,
        "transformation_mode": mode,
        "transformation_target": target,
    }
    if chain is not None:
        fields["transformation_chain"] = chain
    return {
        **fields,
        "source": row.get("source", "unknown"),
        "topic": row.get("topic"),
        "language": row.get("language"),
        "generator_model": generator_model,
        "paraphrase_review": "pending",
    }


def _generate_combined_question(
    row: dict[str, Any], question: str, args: argparse.Namespace,
    config: dict[str, Any], targets: dict[str, list[str]],
    previous: list[str], used_ids: set[str],
) -> list[dict[str, Any]]:
    question_id = row["question_id"]
    dedup = DedupTracker(question, previous)
    allocator = VariantIdAllocator(question_id, used_ids)
    logger = ParaphraseLogger(getattr(args, "log_file", None), getattr(args, "run_id", None), question_id)

    def generate_one(parent: str, mode: str, target: str, path: tuple[int, ...]) -> str:
        for attempt in range(5):
            request_id = uuid4().hex
            error = None
            try:
                result = client.call_model(
                    config, "paraphrases", question=parent, count=1,
                    mode=mode, target=target, reference_question=question,
                    preserve_structure=mode != "structure",
                    existing_paraphrases=dedup.history.copy(), allow_duplicates=True,
                    allow_partial=True, variation_attempt=attempt,
                    model_trace=logger.trace(request_id, mode=mode, target=target, path=path,
                                             variation_attempt=attempt),
                )
            except client.InvalidGeneratorJSON as exc:
                error = str(exc)
                result = {"variants": []}
            except WorkflowError as exc:
                logger.log(event="request_error", mode=mode, target=target, path=path, error=str(exc))
                raise
            variants = result.get("variants")
            if not isinstance(variants, list):
                raise WorkflowError(f"{question_id}: generator returned an invalid paraphrase batch")
            duplicate_examples = []
            accepted = None
            for item in variants:
                transformed = item["question"].strip()
                if dedup.is_duplicate(transformed):
                    duplicate_examples.append(transformed[:300])
                    continue
                dedup.add(transformed)
                accepted = transformed
                break
            logger.log(request_id=request_id, event="batch_result", mode=mode, target=target, path=path,
                       accepted=0 if accepted is None else 1, duplicate_examples=duplicate_examples[:5],
                       variation_attempt=attempt, error=error)
            if accepted is not None:
                return accepted
        raise WorkflowError(
            f"{question_id}: no usable {mode}:{target} variant after 5 attempts at {path}"
        )

    output = []
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
                output.append(_variant_row(
                    row, allocator.next(), question, transformed, "combined", "combined", language,
                    config["generator"]["model"],
                    chain=[
                        {"mode": "structure", "target": structure, "question": structured},
                        {"mode": "style", "target": style, "question": styled},
                        {"mode": "translation", "target": language, "question": transformed},
                    ],
                ))
    return output


def _request_batch(config: dict[str, Any], logger: ParaphraseLogger, question: str, count: int,
                   mode: str, target: str | None, dedup: DedupTracker, attempt: int,
                   accepted_before: int, request_id: str) -> tuple[dict[str, Any], str | None]:
    """Ask for ``count`` variants; unparseable JSON counts as an empty batch, other errors stop the run."""
    kwargs = dict(question=question, count=count,
                  existing_paraphrases=dedup.history.copy(), allow_duplicates=True,
                  allow_partial=True, variation_attempt=attempt,
                  model_trace=logger.trace(request_id, requested=count, variation_attempt=attempt,
                                           mode=mode, target=target))
    if mode != "general":
        kwargs.update(mode=mode, target=target)
    try:
        return client.call_model(config, "paraphrases", **kwargs), None
    except client.InvalidGeneratorJSON as exc:
        return {"variants": [], "model": config["generator"]["model"]}, str(exc)
    except WorkflowError as exc:
        logger.log(event="request_error", requested=count, accepted_before=accepted_before, error=str(exc))
        raise


def _generate_for_question(
    row: dict[str, Any], question: str, args: argparse.Namespace, config: dict[str, Any],
    mode: str, targets, previous: list[str], used_ids: set[str],
) -> list[dict[str, Any]]:
    """Single-stage generation (general, translation, structure or style) for one question."""
    question_id = row["question_id"]
    dedup = DedupTracker(question, previous)
    allocator = VariantIdAllocator(question_id, used_ids)
    logger = ParaphraseLogger(getattr(args, "log_file", None), getattr(args, "run_id", None), question_id)
    output = []
    generated = 0
    stalled_attempts = 0
    batch_size = getattr(args, "batch_size", 5)
    while generated < args.count:
        count = 1 if mode != "general" else min(batch_size, args.count - generated)
        target = targets[generated % len(targets)] if targets else None
        request_id = uuid4().hex
        result, error = _request_batch(config, logger, question, count, mode, target, dedup,
                                       stalled_attempts, generated, request_id)
        if not isinstance(result.get("variants"), list):
            logger.log(event="invalid_batch", requested=count, accepted_before=generated,
                       error="generator returned a non-list variants field")
            raise WorkflowError(f"{question_id}: generator returned an invalid paraphrase batch")
        accepted = 0
        duplicate_examples = []
        for item in result["variants"]:
            if generated >= args.count or accepted >= count:
                break
            transformed = item["question"].strip()
            if dedup.is_duplicate(transformed):
                duplicate_examples.append({"text": transformed[:300], "reason": dedup.reason(transformed)})
                continue
            dedup.add(transformed)
            output.append(_variant_row(
                row, allocator.next(), question, transformed,
                f"{mode}:{target}" if target else item["transformation_type"],
                mode, target, result["model"],
            ))
            generated += 1
            accepted += 1
        logger.log(
            request_id=request_id, event="batch_result",
            requested=count, raw_variants=result.get("raw_variant_count"),
            valid_variants=len(result["variants"]),
            invalid_variants=result.get("invalid_variant_count"),
            accepted=accepted, accepted_total=generated,
            duplicate_count=len(duplicate_examples),
            duplicate_examples=duplicate_examples[:5],
            variation_attempt=stalled_attempts, error=error,
            mode=mode, target=target,
        )
        if accepted:
            stalled_attempts = 0
        else:
            stalled_attempts += 1
            batch_size = max(1, batch_size // 2)
            if stalled_attempts >= 5:
                logger.log(event="stopped", accepted_total=generated,
                           target=args.count, reason="five_empty_attempts")
                raise WorkflowError(
                    f"{question_id}: generator produced no usable new paraphrase in 5 attempts "
                    f"({generated}/{args.count} completed)"
                )
    return output


def _existing_variants(args: argparse.Namespace) -> tuple[dict[str, list[str]], dict[str, str], set[str]]:
    """Wordings, originals and variant ids from an earlier run (``--existing``)."""
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
    return previous, original_by_id, used_ids


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
    previous, original_by_id, used_ids = _existing_variants(args)
    output = []
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
        output.extend(_generate_for_question(
            row, question, args, config, mode, targets, previous.get(question_id, []), used_ids,
        ))
        if getattr(args, "progress", False):
            print(f"Generated {args.count} paraphrases for {question_id}", file=sys.stderr)
    return output
