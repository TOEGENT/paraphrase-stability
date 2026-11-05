"""Turn compared answers into k/N stability labels (per question or per answer anchor)."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from ..annotation.answer_extraction import extract_final_x
from ..constants import ANCHOR_METADATA_KEYS, EQUIVALENCE_STATUSES, UNRESOLVED_STATUSES
from .io import WorkflowError, read_jsonl, required_text, reviewed, unique_ids
from .splits import partition_keys, validate_ratios


def compute_k_n(args: argparse.Namespace) -> list[dict[str, Any]]:
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
        if status not in EQUIVALENCE_STATUSES:
            raise WorkflowError(f"{variant_id}: missing or invalid answer_equivalence")
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
            "answer_judge_rationale": row.get("answer_judge_rationale"),
            "answer_judge_model": row.get("answer_judge_model"),
        })
    results = []
    for group in output.values():
        variants = group["variants"]
        k = sum(row["answer_equivalence"] == "equivalent" for row in variants)
        unresolved = sum(row["answer_equivalence"] in UNRESOLVED_STATUSES for row in variants)
        n = len(variants)
        group.update({"k": k, "N": n, "target": None if unresolved else k / n, "unresolved": unresolved})
        results.append(group)
    return results


def _anchor_splits(group_ids: set[str], splits_dir: Path | None, seed: int,
                   ratios: tuple[float, float, float]) -> dict[str, str]:
    if splits_dir is not None:
        splits: dict[str, str] = {}
        for split in ("train", "val", "test"):
            for row in read_jsonl(splits_dir / f"{split}.jsonl"):
                source_id = required_text(row, "question_id", split)
                if source_id in splits:
                    raise WorkflowError(f"{source_id}: appears in multiple splits")
                splits[source_id] = split
        if set(splits) != group_ids:
            raise WorkflowError("split groups differ from answer anchors")
        return splits
    if len(group_ids) < 3:
        raise WorkflowError("At least three source questions are required for three splits")
    validate_ratios(ratios)
    return {source_id: split
            for split, ids in partition_keys(group_ids, ratios, seed).items()
            for source_id in ids}


def _anchor_row(question_id: str, source_id: str, anchor_id: str, question: str, answer: str,
                extracted_x: str, k: int, n: int, split: str,
                metadata: dict[str, Any]) -> dict[str, Any]:
    """One labelled anchor: ``k`` of the other ``n`` answers to the same equation agree with it."""
    return {
        "question_id": question_id,
        "source_question_id": source_id,
        "anchor_id": anchor_id,
        "question": question,
        "answer": answer,
        "extracted_x": extracted_x,
        "k": k,
        "N": n,
        "target": k / n,
        "split": split,
        **metadata,
    }


def compute_all_anchors(answers_path: Path, splits_dir: Path | None = None, seed: int = 42,
                        ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)) -> list[dict[str, Any]]:
    """Reanchor each equation answer and count equal final x values among the others."""
    input_rows = read_jsonl(answers_path)
    if "comparisons" in input_rows[0]:
        return _compute_verified_anchors(input_rows, splits_dir, seed, ratios)
    groups: dict[str, dict[str, Any]] = {}
    seen_variants: set[str] = set()
    for row in input_rows:
        variant_id = required_text(row, "variant_id", "answer row")
        if variant_id in seen_variants:
            raise WorkflowError(f"duplicate variant_id: {variant_id}")
        seen_variants.add(variant_id)
        question_id = required_text(row, "question_id", variant_id)
        original_question = required_text(row, "question_original", variant_id)
        original_answer = required_text(row, "answer_original", variant_id)
        transformed_question = required_text(row, "question_transformed", variant_id)
        transformed_answer = required_text(row, "answer_transformed", variant_id)
        original_x = extract_final_x(original_answer)
        transformed_x = extract_final_x(transformed_answer)
        if original_x is None or transformed_x is None:
            raise WorkflowError(f"{variant_id}: answer has no unambiguous final x")
        expected = "equivalent" if original_x == transformed_x else "non_equivalent"
        if "answer_equivalence" in row and row["answer_equivalence"] != expected:
            raise WorkflowError(f"{variant_id}: answer_equivalence disagrees with extracted x")
        metadata = {key: row.get(key) for key in ANCHOR_METADATA_KEYS}
        group = groups.get(question_id)
        if group is None:
            group = {"question": original_question, "answer": original_answer,
                     "x": original_x, "metadata": metadata, "anchors": []}
            groups[question_id] = group
        elif (group["question"] != original_question or group["answer"] != original_answer
              or group["metadata"] != metadata):
            raise WorkflowError(f"{question_id}: conflicting original question, answer, or metadata")
        group["anchors"].append((variant_id, transformed_question, transformed_answer, transformed_x))

    splits = _anchor_splits(set(groups), splits_dir, seed, ratios)

    output: list[dict[str, Any]] = []
    for source_question_id, group in groups.items():
        anchors = [("original", group["question"], group["answer"], group["x"]),
                   *group["anchors"]]
        counts = Counter(x for _, _, _, x in anchors)
        n = len(anchors) - 1
        for anchor_id, question, answer, x in anchors:
            k = counts[x] - 1
            output.append(_anchor_row(
                f"{source_question_id}::anchor::{anchor_id}", source_question_id, anchor_id, question, answer, str(x), k, n,
                splits[source_question_id], group["metadata"],
            ))
    return output


def _compute_verified_anchors(rows: list[dict[str, Any]], splits_dir: Path | None, seed: int,
                              ratios: tuple[float, float, float]) -> list[dict[str, Any]]:
    """Sum explicit pairwise statuses produced by equivalence_generation.py."""
    groups: dict[str, list[dict[str, Any]]] = {}
    seen_ids: set[str] = set()
    for row in rows:
        anchor_id = required_text(row, "question_id", "anchor row")
        source_id = required_text(row, "source_question_id", anchor_id)
        required_text(row, "anchor_id", anchor_id)
        required_text(row, "question", anchor_id)
        required_text(row, "answer", anchor_id)
        required_text(row, "extracted_x", anchor_id)
        if anchor_id in seen_ids:
            raise WorkflowError(f"{anchor_id}: duplicate question_id")
        seen_ids.add(anchor_id)
        if not isinstance(row.get("comparisons"), list):
            raise WorkflowError(f"{anchor_id}: missing comparisons")
        groups.setdefault(source_id, []).append(row)

    splits = _anchor_splits(set(groups), splits_dir, seed, ratios)

    output: list[dict[str, Any]] = []
    for source_id, anchors in groups.items():
        ids = {row["anchor_id"] for row in anchors}
        if len(ids) != len(anchors) or "original" not in ids:
            raise WorkflowError(f"{source_id}: missing or duplicate anchor")
        extracted = {row["anchor_id"]: row["extracted_x"] for row in anchors}
        for row in anchors:
            comparisons = row["comparisons"]
            other_ids: set[str] = set()
            k = 0
            for comparison in comparisons:
                if not isinstance(comparison, dict):
                    raise WorkflowError(f"{row['question_id']}: invalid comparison")
                other_id = required_text(comparison, "other_anchor_id", row["question_id"])
                if other_id == row["anchor_id"] or other_id not in ids or other_id in other_ids:
                    raise WorkflowError(f"{row['question_id']}: invalid comparison target")
                other_ids.add(other_id)
                status = comparison.get("answer_equivalence")
                expected = "equivalent" if row["extracted_x"] == extracted[other_id] else "non_equivalent"
                if status != expected:
                    raise WorkflowError(f"{row['question_id']}: comparison disagrees with extracted x")
                k += status == "equivalent"
            if other_ids != ids - {row["anchor_id"]}:
                raise WorkflowError(f"{row['question_id']}: incomplete comparisons")
            n = len(comparisons)
            output.append(_anchor_row(
                row["question_id"], source_id, row["anchor_id"], row["question"], row["answer"], row["extracted_x"],
                k, n, splits[source_id], {key: row.get(key) for key in ANCHOR_METADATA_KEYS},
            ))
    return output


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
        if len(statuses) != len(variants) or any(
            status not in EQUIVALENCE_STATUSES for status in statuses
        ):
            raise WorkflowError(f"{question_id}: invalid answer_equivalence")
        n = len(variants)
        k = statuses.count("equivalent")
        unresolved = sum(status in UNRESOLVED_STATUSES for status in statuses)
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
