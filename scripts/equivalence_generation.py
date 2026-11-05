"""Compare every answer anchor with the other answers to one equation.

Input: JSONL produced by generate_answers.py.
Default output: one row per anchor with pairwise equivalence statuses.
Use --original-only for the former one-row-per-variant output.
Only one final integer, decimal, a/b, or LaTeX fraction is accepted per answer.
Unclear or unfinished responses remain unresolved for human review.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from slm_answer_stability.constants import ANCHOR_METADATA_KEYS
from slm_answer_stability.annotation.answer_extraction import (  # noqa: F401  (re-exported)
    ASSIGNMENT, FINAL, LATEX_FRACTION, NUMBER, PREFIX, _fraction, _unbox, extract_final_x, verify_pair,
)


def _check_paths(input_path: Path, output_path: Path) -> None:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")
    if output_path.exists():
        raise ValueError(f"output already exists: {output_path}")


def _answer_rows(input_path: Path, required_fields: tuple[str, ...]):
    """Yield (line number, row) for each answer row, validating one line at a time."""
    try:
        lines = input_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read {input_path}: {exc}") from exc
    seen_ids: set[str] = set()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at line {number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"line {number}: expected an object")
        variant_id = row.get("variant_id")
        if not isinstance(variant_id, str) or not variant_id or variant_id in seen_ids:
            raise ValueError(f"line {number}: missing or duplicate variant_id")
        seen_ids.add(variant_id)
        if "answer_equivalence" in row:
            raise ValueError(f"line {number}: answer_equivalence already exists")
        for field in required_fields:
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"line {number}: missing {field}")
        yield number, row


def _write_rows(output_path: Path, rows) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def verify_file(input_path: Path, output_path: Path) -> Counter[str]:
    _check_paths(input_path, output_path)
    rows = []
    counts: Counter[str] = Counter()
    for _, row in _answer_rows(input_path, ("answer_original", "answer_transformed")):
        result = verify_pair(row["answer_original"], row["answer_transformed"])
        counts[result["answer_equivalence"]] += 1
        rows.append({**row, **result})
    if not rows:
        raise ValueError("input has no answer rows")
    _write_rows(output_path, rows)
    return counts


def verify_all_anchors(input_path: Path, output_path: Path) -> Counter[str]:
    """Write all directed within-equation comparisons, excluding self-pairs."""
    _check_paths(input_path, output_path)
    groups: dict[str, dict] = {}
    for number, row in _answer_rows(input_path, (
            "question_id", "question_original", "question_transformed",
            "answer_original", "answer_transformed")):
        variant_id = row["variant_id"]
        original_x = extract_final_x(row["answer_original"])
        transformed_x = extract_final_x(row["answer_transformed"])
        if original_x is None or transformed_x is None:
            raise ValueError(f"line {number}: answer has no unambiguous final x")
        source_id = row["question_id"]
        metadata = {key: row.get(key) for key in ANCHOR_METADATA_KEYS}
        group = groups.get(source_id)
        if group is None:
            group = {
                "question_original": row["question_original"],
                "answer_original": row["answer_original"],
                "original_x": original_x,
                "metadata": metadata,
                "variants": [],
            }
            groups[source_id] = group
        elif (group["question_original"] != row["question_original"]
              or group["answer_original"] != row["answer_original"]
              or group["metadata"] != metadata):
            raise ValueError(f"line {number}: conflicting original pair for {source_id}")
        group["variants"].append((variant_id, row["question_transformed"],
                                  row["answer_transformed"], transformed_x))
    if not groups:
        raise ValueError("input has no answer rows")

    counts: Counter[str] = Counter()
    _write_rows(output_path, _anchor_records(groups, counts))
    return counts


def _anchor_records(groups: dict[str, dict], counts: Counter[str]):
    for source_id, group in groups.items():
        anchors = [("original", group["question_original"], group["answer_original"],
                    group["original_x"]), *group["variants"]]
        for anchor_id, question, answer, x in anchors:
            comparisons = []
            for other_id, _, _, other_x in anchors:
                if other_id == anchor_id:
                    continue
                status = "equivalent" if x == other_x else "non_equivalent"
                counts[status] += 1
                comparisons.append({"other_anchor_id": other_id, "answer_equivalence": status})
            yield {
                "question_id": f"{source_id}::anchor::{anchor_id}",
                "source_question_id": source_id,
                "anchor_id": anchor_id,
                "question": question,
                "answer": answer,
                "extracted_x": str(x),
                "comparisons": comparisons,
                "answer_verifier": "exact_final_x_v1",
                **group["metadata"],
            }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--original-only", action="store_true",
                        help="Compare each variant only with the original answer")
    args = parser.parse_args(argv)
    try:
        counts = (verify_file(args.input, args.output) if args.original_only else
                  verify_all_anchors(args.input, args.output))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output), "counts": dict(counts)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
