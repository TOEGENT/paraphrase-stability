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
import re
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path


NUMBER = r"[+-]?\s*(?:\\(?:dfrac|tfrac|frac)\s*\{[+-]?\d+\}\s*\{[+-]?\d+\}|\d+\s*/\s*[+-]?\d+|\d+(?:\.\d+)?|\.\d+)"
FINAL = re.compile(rf"(?:[a-z]\s*=\s*)?(?P<value>{NUMBER})", re.IGNORECASE)
ASSIGNMENT = re.compile(rf"(?<![A-Za-z0-9+-])[a-z]\s*=\s*(?P<value>{NUMBER})(?![\w./])", re.IGNORECASE)
LATEX_FRACTION = re.compile(r"(?P<sign>[+-]?)\s*\\(?:dfrac|tfrac|frac)\s*\{(?P<num>[+-]?\d+)\}\s*\{(?P<den>[+-]?\d+)\}")
PREFIX = re.compile(r"^(?:(?:final\s+answer|answer|the\s+answer\s+is|ответ|итоговый\s+ответ)\s*[:=]?\s*)", re.IGNORECASE)


def _unbox(value: str) -> str:
    """Remove balanced LaTeX boxes, including boxes around fractions."""
    while "\\boxed{" in value:
        start = value.find("\\boxed{")
        opening = start + len("\\boxed")
        depth = 0
        closing = None
        for index in range(opening, len(value)):
            if value[index] == "{":
                depth += 1
            elif value[index] == "}":
                depth -= 1
                if depth == 0:
                    closing = index
                    break
        if closing is None:
            return value
        value = value[:start] + value[opening + 1:closing] + value[closing + 1:]
    return value


def _fraction(value: str) -> Fraction | None:
    value = value.strip().replace(" ", "")
    latex = LATEX_FRACTION.fullmatch(value)
    try:
        if latex:
            sign = -1 if latex["sign"] == "-" else 1
            return sign * Fraction(int(latex["num"]), int(latex["den"]))
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            return Fraction(int(numerator), int(denominator))
        return Fraction(Decimal(value))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def extract_final_x(answer: str) -> Fraction | None:
    """Parse the last answer line and reject contradictory x assignments."""
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if not lines:
        return None
    final_line = _unbox(lines[-1]).replace("**", "").strip()
    final_line = final_line.removeprefix("\\(").removesuffix("\\)")
    final_line = final_line.removeprefix("$").removesuffix("$").strip()
    final_line = PREFIX.sub("", final_line).strip().rstrip(".!; ")
    match = FINAL.fullmatch(final_line)
    if match is None:
        return None
    result = _fraction(match["value"])
    if result is None:
        return None
    for assignment in ASSIGNMENT.finditer(_unbox(answer)):
        other = _fraction(assignment["value"])
        if other is None or other != result:
            return None
    return result


def verify_pair(original: str, transformed: str) -> dict[str, str | None]:
    left = extract_final_x(original)
    right = extract_final_x(transformed)
    if left is None or right is None:
        status = "unclear"
        rationale = "No unambiguous final x value in one or both answers"
    elif left == right:
        status = "equivalent"
        rationale = f"Final x values match: {left} = {right}"
    else:
        status = "non_equivalent"
        rationale = f"Final x values differ: {left} != {right}"
    return {
        "answer_equivalence": status,
        "answer_judge_rationale": rationale,
        "answer_original_extracted": str(left) if left is not None else None,
        "answer_transformed_extracted": str(right) if right is not None else None,
        "answer_verifier": "exact_final_x_v1",
    }


def verify_file(input_path: Path, output_path: Path) -> Counter[str]:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")
    if output_path.exists():
        raise ValueError(f"output already exists: {output_path}")
    try:
        lines = input_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read {input_path}: {exc}") from exc
    rows = []
    seen_ids: set[str] = set()
    counts: Counter[str] = Counter()
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
        for field in ("answer_original", "answer_transformed"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"line {number}: missing {field}")
        result = verify_pair(row["answer_original"], row["answer_transformed"])
        counts[result["answer_equivalence"]] += 1
        rows.append({**row, **result})
    if not rows:
        raise ValueError("input has no answer rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return counts


def verify_all_anchors(input_path: Path, output_path: Path) -> Counter[str]:
    """Write all directed within-equation comparisons, excluding self-pairs."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")
    if output_path.exists():
        raise ValueError(f"output already exists: {output_path}")
    try:
        lines = input_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read {input_path}: {exc}") from exc
    groups: dict[str, dict] = {}
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
        for field in ("question_id", "question_original", "question_transformed",
                      "answer_original", "answer_transformed"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"line {number}: missing {field}")
        original_x = extract_final_x(row["answer_original"])
        transformed_x = extract_final_x(row["answer_transformed"])
        if original_x is None or transformed_x is None:
            raise ValueError(f"line {number}: answer has no unambiguous final x")
        source_id = row["question_id"]
        metadata = {key: row.get(key) for key in ("source", "topic", "language", "slm_model")}
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
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
                record = {
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
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return counts


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
