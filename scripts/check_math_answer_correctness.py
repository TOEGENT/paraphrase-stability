"""Check SLM answers by exact substitution into linear equations for x."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from slm_answer_stability.annotation.answer_extraction import extract_final_x


class UnsupportedEquation(ValueError):
    pass


def _affine(node: ast.AST) -> tuple[Fraction, Fraction]:
    """Return coefficients (x, constant), rejecting nonlinear expressions."""
    if isinstance(node, ast.Name) and node.id == "x":
        return Fraction(1), Fraction(0)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return Fraction(0), Fraction(Decimal(str(node.value)))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        coefficient, constant = _affine(node.operand)
        sign = -1 if isinstance(node.op, ast.USub) else 1
        return sign * coefficient, sign * constant
    if isinstance(node, ast.BinOp):
        left = _affine(node.left)
        right = _affine(node.right)
        if isinstance(node.op, ast.Add):
            return left[0] + right[0], left[1] + right[1]
        if isinstance(node.op, ast.Sub):
            return left[0] - right[0], left[1] - right[1]
        if isinstance(node.op, ast.Mult):
            if left[0] and right[0]:
                raise UnsupportedEquation("nonlinear multiplication")
            return left[0] * right[1] + left[1] * right[0], left[1] * right[1]
        if isinstance(node.op, ast.Div):
            if right[0] or right[1] == 0:
                raise UnsupportedEquation("variable or zero denominator")
            return left[0] / right[1], left[1] / right[1]
    raise UnsupportedEquation("unsupported expression")


def _side(text: str) -> tuple[Fraction, Fraction]:
    # The structured generator writes 3x and -x; Python AST needs explicit *.
    text = re.sub(r"(?<=\d)(?=x\b|\()", "*", text.strip())
    try:
        return _affine(ast.parse(text, mode="eval").body)
    except (SyntaxError, ValueError, OverflowError) as exc:
        raise UnsupportedEquation("invalid equation syntax") from exc


def equation_sides(question: str) -> tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]:
    match = re.fullmatch(r"\s*Solve for x:\s*\$([^$]+)\$\.?(?:\s*)", question, re.IGNORECASE)
    if match is None or match[1].count("=") != 1:
        raise UnsupportedEquation("expected one Solve for x equation in dollar signs")
    left, right = match[1].split("=")
    sides = _side(left), _side(right)
    if sides[0][0] == sides[1][0]:
        raise UnsupportedEquation("equation has no unique solution")
    return sides


def check_answer(question: str, answer: str) -> dict[str, str | None]:
    try:
        left, right = equation_sides(question)
    except UnsupportedEquation as exc:
        return {"status": "unsupported", "extracted_x": None, "reason": str(exc)}
    value = extract_final_x(answer)
    if value is None:
        return {"status": "unparseable", "extracted_x": None,
                "reason": "no unambiguous final numeric x"}
    correct = left[0] * value + left[1] == right[0] * value + right[1]
    return {"status": "correct" if correct else "incorrect", "extracted_x": str(value),
            "reason": "exact substitution matches" if correct else "exact substitution differs"}


def check_file(input_path: Path, output_path: Path) -> dict[str, Counter[str]]:
    if input_path.resolve() == output_path.resolve() or output_path.exists():
        raise ValueError("output must be a new file different from input")
    rows = []
    seen: set[str] = set()
    counts: dict[str, Counter[str]] = {"original_unique": Counter(), "transformed": Counter()}
    original_cache: dict[str, tuple[str, str, dict[str, str | None]]] = {}
    for number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key]
                for key in ("question_id", "variant_id", "question_original", "question_transformed",
                            "answer_original", "answer_transformed")):
            raise ValueError(f"line {number}: missing required text field")
        if row["variant_id"] in seen:
            raise ValueError(f"line {number}: duplicate variant_id")
        seen.add(row["variant_id"])
        if "answer_original_correctness" in row or "answer_transformed_correctness" in row:
            raise ValueError(f"line {number}: correctness already recorded")
        question_id = row["question_id"]
        original = (row["question_original"], row["answer_original"])
        if question_id not in original_cache:
            result = check_answer(*original)
            original_cache[question_id] = (*original, result)
            counts["original_unique"][result["status"]] += 1
        elif original_cache[question_id][:2] != original:
            raise ValueError(f"line {number}: conflicting original question or answer")
        original_result = original_cache[question_id][2]
        transformed_result = check_answer(row["question_transformed"], row["answer_transformed"])
        counts["transformed"][transformed_result["status"]] += 1
        rows.append({**row,
                     "answer_original_correctness": original_result["status"],
                     "answer_transformed_correctness": transformed_result["status"],
                     "answer_original_correctness_reason": original_result["reason"],
                     "answer_transformed_correctness_reason": transformed_result["reason"],
                     "answer_original_checked_x": original_result["extracted_x"],
                     "answer_transformed_checked_x": transformed_result["extracted_x"],
                     "answer_correctness_verifier": "exact_linear_substitution_v1"})
    if not rows:
        raise ValueError("input has no answer rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        counts = check_file(args.input, args.output)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output), "counts": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
