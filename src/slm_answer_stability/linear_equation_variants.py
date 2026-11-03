"""Generate verified equivalent formulations of parsed linear equations."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import sympy
from sympy.solvers.solveset import NonlinearError

from .linear_equation_parser import parse_linear_row


def _system(row: dict) -> tuple[list[sympy.Symbol], sympy.Matrix, sympy.Matrix, sympy.Matrix]:
    parsed = row.get("parsed_linear")
    if not isinstance(parsed, dict):
        parsed = parse_linear_row(row)["parsed_linear"]
    if any(not re.fullmatch(r"[a-z]", name) for name in parsed["variables"]):
        raise ValueError("unexpected variable name")
    variables = [sympy.Symbol(name) for name in parsed["variables"]]
    size = len(variables)
    if size not in (1, 2) or len(set(variables)) != size:
        raise ValueError("expected one or two distinct variables")
    matrix = sympy.Matrix([[sympy.Rational(value) for value in line]
                           for line in parsed["coefficients"]])
    right = sympy.Matrix([sympy.Rational(value) for value in parsed["right_hand_side"]])
    solution = sympy.Matrix([sympy.Rational(value) for value in parsed["solution"]])
    if matrix.shape != (size, size) or right.shape != (size, 1) or solution.shape != (size, 1):
        raise ValueError("invalid parsed_linear dimensions")
    if matrix.det() == 0 or matrix * solution != right:
        raise ValueError("parsed_linear does not describe a unique valid solution")
    if solution[0] != sympy.Rational(row["answer"]):
        raise ValueError("parsed_linear solution disagrees with answer")
    return variables, matrix, right, solution


def _candidate(
    variables: list[sympy.Symbol], matrix: sympy.Matrix, right: sympy.Matrix,
    rng: random.Random,
) -> tuple[str, list[str]]:
    size = len(variables)
    working_matrix = matrix.copy()
    working_right = right.copy()
    steps = []
    if size == 2 and rng.choice((False, True)):
        i = rng.randrange(2)
        j = 1 - i
        factor = rng.choice((-3, -2, -1, 1, 2, 3))
        working_matrix[i, :] = working_matrix[i, :] + factor * working_matrix[j, :]
        working_right[i] += factor * working_right[j]
        steps.append(f"row_{i + 1} += {factor}*row_{j + 1}")
    if size == 2 and rng.choice((False, True)):
        working_matrix.row_swap(0, 1)
        working_right.row_swap(0, 1)
        steps.append("swap_equations")
    equations = []
    for i in range(size):
        factor = rng.choice((-4, -3, -2, -1, 1, 2, 3, 4))
        coefficients = [rng.randint(-5, 5) for _ in variables]
        constant = rng.randint(-12, 12)
        shift = sum((coefficient * variable for coefficient, variable
                     in zip(coefficients, variables)), sympy.Integer(constant))
        left = factor * sum((working_matrix[i, j] * variables[j]
                             for j in range(size)), sympy.Integer(0)) + shift
        other = factor * working_right[i] + shift
        left, other = sympy.expand(left), sympy.expand(other)
        if rng.choice((False, True)):
            left, other = other, left
            steps.append(f"swap_sides_{i + 1}")
        if rng.choice((False, True)):
            left, other = sympy.factor(left), sympy.factor(other)
            steps.append(f"factor_display_{i + 1}")
        equations.append(f"{left} = {other}")
        steps.append(f"scale_{i + 1}_by_{factor}_and_add_same_expression")
    return f"Solve {', '.join(equations)} for {variables[0]}.", steps


def _verified(question: str, variables: list[sympy.Symbol], solution: sympy.Matrix) -> bool:
    # The candidate is produced from exact SymPy expressions, but validate the
    # final text too, since that is what downstream models will receive.
    prefix = "Solve "
    suffix = f" for {variables[0]}."
    if not question.startswith(prefix) or not question.endswith(suffix):
        return False
    body = question[len(prefix):-len(suffix)]
    parts = body.split(", ")
    if len(parts) != len(variables):
        return False
    residuals = []
    locals_ = {str(variable): variable for variable in variables}
    for part in parts:
        sides = part.split(" = ")
        if len(sides) != 2:
            return False
        try:
            left = sympy.sympify(sides[0], locals=locals_)
            right = sympy.sympify(sides[1], locals=locals_)
        except (sympy.SympifyError, TypeError, ValueError):
            return False
        residuals.append(left - right)
    try:
        matrix, right = sympy.linear_eq_to_matrix(residuals, variables)
    except NonlinearError:
        return False
    return matrix.det() != 0 and matrix * solution == right


def generate_variants(row: dict, count: int, seed: int) -> list[dict]:
    if count < 1:
        raise ValueError("variants per question must be positive")
    question_id = row.get("question_id")
    original = row.get("question")
    if not isinstance(question_id, str) or not question_id or not isinstance(original, str) or not original:
        raise ValueError("missing question_id or question")
    variables, matrix, right, solution = _system(row)
    rng = random.Random(f"{seed}:{question_id}")
    seen = {original}
    variants = []
    for _ in range(max(1000, count * 100)):
        if len(variants) == count:
            break
        transformed, steps = _candidate(variables, matrix, right, rng)
        if transformed in seen or not _verified(transformed, variables, solution):
            continue
        seen.add(transformed)
        variants.append({
            "question_id": question_id,
            "variant_id": f"{question_id}-v{len(variants) + 1}",
            "question_original": original,
            "question_transformed": transformed,
            "reference_answer": row["answer"],
            "source": row.get("source", "mathematics_dataset"),
            "topic": row.get("topic", "algebra"),
            "module": row.get("module"),
            "language": "en",
            "transformation_type": "linear_equivalence",
            "transformation_mode": "structure",
            "transformation_steps": steps,
            "paraphrase_review": "pending",
        })
    if len(variants) != count:
        raise ValueError(f"{question_id}: generated only {len(variants)} of {count} variants")
    return variants


def generate_file(input_path: Path, output_path: Path, count: int, seed: int) -> int:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")
    if output_path.exists():
        raise ValueError(f"output already exists: {output_path}")
    rows = []
    seen_ids = set()
    for number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if row["question_id"] in seen_ids:
                raise ValueError(f"duplicate question_id: {row['question_id']}")
            seen_ids.add(row["question_id"])
            rows.extend(generate_variants(row, count, seed))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"line {number}: {exc}") from exc
    if not rows:
        raise ValueError("input has no questions")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variants-per-question", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        count = generate_file(args.input, args.output, args.variants_per_question, args.seed)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output), "variants": count}))
    return 0
