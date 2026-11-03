"""Recover a linear system from Mathematics Dataset structured output."""

from __future__ import annotations

import sympy


def parse_linear_row(row: dict) -> dict:
    structure = row.get("structure")
    source = structure.get("source") if isinstance(structure, dict) else None
    if not isinstance(source, dict):
        raise ValueError("missing structure.source; generate tasks with --format structured")
    equations = source.get("base_equations")
    variables = source.get("variables")
    if not isinstance(equations, list) or not isinstance(variables, list):
        raise ValueError("missing structured base_equations or variables")
    names = [item.get("text") if isinstance(item, dict) else None for item in variables]
    if len(names) not in (1, 2) or any(not isinstance(name, str) for name in names):
        raise ValueError("expected one or two structured variables")
    symbols = [sympy.Symbol(name) for name in names]
    locals_ = dict(zip(names, symbols))
    residuals = []
    for equation in equations:
        if not isinstance(equation, dict) or equation.get("operator") != "Eq":
            raise ValueError("expected structured Eq")
        children = equation.get("children")
        if not isinstance(children, dict):
            raise ValueError("invalid structured equation")
        try:
            left = sympy.sympify(children["left"]["text"], locals=locals_)
            right = sympy.sympify(children["right"]["text"], locals=locals_)
        except (KeyError, TypeError, ValueError, sympy.SympifyError) as exc:
            raise ValueError("invalid structured equation expression") from exc
        residuals.append(left - right)
    if len(residuals) != len(symbols):
        raise ValueError("equation count differs from variable count")
    try:
        matrix, rhs = sympy.linear_eq_to_matrix(residuals, symbols)
    except ValueError as exc:
        raise ValueError("structured equations are not linear") from exc
    if matrix.det() == 0:
        raise ValueError("structured equations have no unique solution")
    solution = matrix.inv() * rhs
    return {**row, "parsed_linear": {
        "variables": names,
        "coefficients": [[str(matrix[i, j]) for j in range(len(symbols))]
                         for i in range(len(symbols))],
        "right_hand_side": [str(value) for value in rhs],
        "solution": [str(value) for value in solution],
    }}
