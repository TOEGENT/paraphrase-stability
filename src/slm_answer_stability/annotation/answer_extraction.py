"""Exact final-value extraction and pairwise comparison for numeric (linear equation) answers.

Only one final integer, decimal, a/b, or LaTeX fraction is accepted per answer; anything
ambiguous or unfinished is left unresolved for human review.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction


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
