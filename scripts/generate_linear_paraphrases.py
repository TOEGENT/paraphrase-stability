"""Generate 100 exact, structurally varied rewrites of each linear equation.

Standalone Python 3.10+: no model calls, dataset imports or third-party packages.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


FAMILIES = (
    "permute_terms", "transfer_terms", "split_constants", "split_coefficients",
    "group_terms", "multiply_sides", "divide_sides", "add_constant",
    "add_linear_expression", "distribute",
)
NONZERO = tuple(range(-12, 0)) + tuple(range(1, 13))
FACTORS = tuple(range(-12, -1)) + tuple(range(2, 13))


@dataclass
class Side:
    text: str
    node: ast.AST
    coefficient: Fraction
    constant: Fraction

    def terms(self) -> list[tuple[int, str]]:
        """Separate signed top-level summands, retaining products and brackets."""
        def visit(node: ast.AST, sign: int) -> list[tuple[int, str]]:
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
                right_sign = -sign if isinstance(node.op, ast.Sub) else sign
                return visit(node.left, sign) + visit(node.right, right_sign)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                return visit(node.operand, -sign if isinstance(node.op, ast.USub) else sign)
            return [(sign, ast.get_source_segment(self.text, node))]
        return visit(self.node, 1)

    def tree(self) -> dict:
        def visit(node: ast.AST) -> dict:
            text = ast.get_source_segment(self.text, node)
            if isinstance(node, ast.Constant):
                return {"operator": "Number", "text": text, "value": str(Fraction(text))}
            if isinstance(node, ast.Name):
                return {"operator": "Symbol", "text": node.id}
            if isinstance(node, ast.UnaryOp):
                return {"operator": type(node.op).__name__, "text": text,
                        "children": {"operand": visit(node.operand)}}
            return {"operator": type(node.op).__name__, "text": text,
                    "children": {"left": visit(node.left), "right": visit(node.right)}}
        return {**visit(self.node),
                "terms": [{"sign": sign, "text": text} for sign, text in self.terms()],
                "coefficient": str(self.coefficient), "constant": str(self.constant)}


def parse_side(text: str) -> Side:
    """Allowlisted AST parsing with exact rational arithmetic; never use eval."""
    text = text.strip()
    if not text or len(text) > 4096:
        raise ValueError("empty expression or expression longer than 4096 characters")

    def visit(node: ast.AST) -> tuple[Fraction, Fraction]:
        if isinstance(node, ast.Name) and node.id == "x":
            return Fraction(1), Fraction(0)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            literal = ast.get_source_segment(text, node)
            if not re.fullmatch(r"(?:\d+(?:\.\d*)?|\.\d+)", literal):
                raise ValueError("use ordinary decimal numbers or fractions")
            return Fraction(0), Fraction(literal)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            a, b = visit(node.operand)
            sign = -1 if isinstance(node.op, ast.USub) else 1
            return sign * a, sign * b
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left)
            c, d = visit(node.right)
            if isinstance(node.op, ast.Add):
                return a + c, b + d
            if isinstance(node.op, ast.Sub):
                return a - c, b - d
            if isinstance(node.op, ast.Mult) and not (a and c):
                return a * d + b * c, b * d
            if isinstance(node.op, ast.Div) and c == 0 and d != 0:
                return a / d, b / d
        raise ValueError("only linear arithmetic in x with +, -, *, / and brackets is supported")

    try:
        node = ast.parse(text, mode="eval").body
        a, b = visit(node)
    except (SyntaxError, TypeError, ZeroDivisionError, OverflowError, RecursionError) as exc:
        raise ValueError("invalid linear expression") from exc
    return Side(text, node, a, b)


def solution(left: Side, right: Side) -> Fraction:
    coefficient = left.coefficient - right.coefficient
    if coefficient == 0:
        raise ValueError("equation must have exactly one solution")
    return (right.constant - left.constant) / coefficient


def structure(left: Side, right: Side) -> dict:
    equation = {"operator": "Eq", "text": f"{left.text} = {right.text}",
                "children": {"left": left.tree(), "right": right.tree()}}
    return {"source": {"variables": [{"operator": "Symbol", "text": "x"}],
                       "base_equations": [equation]}}


def read_equation(question: str) -> tuple[Side, Side, str]:
    """Accept bare equations and both project's question wrappers; normalize to x."""
    text = question.strip().replace("−", "-").replace("×", "*").replace("÷", "/")
    before = re.fullmatch(r"Solve for ([a-z]):\s*\$(.*?)\$\.?", text, re.IGNORECASE)
    after = re.fullmatch(r"Solve\s+(.*?)\s+for ([a-z])\.", text, re.IGNORECASE)
    requested = before[1] if before else after[2] if after else None
    if before or after:
        text = before[2] if before else after[1]
    if text.count("=") != 1:
        raise ValueError("expected one equation, e.g. 2*x + 3 = 7")
    names = set(re.findall(r"[A-Za-z_]", text))
    if len(names) != 1 or not re.fullmatch(r"[a-z]", next(iter(names), "")):
        raise ValueError("expected one lowercase, single-letter variable")
    variable = next(iter(names))
    if requested is not None and requested != variable:
        raise ValueError("requested variable differs from the equation variable")
    # Insert implicit multiplication BEFORE renaming: 2r, 2(...), r(...), (...)r.
    text = re.sub(rf"(?<=\d)\s*(?={variable}\b|\()", "*", text)
    text = re.sub(rf"\b{variable}\s*(?=\()", variable + "*", text)
    text = re.sub(rf"\)\s*(?=\(|{variable}\b)", ")*", text)
    text = re.sub(rf"\b{variable}\b", "x", text)
    # The existing correctness checker reads float AST constants. Render source
    # decimals as exact fractions so its float conversion cannot round them.
    text = re.sub(r"(?<![\w.])(?:\d+\.\d*|\.\d+)(?![\w.])",
                  lambda match: f"({Fraction(match[0])})", text)
    left, right = (parse_side(side) for side in text.split("="))
    solution(left, right)
    return left, right, variable


def render(terms: list[tuple[int, str]]) -> str:
    return " + ".join(f"{'-' if sign < 0 else ''}({term})" for sign, term in terms) or "0"


def transform(left: Side, right: Side, family: str, rng: random.Random,
              attempt: int) -> tuple[str, str, list[str]]:
    l, r = f"({left.text})", f"({right.text})"
    c, p, factor = rng.choice(NONZERO), rng.choice(NONZERO), rng.choice(FACTORS)
    lt, rt = left.terms(), right.terms()
    steps = []
    if family == "permute_terms":
        # Short equations have too few permutations. After trying pure orders,
        # split one summand T into (T-c)+c, then permute. Record this extra step.
        if attempt >= 24:
            sign, term = lt.pop(rng.randrange(len(lt)))
            lt.extend([(sign, f"({term}) - ({c})"), (sign, str(c))])
            steps.append(f"split one summand into (summand - ({c})) + ({c})")
        rng.shuffle(lt)
        rng.shuffle(rt)
        result = render(lt), render(rt)
        steps.append("permute signed summands within each side")
    elif family == "transfer_terms":
        # Add the same constant for diversity even on x=0, then transfer an
        # original signed summand across equality, changing its sign.
        sign, term = lt.pop(rng.randrange(len(lt)))
        lt.append((1, str(c)))
        rt.extend([(1, str(c)), (-sign, term)])
        rng.shuffle(lt)
        rng.shuffle(rt)
        result = render(lt), render(rt)
        steps += [f"add ({c}) to both sides", "transfer a signed summand to the right"]
    elif family == "split_constants":
        choices = [i for i, (_, term) in enumerate(rt) if parse_side(term).coefficient == 0]
        if choices:
            index = rng.choice(choices)
            sign, term = rt.pop(index)
            value = parse_side(term).constant
            rt.extend([(sign, str(c)), (sign, str(value - c))])
            steps.append(f"split a constant into ({c}) and its remainder")
        else:
            rt.extend([(1, str(c)), (-1, str(c))])
            steps.append(f"insert cancelling constants ({c}) and -({c})")
        rng.shuffle(rt)
        result = l, render(rt)
    elif family == "split_coefficients":
        # At least one side must contain x in an equation with a unique root.
        terms = lt if left.coefficient else rt
        indices = [i for i, (_, term) in enumerate(terms) if parse_side(term).coefficient]
        index = rng.choice(indices)
        sign, term = terms.pop(index)
        parsed = parse_side(term)
        # Keep the original term's constant part and the original sign.
        terms.extend([(sign, f"({p})*x"),
                      (sign, f"({parsed.coefficient - p})*x + ({parsed.constant})")])
        rng.shuffle(terms)
        result = render(lt), render(rt)
        steps.append(f"split a variable coefficient using ({p})*x and its remainder")
    elif family == "group_terms":
        # Enrich short expressions with a zero pair before changing grouping.
        lt.extend([(1, str(c)), (-1, str(c))])
        rng.shuffle(lt)
        pivot = rng.randrange(1, len(lt))
        result = f"({render(lt[:pivot])}) + ({render(lt[pivot:])})", r
        steps += [f"insert cancelling constants ({c}) and -({c})", "regroup signed summands"]
    elif family == "multiply_sides":
        result = f"({factor})*{l}", f"({factor})*{r}"
        steps.append(f"multiply both sides by nonzero ({factor})")
    elif family == "divide_sides":
        result = f"{l}/({factor})", f"{r}/({factor})"
        steps.append(f"divide both sides by nonzero ({factor})")
    elif family == "add_constant":
        result = f"{l} + ({c})", f"{r} + ({c})"
        steps.append(f"add ({c}) to both sides")
    elif family == "add_linear_expression":
        shift = f"(({p})*x + ({c}))"
        result = f"{l} + {shift}", f"{r} + {shift}"
        steps.append(f"add {shift} to both sides")
    elif family == "distribute":
        result = f"({factor})*({l} + ({c}))", f"({factor})*{r} + ({factor * c})"
        steps += [f"add ({c}) to both sides", f"multiply both sides by ({factor})",
                  "distribute multiplication over the added constant on the right"]
    else:
        raise ValueError(f"unknown transformation family: {family}")
    if rng.choice((False, True)):
        result = result[::-1]
        steps.append("swap equality sides")
    return result[0], result[1], steps


def generate_variants(row: dict, count: int = 100, seed: int = 42) -> list[dict]:
    if not isinstance(row, dict):
        raise ValueError("expected a JSON object")
    question_id = row.get("question_id")
    question = row.get("question", row.get("equation"))
    if not isinstance(question_id, str) or not question_id.strip():
        raise ValueError("question_id must be a nonempty string")
    if not isinstance(question, str):
        raise ValueError("question or equation must be a string")
    if not 1 <= count <= 100:
        raise ValueError("variants per question must be between 1 and 100")
    left, right, variable = read_equation(question)
    root = solution(left, right)
    if "answer" in row:
        try:
            expected = Fraction(str(row["answer"]))
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError("answer must be an exact integer, decimal or fraction") from exc
        if expected != root:
            raise ValueError("reference answer disagrees with the equation")
    # Validate structured input too, when present. The visible equation remains
    # authoritative, so unrelated/malformed structure cannot silently replace it.
    if "structure" in row:
        try:
            equations = row["structure"]["source"]["base_equations"]
            if len(equations) != 1 or equations[0]["operator"] != "Eq":
                raise ValueError("expected one structured Eq")
            children = equations[0]["children"]
            a, b, _ = read_equation(f"{children['left']['text']} = {children['right']['text']}")
            if solution(a, b) != root:
                raise ValueError("structured equation disagrees with visible equation")
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError("invalid structure.source.base_equations") from exc
    original = f"Solve for x: ${left.text} = {right.text}$."
    rng = random.Random(f"{seed}:{question_id}")
    seen = {re.sub(r"\s+", "", f"{left.text}={right.text}")}
    output = []
    for index, family in enumerate(FAMILIES):
        quota = count // len(FAMILIES) + int(index < count % len(FAMILIES))
        accepted = 0
        for attempt in range(3000):
            if accepted == quota:
                break
            new_left, new_right, steps = transform(left, right, family, rng, attempt)
            key = re.sub(r"\s+", "", f"{new_left}={new_right}")
            if key in seen:
                continue
            # Verify the rendered strings with exact fractions, rather than
            # trusting the transformation rule or numerical substitution alone.
            parsed_left, parsed_right = parse_side(new_left), parse_side(new_right)
            if solution(parsed_left, parsed_right) != root:
                raise ValueError(f"internal equivalence check failed: {family}")
            seen.add(key)
            accepted += 1
            output.append({
                "question_id": question_id,
                "variant_id": f"{question_id}-v{len(output) + 1:03d}",
                "question_original": original,
                "question_transformed": f"Solve for x: ${new_left} = {new_right}$.",
                "reference_answer": str(root),
                "source_question": question,
                "source_variable": variable,
                "source": row.get("source", "provided_linear_equation"),
                "topic": row.get("topic", "algebra"),
                "module": row.get("module", "linear_1d"),
                "language": "en",
                "transformation_type": "linear_equivalence",
                "transformation_mode": "structure",
                "transformation_target": family,
                "transformation_family": family,
                "transformation_steps": steps,
                "structure": structure(parsed_left, parsed_right),
                "equivalence_verifier": "exact_affine_fraction_v1",
                "paraphrase_review": "pending",
            })
        if accepted != quota:
            raise ValueError(f"{question_id}: could only generate {accepted}/{quota} for {family}")
    return output


def generate_file(input_path: Path, output_path: Path, count: int = 100, seed: int = 42) -> int:
    if output_path.exists() or input_path.resolve() == output_path.resolve():
        raise ValueError("output must be a new file different from input")
    output, ids = [], set()
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            variants = generate_variants(row, count, seed)
            if row["question_id"] in ids:
                raise ValueError("duplicate question_id")
            ids.add(row["question_id"])
            output.extend(variants)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"line {line_number}: {exc}") from exc
    if not output:
        raise ValueError("input has no equations")
    write_output(output_path, output)
    return len(output)


def write_output(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise ValueError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="JSONL with question_id and question/equation")
    source.add_argument("--equation", help='One equation, e.g. "2*x + 3 = 7"')
    parser.add_argument("--question-id", default="linear-000001")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants-per-question", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        if args.input is not None:
            total = generate_file(args.input, args.output, args.variants_per_question, args.seed)
        else:
            rows = generate_variants({"question_id": args.question_id, "equation": args.equation},
                                     args.variants_per_question, args.seed)
            write_output(args.output, rows)
            total = len(rows)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output), "variants": total}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
