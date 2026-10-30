"""Build equivalent equation questions from structured, signed terms."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import random
import sys
from pathlib import Path

from .dataset_cli import WorkflowError, write_jsonl


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _at_most_100(value: str) -> int:
    number = _positive_int(value)
    if number > 100:
        raise argparse.ArgumentTypeError("must be at most 100")
    return number


def _term(coefficient: int, variable: bool = False) -> str:
    if variable:
        if coefficient == 1:
            return "x"
        if coefficient == -1:
            return "-x"
        return f"{coefficient}x"
    return str(coefficient)


def _expression(terms: tuple[str, ...]) -> str:
    first, *rest = terms
    result = first
    for term in rest:
        result += f" - {term[1:]}" if term.startswith("-") else f" + {term}"
    return result


def _question(left: tuple[str, ...], right: tuple[str, ...]) -> str:
    return f"Solve for x: ${_expression(left)} = {_expression(right)}$."


def _source_terms(rng: random.Random) -> tuple[tuple[str, ...], tuple[str, ...]]:
    left_size = rng.choice((3, 4))
    right_size = rng.choice((3, 4))
    coefficient = rng.choice(tuple(range(-9, 0)) + tuple(range(1, 10)))
    values = tuple(range(-15, 0)) + tuple(range(1, 16))
    left = (_term(coefficient, variable=True),) + tuple(
        _term(value) for value in rng.sample(values, left_size - 1)
    )
    right = tuple(_term(value) for value in rng.sample(values, right_size))
    return left, right


def _variants(
    left: tuple[str, ...], right: tuple[str, ...], max_variants: int, seed: int, question_id: str
) -> list[tuple[tuple[str, ...], tuple[str, ...], bool]]:
    original = _question(left, right)
    seen = {original}
    candidates = []
    left_orders = list(itertools.permutations(left))
    right_orders = list(itertools.permutations(right))
    for swapped in (False, True):
        for left_order, right_order in itertools.product(left_orders, right_orders):
            # A one-sided shuffle is barely distinguishable from the source.
            # Keep it only when the two whole sides are exchanged.
            if not swapped and (left_order == left or right_order == right):
                continue
            displayed_left, displayed_right = (
                (right_order, left_order) if swapped else (left_order, right_order)
            )
            text = _question(displayed_left, displayed_right)
            if text in seen:
                continue
            seen.add(text)
            candidates.append((displayed_left, displayed_right, swapped))
    if len(candidates) > max_variants:
        rng = random.Random(f"{seed}:{question_id}:variants")
        indices = sorted(rng.sample(range(len(candidates)), max_variants))
        candidates = [candidates[index] for index in indices]
    return candidates


def generate_equation_permutations(
    count: int, seed: int, max_variants: int = 100
) -> list[dict[str, object]]:
    if count < 1 or not 1 <= max_variants <= 100:
        raise ValueError("count must be positive and max_variants must be between 1 and 100")
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    seen_equations: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for index in range(1, count + 1):
        for _ in range(1000):
            left, right = _source_terms(rng)
            key = (tuple(sorted(left)), tuple(sorted(right)))
            if key not in seen_equations:
                seen_equations.add(key)
                break
        else:
            raise ValueError("could not generate another distinct equation")
        digest = hashlib.sha256(repr((seed, left, right)).encode("utf-8")).hexdigest()[:10]
        question_id = f"equation-{index:06d}-{digest}"
        original = _question(left, right)
        for variant_index, (displayed_left, displayed_right, swapped) in enumerate(
            _variants(left, right, max_variants, seed, question_id), start=1
        ):
            rows.append({
                "question_id": question_id,
                "variant_id": f"{question_id}-v{variant_index}",
                "question_original": original,
                "question_transformed": _question(displayed_left, displayed_right),
                "left_terms": list(left),
                "right_terms": list(right),
                "variant_left_terms": list(displayed_left),
                "variant_right_terms": list(displayed_right),
                "sides_swapped": swapped,
                "transformation_type": "side_swap_and_term_permutation" if swapped else "term_permutation",
                "transformation_mode": "structure",
                "source": "synthetic_equation_permutations",
                "topic": "linear_equations",
                "language": "en",
                "paraphrase_review": "pending",
            })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=_positive_int, required=True, help="Number of source equations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-variants", type=_at_most_100, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        rows = generate_equation_permutations(args.count, args.seed, args.max_variants)
        write_jsonl(args.output, rows)
    except (ValueError, WorkflowError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {len(rows)} variants for {args.count} equations to {args.output}")
    return 0
