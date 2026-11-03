import json
import tempfile
import unittest
from pathlib import Path

import sympy

from scripts.generate_math_tasks import generate
from slm_answer_stability.linear_equation_variants import generate_file


class LinearEquationVariantsTests(unittest.TestCase):
    def test_direct_structured_input_and_exact_solutions(self):
        with tempfile.TemporaryDirectory() as directory:
            for module in ("linear_1d", "linear_1d_composed", "linear_2d", "linear_2d_composed"):
                source = Path(directory) / f"{module}.jsonl"
                output = Path(directory) / f"{module}_variants.jsonl"
                generate(f"algebra.{module}", 3, 9, source, "structured")
                self.assertEqual(generate_file(source, output, 8, 42), 24)
                rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len({row["variant_id"] for row in rows}), 24)
                for row in rows:
                    self.assertNotEqual(row["question_original"], row["question_transformed"])
                    question = row["question_transformed"]
                    variable = sympy.Symbol(question.rsplit(" for ", 1)[1][:-1])
                    equations = question.removeprefix("Solve ").rsplit(" for ", 1)[0].split(", ")
                    residuals = []
                    for equation in equations:
                        left, right = equation.split(" = ")
                        residuals.append(sympy.sympify(left) - sympy.sympify(right))
                    solutions = sympy.solve(residuals, dict=True)
                    self.assertEqual(len(solutions), 1)
                    self.assertEqual(solutions[0][variable], sympy.Rational(row["reference_answer"]))

    def test_same_seed_repeats_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jsonl"
            first = Path(directory) / "first.jsonl"
            second = Path(directory) / "second.jsonl"
            generate("algebra.linear_1d", 2, 7, source, "structured")
            generate_file(source, first, 10, 3)
            generate_file(source, second, 10, 3)
            self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
