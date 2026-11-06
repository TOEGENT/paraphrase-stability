"""Prepared checks for the standalone generator and the existing evaluator."""

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.check_math_answer_correctness import check_answer
from scripts.generate_linear_paraphrases import (
    FAMILIES, generate_file, generate_variants, main, read_equation, structure,
)


class LinearParaphrasesTests(unittest.TestCase):
    def test_hundred_unique_balanced_variants_checked_independently(self):
        examples = (
            ("2*x + 3 = 7", "2"),
            ("-x - 8 - 11 = 9 - 12 + 7", "-23"),
            ("3*x - 2 = x + 5", "7/2"),
            ("x/3 + 0.25 = 1/2", "3/4"),
            ("x + 0.10000000000000001 = 1", "89999999999999999/100000000000000000"),
            ("2(x - 3) = 1", "7/2"),
            ("x = 0", "0"),
            ("0 = x", "0"),
            ("Solve -2*r + 4 = 7 for r.", "-3/2"),
        )
        for index, (equation, root) in enumerate(examples):
            with self.subTest(equation=equation):
                rows = generate_variants({"question_id": f"q{index}", "equation": equation})
                self.assertEqual(len(rows), 100)
                texts = [row["question_transformed"] for row in rows]
                self.assertEqual(len(set(texts)), 100)
                self.assertEqual(len({row["variant_id"] for row in rows}), 100)
                self.assertEqual(Counter(row["transformation_family"] for row in rows),
                                 Counter({family: 10 for family in FAMILIES}))
                for row in rows:
                    self.assertEqual(row["reference_answer"], root)
                    self.assertNotEqual(row["question_original"], row["question_transformed"])
                    self.assertEqual(row["paraphrase_review"], "pending")
                    self.assertEqual(row["transformation_target"], row["transformation_family"])
                    for field in ("question_original", "question_transformed"):
                        # The project's verifier has its own affine parser.
                        self.assertEqual(check_answer(row[field], f"x = {root}")["status"],
                                         "correct", row[field])
                    eq = row["structure"]["source"]["base_equations"][0]
                    self.assertEqual(eq["operator"], "Eq")
                    self.assertEqual(set(eq["children"]), {"left", "right"})

    def test_structure_contains_signed_terms_and_exact_coefficients(self):
        left, right, _ = read_equation("2*x - 3 + 4 = 7")
        tree = structure(left, right)["source"]["base_equations"][0]["children"]["left"]
        self.assertEqual(tree["terms"], [
            {"sign": 1, "text": "2*x"}, {"sign": -1, "text": "3"},
            {"sign": 1, "text": "4"},
        ])
        self.assertEqual((tree["coefficient"], tree["constant"]), ("2", "1"))
        row = {"question_id": "q", "question": "Solve 2*x - 3 + 4 = 7 for x.",
               "answer": "3", "structure": structure(left, right)}
        self.assertEqual(len(generate_variants(row)), 100)
        row["structure"]["source"]["base_equations"][0]["children"]["right"]["text"] = "8"
        with self.assertRaisesRegex(ValueError, "disagrees"):
            generate_variants(row)

    def test_reproducibility_and_smaller_balanced_counts(self):
        row = {"question_id": "q", "question": "Solve for x: $2*x + 3 = 7$."}
        self.assertEqual(generate_variants(row), generate_variants(row))
        self.assertNotEqual(generate_variants(row), generate_variants(row, seed=43))
        rows = generate_variants(row, count=17)
        self.assertEqual(len(rows), 17)
        counts = Counter(row["transformation_family"] for row in rows)
        self.assertEqual(set(counts.values()), {1, 2})

    def test_invalid_equations_and_references_are_rejected(self):
        for equation in ("x = x", "x + 1 = x + 2", "x*x = 4", "x**2 = 4", "1/x = 2",
                         "x/(x + 1) = 2", "x + y = 3", "f(x) = 2", "x = 1/0",
                         "x = 1 = 2", "Solve x + 1 = 2 for y."):
            with self.subTest(equation=equation), self.assertRaises(ValueError):
                generate_variants({"question_id": "q", "equation": equation})
        with self.assertRaisesRegex(ValueError, "disagrees"):
            generate_variants({"question_id": "q", "equation": "x = 2", "answer": "3"})

    def test_jsonl_file_protection_and_invalid_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "source.jsonl", Path(directory) / "out.jsonl"
            valid = {"question_id": "q", "question": "2*x + 3 = 7", "answer": "2"}
            source.write_text(json.dumps(valid) + "\n", encoding="utf-8")
            self.assertEqual(generate_file(source, output), 100)
            saved = output.read_bytes()
            with self.assertRaisesRegex(ValueError, "new file"):
                generate_file(source, output)
            self.assertEqual(output.read_bytes(), saved)
            output.unlink()
            bad = {"question_id": "bad", "equation": "x = x"}
            source.write_text(json.dumps(valid) + "\n" + json.dumps(bad) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 2"):
                generate_file(source, output)
            self.assertFalse(output.exists())
            source.write_text((json.dumps(valid) + "\n") * 2, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                generate_file(source, output)
            self.assertFalse(output.exists())

    def test_cli_single_equation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "variants.jsonl"
            args = ["--equation", "2*x + 3 = 7", "--output", str(output)]
            self.assertEqual(main(args), 0)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 100)
            self.assertEqual(main(args), 1)


if __name__ == "__main__":
    unittest.main()
