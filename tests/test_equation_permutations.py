import tempfile
import unittest
from argparse import Namespace
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from slm_answer_stability.dataset_cli import generate_answers, read_jsonl, write_jsonl
from slm_answer_stability.equation_permutations import (
    _question,
    _variants,
    generate_equation_permutations,
    main,
)


class EquationPermutationTests(unittest.TestCase):
    def test_all_unique_combinations_keep_signed_terms_and_instruction(self):
        left = ("-2x", "3", "-5")
        right = ("7", "-4", "9")
        variants = _variants(left, right, 100, 42, "q1")
        self.assertEqual(len(variants), 61)  # (3! - 1)^2 plus 3! * 3! swaps
        self.assertEqual(len({(a, b) for a, b, _ in variants}), 61)
        self.assertTrue(any(swapped for _, _, swapped in variants))
        for variant_left, variant_right, swapped in variants:
            self.assertEqual(Counter(variant_left), Counter(right if swapped else left))
            self.assertEqual(Counter(variant_right), Counter(left if swapped else right))
            if not swapped:
                self.assertNotEqual(variant_left, left)
                self.assertNotEqual(variant_right, right)

    def test_reproducible_limit_and_no_reference_answer(self):
        first = generate_equation_permutations(3, 19, 100)
        self.assertEqual(first, generate_equation_permutations(3, 19, 100))
        self.assertLessEqual(len(first), 300)
        self.assertEqual(len({row["variant_id"] for row in first}), len(first))
        self.assertEqual(len({row["question_id"] for row in first}), 3)
        self.assertTrue(all(1 <= sum(item["question_id"] == row["question_id"] for item in first) <= 100
                            for row in first))
        self.assertTrue(all(row["paraphrase_review"] == "pending" for row in first))
        self.assertTrue(all(row["question_transformed"].startswith("Solve for x: $") for row in first))
        self.assertTrue(all(row["question_transformed"] == _question(
            tuple(row["variant_left_terms"]), tuple(row["variant_right_terms"])
        ) for row in first))
        self.assertTrue(all(
            Counter(row["variant_left_terms"]) == Counter(row["right_terms"] if row["sides_swapped"] else row["left_terms"])
            and Counter(row["variant_right_terms"]) == Counter(row["left_terms"] if row["sides_swapped"] else row["right_terms"])
            for row in first
        ))
        self.assertTrue(all(
            row["sides_swapped"] or (
                row["variant_left_terms"] != row["left_terms"]
                and row["variant_right_terms"] != row["right_terms"]
            ) for row in first
        ))
        self.assertTrue(all("answer" not in key and "ref_ans" not in key
                            for row in first for key in row))
        self.assertTrue(all(row["question_transformed"] != row["question_original"] for row in first))
        self.assertNotEqual(first, generate_equation_permutations(3, 20, 100))

    def test_reviewed_pilot_reaches_slm_without_correctness_check(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "variants.jsonl"
            self.assertEqual(main(["--count", "1", "--seed", "7", "--max-variants", "3",
                                   "--output", str(path)]), 0)
            rows = read_jsonl(path)
            self.assertEqual(len(rows), 3)
            for row in rows:
                row["paraphrase_review"] = "approved"
            approved = Path(directory) / "approved.jsonl"
            write_jsonl(approved, rows)

            def fake_slm(_config, command, **kwargs):
                self.assertEqual(command, "answer")
                self.assertTrue(kwargs["question"].startswith("Solve for x: $"))
                return {"answer": "model output"}

            with patch("slm_answer_stability.dataset_cli.call_model", side_effect=fake_slm) as model:
                answers = generate_answers(Namespace(input=approved), {"slm": {"model": "fixed-slm"}})
            self.assertEqual(len(answers), 3)
            self.assertEqual(model.call_count, 4)  # original once, three variants
            self.assertTrue(all(row["answer_original"] == "model output" and
                                row["answer_transformed"] == "model output" for row in answers))
            self.assertEqual(main(["--count", "1", "--output", str(path)]), 1)


if __name__ == "__main__":
    unittest.main()
