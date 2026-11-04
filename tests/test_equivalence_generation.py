import json
import tempfile
import unittest
from argparse import Namespace
from fractions import Fraction
from pathlib import Path

from scripts.equivalence_generation import extract_final_x, main as equivalence_main, verify_file, verify_pair
from slm_answer_stability.dataset_cli import compute_k_n, main, read_jsonl


class MathAnswerVerifierTests(unittest.TestCase):
    def test_exact_numeric_equivalence(self):
        self.assertEqual(verify_pair("x = \\boxed{\\frac{3}{4}}", "The answer is x = 0.75.")["answer_equivalence"], "equivalent")
        self.assertEqual(verify_pair("x = -1/2", "Answer: -0.5")["answer_equivalence"], "equivalent")
        self.assertEqual(verify_pair("x = 2", "x = 3")["answer_equivalence"], "non_equivalent")

    def test_ambiguous_or_unfinished_answers_remain_unclear(self):
        for answer in ("x = 1\nx = 2", "x = 1 or x = 2", "x = ",
                       "First calculate 3 + 4", "x = 1/0", "No solution"):
            with self.subTest(answer=answer):
                self.assertIsNone(extract_final_x(answer))
                self.assertEqual(verify_pair("x = 1", answer)["answer_equivalence"], "unclear")
        self.assertEqual(extract_final_x("To solve the equation, rearrange terms.\nFinal answer: x = 4."), 4)
        self.assertEqual(extract_final_x("i = -20"), -20)  # Mathematics Dataset uses any one-letter variable

    def test_intermediate_variable_terms_are_not_final_assignments(self):
        self.assertEqual(extract_final_x("6x = -19\nx = -19/6"), Fraction(-19, 6))
        self.assertEqual(extract_final_x("-x = 23\nx = -23"), -23)

    def test_jsonl_output_flows_to_algorithmic_stability_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "answers.jsonl"
            target = Path(directory) / "verified.jsonl"
            rows = [
                {"question_id": "q1", "variant_id": "v1", "question_original": "Solve x.",
                 "question_transformed": "Find x.", "answer_original": "x = 1/2",
                 "answer_transformed": "x = 0.5", "slm_model": "slm"},
                {"question_id": "q1", "variant_id": "v2", "question_original": "Solve x.",
                 "question_transformed": "Find x again.", "answer_original": "x = 1/2",
                 "answer_transformed": "x = 2", "slm_model": "slm"},
            ]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            counts = verify_file(source, target)
            self.assertEqual(counts, {"equivalent": 1, "non_equivalent": 1})
            legacy = Path(directory) / "legacy.jsonl"
            self.assertEqual(equivalence_main(["--original-only", "--input", str(source),
                                               "--output", str(legacy)]), 0)
            self.assertEqual(len(read_jsonl(legacy)), 2)
            results = compute_k_n(Namespace(input=target))
            self.assertEqual((results[0]["k"], results[0]["N"], results[0]["target"]), (1, 2, 0.5))
            grouped = Path(directory) / "grouped.jsonl"
            self.assertEqual(main("compute_k_n", ["--original-only", "--input", str(target),
                                                    "--output", str(grouped)]), 0)
            self.assertEqual(read_jsonl(grouped)[0]["target"], 0.5)
            with self.assertRaisesRegex(ValueError, "already exists"):
                verify_file(source, target)


if __name__ == "__main__":
    unittest.main()
