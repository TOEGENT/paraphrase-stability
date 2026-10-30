import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_math_answer_correctness import check_answer, check_file


class MathCorrectnessTests(unittest.TestCase):
    def test_exact_substitution_with_signed_terms_and_fractions(self):
        question = "Solve for x: $-x - 8 - 11 = 9 - 12 + 7$."
        self.assertEqual(check_answer(question, "x = -23")["status"], "correct")
        self.assertEqual(check_answer(question, "x = -1")["status"], "incorrect")
        self.assertEqual(check_answer(question, "-19 = 4")["status"], "unparseable")
        self.assertEqual(check_answer("Solve for x: $2(x - 3) = 1$.", "x = 7/2")["status"], "correct")
        self.assertEqual(check_answer("Solve for x: $x/2 + 0.25 = 1$.", "x = 3/2")["status"], "correct")

    def test_rejects_nonunique_or_unsupported_equations(self):
        self.assertEqual(check_answer("Solve for x: $x + 1 = x + 2$.", "x = 1")["status"], "unsupported")
        self.assertEqual(check_answer("Solve for x: $x*x = 4$.", "x = 2")["status"], "unsupported")
        self.assertEqual(check_answer("Find x: $x + 1 = 2$.", "x = 1")["status"], "unsupported")

    def test_file_checks_original_once_and_preserves_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "answers.jsonl"
            output = Path(directory) / "checked.jsonl"
            base = {"question_id": "q1", "question_original": "Solve for x: $x + 1 = 2$.",
                    "answer_original": "x = 1"}
            rows = [
                {**base, "variant_id": "v1", "question_transformed": "Solve for x: $2 = 1 + x$.",
                 "answer_transformed": "x = 1"},
                {**base, "variant_id": "v2", "question_transformed": "Solve for x: $x + 1 = 2$.",
                 "answer_transformed": "x = 3"},
            ]
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            counts = check_file(source, output)
            self.assertEqual(counts["original_unique"], {"correct": 1})
            self.assertEqual(counts["transformed"], {"correct": 1, "incorrect": 1})
            checked = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(checked[0]["answer_transformed_correctness"], "correct")
            self.assertEqual(checked[1]["answer_transformed_correctness"], "incorrect")
            self.assertNotIn("reference_answer", checked[0])
            with self.assertRaisesRegex(ValueError, "new file"):
                check_file(source, output)


if __name__ == "__main__":
    unittest.main()
