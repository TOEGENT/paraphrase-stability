import json
import tempfile
import unittest
from pathlib import Path

from scripts.generate_math_tasks import generate


class GenerateMathTasksTests(unittest.TestCase):
    def test_structured_mode_preserves_plain_questions_and_equation_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            plain = Path(directory) / "plain.jsonl"
            structured = Path(directory) / "structured.jsonl"
            generate("algebra.linear_1d", 5, 42, plain)
            generate("algebra.linear_1d", 5, 42, structured, "structured")
            plain_rows = [json.loads(line) for line in plain.read_text(encoding="utf-8").splitlines()]
            structured_rows = [json.loads(line) for line in structured.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(plain_rows, [
                {key: value for key, value in row.items() if key != "structure"}
                for row in structured_rows
            ])
            self.assertEqual(len({row["question"] for row in structured_rows}), 5)
            for row in structured_rows:
                structure = row["structure"]
                self.assertEqual(structure["source"]["equations"][0]["operator"], "Eq")
                self.assertIn("equations", structure["arguments"])


if __name__ == "__main__":
    unittest.main()
