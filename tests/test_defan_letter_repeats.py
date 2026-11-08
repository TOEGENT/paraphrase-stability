"""Regression checks for persisted repeated-letter variants."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_defan_letter_repeats.py"
spec = importlib.util.spec_from_file_location("generate_defan_letter_repeats", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SaveWithResumeTests(unittest.TestCase):
    def test_resume_after_complete_and_partial_rows(self):
        rows = module.generate_rows(
            [{"question_id": "q1", "question_original": "What is your name?"}], 5
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "variants.jsonl"
            output.write_bytes(
                (json.dumps(rows[0]) + "\n" + json.dumps(rows[1])[:12]).encode("utf-8")
            )
            self.assertEqual(module.save_with_resume(output, iter(rows)), (1, 5, 1))
            self.assertEqual(module.save_with_resume(output, iter(rows)), (5, 5, 1))
            saved = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(saved, rows)

    def test_changed_existing_row_is_rejected(self):
        rows = module.generate_rows(
            [{"question_id": "q1", "question_original": "Waa?"}], 10
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "variants.jsonl"
            output.write_text(json.dumps({**rows[0], "question_transformed": "wrong"}) + "\n")
            with self.assertRaisesRegex(ValueError, "does not match"):
                module.save_with_resume(output, rows)


if __name__ == "__main__":
    unittest.main()
