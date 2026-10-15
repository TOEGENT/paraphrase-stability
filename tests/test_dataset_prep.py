import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from slm_answer_stability.dataset_cli import WorkflowError
from slm_answer_stability.dataset_prep import check_dataset, import_questions, split_dataset


def final_row(number):
    return {
        "question_id": f"q{number}", "question": f"Question {number}?", "answer": str(number),
        "review_status": "approved", "variants": [{
            "variant_id": f"q{number}-v1", "question": f"Reworded question {number}?",
            "answer": str(number), "answer_equivalence": "equivalent",
        }], "k": 1, "N": 1, "target": 1.0, "unresolved": 0,
    }


class DatasetPrepTests(unittest.TestCase):
    def test_import_csv_generates_stable_ids_and_pending_review(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "questions.csv"
            source.write_text("question,topic,language\nWhat is two plus two?,math,en\nWhat is three plus three?,math,en\n", encoding="utf-8-sig")
            args = Namespace(input=source, delimiter=None, question_column="question", id_column=None, source="pilot")
            first = import_questions(args)
            self.assertEqual(first, import_questions(args))
            self.assertEqual(first[0]["question_review"], "pending")
            self.assertEqual(first[0]["source"], "pilot")
            self.assertEqual(first[0]["topic"], "math")
            self.assertEqual(first[0]["language"], "en")
            source.write_text("question\nSame?\n same? \n", encoding="utf-8")
            with self.assertRaisesRegex(WorkflowError, "duplicate question"):
                import_questions(args)

    def test_split_is_deterministic_and_keeps_groups_intact(self):
        rows = [final_row(number) for number in range(11)]
        first = split_dataset(rows, (0.6, 0.2, 0.2), 23)
        self.assertEqual(first, split_dataset(list(reversed(rows)), (0.6, 0.2, 0.2), 23))
        self.assertEqual(sum(len(part) for part in first.values()), len(rows))
        self.assertTrue(all(first[name] for name in ("train", "val", "test")))
        self.assertEqual({row["question_id"] for part in first.values() for row in part}, {row["question_id"] for row in rows})
        self.assertTrue(all(row["split"] == name for name, part in first.items() for row in part))

    def test_check_catches_threshold_and_incorrect_target(self):
        rows = [final_row(1), final_row(2)]
        report = check_dataset(rows, 3, 2, 5)
        self.assertFalse(report["passed"])
        self.assertTrue(any("questions" in problem for problem in report["problems"]))
        rows[0]["target"] = 0.0
        self.assertTrue(any("target does not equal k/N" in problem for problem in check_dataset(rows, 2, 1, 2)["problems"]))
        with self.assertRaisesRegex(WorkflowError, "target"):
            split_dataset(rows, (0.6, 0.2, 0.2), 23)


if __name__ == "__main__":
    unittest.main()
