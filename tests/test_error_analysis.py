import json
import tempfile
import unittest
from pathlib import Path

from slm_answer_stability.dataset_cli import WorkflowError, read_jsonl, write_jsonl
from slm_answer_stability.evaluation.error_analysis import analyze, write_report
from slm_answer_stability.models.embeddings import sha256


def source_row(question_id, split, question, target, source):
    status = "equivalent" if target == 1.0 else "non_equivalent"
    return {
        "question_id": question_id, "split": split, "question": question,
        "answer": "original answer", "review_status": "approved",
        "source": source, "topic": "math", "language": "ru",
        "variants": [{"variant_id": question_id + "-v1", "question": "rewritten", "answer": "variant answer", "answer_equivalence": status, "transformation_type": "lexical"}],
        "k": int(target), "N": 1, "target": target, "unresolved": 0,
    }


class ErrorAnalysisTests(unittest.TestCase):
    def test_systematic_bias_duplicate_and_truncation_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = root / "splits"
            features = root / "features"
            splits.mkdir()
            features.mkdir()
            train = [source_row("train-1", "train", "Repeated question", 0.0, "pilot")]
            val = [
                source_row("val-1", "val", "Repeated question", 1.0, "source-a"),
                source_row("val-2", "val", "Unique second", 1.0, "source-a"),
                source_row("val-3", "val", "Unique third", 0.0, "source-b"),
            ]
            write_jsonl(splits / "train.jsonl", train)
            write_jsonl(splits / "val.jsonl", val)
            predictions = [{
                "question_id": row["question_id"], "split": "val",
                "question": row["question"], "answer": row["answer"],
                "k": row["k"], "N": row["N"], "target": row["target"],
                "prediction": 0.1, "constant_prediction": 0.0,
                "absolute_error": abs(0.1 - row["target"]),
            } for row in val]
            prediction_file = root / "predictions.jsonl"
            write_jsonl(prediction_file, predictions)
            lengths = [{
                "question_id": row["question_id"], "question_tokens": 513 if index == 0 else 10,
                "answer_tokens": 5, "question_truncated": index == 0,
                "answer_truncated": False,
            } for index, row in enumerate(val)]
            lengths_file = features / "val_lengths.jsonl"
            write_jsonl(lengths_file, lengths)
            (features / "manifest.json").write_text(json.dumps({
                "splits": {"val": {
                    "source_sha256": sha256(splits / "val.jsonl"),
                    "lengths_sha256": sha256(lengths_file),
                }}
            }), encoding="utf-8")
            summary, groups, calibration, top = analyze(prediction_file, splits, features, 0.01, 0.5, 2, 5, 2, False)
            self.assertEqual(summary["exact_train_question_duplicates"], ["val-1"])
            self.assertTrue(summary["large_bias_groups"])
            self.assertTrue(summary["truncation_available"])
            self.assertEqual(top[0]["question_id"], "val-1")
            self.assertTrue(top[0]["truncated"])
            self.assertEqual(sum(row["questions"] for row in calibration), 3)
            output = root / "analysis"
            write_report(output, summary, groups, calibration, top)
            self.assertEqual(len(read_jsonl(output / "top_errors.jsonl")), 2)
            self.assertTrue((output / "group_metrics.csv").exists())
            self.assertTrue((output / "report.md").exists())

    def test_rejects_mismatched_target_and_unapproved_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = root / "splits"
            splits.mkdir()
            train = [source_row("train-1", "train", "train question", 0.0, "pilot")]
            val = [source_row("val-1", "val", "val question", 1.0, "pilot")]
            write_jsonl(splits / "train.jsonl", train)
            write_jsonl(splits / "val.jsonl", val)
            prediction_file = root / "predictions.jsonl"
            row = {"question_id": "val-1", "split": "val", "question": "val question", "answer": "original answer", "k": 1, "N": 1, "target": 0.0, "prediction": 0.5, "constant_prediction": 0.0}
            write_jsonl(prediction_file, [row])
            with self.assertRaisesRegex(WorkflowError, "target or constant"):
                analyze(prediction_file, splits, None, 0.01, 0.2, 1, 5, 1, False)
            test_file = root / "test_predictions.jsonl"
            write_jsonl(test_file, [{**row, "split": "test"}])
            with self.assertRaisesRegex(WorkflowError, "--final"):
                analyze(test_file, splits, None, 0.01, 0.2, 1, 5, 1, False)


if __name__ == "__main__":
    unittest.main()
