import tempfile
import unittest
from pathlib import Path

from scripts.equivalence_generation import main as equivalence_main, verify_all_anchors
from slm_answer_stability.dataset_cli import (
    WorkflowError, compute_all_anchors, main, read_jsonl, write_jsonl,
)


class AllAnchorTests(unittest.TestCase):
    def test_reanchors_every_response_and_preserves_group_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = root / "splits"
            splits.mkdir()
            for split, question_id in (("train", "q1"), ("val", "q2"), ("test", "q3")):
                write_jsonl(splits / f"{split}.jsonl", [{"question_id": question_id}])
            rows = [
                {"question_id": "q1", "variant_id": f"v{i}",
                 "question_original": "x + 2 = 4", "answer_original": "x = 2",
                 "question_transformed": f"version {i}", "answer_transformed": answer,
                 "answer_equivalence": "equivalent" if answer == "x = 2" else "non_equivalent"}
                for i, answer in enumerate(("x = 2", "x = 3", "x = 2"), 1)
            ]
            rows += [
                {"question_id": question_id, "variant_id": variant_id,
                 "question_original": "x = 1", "answer_original": "x = 1",
                 "question_transformed": "1 = x", "answer_transformed": "x = 1",
                 "answer_equivalence": "equivalent"}
                for question_id, variant_id in (("q2", "v4"), ("q3", "v5"))
            ]
            source = root / "answers.jsonl"
            write_jsonl(source, rows)
            output = compute_all_anchors(source, splits)
            self.assertEqual(len(output), 8)
            self.assertEqual(len({row["question_id"] for row in output}), 8)
            q1 = [row for row in output if row["source_question_id"] == "q1"]
            self.assertEqual([(row["anchor_id"], row["k"], row["N"], row["target"])
                              for row in q1],
                             [("original", 2, 3, 2 / 3), ("v1", 2, 3, 2 / 3),
                              ("v2", 0, 3, 0), ("v3", 2, 3, 2 / 3)])
            self.assertEqual({row["split"] for row in q1}, {"train"})
            self.assertEqual({row["split"] for row in output if row["source_question_id"] == "q2"}, {"val"})
            self.assertEqual({row["split"] for row in output if row["source_question_id"] == "q3"}, {"test"})

            cli_output = root / "all_anchors.jsonl"
            self.assertEqual(main("compute_k_n", ["--input", str(source),
                                                    "--splits-dir", str(splits),
                                                    "--output", str(cli_output)]), 0)
            self.assertEqual(read_jsonl(cli_output), output)

            raw = root / "raw_answers.jsonl"
            for row in rows:
                del row["answer_equivalence"]
            write_jsonl(raw, rows)
            self.assertEqual(compute_all_anchors(raw, splits), output)
            pairwise = root / "pairwise.jsonl"
            self.assertEqual(equivalence_main(["--input", str(raw), "--output", str(pairwise)]), 0)
            self.assertEqual(len(read_jsonl(pairwise)), 8)
            self.assertEqual(compute_all_anchors(pairwise, splits), output)
            self.assertEqual(sum(len(row["comparisons"]) for row in read_jsonl(pairwise)), 16)
            automatic = root / "automatic.jsonl"
            self.assertEqual(main("compute_k_n", ["--input", str(pairwise),
                                                    "--output", str(automatic)]), 0)
            auto_rows = read_jsonl(automatic)
            self.assertEqual(len(auto_rows), 8)
            self.assertEqual({row["source_question_id"] for row in auto_rows}, {"q1", "q2", "q3"})
            self.assertEqual(len({row["split"] for row in auto_rows}), 3)
            self.assertTrue(all(len({row["split"] for row in auto_rows
                                     if row["source_question_id"] == source_id}) == 1
                                for source_id in ("q1", "q2", "q3")))

            for row in rows:
                row["answer_equivalence"] = (
                    "equivalent" if row["answer_original"] == row["answer_transformed"]
                    else "non_equivalent"
                )

            rows[0]["answer_equivalence"] = "non_equivalent"
            incorrect = root / "incorrect.jsonl"
            write_jsonl(incorrect, rows)
            with self.assertRaisesRegex(WorkflowError, "disagrees with extracted x"):
                compute_all_anchors(incorrect, splits)


if __name__ == "__main__":
    unittest.main()
