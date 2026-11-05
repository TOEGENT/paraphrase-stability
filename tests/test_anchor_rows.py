import tempfile
import unittest
from pathlib import Path

from slm_answer_stability.dataset_cli import compute_all_anchors, write_jsonl


def pairwise_rows(question_ids):
    """Verified (pairwise) anchor rows for one equation whose answers are x=2, x=2, x=3."""
    xs = {"original": "2", "v1": "2", "v2": "3"}
    return [{
        "question_id": question_ids[anchor], "source_question_id": "eq1", "anchor_id": anchor,
        "question": f"question {anchor}", "answer": f"x = {x}", "extracted_x": x,
        "comparisons": [{"other_anchor_id": other, "answer_equivalence":
                         "equivalent" if xs[other] == x else "non_equivalent"}
                        for other in xs if other != anchor],
        "source": "synthetic", "topic": "algebra", "language": "en", "slm_model": "m",
    } for anchor, x in xs.items()]


class VerifiedAnchorRows(unittest.TestCase):
    def test_pairwise_mode_keeps_the_input_question_id(self):
        ids = {"original": "custom-0", "v1": "custom-1", "v2": "custom-2"}
        # three groups are needed for three splits, so use three equations
        with tempfile.TemporaryDirectory() as directory:
            rows = []
            for group in ("eq1", "eq2", "eq3"):
                for row in pairwise_rows({a: f"{group}-{a}-custom" for a in ids}):
                    rows.append({**row, "source_question_id": group,
                                 "question_id": f"{group}-{row['anchor_id']}-custom"})
            source = Path(directory) / "pairwise.jsonl"
            write_jsonl(source, rows)
            result = compute_all_anchors(source)
        self.assertEqual(sorted(row["question_id"] for row in result),
                         sorted(row["question_id"] for row in rows))
        self.assertEqual({row["anchor_id"]: (row["k"], row["N"]) for row in result
                          if row["source_question_id"] == "eq1"},
                         {"original": (1, 2), "v1": (1, 2), "v2": (0, 2)})


if __name__ == "__main__":
    unittest.main()
