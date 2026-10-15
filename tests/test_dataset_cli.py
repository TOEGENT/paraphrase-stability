import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from slm_answer_stability.dataset_cli import (
    WorkflowError,
    compute_stability,
    evaluate_paraphrases,
    finalize,
    generate_answers,
    generate_paraphrases,
    generate_questions,
    read_jsonl,
    recompute,
    write_jsonl,
)


CONFIG = {
    "slm": {"model": "fixed-slm"},
    "generator": {"model": "generator"},
    "judge": {"model": "judge"},
}


class DatasetWorkflowTests(unittest.TestCase):
    def test_question_generation_flags_existing_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing.jsonl"
            write_jsonl(existing, [{"question_id": "old", "question": "What is two plus two?"}])
            with patch("slm_answer_stability.dataset_cli.call_model", return_value={
                "questions": [{"question": "  what is two plus two?  "}], "model": "generator"
            }) as model:
                rows = generate_questions(Namespace(existing=existing, topic="math", count=1), CONFIG)
            self.assertEqual(rows[0]["duplicate_status"], "exact")
            self.assertEqual(rows[0]["question_review"], "pending")
            self.assertEqual(model.call_count, 1)

    def test_human_gates_then_answer_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            questions = base / "questions.jsonl"
            write_jsonl(questions, [{"question_id": "q1", "question": "2+2?", "question_review": "pending"}])
            with self.assertRaisesRegex(WorkflowError, "question_review"):
                generate_paraphrases(Namespace(input=questions, count=1), CONFIG)
            approved = base / "approved.jsonl"
            write_jsonl(approved, [{"question_id": "q1", "question": "2+2?", "question_review": "approved"}])
            with patch("slm_answer_stability.dataset_cli.call_model", return_value={
                "variants": [{"question": "What is two plus two?", "transformation_type": "lexical"}],
                "model": "generator",
            }):
                variants = generate_paraphrases(Namespace(input=approved, count=1), CONFIG)
            self.assertEqual(variants[0]["paraphrase_review"], "pending")
            variants[0]["paraphrase_review"] = "approved"
            variant_file = base / "variants.jsonl"
            write_jsonl(variant_file, variants)
            with patch("slm_answer_stability.dataset_cli.call_model", return_value={
                "status": "equivalent", "rationale_short": "Same question", "model": "judge"
            }):
                evaluated = evaluate_paraphrases(Namespace(input=variant_file), CONFIG)
            self.assertEqual(evaluated[0]["equivalence_review"], "pending")
            evaluated[0]["equivalence_review"] = "approved"
            evaluated_file = base / "evaluated.jsonl"
            write_jsonl(evaluated_file, evaluated)
            with patch("slm_answer_stability.dataset_cli.call_model", side_effect=[
                {"answer": "4", "model": "fixed-slm"},
                {"answer": "Four", "model": "fixed-slm"},
            ]):
                answers = generate_answers(Namespace(input=evaluated_file), CONFIG)
            self.assertEqual(answers[0]["answer_original"], "4")
            self.assertEqual(answers[0]["answer_transformed"], "Four")

    def test_unclear_target_requires_review_and_recompute(self):
        with tempfile.TemporaryDirectory() as directory:
            answer_file = Path(directory) / "answers.jsonl"
            write_jsonl(answer_file, [{
                "question_id": "q1", "variant_id": "q1-v1", "question_original": "2+2?",
                "question_transformed": "Two plus two?", "answer_original": "4",
                "answer_transformed": "Four",
            }])
            with patch("slm_answer_stability.dataset_cli.call_model", return_value={
                "status": "unclear", "rationale_short": "Review", "model": "judge"
            }):
                results = compute_stability(Namespace(input=answer_file), CONFIG)
            self.assertIsNone(results[0]["target"])
            self.assertEqual(results[0]["N"], 1)
            results[0]["variants"][0]["answer_equivalence"] = "equivalent"
            reviewed_file = Path(directory) / "reviewed.jsonl"
            write_jsonl(reviewed_file, results)
            final = recompute(Namespace(input=reviewed_file))
            self.assertEqual(final[0]["target"], 1.0)
            self.assertEqual(final[0]["k"], 1)
            self.assertEqual(read_jsonl(reviewed_file)[0]["target"], None)
            with self.assertRaisesRegex(WorkflowError, "review_status"):
                finalize(final)
            final[0]["review_status"] = "approved"
            self.assertEqual(len(finalize(final)), 1)


if __name__ == "__main__":
    unittest.main()
