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
            self.assertEqual(model.call_args.kwargs["existing_questions"], ["What is two plus two?"])

    def test_generates_200_questions_in_batches(self):
        next_number = 0

        def model_response(_config, command, **kwargs):
            nonlocal next_number
            self.assertEqual(command, "questions")
            questions = []
            for _ in range(kwargs["count"]):
                next_number += 1
                questions.append({"question": f"What is {next_number} plus 1?"})
            return {"questions": questions, "model": "generator"}

        with patch("slm_answer_stability.dataset_cli.call_model", side_effect=model_response) as model:
            rows = generate_questions(Namespace(existing=None, topic="math", count=200, batch_size=20, duplicate_check="exact"), CONFIG)
        self.assertEqual(len(rows), 200)
        self.assertEqual(model.call_count, 10)
        self.assertEqual([call.kwargs["count"] for call in model.call_args_list], [20] * 10)
        self.assertEqual(model.call_args_list[0].kwargs["existing_questions"], [])
        self.assertEqual(
            model.call_args_list[1].kwargs["existing_questions"],
            [f"What is {number} plus 1?" for number in range(1, 21)],
        )
        self.assertEqual(len(model.call_args_list[-1].kwargs["existing_questions"]), 180)
        self.assertTrue(all(row["duplicate_status"] == "unchecked" for row in rows))

    def test_default_duplicate_check_does_not_call_judge(self):
        with patch("slm_answer_stability.dataset_cli.call_model", return_value={
            "questions": [{"question": "What is 3 plus 4?"}], "model": "generator"
        }) as model:
            rows = generate_questions(Namespace(existing=None, topic="math", count=1), CONFIG)
        self.assertEqual(rows[0]["duplicate_status"], "unchecked")
        self.assertEqual([call.args[1] for call in model.call_args_list], ["questions"])

    def test_invalid_json_retries_with_smaller_batches(self):
        next_number = 0

        def model_response(_config, command, **kwargs):
            nonlocal next_number
            self.assertEqual(command, "questions")
            if kwargs["count"] > 2:
                raise WorkflowError("questions: generator returned invalid JSON: Unterminated string")
            rows = []
            for _ in range(kwargs["count"]):
                next_number += 1
                rows.append({"question": f"Question {next_number}?"})
            return {"questions": rows, "model": "generator"}

        with patch("slm_answer_stability.dataset_cli.call_model", side_effect=model_response) as model:
            rows = generate_questions(Namespace(existing=None, topic="math", count=5,
                                                batch_size=5, duplicate_check="exact"), CONFIG)
        self.assertEqual(len(rows), 5)
        self.assertEqual([call.kwargs["count"] for call in model.call_args_list], [5, 2, 2, 1])
        self.assertEqual(model.call_args_list[2].kwargs["existing_questions"],
                         ["Question 1?", "Question 2?"])

    def test_human_gates_then_answer_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            questions = base / "questions.jsonl"
            write_jsonl(questions, [{"question_id": "q1", "question": "2+2?", "question_review": "pending"}])
            with self.assertRaisesRegex(WorkflowError, "question_review"):
                generate_paraphrases(Namespace(input=questions, count=1), CONFIG)
            approved = base / "approved.jsonl"
            write_jsonl(approved, [{"question_id": "q1", "question": "2+2?", "topic": "math", "language": "en", "question_review": "approved"}])
            with patch("slm_answer_stability.dataset_cli.call_model", return_value={
                "variants": [{"question": "What is two plus two?", "transformation_type": "lexical"}],
                "model": "generator",
            }):
                variants = generate_paraphrases(Namespace(input=approved, count=1), CONFIG)
            self.assertEqual(variants[0]["paraphrase_review"], "pending")
            self.assertEqual(variants[0]["topic"], "math")
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
            self.assertEqual(answers[0]["language"], "en")

    def test_unclear_target_requires_review_and_recompute(self):
        with tempfile.TemporaryDirectory() as directory:
            answer_file = Path(directory) / "answers.jsonl"
            write_jsonl(answer_file, [{
                "question_id": "q1", "variant_id": "q1-v1", "question_original": "2+2?",
                "question_transformed": "Two plus two?", "answer_original": "4",
                "answer_transformed": "Four", "topic": "math", "language": "en",
            }])
            with patch("slm_answer_stability.dataset_cli.call_model", return_value={
                "status": "unclear", "rationale_short": "Review", "model": "judge"
            }):
                results = compute_stability(Namespace(input=answer_file), CONFIG)
            self.assertIsNone(results[0]["target"])
            self.assertEqual(results[0]["N"], 1)
            self.assertEqual(results[0]["topic"], "math")
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
