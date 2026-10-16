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
    read_targets,
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

    def test_paraphrase_batches_use_only_same_question_history(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            questions = base / "questions.jsonl"
            write_jsonl(questions, [
                {"question_id": "q1", "question": "One?", "question_review": "approved"},
                {"question_id": "q2", "question": "Two?", "question_review": "approved"},
            ])
            existing = base / "existing.jsonl"
            write_jsonl(existing, [{"question_id": "q1", "variant_id": "q1-v1",
                                    "question_original": "One?", "question_transformed": "First?"}])
            number = 0

            def model_response(_config, command, **kwargs):
                nonlocal number
                self.assertEqual(command, "paraphrases")
                self.assertEqual(kwargs["count"], 1)
                number += 1
                return {"variants": [{"question": f"Rewrite {number}?",
                                       "transformation_type": "lexical"}], "model": "generator"}

            with patch("slm_answer_stability.dataset_cli.call_model", side_effect=model_response) as model:
                rows = generate_paraphrases(Namespace(input=questions, existing=existing,
                                                       count=2, batch_size=1), CONFIG)
            self.assertEqual([row["variant_id"] for row in rows],
                             ["q1-v2", "q1-v3", "q2-v1", "q2-v2"])
            self.assertEqual([call.kwargs["existing_paraphrases"] for call in model.call_args_list],
                             [["First?"], ["First?", "Rewrite 1?"], [], ["Rewrite 3?"]])

    def test_paraphrase_modes_cycle_targets_and_record_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            questions = Path(directory) / "questions.jsonl"
            write_jsonl(questions, [{"question_id": "q1", "question": "Кто?",
                                     "question_review": "approved"}])
            answers = iter(["Who?", "Wer?", "Who exactly?"])

            def model_response(_config, _command, **kwargs):
                return {"variants": [{"question": next(answers),
                                      "transformation_type": "model label"}], "model": "generator"}

            with patch("slm_answer_stability.dataset_cli.call_model", side_effect=model_response) as model:
                rows = generate_paraphrases(Namespace(input=questions, count=3, batch_size=5,
                                                       mode="translation",
                                                       targets=["English", "German"]), CONFIG)
            self.assertEqual([call.kwargs["target"] for call in model.call_args_list],
                             ["English", "German", "English"])
            self.assertEqual([call.kwargs["count"] for call in model.call_args_list], [1, 1, 1])
            self.assertEqual([row["transformation_type"] for row in rows],
                             ["translation:English", "translation:German", "translation:English"])
            self.assertEqual(rows[1]["transformation_mode"], "translation")
            self.assertEqual(rows[1]["transformation_target"], "German")

    def test_translation_uses_language_file_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            questions = Path(directory) / "questions.jsonl"
            write_jsonl(questions, [{"question_id": "q1", "question": "Кто?",
                                     "question_review": "approved"}])
            languages = Path(directory) / "languages.txt"
            languages.write_text("# comment\nItalian\nSpanish\n", encoding="utf-8")
            with patch.dict("slm_answer_stability.dataset_cli.TARGET_FILES",
                            {"translation": languages}):
                with patch("slm_answer_stability.dataset_cli.call_model", return_value={
                    "variants": [{"question": "Chi?", "transformation_type": "translation"}],
                    "model": "generator",
                }) as model:
                    generate_paraphrases(Namespace(input=questions, count=1,
                                                   mode="translation"), CONFIG)
            self.assertEqual(model.call_args.kwargs["target"], "Italian")

    def test_target_file_rejects_case_insensitive_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "targets.txt"
            path.write_text("English\nenglish\n", encoding="utf-8")
            with self.assertRaisesRegex(WorkflowError, "Duplicate targets"):
                read_targets(path)

    def test_paraphrase_mode_retries_same_target_after_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            questions = Path(directory) / "questions.jsonl"
            write_jsonl(questions, [{"question_id": "q1", "question": "Кто?",
                                     "question_review": "approved"}])
            responses = [
                {"variants": [{"question": "Кто?", "transformation_type": "translation"}],
                 "model": "generator"},
                {"variants": [{"question": "Who?", "transformation_type": "translation"}],
                 "model": "generator"},
                {"variants": [{"question": "Wer?", "transformation_type": "translation"}],
                 "model": "generator"},
            ]
            with patch("slm_answer_stability.dataset_cli.call_model", side_effect=responses) as model:
                rows = generate_paraphrases(Namespace(input=questions, count=2,
                                                       mode="translation",
                                                       targets=["English", "German"]), CONFIG)
            self.assertEqual([call.kwargs["target"] for call in model.call_args_list],
                             ["English", "English", "German"])
            self.assertEqual([row["question_transformed"] for row in rows], ["Who?", "Wer?"])

    def test_combined_mode_generates_cartesian_cascade(self):
        with tempfile.TemporaryDirectory() as directory:
            questions = Path(directory) / "questions.jsonl"
            write_jsonl(questions, [{"question_id": "q1", "question": "Original?",
                                     "question_review": "approved"}])
            calls = []

            def model_response(_config, _command, **kwargs):
                calls.append(kwargs)
                transformed = f'{kwargs["question"]} [{kwargs["mode"]}:{kwargs["target"]}]'
                return {"variants": [{"question": transformed,
                                      "transformation_type": kwargs["mode"]}],
                        "model": "generator"}

            args = Namespace(input=questions, mode="combined", count=2,
                             structures=["table", "JSON"],
                             styles=["literary", "scientific"],
                             languages=["English", "German"])
            with patch("slm_answer_stability.dataset_cli.call_model", side_effect=model_response):
                rows = generate_paraphrases(args, CONFIG)
            self.assertEqual(len(rows), 8)
            self.assertEqual(len(calls), 14)
            self.assertEqual([call["mode"] for call in calls].count("structure"), 2)
            self.assertEqual([call["mode"] for call in calls].count("style"), 4)
            self.assertEqual([call["mode"] for call in calls].count("translation"), 8)
            self.assertEqual(calls[1]["question"], rows[0]["transformation_chain"][0]["question"])
            self.assertEqual(calls[2]["question"], rows[0]["transformation_chain"][1]["question"])
            self.assertEqual(rows[0]["question_transformed"],
                             rows[0]["transformation_chain"][2]["question"])
            self.assertEqual([row["variant_id"] for row in rows],
                             [f"q1-v{i}" for i in range(1, 9)])
            self.assertEqual({row["transformation_type"] for row in rows}, {"combined"})

    def test_combined_mode_requires_distinct_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            questions = Path(directory) / "questions.jsonl"
            write_jsonl(questions, [{"question_id": "q1", "question": "Original?",
                                     "question_review": "approved"}])
            with self.assertRaisesRegex(WorkflowError, "distinct translation targets"):
                generate_paraphrases(Namespace(input=questions, mode="combined", count=2,
                                               languages=["English", "english"]), CONFIG)

    def test_combined_mode_reads_all_target_files(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            questions = base / "questions.jsonl"
            write_jsonl(questions, [{"question_id": "q1", "question": "Original?",
                                     "question_review": "approved"}])
            files = {}
            for mode, name in (("structure", "JSON"), ("style", "scientific"),
                               ("translation", "French")):
                path = base / f"{mode}.txt"
                path.write_text(f"# available\n{name}\n", encoding="utf-8")
                files[mode] = path

            def model_response(_config, _command, **kwargs):
                return {"variants": [{"question": f'{kwargs["question"]} {kwargs["target"]}',
                                      "transformation_type": kwargs["mode"]}],
                        "model": "generator"}

            with patch.dict("slm_answer_stability.dataset_cli.TARGET_FILES", files):
                with patch("slm_answer_stability.dataset_cli.call_model",
                           side_effect=model_response):
                    rows = generate_paraphrases(Namespace(input=questions, mode="combined",
                                                           count=1), CONFIG)
            self.assertEqual([step["target"] for step in rows[0]["transformation_chain"]],
                             ["JSON", "scientific", "French"])

    def test_paraphrase_generation_recovers_from_duplicate_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            questions = Path(directory) / "questions.jsonl"
            write_jsonl(questions, [{"question_id": "q1", "question": "Original?",
                                     "question_review": "approved"}])
            responses = [
                {"variants": [{"question": "Original?", "transformation_type": "lexical"},
                              {"question": "Rewrite A?", "transformation_type": "lexical"}],
                 "model": "generator"},
                {"variants": [{"question": "Rewrite A?", "transformation_type": "lexical"}],
                 "model": "generator"},
                {"variants": [{"question": "Rewrite B?", "transformation_type": "syntactic"}],
                 "model": "generator"},
            ]
            with patch("slm_answer_stability.dataset_cli.call_model", side_effect=responses) as model:
                rows = generate_paraphrases(Namespace(input=questions, count=2, batch_size=2), CONFIG)
            self.assertEqual([row["question_transformed"] for row in rows],
                             ["Rewrite A?", "Rewrite B?"])
            self.assertEqual([call.kwargs["count"] for call in model.call_args_list], [2, 1, 1])
            self.assertEqual(model.call_args_list[2].kwargs["variation_attempt"], 1)

    def test_paraphrase_generation_fills_incomplete_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            questions = Path(directory) / "questions.jsonl"
            write_jsonl(questions, [{"question_id": "q1", "question": "Original?",
                                     "question_review": "approved"}])
            responses = [
                {"variants": [{"question": "Rewrite A?", "transformation_type": "lexical"}],
                 "model": "generator"},
                {"variants": [{"question": "Rewrite B?", "transformation_type": "syntactic"}],
                 "model": "generator"},
            ]
            with patch("slm_answer_stability.dataset_cli.call_model", side_effect=responses) as model:
                rows = generate_paraphrases(Namespace(input=questions, count=2, batch_size=2), CONFIG)
            self.assertEqual([row["question_transformed"] for row in rows],
                             ["Rewrite A?", "Rewrite B?"])
            self.assertEqual([call.kwargs["count"] for call in model.call_args_list], [2, 1])

    def test_paraphrase_generation_checks_extra_candidates_after_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            questions = Path(directory) / "questions.jsonl"
            write_jsonl(questions, [{"question_id": "q1", "question": "Original?",
                                     "question_review": "approved"}])
            response = {"variants": [
                {"question": "Original?", "transformation_type": "lexical"},
                {"question": "Rewrite A?", "transformation_type": "syntactic"},
                {"question": "Rewrite B?", "transformation_type": "stylistic"},
            ], "model": "generator", "raw_variant_count": 3, "invalid_variant_count": 0}
            with patch("slm_answer_stability.dataset_cli.call_model", return_value=response) as model:
                rows = generate_paraphrases(Namespace(input=questions, count=1, batch_size=1), CONFIG)
            self.assertEqual([row["question_transformed"] for row in rows], ["Rewrite A?"])
            self.assertEqual(model.call_count, 1)

    def test_paraphrase_generation_retries_invalid_json_with_smaller_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            questions = Path(directory) / "questions.jsonl"
            write_jsonl(questions, [{"question_id": "q1", "question": "Original?",
                                     "question_review": "approved"}])
            responses = [
                WorkflowError("paraphrases: generator returned invalid JSON: Unterminated string"),
                {"variants": [{"question": "Rewrite A?", "transformation_type": "lexical"}],
                 "model": "generator"},
                {"variants": [{"question": "Rewrite B?", "transformation_type": "syntactic"}],
                 "model": "generator"},
            ]
            with patch("slm_answer_stability.dataset_cli.call_model", side_effect=responses) as model:
                rows = generate_paraphrases(Namespace(input=questions, count=2, batch_size=2), CONFIG)
            self.assertEqual(len(rows), 2)
            self.assertEqual([call.kwargs["count"] for call in model.call_args_list], [2, 1, 1])

    def test_paraphrase_failure_keeps_attempt_log(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            questions = base / "questions.jsonl"
            log_file = base / "paraphrases.log.jsonl"
            write_jsonl(questions, [{"question_id": "q1", "question": "Original?",
                                     "question_review": "approved"}])
            response = {"variants": [{"question": "Original?", "transformation_type": "lexical"}],
                        "model": "generator", "raw_variant_count": 1, "invalid_variant_count": 0}
            with patch("slm_answer_stability.dataset_cli.call_model", return_value=response):
                with self.assertRaisesRegex(WorkflowError, "5 attempts"):
                    generate_paraphrases(Namespace(input=questions, count=1, batch_size=1,
                                                   log_file=log_file, run_id="run-1"), CONFIG)
            events = read_jsonl(log_file)
            self.assertEqual(len(events), 6)
            self.assertEqual([event["accepted"] for event in events[:5]], [0] * 5)
            self.assertEqual(events[0]["duplicate_examples"][0]["reason"], "original")
            self.assertEqual(events[-1]["event"], "stopped")

    def test_paraphrase_attempt_log_contains_model_prompt_and_response(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            questions = base / "questions.jsonl"
            log_file = base / "paraphrases.log.jsonl"
            write_jsonl(questions, [{"question_id": "q1", "question": "Original?",
                                     "question_review": "approved"}])

            def model_response(_config, _command, **kwargs):
                kwargs["model_trace"]({"event": "request", "url": "http://local",
                                       "payload": {"messages": [{"content": "full prompt"}]}})
                kwargs["model_trace"]({"event": "response", "content": "raw model reply",
                                       "finish_reason": "stop"})
                return {"variants": [{"question": "Rewrite?", "transformation_type": "lexical"}],
                        "model": "generator"}

            with patch("slm_answer_stability.dataset_cli.call_model", side_effect=model_response):
                generate_paraphrases(Namespace(input=questions, count=1, batch_size=1,
                                               log_file=log_file, run_id="run-1"), CONFIG)
            events = read_jsonl(log_file)
            self.assertEqual([event["event"] for event in events],
                             ["llm_request", "llm_response", "batch_result"])
            self.assertEqual(events[0]["payload"]["messages"][0]["content"], "full prompt")
            self.assertEqual(events[1]["content"], "raw model reply")
            self.assertEqual(events[0]["request_id"], events[1]["request_id"])

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
