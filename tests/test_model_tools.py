import io
import json
import unittest
from argparse import Namespace
from unittest.mock import patch

from slm_answer_stability.model_tools import ModelToolError, chat, run


CONFIG = {
    "base_url": "http://localhost:11434",
    "slm": {"model": "fixed-slm", "temperature": 0},
    "generator": {"model": "generator"},
    "judge": {"model": "judge"},
}


class MockResponse:
    def __init__(self, value):
        self.stream = io.BytesIO(json.dumps(value).encode("utf-8"))

    def __enter__(self):
        return self.stream

    def __exit__(self, *_args):
        self.stream.close()


class ModelToolsTests(unittest.TestCase):
    def test_openai_compatible_question_request(self):
        config = {**CONFIG, "api_style": "openai", "base_url": "http://localhost:9090/v1"}
        content = json.dumps({"questions": [{"question": "Сколько будет 2 + 2?", "topic": "математика"}]}, ensure_ascii=False)
        response = {"choices": [{"message": {"content": content}}]}
        with patch("slm_answer_stability.model_tools.urlopen", return_value=MockResponse(response)) as request:
            result = run(Namespace(command="questions", topic="математика", count=1), config)
        self.assertEqual(result["questions"][0]["question"], "Сколько будет 2 + 2?")
        self.assertEqual(request.call_args.args[0].full_url, "http://localhost:9090/v1/chat/completions")
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["max_tokens"], 256)
        self.assertNotIn("options", payload)
        self.assertIn("Existing questions to avoid: []", payload["messages"][1]["content"])

    def test_question_request_includes_existing_questions(self):
        value = {"questions": [{"question": "Сколько граней у куба?"}]}
        with patch("slm_answer_stability.model_tools.urlopen", return_value=MockResponse({"message": {"content": json.dumps(value)}})) as request:
            run(Namespace(command="questions", topic="математика", count=1,
                          existing_questions=["Сколько будет 2 + 2?"]), CONFIG)
        payload = json.loads(request.call_args.args[0].data)
        self.assertIn("Сколько будет 2 + 2?", payload["messages"][1]["content"])
        self.assertIn("changing only numbers or names is not enough", payload["messages"][0]["content"])

    def test_slm_uses_nonstreaming_fixed_decoding(self):
        with patch("slm_answer_stability.model_tools.urlopen", return_value=MockResponse({"message": {"content": "Анна"}})) as request:
            result = run(Namespace(command="answer", question="Кто?"), CONFIG)
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(result["answer"], "Анна")
        self.assertEqual(payload["model"], "fixed-slm")
        self.assertEqual(payload["options"]["temperature"], 0)
        self.assertIs(payload["stream"], False)

    def test_paraphrase_rejects_duplicate(self):
        value = {"variants": [{"question": "Кто?", "transformation_type": "lexical"}]}
        with patch("slm_answer_stability.model_tools.urlopen", return_value=MockResponse({"message": {"content": json.dumps(value)}})):
            with self.assertRaisesRegex(ModelToolError, "duplicate"):
                run(Namespace(command="paraphrases", question="Кто?", count=1), CONFIG)

    def test_paraphrase_request_includes_existing_variants(self):
        value = {"variants": [{"question": "Кто это?", "transformation_type": "lexical"}]}
        with patch("slm_answer_stability.model_tools.urlopen", return_value=MockResponse({"message": {"content": json.dumps(value)}})) as request:
            run(Namespace(command="paraphrases", question="Кто?", count=1,
                          existing_paraphrases=["Кто именно?"]), CONFIG)
        payload = json.loads(request.call_args.args[0].data)
        self.assertIn("Кто именно?", payload["messages"][1]["content"])
        self.assertIn("В коробке 12 красных шаров", payload["messages"][0]["content"])
        self.assertIn("Вычислите 18 + (4 × 6)", payload["messages"][0]["content"])
        self.assertIn("Какой город является столицей Италии", payload["messages"][0]["content"])

    def test_translation_prompt_uses_target_language(self):
        value = {"variants": [{"question": "Who is it?", "transformation_type": "translation"}]}
        with patch("slm_answer_stability.model_tools.urlopen", return_value=MockResponse({"message": {"content": json.dumps(value)}})) as request:
            run(Namespace(command="paraphrases", question="Кто это?", count=1,
                          mode="translation", target="English"), CONFIG)
        payload = json.loads(request.call_args.args[0].data)
        self.assertIn("Translate the entire question", payload["messages"][0]["content"])
        self.assertIn("Target: English", payload["messages"][1]["content"])
        self.assertNotIn("same language as the original", payload["messages"][0]["content"])

    def test_structure_prompt_names_requested_format(self):
        value = {"variants": [{"question": '{"request":"Who?"}',
                               "transformation_type": "structure"}]}
        with patch("slm_answer_stability.model_tools.urlopen", return_value=MockResponse({"message": {"content": json.dumps(value)}})) as request:
            run(Namespace(command="paraphrases", question="Кто?", count=1,
                          mode="structure", target="JSON"), CONFIG)
        payload = json.loads(request.call_args.args[0].data)
        self.assertIn("Target: JSON", payload["messages"][1]["content"])
        self.assertIn("explicit request for the answer", payload["messages"][0]["content"])

    def test_combined_stage_keeps_structure_and_original_reference(self):
        value = {"variants": [{"question": '{"request":"Who?"}',
                               "transformation_type": "translation"}]}
        with patch("slm_answer_stability.model_tools.urlopen", return_value=MockResponse({"message": {"content": json.dumps(value)}})) as request:
            run(Namespace(command="paraphrases", question='{"request":"Кто?"}', count=1,
                          mode="translation", target="English", preserve_structure=True,
                          reference_question="Кто?"), CONFIG)
        payload = json.loads(request.call_args.args[0].data)
        self.assertIn("Preserve the input's table, CSV, JSON", payload["messages"][0]["content"])
        self.assertIn("Source question for checking facts only: Кто?",
                      payload["messages"][1]["content"])

    def test_paraphrase_trace_records_exact_request_and_raw_response(self):
        value = {"variants": [{"question": "Кто это?", "transformation_type": "lexical"}]}
        raw = json.dumps(value, ensure_ascii=False)
        events = []
        config = {**CONFIG, "generator": {"model": "generator", "temperature": 0.7, "seed": 42}}
        with patch("slm_answer_stability.model_tools.urlopen", return_value=MockResponse({"message": {"content": raw}})):
            run(Namespace(command="paraphrases", question="Кто?", count=1,
                          existing_paraphrases=["Кто именно?"], variation_attempt=2,
                          model_trace=events.append), config)
        self.assertEqual([event["event"] for event in events], ["request", "response"])
        self.assertIn("Кто именно?", events[0]["payload"]["messages"][1]["content"])
        self.assertIn("Variation attempt: 2", events[0]["payload"]["messages"][1]["content"])
        self.assertIn("Change the sentence type or clause structure", events[0]["payload"]["messages"][1]["content"])
        self.assertEqual(events[0]["payload"]["options"]["temperature"], 0.7)
        self.assertEqual(events[0]["payload"]["options"]["seed"], 42)
        self.assertEqual(events[1]["content"], raw)

    def test_paraphrase_batch_can_return_duplicate_for_workflow_filtering(self):
        value = {"variants": [{"question": "Кто?", "transformation_type": "lexical"}]}
        with patch("slm_answer_stability.model_tools.urlopen", return_value=MockResponse({"message": {"content": json.dumps(value)}})):
            result = run(Namespace(command="paraphrases", question="Кто?", count=1,
                                   allow_duplicates=True, variation_attempt=1), CONFIG)
        self.assertEqual(result["variants"][0]["question"], "Кто?")

    def test_paraphrase_batch_keeps_valid_items_from_incomplete_response(self):
        value = {"variants": [
            {"question": "Кто это?", "transformation_type": "lexical"},
            {"question": "Пустой тип", "transformation_type": ""},
        ]}
        with patch("slm_answer_stability.model_tools.urlopen", return_value=MockResponse({"message": {"content": json.dumps(value)}})):
            result = run(Namespace(command="paraphrases", question="Кто?", count=3,
                                   allow_partial=True, allow_duplicates=True), CONFIG)
        self.assertEqual([row["question"] for row in result["variants"]], ["Кто это?"])

    def test_paraphrase_batch_keeps_extra_variants_for_workflow_selection(self):
        value = {"variants": [
            {"question": "Кто?", "transformation_type": "lexical"},
            {"question": "Кто это?", "transformation_type": "syntactic"},
        ]}
        with patch("slm_answer_stability.model_tools.urlopen", return_value=MockResponse({"message": {"content": json.dumps(value)}})):
            result = run(Namespace(command="paraphrases", question="Кто?", count=1,
                                   allow_partial=True, allow_duplicates=True), CONFIG)
        self.assertEqual(result["raw_variant_count"], 2)
        self.assertEqual(len(result["variants"]), 2)

    def test_judge_answer_returns_status(self):
        value = {"status": "unclear", "rationale_short": "Контекста недостаточно"}
        with patch("slm_answer_stability.model_tools.urlopen", return_value=MockResponse({"message": {"content": json.dumps(value)}})) as request:
            result = run(Namespace(command="judge-answer", question="Кто?", answer="Анна", rewritten_answer="Борис"), CONFIG)
        payload = json.loads(request.call_args.args[0].data)
        self.assertEqual(result["status"], "unclear")
        self.assertEqual(payload["model"], "judge")
        self.assertEqual(payload["format"], "json")


if __name__ == "__main__":
    unittest.main()
