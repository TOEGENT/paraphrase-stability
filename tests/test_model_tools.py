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
