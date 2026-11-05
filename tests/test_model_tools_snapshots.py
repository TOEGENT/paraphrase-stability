"""Pin the exact HTTP requests model_tools sends, so refactors cannot change model input.

Expected values live in tests/golden/model_tools_requests.json. Regenerate them only
when a prompt or payload is changed on purpose:
    PYTHONPATH=src python tests/test_model_tools_snapshots.py --record
"""

import io
import json
import sys
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from slm_answer_stability.model_tools import run

GOLDEN = Path(__file__).parent / "golden" / "model_tools_requests.json"

BASE = {
    "slm": {"model": "fixed-slm", "temperature": 0, "seed": 42, "num_predict": 128},
    "generator": {"model": "generator", "temperature": 0.7, "seed": 42, "num_predict": 1024},
    "judge": {"model": "judge", "temperature": 0, "seed": 42, "num_predict": 256},
}
STYLES = {
    "ollama": {"base_url": "http://localhost:11434", **BASE},
    "openai": {"base_url": "http://localhost:9090/v1", "api_style": "openai", **BASE,
               "slm": {**BASE["slm"], "system": "Answer briefly.",
                       "chat_template_kwargs": {"enable_thinking": False}}},
}

VARIANT = {"variants": [{"question": "Кто это такой?", "transformation_type": "lexical"}]}
STRUCT = {"variants": [{"question": '{"request":"Who?"}', "transformation_type": "structure"}]}
JUDGE = {"status": "equivalent", "rationale_short": "same"}

# name -> (response content for the mocked model, run() arguments)
CASES = {
    "answer": ("x = 3", dict(command="answer", question="Solve 2*x = 6 for x.")),
    "questions": (
        {"questions": [{"question": "Сколько граней у куба?", "topic": "математика"}]},
        dict(command="questions", topic="математика", count=1)),
    "questions_existing": (
        {"questions": [{"question": "Сколько граней у куба?", "topic": "математика"}]},
        dict(command="questions", topic="математика", count=1,
             existing_questions=["Сколько будет 2 + 2?"])),
    "paraphrase_general": (VARIANT, dict(command="paraphrases", question="Кто?", count=1)),
    "paraphrase_existing_attempt": (VARIANT, dict(
        command="paraphrases", question="Кто?", count=1,
        existing_paraphrases=["Кто именно?"], variation_attempt=7)),
    "paraphrase_translation": (VARIANT, dict(
        command="paraphrases", question="Кто это?", count=1, mode="translation", target="English")),
    "paraphrase_structure": (STRUCT, dict(
        command="paraphrases", question="Кто?", count=1, mode="structure", target="JSON")),
    "paraphrase_style": (VARIANT, dict(
        command="paraphrases", question="Кто?", count=1, mode="style", target="formal")),
    "paraphrase_combined_stage": (VARIANT, dict(
        command="paraphrases", question='{"request":"Кто?"}', count=1, mode="translation",
        target="English", preserve_structure=True, reference_question="Кто?",
        allow_partial=True, allow_duplicates=True)),
    "judge_question": (JUDGE, dict(
        command="judge-question", question="Кто?", rewritten_question="Назовите человека.")),
    "judge_answer": (JUDGE, dict(
        command="judge-answer", question="Кто?", answer="Анна", rewritten_answer="Борис")),
}


class MockResponse:
    def __init__(self, value):
        self.stream = io.BytesIO(json.dumps(value).encode("utf-8"))

    def __enter__(self):
        return self.stream

    def __exit__(self, *_args):
        self.stream.close()


def capture():
    """Run every case against a mocked urlopen and return the requests sent."""
    result = {}
    for style, config in STYLES.items():
        for name, (content, arguments) in CASES.items():
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            body = ({"message": {"content": text}} if style == "ollama"
                    else {"choices": [{"message": {"content": text}}]})
            with patch("slm_answer_stability.model_tools.urlopen",
                       return_value=MockResponse(body)) as request:
                output = run(Namespace(**arguments), config)
            sent = request.call_args.args[0]
            result[f"{style}/{name}"] = {
                "url": sent.full_url,
                "headers": sorted(sent.header_items()),
                "body": sent.data.decode("utf-8"),
                "timeout": request.call_args.kwargs["timeout"],
                "result": output,
            }
    return result


class ModelToolRequestSnapshots(unittest.TestCase):
    def test_requests_and_results_are_unchanged(self):
        expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
        actual = json.loads(json.dumps(capture()))
        self.assertEqual(sorted(actual), sorted(expected))
        for key in expected:
            with self.subTest(key):
                self.assertEqual(actual[key], expected[key])


if __name__ == "__main__":
    if "--record" in sys.argv:
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(capture(), ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                          encoding="utf-8")
        print(f"recorded {GOLDEN}")
    else:
        unittest.main()
