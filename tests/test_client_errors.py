import json
import unittest
from unittest.mock import patch

from slm_answer_stability.data.io import WorkflowError
from slm_answer_stability.generation import client
from slm_answer_stability.model_tools import ModelToolError, WrongQuestionCount, parse_object


def raising(error):
    return patch("slm_answer_stability.generation.client.run_model", side_effect=error)


class CallModelErrorMapping(unittest.TestCase):
    def invalid_json(self, role):
        try:
            parse_object("not json", role)
        except ModelToolError as exc:
            return exc

    def test_generator_json_error_is_recoverable_and_keeps_its_message(self):
        original = self.invalid_json("generator")
        with raising(original), self.assertRaises(client.InvalidGeneratorJSON) as raised:
            client.call_model({}, "paraphrases")
        self.assertEqual(str(raised.exception), f"paraphrases: {original}")
        self.assertIn("generator returned invalid JSON", str(raised.exception))

    def test_judge_json_error_is_an_ordinary_workflow_error(self):
        with raising(self.invalid_json("judge")), self.assertRaises(WorkflowError) as raised:
            client.call_model({}, "judge-answer")
        self.assertNotIsInstance(raised.exception, client.InvalidGeneratorJSON)
        self.assertIn("judge returned invalid JSON", str(raised.exception))

    def test_wrong_question_count_is_an_incomplete_batch(self):
        message = "generator returned the wrong number of questions or an invalid question"
        with raising(WrongQuestionCount(message)), self.assertRaises(client.IncompleteBatch) as raised:
            client.call_model({}, "questions")
        self.assertEqual(str(raised.exception), f"questions: {message}")

    def test_other_model_errors_are_not_retried(self):
        with raising(ModelToolError("generator request failed")), self.assertRaises(WorkflowError) as raised:
            client.call_model({}, "paraphrases")
        self.assertNotIsInstance(raised.exception, (client.InvalidGeneratorJSON, client.IncompleteBatch))

    def test_invalid_json_that_is_not_an_object_is_not_retried(self):
        with self.assertRaises(ModelToolError) as raised:
            parse_object(json.dumps([1, 2]), "generator")
        with raising(raised.exception), self.assertRaises(WorkflowError) as mapped:
            client.call_model({}, "paraphrases")
        self.assertNotIsInstance(mapped.exception, client.InvalidGeneratorJSON)


if __name__ == "__main__":
    unittest.main()
