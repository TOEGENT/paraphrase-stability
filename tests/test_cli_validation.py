"""Pin every argparse validation message of dataset_cli.main, including which check wins."""

import contextlib
import io
import unittest

from slm_answer_stability.dataset_cli import main

OUT = "out.jsonl"
CASES = [
    # (command, argv, expected stderr fragment) - argv excludes --output, which is always added
    ("generate_questions", ["--topic", "t", "--count", "0"], "--count must be between 1 and 10000"),
    ("generate_questions", ["--topic", "t", "--count", "10001"], "--count must be between 1 and 10000"),
    ("generate_questions", ["--topic", "t", "--count", "5", "--batch-size", "0"],
     "--batch-size must be between 1 and 100"),
    ("generate_questions", ["--topic", "t", "--count", "0", "--batch-size", "0"],
     "--count must be between 1 and 10000"),
    ("generate_paraphrases", ["--input", "i", "--count", "0"], "--count must be between 1 and 100"),
    ("generate_paraphrases", ["--count", "0"], "--count must be between 1 and 100"),
    ("generate_paraphrases", ["--input", "i", "--count", "2", "--batch-size", "101"],
     "--batch-size must be between 1 and 100"),
    ("generate_paraphrases", ["--input", "i", "--count", "11", "--mode", "combined"],
     "--mode combined requires --count at most 10 (count cubed outputs)"),
    ("generate_paraphrases", ["--input", "i", "--count", "2", "--targets", "A"],
     "--targets requires --mode translation, structure, or style"),
    ("generate_paraphrases", ["--input", "i", "--count", "2", "--mode", "combined", "--targets", "A"],
     "--mode combined uses --structures, --styles, and --languages"),
    ("generate_paraphrases", ["--input", "i", "--count", "2", "--mode", "style", "--languages", "English"],
     "--structures, --styles, and --languages require --mode combined"),
    ("generate_paraphrases", ["--input", "i", "--count", "3", "--mode", "combined",
                              "--structures", "A", "B", "--styles", "C", "D", "E",
                              "--languages", "F", "G", "H"],
     "combined mode needs at least 3 structure targets"),
    ("generate_paraphrases", ["--input", "i", "--count", "2", "--mode", "combined",
                              "--structures", "A", "a", "--styles", "C", "D", "--languages", "F", "G"],
     "combined mode needs distinct structure targets"),
    ("generate_paraphrases", ["--count", "2"], "--input is required"),
    ("generate_answers", [], "--input is required"),
    ("equivalence_generation_ai", [], "--input is required"),
    ("compute_k_n", [], "--input is required"),
    ("compute_k_n", ["--finalize"], "--input is required"),
    ("compute_k_n", ["--input", "i", "--finalize"], "--finalize requires --recompute"),
    ("compute_k_n", ["--input", "i", "--original-only", "--splits-dir", "s"],
     "--splits-dir is only for the default all-anchor mode"),
    ("compute_k_n", ["--input", "i", "--recompute"], "--recompute and --finalize require --original-only"),
    ("compute_k_n", ["--input", "i", "--recompute", "--finalize"],
     "--recompute and --finalize require --original-only"),
]


class CliValidation(unittest.TestCase):
    def test_validation_messages_and_precedence(self):
        for command, argv, expected in CASES:
            with self.subTest(command=command, argv=argv):
                err = io.StringIO()
                with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as raised:
                    main(command, [*argv, "--output", OUT])
                self.assertEqual(raised.exception.code, 2)
                self.assertTrue(err.getvalue().rstrip().endswith(f"error: {expected}"), err.getvalue())

    def test_every_command_accepts_its_documented_flags(self):
        for command, argv in (
            ("generate_questions", ["--topic", "t", "--count", "1", "--batch-size", "1",
                                    "--duplicate-check", "semantic", "--existing", "e"]),
            ("generate_paraphrases", ["--input", "i", "--count", "1", "--batch-size", "1", "--mode", "style",
                                      "--targets", "A", "--existing", "e", "--log-file", "l"]),
            ("generate_answers", ["--input", "i"]),
            ("equivalence_generation_ai", ["--input", "i"]),
            ("compute_k_n", ["--input", "i", "--seed", "1", "--train", "0.8", "--val", "0.1", "--test", "0.1",
                             "--splits-dir", "s"]),
        ):
            with self.subTest(command=command):
                err = io.StringIO()
                with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                    # A missing config or input file is a handled error (exit 1), never an argparse error.
                    code = main(command, [*argv, "--output", "/nonexistent-dir-for-test/out.jsonl",
                                          *([] if command == "compute_k_n" else
                                            ["--config", "/nonexistent-config.json"])])
                self.assertEqual(code, 1, err.getvalue())


if __name__ == "__main__":
    unittest.main()
