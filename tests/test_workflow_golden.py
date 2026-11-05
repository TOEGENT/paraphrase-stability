"""Golden test for the whole file-based workflow, driven through the real CLI entry points.

A fake model server (patched at model_tools.urlopen, the lowest seam, so internal module moves
do not matter) answers deterministically, including invalid JSON and duplicate paraphrases to
exercise the retry paths. Every output file, return code and error message is compared with
tests/golden/workflow_outputs.json. Random ids and timestamps are normalized away.

Regenerate only when behaviour is changed on purpose:
    PYTHONPATH=src:. python tests/test_workflow_golden.py --record
"""

import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from urllib.error import URLError
from unittest.mock import patch

from scripts.equivalence_generation import main as equivalence_main
from slm_answer_stability import dataset_cli, dataset_prep
from slm_answer_stability.dataset_cli import read_jsonl, write_jsonl

GOLDEN = Path(__file__).parent / "golden" / "workflow_outputs.json"
CONFIG = {"base_url": "http://fake", "slm": {"model": "fixed-slm"},
          "generator": {"model": "generator"}, "judge": {"model": "judge"}}


class FakeServer:
    """Deterministic stand-in for the model server."""

    def __init__(self, short_first_questions=False, bad_questions=False):
        self.short_first_questions = short_first_questions
        self.bad_questions = bad_questions
        self.question_calls = 0
        self.paraphrase_calls = 0
        self.variants = 0

    def reply(self, payload: dict) -> str:
        model, system, user = (payload["model"], payload["messages"][0]["content"],
                               payload["messages"][1]["content"])
        if model == "fixed-slm":
            return f"x = {sum(map(ord, user)) % 3}"
        if model == "judge" and "Original answer:" not in user:  # judge-question: a mix of statuses
            status = ("equivalent", "non_equivalent", "unclear")[sum(map(ord, user)) % 3]
            return json.dumps({"status": status, "rationale_short": "question check"})
        if model == "judge":
            original, rewritten = re.search(
                r"Original answer: (.*)\nAnswer to rewritten question: (.*)", user).groups()
            return json.dumps({"status": "equivalent" if original == rewritten else "non_equivalent",
                               "rationale_short": "same" if original == rewritten else "different"})
        if "Create unambiguous questions" in system:
            self.question_calls += 1
            count = int(re.search(r"Number of questions: (\d+)", user).group(1))
            if self.question_calls == 1 and not self.short_first_questions:
                return "not json"
            if self.bad_questions:
                return "not json"
            if self.question_calls == 1:  # one question short: an incomplete batch
                count -= 1
            return json.dumps({"questions": [
                {"question": f"Generated question {self.question_calls}.{i}?", "topic": "math"}
                for i in range(count)]})
        self.paraphrase_calls += 1
        original, count = re.search(
            r"Original question: (.*)\nNumber of variants: (\d+)", user).groups()
        if "NETFAIL" in original:
            raise URLError("down")
        if "STALL" in original:  # only ever returns the original wording, so nothing is new
            return json.dumps({"variants": [{"question": original, "transformation_type": "lexical"}] * int(count)})
        if self.paraphrase_calls % 7 == 3:
            return "not json"
        items = []
        for _ in range(int(count)):
            self.variants += 1
            text = original if self.variants % 4 == 0 else f"{original} [v{self.variants}]"
            items.append({"question": text, "transformation_type": "lexical"})
        return json.dumps({"variants": items})


class Response:
    def __init__(self, text):
        self.stream = io.BytesIO(json.dumps({"message": {"content": text}}).encode("utf-8"))

    def __enter__(self):
        return self.stream

    def __exit__(self, *_args):
        self.stream.close()


def normalizer(tmp: str):
    ids: dict[str, str] = {}

    def walk(value, key=None):
        if isinstance(value, dict):
            return {k: walk(v, k) for k, v in value.items() if k != "timestamp_utc"}
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, str):
            value = value.replace(tmp, "<TMP>")
            if key in ("run_id", "request_id"):
                return ids.setdefault(value, f"<id{len(ids)}>")
            return re.sub(r"synthetic-[0-9a-f]{12}",
                          lambda match: ids.setdefault(match.group(0), f"<id{len(ids)}>"), value)
        return value
    return walk


def review(path: Path, field: str, reject_every: int = 0, rename: str | None = None) -> None:
    """Approve (or reject every Nth) row; optionally replace random ids so later steps are stable."""
    rows = read_jsonl(path)
    for index, row in enumerate(rows, 1):
        row[field] = "rejected" if reject_every and index % reject_every == 0 else "approved"
        if rename:
            row["question_id"] = f"{rename}{index}"
    path.unlink()
    write_jsonl(path, rows)


def anchor_answers() -> list[dict]:
    """Twelve equations with numeric answers: one original and five variant answers each."""
    rows = []
    for e in range(12):
        original = f"x = {e % 4}"
        for j in range(1, 6):
            rows.append({
                "question_id": f"eq{e}", "variant_id": f"eq{e}-v{j}",
                "question_original": f"Solve {e + 2}*x = {(e + 2) * (e % 4)}",
                "question_transformed": f"Find x if {e + 2}x = {(e + 2) * (e % 4)} ({j})",
                "answer_original": original, "answer_transformed": f"x = {(e + j * (e % 3)) % 4}",
                "source": "synthetic", "topic": "algebra", "language": "en", "slm_model": "fixed-slm",
            })
    return rows


def run_workflow(tmp: Path) -> dict:
    norm = normalizer(str(tmp))
    out: dict = {}
    config = tmp / "config.json"
    config.write_text(json.dumps(CONFIG), encoding="utf-8")
    server = FakeServer()

    def call(name, function, *arguments, files=()):
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            code = function(*arguments)
        out[name] = {"code": code, "stderr": re.sub(r"\d+:\d\d:\d\d", "H:MM:SS", norm(err.getvalue())),
                     "files": {}}
        for file in files:
            if Path(file).exists():
                text = Path(file).read_text(encoding="utf-8")
                out[name]["files"][Path(file).name] = (
                    [norm(json.loads(line)) for line in text.splitlines() if line.strip()]
                    if str(file).endswith(".jsonl") else norm(json.loads(text)))

    def fake_urlopen(request, timeout=None):
        return Response(server.reply(json.loads(request.data)))

    with patch("slm_answer_stability.model_tools.urlopen", side_effect=fake_urlopen):
        cfg = ["--config", str(config)]
        questions = tmp / "questions.jsonl"
        call("generate_questions", dataset_cli.main, "generate_questions",
             [*cfg, "--topic", "math", "--count", "5", "--batch-size", "2", "--output", str(questions)],
             files=[questions])
        call("generate_questions_exists", dataset_cli.main, "generate_questions",
             [*cfg, "--topic", "math", "--count", "1", "--output", str(questions)])
        call("paraphrases_unreviewed", dataset_cli.main, "generate_paraphrases",
             [*cfg, "--input", str(questions), "--count", "1", "--output", str(tmp / "unreviewed.jsonl")])
        review(questions, "question_review", reject_every=4, rename="q")

        general = tmp / "para_general.jsonl"
        call("paraphrases_general", dataset_cli.main, "generate_paraphrases",
             [*cfg, "--input", str(questions), "--count", "5", "--batch-size", "4", "--output", str(general)],
             files=[general, general.with_suffix(".log.jsonl")])
        for mode, targets in (("translation", ["English", "German"]), ("structure", ["JSON", "CSV"]),
                              ("style", ["formal", "casual"])):
            path = tmp / f"para_{mode}.jsonl"
            call(f"paraphrases_{mode}", dataset_cli.main, "generate_paraphrases",
                 [*cfg, "--input", str(questions), "--mode", mode, "--targets", *targets,
                  "--count", "3", "--output", str(path)],
                 files=[path, path.with_suffix(".log.jsonl")])
        combined = tmp / "para_combined.jsonl"
        call("paraphrases_combined", dataset_cli.main, "generate_paraphrases",
             [*cfg, "--input", str(questions), "--mode", "combined", "--count", "2",
              "--structures", "JSON", "CSV", "--styles", "formal", "casual",
              "--languages", "English", "German", "--existing", str(general),
              "--output", str(combined)],
             files=[combined, combined.with_suffix(".log.jsonl")])

        existing = tmp / "existing_questions.jsonl"
        write_jsonl(existing, [{"question_id": "old1", "question": "Existing question one?"},
                               {"question_id": "old2", "question": "Generated question 9.0?"}])
        semantic = tmp / "semantic.jsonl"
        call("generate_questions_semantic", dataset_cli.main, "generate_questions",
             [*cfg, "--topic", "math", "--count", "3", "--batch-size", "3", "--duplicate-check", "semantic",
              "--existing", str(existing), "--output", str(semantic)], files=[semantic])

        review(general, "paraphrase_review", reject_every=5)
        answers = tmp / "answers.jsonl"
        call("generate_answers", dataset_cli.main, "generate_answers",
             [*cfg, "--input", str(general), "--output", str(answers)], files=[answers])
        judged = tmp / "judged.jsonl"
        call("equivalence_ai", dataset_cli.main, "equivalence_generation_ai",
             [*cfg, "--input", str(answers), "--output", str(judged)], files=[judged])

    groups = tmp / "groups.jsonl"
    call("compute_k_n_original_only", dataset_cli.main, "compute_k_n",
         ["--input", str(judged), "--original-only", "--output", str(groups)], files=[groups])
    review(groups, "review_status")
    recomputed = tmp / "final.jsonl"
    call("recompute_finalize", dataset_cli.main, "compute_k_n",
         ["--input", str(groups), "--original-only", "--recompute", "--finalize",
          "--output", str(recomputed)], files=[recomputed])

    splits = tmp / "splits"
    call("check_dataset", dataset_prep.main, "check_dataset",
         ["--input", str(recomputed), "--min-questions", "2", "--min-variants-per-question", "1",
          "--report", str(tmp / "check.json")], files=[tmp / "check.json"])
    call("split_dataset", dataset_prep.main, "split_dataset",
         ["--input", str(recomputed), "--output-dir", str(splits), "--train", "0.6", "--val", "0.2",
          "--test", "0.2", "--seed", "5"],
         files=[splits / "train.jsonl", splits / "val.jsonl", splits / "test.jsonl",
                splits / "split_manifest.json"])
    csv = tmp / "import.csv"
    csv.write_text("question,topic\nWhat is 2+2?,math\nWhat is 3+3?,math\n", encoding="utf-8")
    call("import_questions", dataset_prep.main, "import_questions",
         ["--input", str(csv), "--output", str(tmp / "imported.jsonl"), "--source", "pilot"],
         files=[tmp / "imported.jsonl"])

    raw = tmp / "anchor_answers.jsonl"
    write_jsonl(raw, anchor_answers())
    pairwise = tmp / "pairwise.jsonl"
    call("equivalence_all_anchors", equivalence_main,
         ["--input", str(raw), "--output", str(pairwise)], files=[pairwise])
    legacy = tmp / "legacy.jsonl"
    call("equivalence_original_only", equivalence_main,
         ["--input", str(raw), "--output", str(legacy), "--original-only"], files=[legacy])
    call("equivalence_exists", equivalence_main, ["--input", str(raw), "--output", str(legacy)])
    for name, source, extra in (
        ("anchors_from_pairwise", pairwise, ["--seed", "7", "--train", "0.6", "--val", "0.2", "--test", "0.2"]),
        ("anchors_from_raw", raw, []),
        ("anchors_few_groups", raw, ["--train", "0.5", "--val", "0.3", "--test", "0.3"]),
    ):
        path = tmp / f"{name}.jsonl"
        call(name, dataset_cli.main, "compute_k_n",
             ["--input", str(source), "--output", str(path), *extra], files=[path])
    # Eleven groups with 0.6/0.2/0.2 leave tied fractional remainders, so the tie-break is pinned.
    tied = tmp / "tied_answers.jsonl"
    write_jsonl(tied, [row for row in anchor_answers() if int(row["question_id"][2:]) < 11])
    path = tmp / "anchors_tied_ratios.jsonl"
    call("anchors_tied_ratios", dataset_cli.main, "compute_k_n",
         ["--input", str(tied), "--output", str(path), "--seed", "3", "--train", "0.6",
          "--val", "0.2", "--test", "0.2"], files=[path])
    final = tmp / "final_eleven.jsonl"
    write_jsonl(final, [{
        "question_id": f"q{n}", "question": f"Question {n}?", "answer": str(n), "review_status": "approved",
        "variants": [{"variant_id": f"q{n}-v1", "question": f"Reworded {n}?", "answer": str(n),
                      "answer_equivalence": "equivalent"}],
        "k": 1, "N": 1, "target": 1.0, "unresolved": 0} for n in range(11)])
    eleven = tmp / "split_eleven"
    call("split_dataset_tied", dataset_prep.main, "split_dataset",
         ["--input", str(final), "--output-dir", str(eleven), "--train", "0.6", "--val", "0.2",
          "--test", "0.2", "--seed", "23"],
         files=[eleven / "train.jsonl", eleven / "val.jsonl", eleven / "test.jsonl"])
    anchor_splits = tmp / "anchor_splits"
    anchor_splits.mkdir()
    for split, ids in (("train", ["eq0", "eq1", "eq2", "eq3", "eq4", "eq5", "eq6", "eq7"]),
                       ("val", ["eq8", "eq9"]), ("test", ["eq10", "eq11"])):
        write_jsonl(anchor_splits / f"{split}.jsonl", [{"question_id": i} for i in ids])
    path = tmp / "anchors_given_splits.jsonl"
    call("anchors_given_splits", dataset_cli.main, "compute_k_n",
         ["--input", str(raw), "--splits-dir", str(anchor_splits), "--output", str(path)], files=[path])
    # Failure paths of question and paraphrase generation, each with a fresh server.
    def fresh(server_args):
        server.__init__(**server_args)

    with patch("slm_answer_stability.model_tools.urlopen", side_effect=fake_urlopen):
        cfg = ["--config", str(config)]
        fresh({"short_first_questions": True})
        short = tmp / "questions_short.jsonl"
        call("questions_incomplete_batch_recovers", dataset_cli.main, "generate_questions",
             [*cfg, "--topic", "math", "--count", "4", "--batch-size", "4", "--output", str(short)],
             files=[short])
        fresh({"bad_questions": True})
        call("questions_unrecoverable", dataset_cli.main, "generate_questions",
             [*cfg, "--topic", "math", "--count", "3", "--batch-size", "4",
              "--output", str(tmp / "questions_bad.jsonl")])
        for name, text in (("stall", "STALL question?"), ("netfail", "NETFAIL question?")):
            source = tmp / f"{name}_questions.jsonl"
            write_jsonl(source, [{"question_id": f"{name}1", "question": text, "question_review": "approved"}])
            for mode_name, extra in (("general", ["--count", "2"]),
                                     ("translation", ["--count", "2", "--mode", "translation", "--targets", "English"]),
                                     ("combined", ["--count", "2", "--mode", "combined", "--structures", "A", "B",
                                                   "--styles", "C", "D", "--languages", "E", "F"])):
                fresh({})
                path = tmp / f"{name}_{mode_name}.jsonl"
                call(f"paraphrases_{name}_{mode_name}", dataset_cli.main, "generate_paraphrases",
                     [*cfg, "--input", str(source), "--output", str(path), *extra],
                     files=[path.with_suffix(".log.jsonl")])
    return out


@unittest.skipUnless(GOLDEN.exists(), "record the golden file first")
class WorkflowGolden(unittest.TestCase):
    def test_outputs_are_unchanged(self):
        expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            actual = json.loads(json.dumps(run_workflow(Path(tmp))))
        self.assertEqual(list(actual), list(expected))
        for step in expected:
            with self.subTest(step):
                self.assertEqual(actual[step], expected[step])
                # Key order is part of the output files, so compare the serialized form too.
                self.assertEqual(json.dumps(actual[step], ensure_ascii=False),
                                 json.dumps(expected[step], ensure_ascii=False))


if __name__ == "__main__":
    if "--record" in sys.argv:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_workflow(Path(tmp))
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"recorded {GOLDEN}: " + ", ".join(f"{k}={v['code']}" for k, v in result.items()))
    else:
        unittest.main()
