"""Generate repeated-letter variants from imported DefAn paraphrases JSONL.

Run from the repository root (first import DefAn if the JSONL does not exist):

    uv run python scripts/import_defan_paraphrases.py --output work/paraphrases_defan.jsonl
    uv run python scripts/generate_defan_letter_repeats.py --input work/paraphrases_defan.jsonl --output work/letter_repeats_defan.jsonl --count 100

The output uses the generate_answers.py variant schema. Review the variants and
change paraphrase_review from "pending" to "approved" before generating answers.
Run the same command again to resume an interrupted output file. Generation
loads input into memory but yields and saves each generated variant immediately.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
import os
from pathlib import Path
from typing import Iterable


def eligible_positions(question: str) -> list[int]:
    """Return ASCII-letter positions except the first letter of the first word."""
    positions = [index for index, char in enumerate(question) if char.isascii() and char.isalpha()]
    return positions[1:]


def repeat_letters(question: str, positions: tuple[int, ...]) -> str:
    selected = set(positions)
    return "".join(char * 2 if index in selected else char for index, char in enumerate(question))


def iter_generated_rows(rows: list[dict], count: int) -> Iterable[dict]:
    originals: dict[str, dict] = {}
    for row in rows:
        question_id = row["question_id"]
        question = row["question_original"]
        if question_id in originals:
            if originals[question_id]["question_original"] != question:
                raise ValueError(f"Conflicting original question for {question_id}")
        else:
            originals[question_id] = row

    for question_id, source in originals.items():
        question = source["question_original"]
        positions = eligible_positions(question)
        if not positions:
            continue
        number = 0
        seen: set[str] = set()
        for size in range(1, len(positions) + 1):
            for selection in combinations(positions, size):
                transformed = repeat_letters(question, selection)
                if transformed in seen:
                    continue
                seen.add(transformed)
                number += 1
                yield {
                    "question_id": question_id,
                    "variant_id": f"{question_id}-repeat-v{number}",
                    "question_original": question,
                    "question_transformed": transformed,
                    "transformation_type": "letter_repeat",
                    "transformation_mode": "letter_repeat",
                    "repeat_positions": list(selection),
                    "source": source.get("source", "defan"),
                    "topic": source.get("topic"),
                    "language": source.get("language", "en"),
                    "paraphrase_review": "pending",
                }
                if number == count:
                    break
            if number == count:
                break


def generate_rows(rows: list[dict], count: int) -> list[dict]:
    """Materialize variants for callers that need a list, such as tests."""
    return list(iter_generated_rows(rows, count))


def save_with_resume(path: Path, rows: Iterable[dict]) -> tuple[int, int, int]:
    """Persist each row; on restart verify and skip the already saved prefix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = 0
    total = 0
    questions: set[str] = set()
    appending = False
    with path.open("a+b") as stream:
        stream.seek(0)
        for row in rows:
            total += 1
            questions.add(row["question_id"])
            if not appending:
                start = stream.tell()
                line = stream.readline()
                if line and not line.endswith(b"\n"):
                    # A crash may leave one incomplete final record; regenerate it.
                    stream.truncate(start)
                    stream.flush()
                    os.fsync(stream.fileno())
                    line = b""
                if line:
                    try:
                        existing = json.loads(line)
                    except (UnicodeError, json.JSONDecodeError) as exc:
                        raise ValueError(f"Invalid saved JSONL row {total} in {path}") from exc
                    if existing != row:
                        raise ValueError(f"Saved JSONL row {total} does not match generated data: {path}")
                    saved += 1
                    continue
                appending = True
                stream.seek(0, os.SEEK_END)
            stream.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        if not appending and stream.readline():
            raise ValueError(f"Saved JSONL has more rows than generated data: {path}")
    return saved, total, len(questions)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("work/paraphrases_defan.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("work/letter_repeats_defan.jsonl"))
    parser.add_argument("--count", type=int, default=100, help="Maximum variants per original question")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")
    with args.input.open(encoding="utf-8-sig") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    try:
        saved, total, questions = save_with_resume(
            args.output, iter_generated_rows(rows, args.count)
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Saved {total} variants ({saved} already present) for "
          f"{questions} questions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
