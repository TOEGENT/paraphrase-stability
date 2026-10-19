"""Convert DefAn's adjacent 15-question blocks to paraphrases JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DOMAIN_TOPICS = {
    1: "sports",
    2: "census_australia",
    3: "nobel_prize",
    4: "entertainment",
    5: "world_organizations",
    6: "qs_ranking",
    7: "conference_venue",
}
GROUP_SIZE = 15


def convert_file(path: Path, domain: int) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig") as source:
        rows = json.load(source)
    if not isinstance(rows, list) or not rows or len(rows) % GROUP_SIZE:
        raise ValueError(f"{path}: expected a nonempty array of complete {GROUP_SIZE}-row groups")

    output = []
    for offset in range(0, len(rows), GROUP_SIZE):
        group = rows[offset:offset + GROUP_SIZE]
        questions = []
        for index, row in enumerate(group, offset + 1):
            if not isinstance(row, dict) or not isinstance(row.get("questions"), str):
                raise ValueError(f"{path}: row {index} has no question text")
            question = row["questions"].strip()
            if not question:
                raise ValueError(f"{path}: row {index} has an empty question")
            questions.append(question)

        question_id = f"defan-d{domain}-g{offset // GROUP_SIZE + 1:05d}"
        original = questions[0]
        for number, question in enumerate(questions[1:], 1):
            output.append({
                "question_id": question_id,
                "variant_id": f"{question_id}-v{number}",
                "question_original": original,
                "question_transformed": question,
                "transformation_type": "defan",
                "source": "defan",
                "topic": DOMAIN_TOPICS[domain],
                "language": "en",
                "paraphrase_review": "pending",
            })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("datasets/DefAn/DefAn-public"))
    parser.add_argument("--output", type=Path, default=Path("work/paraphrases_defan.jsonl"))
    args = parser.parse_args()

    try:
        if args.output.exists():
            raise ValueError(f"Output already exists: {args.output}")
        rows = []
        for domain in DOMAIN_TOPICS:
            path = args.input_dir / f"QA_domain_{domain}_public.json"
            rows.extend(convert_file(path, domain))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as target:
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        print(f"DefAn import failed: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(rows)} paraphrases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
