"""Generate an exact number of unique Mathematics Dataset questions as JSONL."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


VENDORED_ROOT = Path(__file__).resolve().parents[1] / "datasets" / "mathematics_dataset"


def _modules():
    if not (VENDORED_ROOT / "mathematics_dataset" / "modules" / "modules.py").is_file():
        raise RuntimeError(f"Mathematics Dataset source missing: {VENDORED_ROOT}")
    sys.path.insert(0, str(VENDORED_ROOT))
    try:
        from mathematics_dataset.modules import modules
        from mathematics_dataset import generate_settings
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(f"Missing Mathematics Dataset dependency: {exc}; run uv sync") from exc
    return modules, generate_settings, np


def generate(topic: str, count: int, seed: int, output: Path,
             output_format: str = "plain") -> dict[str, int]:
    if count < 1:
        raise ValueError("--count must be positive")
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    if output_format not in ("plain", "structured"):
        raise ValueError("--format must be plain or structured")
    modules, settings, np = _modules()
    from mathematics_dataset import example
    random.seed(seed)
    np.random.seed(seed)
    groups = modules.train(lambda entropy: entropy)
    choices = {
        f"{group}.{name}": function
        for group, group_modules in groups.items()
        for name, function in group_modules.items()
    }
    selected = (
        {name: function for name, function in choices.items() if name.startswith(topic + ".")}
        if topic in groups else {topic: choices[topic]} if topic in choices else {}
    )
    if not selected:
        raise ValueError(f"unknown topic: {topic}; use --list-topics")
    selected_items = sorted(selected.items())
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    attempts = 0
    max_attempts = max(1000, count * 100)
    while len(rows) < count and attempts < max_attempts:
        attempts += 1
        module_name, function = random.choice(selected_items)
        if output_format == "structured":
            with example.capture_question_structure() as captured:
                problem = function()
        else:
            problem = function()
        question, answer = str(problem.question), str(problem.answer)
        if (len(question) > settings.MAX_QUESTION_LENGTH
                or len(answer) > settings.MAX_ANSWER_LENGTH or question in seen):
            continue
        seen.add(question)
        group, name = module_name.split(".", 1)
        row: dict[str, object] = {"question_id": f"math-{len(rows) + 1:06d}",
                                  "source": "mathematics_dataset", "topic": group,
                                  "module": name, "question": question, "answer": answer}
        if output_format == "structured":
            if not captured:
                raise RuntimeError(f"module {module_name} did not call example.question")
            row["structure"] = captured[-1]
        rows.append(row)
        counts[module_name] = counts.get(module_name, 0) + 1
    if len(rows) != count:
        raise RuntimeError(f"generated {len(rows)} unique questions after {attempts} attempts; no output written")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", help="Group (algebra) or module (algebra.linear_1d)")
    parser.add_argument("--count", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--format", choices=("plain", "structured"), default="plain")
    parser.add_argument("--list-topics", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.list_topics:
            modules, _, _ = _modules()
            for group, functions in modules.train(lambda entropy: entropy).items():
                print(group)
                for name in sorted(functions):
                    print(f"  {group}.{name}")
            return 0
        if not args.topic or args.count is None or args.output is None:
            parser.error("--topic, --count and --output are required unless --list-topics is used")
        counts = generate(args.topic, args.count, args.seed, args.output, args.format)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(args.output), "count": args.count,
                      "modules": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
