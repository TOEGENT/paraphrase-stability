"""Crude live check of repeated SLM answers at temperature zero."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

from slm_answer_stability.model_tools import ModelToolError, load_config, run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/model_tools.openai-local.json"))
    parser.add_argument("--question", default="в чём смысл жизни")
    parser.add_argument("--count", type=int, default=3)
    args = parser.parse_args()
    if args.count < 2:
        parser.error("--count must be at least 2")

    try:
        config = load_config(args.config)
        config["slm"] = {**config["slm"], "temperature": 0}
        print(f"Model: {config['slm']['model']}; temperature: 0; repeats: {args.count}")
        print(f"Question: {args.question}")
        answers = []
        for number in range(1, args.count + 1):
            started = perf_counter()
            answer = run(argparse.Namespace(command="answer", question=args.question), config)["answer"]
            answers.append(answer)
            print(f"{number}/{args.count} ({perf_counter() - started:.1f}s): {json.dumps(answer, ensure_ascii=False)}", flush=True)
    except ModelToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    distinct = len(set(answers))
    print(f"Distinct answers: {distinct}/{args.count}")
    return 0 if distinct == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
