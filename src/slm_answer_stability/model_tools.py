"""Small CLI for the three model roles used while building the dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


STATUSES = {"equivalent", "non_equivalent", "unclear", "insufficient_context"}


class ModelToolError(Exception):
    """A configuration, model, or response error suitable for the CLI."""


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelToolError(f"Cannot read config {path}: {exc}") from exc
    if not isinstance(config, dict) or not isinstance(config.get("base_url"), str):
        raise ModelToolError("Config needs a string base_url")
    for role in ("slm", "generator", "judge"):
        value = config.get(role)
        if not isinstance(value, dict) or not isinstance(value.get("model"), str) or not value["model"].strip():
            raise ModelToolError(f"Config needs {role}.model")
    return config


def chat(config: dict[str, Any], role: str, system: str, user: str, *, structured: bool) -> str:
    settings = config[role]
    options = {
        "temperature": settings.get("temperature", 0.0),
        "seed": settings.get("seed", 0),
        "num_predict": settings.get("num_predict", 256),
    }
    payload: dict[str, Any] = {
        "model": settings["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": options,
    }
    if structured:
        payload["format"] = "json"
    url = config["base_url"].rstrip("/") + "/api/chat"
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=float(config.get("timeout_seconds", 120))) as response:
            result = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise ModelToolError(f"{role} request failed (HTTP {exc.code}): {detail}") from exc
    except (URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise ModelToolError(f"{role} request failed at {url}: {exc}") from exc
    message = result.get("message") if isinstance(result, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ModelToolError(f"{role} returned no message content")
    return content.strip()


def parse_object(raw: str, role: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelToolError(f"{role} returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelToolError(f"{role} must return a JSON object")
    return value


def positive_count(value: str) -> int:
    count = int(value)
    if count < 1 or count > 100:
        raise argparse.ArgumentTypeError("count must be between 1 and 100")
    return count


def run(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    if args.command == "answer":
        raw = chat(
            config,
            "slm",
            config["slm"].get("system", "Answer the question directly and briefly."),
            args.question,
            structured=False,
        )
        return {"answer": raw, "model": config["slm"]["model"]}

    if args.command == "questions":
        raw = chat(
            config,
            "generator",
            "Create unambiguous questions that each require one short final answer. "
            "Return JSON with a questions array of objects containing question and topic. "
            "These are synthetic tasks; do not claim they came from a real dataset.",
            f"Topic: {args.topic}\nNumber of questions: {args.count}",
            structured=True,
        )
        value = parse_object(raw, "generator")
        rows = value.get("questions")
        if not isinstance(rows, list) or len(rows) != args.count or any(
            not isinstance(row, dict) or not isinstance(row.get("question"), str) or not row["question"].strip()
            for row in rows
        ):
            raise ModelToolError("generator returned the wrong number of questions or an invalid question")
        return {"questions": rows, "source": "synthetic", "model": config["generator"]["model"]}

    if args.command == "paraphrases":
        raw = chat(
            config,
            "generator",
            "Rewrite the question without changing facts, constraints, negation, or answer format. "
            "Return JSON with a variants array of objects containing question and transformation_type. "
            "Use distinct lexical, syntactic, or stylistic transformations; do not solve the question.",
            f"Original question: {args.question}\nNumber of variants: {args.count}",
            structured=True,
        )
        value = parse_object(raw, "generator")
        rows = value.get("variants")
        if not isinstance(rows, list) or len(rows) != args.count or any(
            not isinstance(row, dict) or not isinstance(row.get("question"), str) or not row["question"].strip()
            or not isinstance(row.get("transformation_type"), str)
            for row in rows
        ):
            raise ModelToolError("generator returned the wrong number of variants or an invalid variant")
        if len({row["question"].strip() for row in rows} | {args.question.strip()}) != args.count + 1:
            raise ModelToolError("generator returned a duplicate or the original question")
        return {"variants": rows, "model": config["generator"]["model"]}

    if args.command == "judge-question":
        system = (
            "Decide whether the rewritten question preserves all facts, constraints, negation, "
            "and the requested answer of the original question. Return JSON with status and rationale_short. "
            "status must be equivalent, non_equivalent, unclear, or insufficient_context."
        )
        user = f"Original question: {args.question}\nRewritten question: {args.rewritten_question}"
    else:
        system = (
            "Decide whether two answers have the same meaning in the context of the original question. "
            "Correctness is separate from equivalence. Return JSON with status and rationale_short. "
            "status must be equivalent, non_equivalent, unclear, or insufficient_context."
        )
        user = (
            f"Question: {args.question}\nOriginal answer: {args.answer}\n"
            f"Answer to rewritten question: {args.rewritten_answer}"
        )
    value = parse_object(chat(config, "judge", system, user, structured=True), "judge")
    if value.get("status") not in STATUSES or not isinstance(value.get("rationale_short"), str):
        raise ModelToolError("judge returned an invalid status or missing rationale_short")
    return {"status": value["status"], "rationale_short": value["rationale_short"], "model": config["judge"]["model"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/model_tools.example.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    answer = commands.add_parser("answer", help="Get a fixed SLM answer")
    answer.add_argument("--question", required=True)
    questions = commands.add_parser("questions", help="Generate synthetic source questions")
    questions.add_argument("--topic", required=True)
    questions.add_argument("--count", required=True, type=positive_count)
    paraphrases = commands.add_parser("paraphrases", help="Generate meaning-preserving rewrites")
    paraphrases.add_argument("--question", required=True)
    paraphrases.add_argument("--count", required=True, type=positive_count)
    judge_question = commands.add_parser("judge-question", help="Check question equivalence")
    judge_question.add_argument("--question", required=True)
    judge_question.add_argument("--rewritten-question", required=True)
    judge_answer = commands.add_parser("judge-answer", help="Check answer equivalence")
    judge_answer.add_argument("--question", required=True)
    judge_answer.add_argument("--answer", required=True)
    judge_answer.add_argument("--rewritten-answer", required=True)
    args = parser.parse_args(argv)
    try:
        result = run(args, load_config(args.config))
    except ModelToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
