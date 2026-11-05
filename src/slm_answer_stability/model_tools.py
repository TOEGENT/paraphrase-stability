"""Small CLI for the three model roles used while building the dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .constants import EQUIVALENCE_STATUSES
from .text import normalize


STATUSES = set(EQUIVALENCE_STATUSES)

PARAPHRASE_SYSTEM_PROMPT = (
    "Create standalone paraphrases in the same language as the original question. "
    "Preserve every entity, number, unit, condition, operation order, negation, and the requested answer. "
    "Change the wording substantially: move the requested quantity to the beginning, reorder clauses, "
    "switch between a question and an instruction, change sentence structure, or use an exactly "
    "equivalent mathematical notation where appropriate. Choose natural constructions rather than "
    "just replacing one word with a synonym. Do not add facts, solve the task, or copy a previous variant. "
    "Examples of allowed changes (illustrations only; do not reuse their facts or wording):\n"
    "Original: В коробке 12 красных шаров и 8 синих. Сколько шаров всего?\n"
    "Variants: Сколько всего шаров в коробке, если красных в ней 12, а синих 8? | "
    "Коробка содержит 8 синих и 12 красных шаров; найдите общее число шаров.\n"
    "Original: Чему равна сумма 18 и произведения 4 на 6?\n"
    "Variants: Вычислите 18 + (4 × 6). | Если к 18 прибавить результат умножения 4 на 6, "
    "какое число получится?\n"
    "Original: Какой город является столицей Италии?\n"
    "Variants: Назовите столицу Италии. | Укажите город, являющийся столицей Италии.\n"
    "Return JSON with a variants array of objects containing question and transformation_type. "
    "Each output must differ from the supplied existing paraphrases, including near-identical wording."
)

PARAPHRASE_RETRY_GUIDANCE = (
    "Use a new wording pattern for this attempt.",
    "Move the requested quantity to the beginning and place the conditions afterward.",
    "Change the sentence type or clause structure; avoid a simple synonym substitution.",
    "Use an exact symbolic formulation if suitable, or a different natural-language construction.",
    "Choose a different grammatical subject and reorder the facts without changing them.",
)

PARAPHRASE_MODES = ("general", "translation", "structure", "style")
PARAPHRASE_MODE_GUIDANCE = {
    "translation": (
        "Translate the entire question into the requested target language. "
        "Use that language for both facts and the final request; preserve names, numbers, "
        "units, logical relationships, and answer format. Do not include the original text."
    ),
    "structure": (
        "Recast the question in the requested structure. Use a real table, CSV, JSON, "
        "or list when requested, including every fact and an explicit request for the answer. "
        "The structure is part of the question text, not the response envelope."
    ),
    "style": (
        "Rewrite the whole question in the requested register. Change diction and sentence "
        "construction substantially while keeping every fact and the requested answer. "
        "Do not add a story, explanation, solution, or new assumptions."
    ),
}


class ModelToolError(Exception):
    """A configuration, model, or response error suitable for the CLI."""


class InvalidModelJSON(ModelToolError):
    """The model's reply could not be parsed as JSON."""

    def __init__(self, role: str, detail: Exception):
        super().__init__(f"{role} returned invalid JSON: {detail}")
        self.role = role


class WrongQuestionCount(ModelToolError):
    """The generator returned a different number of questions than requested."""


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelToolError(f"Cannot read config {path}: {exc}") from exc
    if not isinstance(config, dict) or not isinstance(config.get("base_url"), str):
        raise ModelToolError("Config needs a string base_url")
    if config.get("api_style", "ollama") not in ("ollama", "openai"):
        raise ModelToolError("api_style must be ollama or openai")
    for role in ("slm", "generator", "judge"):
        value = config.get(role)
        if not isinstance(value, dict) or not isinstance(value.get("model"), str) or not value["model"].strip():
            raise ModelToolError(f"Config needs {role}.model")
    return config


def _build_request(config: dict[str, Any], role: str, system: str, user: str,
                   structured: bool) -> tuple[str, dict[str, Any]]:
    """Return the endpoint URL and JSON payload for one chat call."""
    settings = config[role]
    api_style = config.get("api_style", "ollama")
    payload: dict[str, Any] = {
        "model": settings["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }
    if api_style == "ollama":
        payload["options"] = {
            "temperature": settings.get("temperature", 0.0),
            "seed": settings.get("seed", 0),
            "num_predict": settings.get("num_predict", 256),
        }
        if structured:
            payload["format"] = "json"
        url = config["base_url"].rstrip("/") + "/api/chat"
    else:
        payload.update({
            "temperature": settings.get("temperature", 0.0),
            "seed": settings.get("seed", 0),
            "max_tokens": settings.get("num_predict", 256),
        })
        if "chat_template_kwargs" in settings:
            payload["chat_template_kwargs"] = settings["chat_template_kwargs"]
        if structured:
            payload["response_format"] = {"type": "json_object"}
        url = config["base_url"].rstrip("/") + "/chat/completions"
    return url, payload


def _parse_response(result: Any, api_style: str) -> tuple[Any, Any]:
    """Return (message content, finish reason) from a decoded chat response."""
    if api_style == "ollama":
        message = result.get("message") if isinstance(result, dict) else None
        finish_reason = result.get("done_reason") if isinstance(result, dict) else None
    else:
        choices = result.get("choices") if isinstance(result, dict) else None
        first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
        message = first.get("message") if first is not None else None
        finish_reason = first.get("finish_reason") if first is not None else None
    content = message.get("content") if isinstance(message, dict) else None
    return content, finish_reason


def chat(config: dict[str, Any], role: str, system: str, user: str, *, structured: bool,
         trace: Callable[[dict[str, Any]], None] | None = None) -> str:
    api_style = config.get("api_style", "ollama")
    url, payload = _build_request(config, role, system, user, structured)
    if trace is not None:
        trace({"event": "request", "url": url, "payload": payload})
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
    content, finish_reason = _parse_response(result, api_style)
    if trace is not None:
        trace({"event": "response", "content": content, "finish_reason": finish_reason})
    if not isinstance(content, str) or not content.strip():
        raise ModelToolError(f"{role} returned no message content")
    return content.strip()


def parse_object(raw: str, role: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidModelJSON(role, exc) from exc
    if not isinstance(value, dict):
        raise ModelToolError(f"{role} must return a JSON object")
    return value


def positive_count(value: str) -> int:
    count = int(value)
    if count < 1 or count > 100:
        raise argparse.ArgumentTypeError("count must be between 1 and 100")
    return count


def answer(config: dict[str, Any], question: str) -> dict[str, Any]:
    raw = chat(
        config,
        "slm",
        config["slm"].get("system", "Answer the question directly and briefly."),
        question,
        structured=False,
    )
    return {"answer": raw, "model": config["slm"]["model"]}


def generate_questions(config: dict[str, Any], topic: str, count: int,
                       existing_questions: list[str] | None = None) -> dict[str, Any]:
    if existing_questions is None:
        existing_questions = []
    if not isinstance(existing_questions, list) or any(
        not isinstance(question, str) for question in existing_questions
    ):
        raise ModelToolError("existing_questions must be a list of question strings")
    raw = chat(
        config,
        "generator",
        "Create unambiguous questions that each require one short final answer. "
        "Write questions in the same language as the topic. "
        "For mathematical expressions, use explicit parentheses whenever operation order could be ambiguous. "
        "Make the new questions distinct from the supplied existing questions, including their problem "
        "types and solution patterns; changing only numbers or names is not enough. "
        "Also vary the questions within the new batch. "
        "Return JSON with a questions array of objects containing question and topic. "
        "These are synthetic tasks; do not claim they came from a real dataset.",
        f"Topic: {topic}\nNumber of questions: {count}\n"
        f"Existing questions to avoid: {json.dumps(existing_questions, ensure_ascii=False)}",
        structured=True,
    )
    value = parse_object(raw, "generator")
    rows = value.get("questions")
    if not isinstance(rows, list) or len(rows) != count or any(
        not isinstance(row, dict) or not isinstance(row.get("question"), str) or not row["question"].strip()
        for row in rows
    ):
        raise WrongQuestionCount("generator returned the wrong number of questions or an invalid question")
    return {"questions": rows, "source": "synthetic", "model": config["generator"]["model"]}


def generate_paraphrases(
    config: dict[str, Any], question: str, count: int, *, mode: str = "general",
    target: str | None = None, existing_paraphrases: list[str] | None = None,
    variation_attempt: int = 0, reference_question: str | None = None,
    preserve_structure: bool = False, allow_partial: bool = False,
    allow_duplicates: bool = False,
    model_trace: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if existing_paraphrases is None:
        existing_paraphrases = []
    if mode not in PARAPHRASE_MODES or (mode == "general") != (target is None):
        raise ModelToolError("Invalid paraphrase mode or target")
    if target is not None and (not isinstance(target, str) or not target.strip()):
        raise ModelToolError("Paraphrase target must be a non-empty string")
    if not isinstance(existing_paraphrases, list) or any(
        not isinstance(value, str) for value in existing_paraphrases
    ):
        raise ModelToolError("existing_paraphrases must be a list of strings")
    system_prompt = PARAPHRASE_SYSTEM_PROMPT
    if mode != "general":
        system_prompt = (
            "Create standalone, semantically equivalent versions of the original question. "
            "Preserve every entity, number, unit, condition, operation order, negation, "
            "and the requested answer. Do not add facts, hints, or a solution. "
            f"{PARAPHRASE_MODE_GUIDANCE[mode]} "
            "Return JSON with a variants array of objects containing question and "
            "transformation_type. Each variant must differ from previous variants."
        )
        if preserve_structure:
            system_prompt += (
                " Preserve the input's table, CSV, JSON, or list structure. "
                "For JSON, keep structural keys and translate or restyle human-readable values."
            )
    user_prompt = (
        f"Original question: {question}\nNumber of variants: {count}\n"
        + (f"Transformation mode: {mode}\nTarget: {target}\n" if target is not None else "")
        + f"Existing paraphrases to avoid: {json.dumps(existing_paraphrases, ensure_ascii=False)}\n"
        + f"Variation attempt: {variation_attempt}. "
        + PARAPHRASE_RETRY_GUIDANCE[min(variation_attempt, len(PARAPHRASE_RETRY_GUIDANCE) - 1)]
    )
    if reference_question:
        user_prompt += ("\nSource question for checking facts only: "
                        f"{reference_question}\nTransform the immediate input above.")
    raw = chat(
        config,
        "generator",
        system_prompt,
        user_prompt,
        structured=True,
        trace=model_trace,
    )
    value = parse_object(raw, "generator")
    rows = value.get("variants")
    raw_variant_count = len(rows) if isinstance(rows, list) else 0
    valid = [] if not isinstance(rows, list) else [row for row in rows if (
        isinstance(row, dict) and isinstance(row.get("question"), str) and row["question"].strip()
        and isinstance(row.get("transformation_type"), str) and row["transformation_type"].strip()
    )]
    if allow_partial:
        rows = valid
    elif not isinstance(rows, list) or len(rows) != count or len(valid) != count:
        raise ModelToolError("generator returned the wrong number of variants or an invalid variant")
    known = {normalize(value) for value in [question, *existing_paraphrases]}
    generated = {normalize(row["question"]) for row in rows}
    if not allow_duplicates and (len(generated) != count or known & generated):
        raise ModelToolError("generator returned a duplicate or the original question")
    return {"variants": rows, "model": config["generator"]["model"],
            "raw_variant_count": raw_variant_count,
            "invalid_variant_count": raw_variant_count - len(valid)}


def _judge(config: dict[str, Any], system: str, user: str) -> dict[str, Any]:
    value = parse_object(chat(config, "judge", system, user, structured=True), "judge")
    if value.get("status") not in STATUSES or not isinstance(value.get("rationale_short"), str):
        raise ModelToolError("judge returned an invalid status or missing rationale_short")
    return {"status": value["status"], "rationale_short": value["rationale_short"], "model": config["judge"]["model"]}


def judge_question(config: dict[str, Any], question: str, rewritten_question: str) -> dict[str, Any]:
    system = (
        "Decide whether the rewritten question preserves all facts, constraints, negation, "
        "and the requested answer of the original question. Return JSON with status and rationale_short. "
        "status must be equivalent, non_equivalent, unclear, or insufficient_context."
    )
    user = f"Original question: {question}\nRewritten question: {rewritten_question}"
    return _judge(config, system, user)


def judge_answer(config: dict[str, Any], question: str, answer: str, rewritten_answer: str) -> dict[str, Any]:
    system = (
        "Decide whether two answers have the same meaning in the context of the original question. "
        "Correctness is separate from equivalence. Return JSON with status and rationale_short. "
        "status must be equivalent, non_equivalent, unclear, or insufficient_context."
    )
    user = (
        f"Question: {question}\nOriginal answer: {answer}\n"
        f"Answer to rewritten question: {rewritten_answer}"
    )
    return _judge(config, system, user)


def run(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one CLI-style command; optional arguments are read from ``args`` with defaults."""
    if args.command == "answer":
        return answer(config, args.question)
    if args.command == "questions":
        return generate_questions(
            config, args.topic, args.count, getattr(args, "existing_questions", []))
    if args.command == "paraphrases":
        return generate_paraphrases(
            config, args.question, args.count,
            mode=getattr(args, "mode", "general"),
            target=getattr(args, "target", None),
            existing_paraphrases=getattr(args, "existing_paraphrases", []),
            variation_attempt=getattr(args, "variation_attempt", 0),
            reference_question=getattr(args, "reference_question", None),
            preserve_structure=getattr(args, "preserve_structure", False),
            allow_partial=getattr(args, "allow_partial", False),
            allow_duplicates=getattr(args, "allow_duplicates", False),
            model_trace=getattr(args, "model_trace", None),
        )
    if args.command == "judge-question":
        return judge_question(config, args.question, args.rewritten_question)
    if args.command == "judge-answer":
        return judge_answer(config, args.question, args.answer, args.rewritten_answer)
    raise ModelToolError(f"Unknown command {args.command}")


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
    paraphrases.add_argument("--mode", choices=PARAPHRASE_MODES, default="general")
    paraphrases.add_argument("--target")
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
