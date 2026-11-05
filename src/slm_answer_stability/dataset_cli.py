"""File-based, human-reviewed dataset workflow built on model_tools.

The stages live in ``generation``, ``annotation`` and ``data``; this module owns the command
line and re-exports the public names that scripts and tests import from here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from uuid import uuid4

from .annotation.equivalence_ai import generate_answer_equivalence_ai
from .data.anchors import compute_all_anchors, compute_k_n, finalize, recompute
from .data.io import (WorkflowError, append_log, normalize, read_jsonl, required_text,
                      reviewed, unique_ids, write_jsonl)
from .generation.answers import generate_answers
from .generation.client import call_model
from .generation.paraphrases import (TARGET_FILES, _combined_targets, generate_paraphrases,
                                     read_targets)
from .generation.questions import generate_questions
from .model_tools import PARAPHRASE_MODES, ModelToolError, load_config

__all__ = [
    "TARGET_FILES", "WorkflowError", "append_log", "call_model", "compute_all_anchors",
    "compute_k_n", "finalize", "generate_answer_equivalence_ai", "generate_answers",
    "generate_paraphrases", "generate_questions", "main", "normalize", "read_jsonl",
    "read_targets", "recompute", "required_text", "reviewed", "unique_ids", "write_jsonl",
]


def _add_args_generate_questions(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--topic", required=True)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--duplicate-check", choices=("exact", "semantic"), default="exact")
    parser.add_argument("--existing", type=Path)


def _add_args_generate_paraphrases(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--mode", choices=(*PARAPHRASE_MODES, "combined"), default="general")
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--structures", nargs="+")
    parser.add_argument("--styles", nargs="+")
    parser.add_argument("--languages", nargs="+")
    parser.add_argument("--existing", type=Path)
    parser.add_argument("--log-file", type=Path)


def _add_args_compute_k_n(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--original-only", action="store_true",
                        help="Compute one grouped k/N for each original question")
    parser.add_argument("--splits-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)


def _check_generate_questions(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not 1 <= args.count <= 10000:
        parser.error("--count must be between 1 and 10000")
    if not 1 <= args.batch_size <= 100:
        parser.error("--batch-size must be between 1 and 100")


def _check_generate_paraphrases(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not 1 <= args.count <= 100:
        parser.error("--count must be between 1 and 100")
    if args.mode == "combined" and args.count > 10:
        parser.error("--mode combined requires --count at most 10 (count cubed outputs)")
    if not 1 <= args.batch_size <= 100:
        parser.error("--batch-size must be between 1 and 100")
    if args.mode == "general" and args.targets:
        parser.error("--targets requires --mode translation, structure, or style")
    if args.mode == "combined" and args.targets:
        parser.error("--mode combined uses --structures, --styles, and --languages")
    if args.mode != "combined" and any((args.structures, args.styles, args.languages)):
        parser.error("--structures, --styles, and --languages require --mode combined")
    if args.mode == "combined":
        try:
            _combined_targets(args)
        except WorkflowError as exc:
            parser.error(str(exc))


def _check_compute_k_n(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.finalize and not args.recompute:
        parser.error("--finalize requires --recompute")
    if args.original_only and args.splits_dir is not None:
        parser.error("--splits-dir is only for the default all-anchor mode")
    if not args.original_only and (args.recompute or args.finalize):
        parser.error("--recompute and --finalize require --original-only")


# Checks that run before / after the shared "--input is required" check; the order is part of the CLI.
_CHECKS_BEFORE_INPUT = {"generate_questions": _check_generate_questions,
                        "generate_paraphrases": _check_generate_paraphrases}
_CHECKS_AFTER_INPUT = {"compute_k_n": _check_compute_k_n}
_ADD_ARGS = {"generate_questions": _add_args_generate_questions,
             "generate_paraphrases": _add_args_generate_paraphrases,
             "compute_k_n": _add_args_compute_k_n}


def _run_generate_questions(args, config):
    args.progress = True
    return generate_questions(args, config)


def _run_generate_paraphrases(args, config):
    args.progress = True
    args.log_file = args.log_file or args.output.with_suffix(".log.jsonl")
    args.run_id = uuid4().hex
    print(f"Paraphrase log: {args.log_file}", file=sys.stderr)
    return generate_paraphrases(args, config)


def _run_generate_answers(args, config):
    args.progress = True
    return generate_answers(args, config)


def _run_equivalence_generation_ai(args, config):
    return generate_answer_equivalence_ai(args, config)


def _run_compute_k_n(args, config):
    rows = (recompute(args) if args.recompute else compute_k_n(args)) if args.original_only else compute_all_anchors(
        args.input, args.splits_dir, args.seed, (args.train, args.val, args.test)
    )
    return finalize(rows) if args.finalize else rows


_RUN = {"generate_questions": _run_generate_questions,
        "generate_paraphrases": _run_generate_paraphrases,
        "generate_answers": _run_generate_answers,
        "equivalence_generation_ai": _run_equivalence_generation_ai,
        "compute_k_n": _run_compute_k_n}


def main(command: str, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    if command != "compute_k_n":
        parser.add_argument("--config", type=Path, default=Path("configs/model_tools.example.json"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    if command in _ADD_ARGS:
        _ADD_ARGS[command](parser)
    args = parser.parse_args(argv)
    if command in _CHECKS_BEFORE_INPUT:
        _CHECKS_BEFORE_INPUT[command](args, parser)
    if command != "generate_questions" and args.input is None:
        parser.error("--input is required")
    if command in _CHECKS_AFTER_INPUT:
        _CHECKS_AFTER_INPUT[command](args, parser)
    try:
        config = load_config(args.config) if command != "compute_k_n" else None
        if command not in _RUN:
            raise WorkflowError(f"Unknown command {command}")
        rows = _RUN[command](args, config)
        if command != "generate_answers":
            write_jsonl(args.output, rows)
    except (WorkflowError, ModelToolError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {len(rows)} records to {args.output}")
    return 0
