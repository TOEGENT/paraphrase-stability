# Refactor report

Branch `refactor`. Scope: structure and imports only. Nothing that reaches the SLM or the MLP was meant to change, and the checks below were built to prove that.

## Why

- `dataset_cli.py` (907 lines) held I/O helpers, five workflows, k/N and split logic and the argparse for four commands. Four other modules imported their basics from it.
- Real duplication: split allocation twice, anchor-row construction twice, JSONL reading and "output must not exist" checks in five variants, about 14 hardcoded status strings.
- `src/` imported from `scripts/` (with `try/except ModuleNotFoundError` fallbacks).
- Retry decisions in paraphrase and question generation were made by substring-matching exception messages.
- `model_tools.run(Namespace, config)` dispatched on a string, and `scripts/group_cv.py` carried real logic outside `src/`.
- `sympy` and `scikit-learn` were used but not declared; the install hint named an extra (`.[mlp]`) that does not exist.

## What changed

### `model_tools.py` and `group_cv`
- Each role is now a plain function: `answer`, `generate_questions`, `generate_paraphrases`, `judge_question`, `judge_answer`. `run(Namespace, config)` is a thin dispatcher with the same defaults. Prompt text and payloads are unchanged.
- `chat()` is split into `_build_request` and `_parse_response`.
- Typed errors `InvalidModelJSON` (carries the role) and `WrongQuestionCount`; messages are unchanged.
- `group_cv` moved to `src/slm_answer_stability/evaluation/group_cv.py`; `scripts/group_cv.py` is a 7-line wrapper and the command line is unchanged. Only imports changed (`DIMENSION` instead of a hardcoded 768, sklearn imported lazily for `--extended`). `root()` and its `eval` are deliberately untouched, so the numbers in `scripts/README/role_4_mlp/REPORT.md` stay reproducible.
- `text.normalize` is the single whitespace normalizer.

### `dataset_cli.py` split (now 172 lines: argparse plus re-exports)

| New module | Contents |
|---|---|
| `constants.py` | status tuples, `SPLITS`, `ANCHOR_METADATA_KEYS` |
| `data/io.py` | `WorkflowError` (now a `ValueError` subclass), `read_jsonl`, `write_jsonl`, `append_log`, `required_text`, `reviewed`, `unique_ids`, `ensure_new_output` |
| `data/splits.py` | `partition_keys`, `validate_ratios` (one allocation, was copy-pasted) |
| `data/anchors.py` | `compute_k_n`, `compute_all_anchors`, `recompute`, `finalize`, shared `_anchor_row` |
| `generation/client.py` | `call_model`, `InvalidGeneratorJSON`, `IncompleteBatch` |
| `generation/questions.py`, `paraphrases.py`, `answers.py` | the three generation stages |
| `annotation/answer_extraction.py` | `extract_final_x`, `verify_pair` (moved out of `scripts/`) |
| `annotation/equivalence_ai.py` | LLM answer judge |

- `dataset_cli` re-exports every name that scripts and tests imported before, so old imports keep working.
- `generation/paraphrases.py`: the 150-line function is split into `_generate_for_question`, `_generate_combined_question`, `_request_batch`, `_existing_variants`, with helpers `VariantIdAllocator`, `DedupTracker`, `ParaphraseLogger`, `_variant_row`. Retries use `except InvalidGeneratorJSON` / `IncompleteBatch` instead of message text.
- `scripts/equivalence_generation.py`: re-exports the moved functions; `verify_file` and `verify_all_anchors` share `_check_paths`, `_answer_rows`, `_write_rows` (messages with line numbers kept).
- Dependencies: `scikit-learn>=1.3` and `sympy>=1.14` declared; `uv.lock` regenerated; install hints now say `uv sync (or pip install -e .)`.

## Intentional behaviour differences (all error paths or wording)

- An unknown `model_tools.run` command raises `ModelToolError` (was `AttributeError`).
- `WorkflowError` subclasses `ValueError`.
- `check_dataset --report` on an existing file says "...; choose a new file".
- Install-hint wording.
- Two tests that hand-built the "invalid JSON" error now raise `InvalidGeneratorJSON`, and all `call_model` patches target `slm_answer_stability.generation.client.call_model`.
- `uv.lock`: no package version changed (72 packages before and after); uv 0.11.0 re-wrote dependency edges and platform markers, mainly the nvidia packages pulled in by `torch`.

## How "no behaviour change" was checked

All checks use a fake model server and synthetic data. The real SLM server, real `work/` data and GPU runs were **not** used.

- `tests/test_model_tools_snapshots.py` + `tests/golden/model_tools_requests.json`: exact URL, headers, body and result for 11 commands x 2 API styles, recorded from the unmodified code.
- `tests/test_workflow_golden.py` + `tests/golden/workflow_outputs.json`: about 30 steps through the real CLI entry points (questions, paraphrases in every mode, answers, AI equivalence, k/N, recompute/finalize, splits, import, pairwise anchors, retry/stall/network-error paths). Every output file, log event, error message and key order is compared. Recorded from the original `dataset_cli`.
- `tests/test_cli_validation.py`: every argparse message and the order in which checks win.
- `tests/test_group_cv.py` + `tests/golden/group_cv_summary.json`: all three CV modes on synthetic data, recorded from the original script.
- `tests/test_client_errors.py`, `tests/test_anchor_rows.py`: the new error mapping and the pairwise `question_id` pass-through.
- About 30 deliberate source mutations (tie-breaks, log fields, id allocation, retry exception types, `k`, ...) were each caught by the suite.
- Static check: `ruff --select F` found a real bug (missing import in the semantic-duplicate path), now fixed and covered.

Suite: 68 tests. 3 errors remain, identical on the original code: they need the `datasets/mathematics_dataset` submodule, which is empty because the repo has gitlinks but no `.gitmodules`.

To re-record a golden file (only for an intended behaviour change):
`PYTHONPATH=src:. python tests/test_workflow_golden.py --record` (similarly `tests/test_model_tools_snapshots.py --record`, `tests/test_group_cv.py --record`). Use `PYTHONDONTWRITEBYTECODE=1` when mutating sources quickly: a same-size edit within one second can be hidden by a stale `.pyc`.

## Not done / follow-ups

- No linter, formatter or CI config; the README links to missing `docs/`, `prompts/` and `README_*.md` files; no `.gitmodules`.
- `tests/test_dataset_cli.py` (525 lines) was not split per module; only its patch targets changed.
- k/N: only the anchor-row builder is shared; the raw-rows and pairwise validation loops are still separate.
- `group_cv.fit_mlp` and `models/training.py` still duplicate a training loop (merging would change reported numbers); `eval` in `group_cv.root()` remains.
- `scripts/test_temperature_zero.py` is a utility with a test-like name.
- Pre-existing lint notes: unused `SPLITS` import in `models/training.py`, unused imports in two tests.
- Suggested before relying on it: one small real run (real server, real data) before vs after, same seed, diff the outputs.
