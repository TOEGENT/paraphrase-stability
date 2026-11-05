# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research scaffold (package `slm_answer_stability`, in `src/`) that tests whether a small MLP can predict, from the embeddings of one (question, SLM answer) pair, how stable that answer is under meaning-preserving paraphrases. The label is `target = k/N`: for an "anchor" answer, `N` is the number of *other* paraphrases of the same task that have answers, and `k` is how many of those answers are equivalent to the anchor. The MLP does a **regression** on `k/N` (not classification, and not a correctness check). The README and most docs are in Russian.

## Commands

Dependencies are managed with `uv` (`uv sync`; Python >=3.10; torch, numpy, sentence-transformers). There is no linter or build step configured.

```
uv run python -m unittest discover -s tests          # all tests
uv run python -m unittest tests.test_dataset_cli     # one module
uv run python -m unittest tests.test_dataset_cli.SomeClass.test_name   # one test
```

Tests use `unittest` (not pytest). `tests/test_mlp_pipeline.py` skips itself if numpy/torch are missing. `tests/README.md` is a stale placeholder.

Run everything from the repo root. Scripts in `scripts/` are thin entry points: most call `slm_answer_stability.dataset_cli.main("<command>")`, so the logic lives in `src/`, not in the scripts.

## Pipeline architecture

Two routes share the same stages: source questions → paraphrases/variants → SLM answers → answer-equivalence → `k/N` anchors → split → embeddings → MLP.

- **Math route** (`algebra.linear_1d` / `linear_2d`, current focus): `generate_math_tasks.py` → `generate_linear_equation_variants.py` (programmatic, SymPy-verified variants; no LLM) → human review → `generate_answers.py` → `check_math_answer_correctness.py` → `equivalence_generation.py` (rule-based, compares extracted variable value) → `compute_k_N.py` (default: every answer is an anchor, giving 11 rows per source with N=10) → `build_embeddings.py` → `train_mlp.py` / `predict_mlp.py`. The full copy-pasteable command sequence is in `README.md`.
- **Non-math route**: `generate_questions.py` / `import_questions.py` / `import_defan_paraphrases.py` → `generate_paraphrases.py` (LLM) → `generate_answers.py` → `equivalence_generation_ai.py` (LLM judge) → `compute_k_N.py --original-only` → `check_dataset.py` / `split_dataset.py` → embeddings → MLP.

Key points that span files:

- `src/slm_answer_stability/model_tools.py` is the single client for the SLM / paraphrase generator / judge roles, configured by JSON (`configs/model_tools.example.json`, `model_tools.openai-local.json`: `base_url`, `api_style` (`ollama` is the default if omitted; the example configs set `openai`), and `slm`/`generator`/`judge` blocks). It talks HTTP to an OpenAI-style (or Ollama-style) server; tests mock it. Each role is a plain function (`answer`, `generate_questions`, `generate_paraphrases`, `judge_question`, `judge_answer`); `run(Namespace, config)` is only a dispatcher over them. `tests/test_model_tools_snapshots.py` pins the exact HTTP payloads against `tests/golden/model_tools_requests.json`, so any prompt or payload change must be deliberate (re-record with `--record`).
- The file-based workflow is split by stage: `generation/` (`questions`, `paraphrases`, `answers`), `annotation/` (`equivalence_ai`, `answer_extraction` for the exact numeric comparison), and `data/` (`io` with `WorkflowError`/`read_jsonl`/`write_jsonl`/`ensure_new_output`, `splits`, `anchors` for k/N). `dataset_cli.py` only holds the argparse `main(command)` and re-exports the public names, so older imports from it keep working. `constants.py` has the shared status/metadata names. Data flows between stages as JSONL files, with a human-review gate (`paraphrase_review: pending → approved/rejected`) before answers are generated.
- **Mock the model at one place:** every stage calls `generation.client.call_model` through the module (`client.call_model(...)`), so tests patch `slm_answer_stability.generation.client.call_model` (or `model_tools.urlopen` for full-stack runs). The client turns `model_tools` errors into `WorkflowError` subclasses: `InvalidGeneratorJSON` and `IncompleteBatch` mean "retry with a smaller batch"; don't match on message text.
- `tests/test_workflow_golden.py` drives the real CLI entry points through a fake model server and compares every output file, log and error message (including key order) with `tests/golden/workflow_outputs.json`. Treat a failure there as a behaviour change; re-record with `--record` only on purpose. Beware stale `.pyc` when mutating source files quickly in the same second (`PYTHONDONTWRITEBYTECODE=1`).
- **Scripts never overwrite existing outputs** — every `--output` / `--output-dir` must be a new path. `work/` (gitignored, as are `*.jsonl`, `*.pt`, `*.npz`) is the scratch area for runs.
- **Splits must be grouped by `source_question_id`** so anchors from the same equation never straddle train/val/test (`compute_all_anchors` assigns splits; `group_cv.py` does grouped CV). Don't introduce row-level random splitting.
- `linear_equation_parser.py` / `linear_equation_variants.py` recover structure from Mathematics Dataset strings; the answer parser accepts any one-letter variable, not just `x`.
- `models/embeddings.py` builds frozen `intfloat/multilingual-e5-base` embeddings (question and answer of the anchor only); `models/training.py` trains the MLP (`models/mlp.py`) and compares it with a constant-mean baseline on val. `configs/mlp.example.json` holds hyperparameters. Use `--device cpu` when CUDA is unavailable. `evaluation/error_analysis.py` backs `analyze_predictions.py`. `evaluation/group_cv.py` backs `scripts/group_cv.py` (grouped CV of the MLP vs baselines); `tests/test_group_cv.py` pins its output on synthetic data.

## Repo quirks

- `datasets/DefAn` and `datasets/mathematics_dataset` are git submodule gitlinks, but there is **no `.gitmodules`**, so they are empty after clone and `git submodule` fails. `generate_math_tasks.py` imports the vendored DeepMind generator from `datasets/mathematics_dataset` and errors if it's missing.
- `README.md` links to `docs/`, `prompts/` and `scripts/README/README_*.md` files that are not in the repo. Per-role status notes live in `scripts/README/role_*/STATE.md`; `role_4_mlp/REPORT.md` has MLP-vs-baseline results on linear equations. Treat these as reports of plans/results, not guarantees about what exists in `work/`.
- `paper/` holds the article outline and draft results.
