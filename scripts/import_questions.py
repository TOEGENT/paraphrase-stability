"""Import CSV/TSV questions into the reviewed JSONL workflow."""

from slm_answer_stability.dataset_prep import main


if __name__ == "__main__":
    raise SystemExit(main("import_questions"))
