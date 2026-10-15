"""Check approved dataset integrity and configured minimum sizes."""

from slm_answer_stability.dataset_prep import main


if __name__ == "__main__":
    raise SystemExit(main("check_dataset"))
