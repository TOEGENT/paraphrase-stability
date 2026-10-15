"""Evaluate a frozen MLP on validation or final test data."""

from slm_answer_stability.models.training import main


if __name__ == "__main__":
    raise SystemExit(main("predict"))
