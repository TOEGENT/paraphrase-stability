"""Compute k/N for every answer anchor by default, or the original only."""

from slm_answer_stability.dataset_cli import main


if __name__ == "__main__":
    raise SystemExit(main("compute_k_n"))
