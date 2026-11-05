"""Pin group_cv results on a small synthetic anchor set (CPU, fixed seeds).

Expected values live in tests/golden/group_cv_summary.json. Regenerate them only when the
experiment itself is changed on purpose:
    python tests/test_group_cv.py --record
"""

import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from slm_answer_stability.dataset_cli import write_jsonl

try:
    import numpy as np
    import sklearn  # noqa: F401
    import torch  # noqa: F401
except ImportError:
    np = None

GOLDEN = Path(__file__).parent / "golden" / "group_cv_summary.json"
DIM = 768
SPLIT_OF = {0: "train", 1: "train", 2: "train", 3: "train", 4: "val", 5: "val", 6: "test", 7: "test"}
ROOTS = [1, 1, 2, 3, 4, 5, 6, 7]  # equations 0 and 1 share a root, so --group-by root merges them
CONFIG = {"hidden_dim": 8, "dropout": 0.1, "learning_rate": 0.01, "weight_decay": 0.01,
          "batch_size": 8, "max_epochs": 5, "patience": 3, "seed": 0,
          "loss_weighting": "equal", "max_n_weight": None}


def write_fixture(directory: Path) -> tuple[Path, Path, Path]:
    """Eight equations, each with an original answer and three variant answers."""
    splits, features = directory / "splits", directory / "features"
    features.mkdir()
    rng = np.random.default_rng(0)
    rows = {"train": [], "val": [], "test": []}
    for i, root in enumerate(ROOTS):
        a, b = 2 + i % 3, i
        question = f"Solve {a}*x + {b} = {a * root + b} for x."
        # Odd equations: three answers agree; even equations: two agree. The rest differ.
        values = [root, root, root if i % 2 else root + 1, root + 2]
        for j, x in enumerate(values):
            anchor_id = "original" if j == 0 else f"v{j}"
            k = values.count(x) - 1
            rows[SPLIT_OF[i]].append({
                "question_id": f"eq{i}::anchor::{anchor_id}", "source_question_id": f"eq{i}",
                "anchor_id": anchor_id, "question": question, "answer": f"x = {x}",
                "extracted_x": str(x), "k": k, "N": 3, "target": k / 3, "split": SPLIT_OF[i],
            })
    for split, split_rows in rows.items():
        write_jsonl(splits / f"{split}.jsonl", split_rows)
        np.savez(features / f"{split}.npz",
                 question_id=np.array([row["question_id"] for row in split_rows]),
                 question=rng.normal(size=(len(split_rows), DIM)).astype(np.float32),
                 answer=rng.normal(size=(len(split_rows), DIM)).astype(np.float32))
    config = directory / "mlp.json"
    config.write_text(json.dumps(CONFIG), encoding="utf-8")
    return splits, features, config


def run_all() -> dict:
    from slm_answer_stability.evaluation.group_cv import main

    result = {}
    with tempfile.TemporaryDirectory() as tmp:
        splits, features, config = write_fixture(Path(tmp))
        for name, extra in (("source", []), ("root", ["--group-by", "root"]),
                            ("extended", ["--extended"])):
            output = Path(tmp) / f"{name}.json"
            argv = ["group_cv", "--splits-dir", str(splits), "--features-dir", str(features),
                    "--config", str(config), "--output", str(output), "--folds", "2",
                    "--seeds", "0", "--device", "cpu", *extra]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            result[name] = json.loads(output.read_text(encoding="utf-8"))
    return result


def assert_close(test: unittest.TestCase, actual, expected, path="") -> None:
    if isinstance(expected, dict):
        test.assertEqual(sorted(actual), sorted(expected), path)
        for key in expected:
            assert_close(test, actual[key], expected[key], f"{path}/{key}")
    elif isinstance(expected, list):
        test.assertEqual(len(actual), len(expected), path)
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_close(test, left, right, f"{path}[{index}]")
    elif isinstance(expected, float) and math.isnan(expected):
        test.assertTrue(isinstance(actual, float) and math.isnan(actual), path)
    elif isinstance(expected, float):
        test.assertAlmostEqual(actual, expected, places=5, msg=path)
    else:
        test.assertEqual(actual, expected, path)


@unittest.skipUnless(np is not None, "needs numpy, torch and scikit-learn")
class GroupCVSnapshot(unittest.TestCase):
    def test_summaries_are_unchanged(self):
        expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
        assert_close(self, json.loads(json.dumps(run_all())), expected)


if __name__ == "__main__":
    if "--record" in sys.argv:
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(run_all(), indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"recorded {GOLDEN}")
    else:
        unittest.main()
