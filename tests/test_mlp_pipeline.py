import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from slm_answer_stability.dataset_cli import WorkflowError, write_jsonl
from slm_answer_stability.models.embeddings import DIMENSION, encode_splits, sha256
from slm_answer_stability.models.training import load_features, predict, train

try:
    import numpy as np
    import torch
except ImportError:
    np = None
    torch = None


@unittest.skipUnless(np is not None and torch is not None, "Install the mlp optional dependencies")
class MLPPipelineTests(unittest.TestCase):
    def test_embedding_stage_uses_only_original_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = root / "splits"
            splits.mkdir()
            for split in ("train", "val", "test"):
                write_jsonl(splits / f"{split}.jsonl", [{
                    "question_id": split, "question": f"Original question {split}",
                    "answer": f"Original answer {split}", "split": split,
                    "review_status": "approved", "variants": [{
                        "variant_id": f"{split}-v1", "question": "Secret paraphrase",
                        "answer": "Secret transformed answer", "answer_equivalence": "equivalent",
                    }], "k": 1, "N": 1, "target": 1.0, "unresolved": 0,
                }])
            model = MagicMock()
            model.get_sentence_embedding_dimension.return_value = DIMENSION
            model.max_seq_length = 512
            model.tokenizer.side_effect = lambda text, **kwargs: {"input_ids": [1, 2]}
            model.encode.side_effect = lambda texts, **kwargs: np.ones((len(texts), DIMENSION), dtype=np.float32)
            with patch("sentence_transformers.SentenceTransformer", return_value=model):
                manifest = encode_splits(splits, root / "features", "a" * 40, "cpu", 2)
            encoded_texts = [text for call in model.encode.call_args_list for text in call.args[0]]
            self.assertEqual(len(encoded_texts), 6)
            self.assertTrue(all(text.startswith("query: Original ") for text in encoded_texts))
            self.assertFalse(any("Secret" in text for text in encoded_texts))
            self.assertEqual(manifest["splits"]["train"]["rows"], 1)

    def test_train_predict_and_reject_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            splits = root / "splits"
            features = root / "features"
            splits.mkdir()
            features.mkdir()
            manifest = {
                "model": "intfloat/multilingual-e5-base", "dimension": DIMENSION,
                "prefix": "query: ", "normalize_embeddings": True,
                "max_seq_length": 512, "splits": {},
            }
            for split, count in (("train", 8), ("val", 3), ("test", 3)):
                rows = []
                for number in range(count):
                    question_id = f"{split}-{number}"
                    equivalent = number % 2 == 0
                    rows.append({
                        "question_id": question_id, "question": f"Question {question_id}",
                        "answer": "yes" if equivalent else "no", "review_status": "approved",
                        "split": split, "variants": [{
                            "variant_id": f"{question_id}-v1", "question": "Reworded question",
                            "answer": "same", "answer_equivalence": "equivalent" if equivalent else "non_equivalent",
                        }],
                        "k": int(equivalent), "N": 1, "target": float(equivalent), "unresolved": 0,
                    })
                source = splits / f"{split}.jsonl"
                write_jsonl(source, rows)
                if split == "val":
                    val_rows = rows
                vectors = np.zeros((count, DIMENSION), dtype=np.float32)
                vectors[:, 0] = np.arange(count, dtype=np.float32)
                archive = features / f"{split}.npz"
                np.savez_compressed(archive, question=vectors, answer=vectors, question_id=np.asarray([row["question_id"] for row in rows], dtype=str))
                manifest["splits"][split] = {"source_sha256": sha256(source), "features_sha256": sha256(archive)}
            (features / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (splits / "split_manifest.json").write_text("{}", encoding="utf-8")
            config = root / "config.json"
            config.write_text(json.dumps({
                "hidden_dim": 8, "dropout": 0.0, "learning_rate": 0.01,
                "weight_decay": 0.0, "batch_size": 4, "max_epochs": 4,
                "patience": 2, "seed": 42, "loss_weighting": "equal", "max_n_weight": None,
            }), encoding="utf-8")
            model_dir = root / "model"
            report = train(splits, features, model_dir, config, "cpu")
            self.assertEqual(report["train_questions"], 8)
            self.assertTrue((model_dir / "model.pt").exists())
            result = predict(splits, features, model_dir, root / "validation", "val", False, "cpu")
            self.assertEqual(result["questions"], 3)
            self.assertTrue(0 <= result["mlp"]["mae"] <= 1)
            with self.assertRaisesRegex(WorkflowError, "--final"):
                predict(splits, features, model_dir, root / "test_run", "test", False, "cpu")
            source = splits / "val.jsonl"
            source.write_text(source.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(WorkflowError, "source JSONL changed"):
                load_features(splits, features, "val", val_rows)


if __name__ == "__main__":
    unittest.main()
