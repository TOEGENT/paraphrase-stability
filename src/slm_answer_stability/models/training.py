"""Train and evaluate the MLP without exposing paraphrases to its features."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

from slm_answer_stability.dataset_cli import WorkflowError, write_jsonl
from slm_answer_stability.models.embeddings import DIMENSION, SPLITS, load_splits, sha256
from slm_answer_stability.models.mlp import build_model


def load_features(split_dir: Path, features_dir: Path, split: str, rows: list[dict]) -> tuple[Any, list[int], list[float]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise WorkflowError("Install model dependencies: pip install -e '.[mlp]'") from exc
    manifest_path = features_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WorkflowError(f"Cannot read feature manifest: {exc}") from exc
    if manifest.get("model") != "intfloat/multilingual-e5-base" or manifest.get("dimension") != DIMENSION:
        raise WorkflowError("Feature manifest has a different embedding model or dimension")
    if manifest.get("prefix") != "query: " or manifest.get("normalize_embeddings") is not True or manifest.get("max_seq_length") != 512:
        raise WorkflowError("Feature manifest has incompatible preprocessing")
    details = manifest.get("splits", {}).get(split, {})
    if details.get("source_sha256") != sha256(split_dir / f"{split}.jsonl"):
        raise WorkflowError(f"{split}: source JSONL changed since embedding")
    feature_file = features_dir / f"{split}.npz"
    if details.get("features_sha256") != sha256(feature_file):
        raise WorkflowError(f"{split}: feature file changed since embedding")
    try:
        with np.load(feature_file, allow_pickle=False) as archive:
            question = archive["question"]
            answer = archive["answer"]
            ids = archive["question_id"].tolist()
    except (OSError, ValueError, KeyError) as exc:
        raise WorkflowError(f"{split}: cannot load features: {exc}") from exc
    expected_ids = [row["question_id"] for row in rows]
    if ids != expected_ids:
        raise WorkflowError(f"{split}: question_id order does not match JSONL")
    if question.shape != (len(rows), DIMENSION) or answer.shape != (len(rows), DIMENSION):
        raise WorkflowError(f"{split}: wrong feature shape")
    if question.dtype != np.float32 or answer.dtype != np.float32:
        raise WorkflowError(f"{split}: expected float32 features")
    if not np.isfinite(question).all() or not np.isfinite(answer).all():
        raise WorkflowError(f"{split}: non-finite features")
    x = np.concatenate((question, answer), axis=1)
    n = [row["N"] for row in rows]
    y = [row["target"] for row in rows]
    return x, n, y


def read_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WorkflowError(f"Cannot read config {path}: {exc}") from exc
    required = {"hidden_dim", "dropout", "learning_rate", "weight_decay", "batch_size", "max_epochs", "patience", "seed", "loss_weighting", "max_n_weight"}
    if not isinstance(config, dict) or required - config.keys():
        raise WorkflowError(f"Config requires {', '.join(sorted(required))}")
    integer_fields = ("hidden_dim", "batch_size", "max_epochs", "patience", "seed")
    if any(type(config[key]) is not int for key in integer_fields):
        raise WorkflowError("Dimension, batch size, epochs, patience and seed must be integers")
    if min(config[key] for key in integer_fields if key != "seed") < 1:
        raise WorkflowError("Dimension, batch size, epochs and patience must be positive")
    if any(type(config[key]) not in (int, float) for key in ("dropout", "learning_rate", "weight_decay")):
        raise WorkflowError("Dropout, learning rate and weight decay must be numbers")
    if not 0 <= config["dropout"] < 1 or config["learning_rate"] <= 0 or config["weight_decay"] < 0:
        raise WorkflowError("Invalid dropout, learning rate or weight decay")
    if config["loss_weighting"] not in ("equal", "binomial"):
        raise WorkflowError("loss_weighting must be equal or binomial")
    cap = config["max_n_weight"]
    if cap is not None and (type(cap) is not int or cap < 1):
        raise WorkflowError("max_n_weight must be null or a positive integer")
    return config


def metrics(targets: list[float], predictions: list[float]) -> dict[str, float]:
    if not targets or len(targets) != len(predictions):
        raise WorkflowError("Targets and predictions must have the same positive length")
    errors = [prediction - target for target, prediction in zip(targets, predictions)]
    return {"mae": sum(abs(error) for error in errors) / len(errors), "rmse": math.sqrt(sum(error * error for error in errors) / len(errors))}


def _device(name: str):
    import torch

    if name.startswith("cuda") and not torch.cuda.is_available():
        raise WorkflowError("CUDA requested but not available")
    return torch.device(name)


def _predict(model, x, device, batch_size: int) -> list[float]:
    import torch

    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch = torch.as_tensor(x[start:start + batch_size], device=device)
            output.extend(torch.sigmoid(model(batch).squeeze(-1)).cpu().tolist())
    return output


def _loss(logits, targets, counts, config):
    import torch
    from torch.nn import functional as F

    per_row = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if config["loss_weighting"] == "equal":
        return per_row.mean()
    weights = counts.float()
    if config["max_n_weight"] is not None:
        weights = torch.clamp(weights, max=config["max_n_weight"])
    return (per_row * weights).sum() / weights.sum()


def train(split_dir: Path, features_dir: Path, output_dir: Path, config_path: Path, device_name: str) -> dict[str, Any]:
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise WorkflowError("Install model dependencies: pip install -e '.[mlp]'") from exc
    config = read_config(config_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise WorkflowError(f"Output directory is not empty: {output_dir}")
    rows = load_splits(split_dir)
    x_train, n_train, y_train = load_features(split_dir, features_dir, "train", rows["train"])
    x_val, n_val, y_val = load_features(split_dir, features_dir, "val", rows["val"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = _device(device_name)
    model = build_model(DIMENSION, config["hidden_dim"], config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    train_x = torch.as_tensor(x_train, device=device)
    train_y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    train_n = torch.as_tensor(n_train, dtype=torch.float32, device=device)
    val_x = torch.as_tensor(x_val, device=device)
    val_y = torch.as_tensor(y_val, dtype=torch.float32, device=device)
    val_n = torch.as_tensor(n_val, dtype=torch.float32, device=device)
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    history = []
    stale = 0
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(train_x), device=device)
        for indices in order.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_x[indices]).squeeze(-1)
            loss = _loss(logits, train_y[indices], train_n[indices], config)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            train_loss = _loss(model(train_x).squeeze(-1), train_y, train_n, config).item()
            val_loss = _loss(model(val_x).squeeze(-1), val_y, val_n, config).item()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config["patience"]:
                break
    model.load_state_dict(best_state)
    predictions = _predict(model, x_val, device, config["batch_size"])
    constant = sum(y_train) / len(y_train)
    report = {
        "train_questions": len(y_train), "val_questions": len(y_val),
        "best_epoch": best_epoch, "best_val_loss": best_loss,
        "val_mlp": metrics(y_val, predictions),
        "val_constant": metrics(y_val, [constant] * len(y_val)),
        "constant_prediction": constant,
        "loss_weighting": config["loss_weighting"], "max_n_weight": config["max_n_weight"],
        "test_used": False, "history": history,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    metadata = {
        "config": config, "embedding_manifest_sha256": sha256(features_dir / "manifest.json"),
        "embedding_model": "intfloat/multilingual-e5-base", "embedding_dim": DIMENSION,
        "split_manifest_sha256": sha256(split_dir / "split_manifest.json"),
        "constant_prediction": constant, "best_epoch": best_epoch,
    }
    (output_dir / "model.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def predict(split_dir: Path, features_dir: Path, model_dir: Path, output_dir: Path, split: str, final: bool, device_name: str) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise WorkflowError("Install model dependencies: pip install -e '.[mlp]'") from exc
    if split not in ("val", "test") or (split == "test" and not final):
        raise WorkflowError("Use --split val, or --split test with --final")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise WorkflowError(f"Output directory is not empty: {output_dir}")
    rows = load_splits(split_dir)[split]
    x, counts, targets = load_features(split_dir, features_dir, split, rows)
    metadata = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    if metadata["embedding_manifest_sha256"] != sha256(features_dir / "manifest.json"):
        raise WorkflowError("Embedding manifest differs from training run")
    if metadata["split_manifest_sha256"] != sha256(split_dir / "split_manifest.json"):
        raise WorkflowError("Split manifest differs from training run")
    device = _device(device_name)
    config = metadata["config"]
    model = build_model(DIMENSION, config["hidden_dim"], config["dropout"]).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device, weights_only=True))
    predictions = _predict(model, x, device, config["batch_size"])
    constant = metadata["constant_prediction"]
    report = {
        "split": split, "questions": len(rows), "mlp": metrics(targets, predictions),
        "constant": metrics(targets, [constant] * len(rows)),
        "embedding_time_included": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows = [
        {
            "question_id": row["question_id"], "split": split,
            "question": row["question"], "answer": row["answer"],
            "k": row["k"], "N": row["N"], "target": row["target"],
            "prediction": prediction,
            "absolute_error": abs(prediction - row["target"]),
            "constant_prediction": constant,
        }
        for row, prediction in zip(rows, predictions)
    ]
    write_jsonl(output_dir / "predictions.jsonl", prediction_rows)
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["question_id", "k", "N", "target", "prediction", "absolute_error", "constant_prediction"])
        writer.writeheader()
        for row in prediction_rows:
            writer.writerow({field: row[field] for field in writer.fieldnames})
    (output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(command: str, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    if command == "train":
        parser.add_argument("--config", type=Path, default=Path("configs/mlp.example.json"))
    elif command == "predict":
        parser.add_argument("--model-dir", type=Path, required=True)
        parser.add_argument("--split", choices=("val", "test"), default="val")
        parser.add_argument("--final", action="store_true")
    else:
        raise ValueError(command)
    args = parser.parse_args(argv)
    try:
        report = (
            train(args.splits_dir, args.features_dir, args.output_dir, args.config, args.device)
            if command == "train" else
            predict(args.splits_dir, args.features_dir, args.model_dir, args.output_dir, args.split, args.final, args.device)
        )
    except (WorkflowError, OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0
