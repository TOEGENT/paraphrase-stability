"""Frozen multilingual-e5-base embeddings for original question/answer pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from slm_answer_stability.dataset_cli import WorkflowError, read_jsonl, required_text, write_jsonl
from slm_answer_stability.dataset_prep import validate_final

MODEL = "intfloat/multilingual-e5-base"
DIMENSION = 768
PREFIX = "query: "
SPLITS = ("train", "val", "test")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_splits(directory: Path) -> dict[str, list[dict]]:
    result = {}
    all_ids: set[str] = set()
    for split in SPLITS:
        rows = read_jsonl(directory / f"{split}.jsonl")
        problems = validate_final(rows)
        if problems:
            raise WorkflowError(f"{split}: " + "; ".join(problems))
        ids = {row["question_id"] for row in rows}
        if ids & all_ids:
            raise WorkflowError(f"{split}: question_id overlaps another split")
        all_ids.update(ids)
        for row in rows:
            if row.get("split") != split:
                raise WorkflowError(f"{row['question_id']}: expected split={split}")
        result[split] = rows
    return result


def encode_splits(split_dir: Path, output_dir: Path, revision: str | None, device: str, batch_size: int) -> dict:
    if batch_size < 1:
        raise WorkflowError("--batch-size must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise WorkflowError(f"Output directory is not empty: {output_dir}")
    rows_by_split = load_splits(split_dir)
    try:
        import numpy as np
        from huggingface_hub import model_info
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise WorkflowError("Install model dependencies: pip install -e '.[mlp]'") from exc
    if revision is None:
        try:
            revision = model_info(MODEL).sha
        except Exception as exc:
            raise WorkflowError(f"Cannot resolve model revision: {exc}") from exc
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise WorkflowError("--revision must be a full 40-character model commit hash")
    model = SentenceTransformer(MODEL, revision=revision, device=device, trust_remote_code=False)
    if model.get_sentence_embedding_dimension() != DIMENSION:
        raise WorkflowError(f"Expected {DIMENSION}-dimensional {MODEL} embeddings")
    max_length = model.max_seq_length
    if max_length != 512:
        raise WorkflowError(f"Expected model max sequence length 512, got {max_length}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": MODEL, "revision": revision, "dimension": DIMENSION,
        "prefix": PREFIX, "pooling": "model_default_mean", "normalize_embeddings": True,
        "max_seq_length": max_length, "splits": {},
    }
    for split, rows in rows_by_split.items():
        ids = [required_text(row, "question_id", split) for row in rows]
        questions = [PREFIX + required_text(row, "question", row["question_id"]) for row in rows]
        answers = [PREFIX + required_text(row, "answer", row["question_id"]) for row in rows]
        token_lengths = [len(model.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]) for text in questions + answers]
        question_vectors = model.encode(questions, prompt="", batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        answer_vectors = model.encode(answers, prompt="", batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        question_vectors = np.asarray(question_vectors, dtype=np.float32)
        answer_vectors = np.asarray(answer_vectors, dtype=np.float32)
        if question_vectors.shape != (len(rows), DIMENSION) or answer_vectors.shape != (len(rows), DIMENSION):
            raise WorkflowError(f"{split}: wrong embedding shape")
        if not np.isfinite(question_vectors).all() or not np.isfinite(answer_vectors).all():
            raise WorkflowError(f"{split}: non-finite embeddings")
        feature_file = output_dir / f"{split}.npz"
        np.savez_compressed(feature_file, question=question_vectors, answer=answer_vectors, question_id=np.asarray(ids, dtype=str))
        lengths_file = output_dir / f"{split}_lengths.jsonl"
        length_rows = [
            {
                "question_id": question_id,
                "question_tokens": token_lengths[index],
                "answer_tokens": token_lengths[len(rows) + index],
                "question_truncated": token_lengths[index] > max_length,
                "answer_truncated": token_lengths[len(rows) + index] > max_length,
            }
            for index, question_id in enumerate(ids)
        ]
        write_jsonl(lengths_file, length_rows)
        manifest["splits"][split] = {
            "rows": len(rows), "source_sha256": sha256(split_dir / f"{split}.jsonl"),
            "features_sha256": sha256(feature_file),
            "lengths_sha256": sha256(lengths_file),
            "truncated_questions": sum(length > max_length for length in token_lengths[:len(rows)]),
            "truncated_answers": sum(length > max_length for length in token_lengths[len(rows):]),
        }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", help="Fixed Hugging Face commit hash; otherwise resolve and record current commit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)
    try:
        manifest = encode_splits(args.splits_dir, args.output_dir, args.revision, args.device, args.batch_size)
    except (WorkflowError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output_dir": str(args.output_dir), "counts": {k: v["rows"] for k, v in manifest["splits"].items()}}, ensure_ascii=False))
    return 0
