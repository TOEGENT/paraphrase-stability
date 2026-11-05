"""JSONL reading and writing, run logs and row validation shared by every workflow stage."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..text import normalize  # noqa: F401  (re-exported with the other row helpers)


class WorkflowError(ValueError):
    """Invalid input or an incomplete human review."""


def ensure_new_output(path: Path) -> None:
    """Refuse to overwrite an earlier result."""
    if path.exists():
        raise WorkflowError(f"Output already exists: {path}; choose a new file")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise WorkflowError(f"Cannot read {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WorkflowError(f"Invalid JSON in {path}:{number}: {exc}") from exc
        if not isinstance(row, dict):
            raise WorkflowError(f"Expected a JSON object in {path}:{number}")
        rows.append(row)
    if not rows:
        raise WorkflowError(f"No records in {path}")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_new_output(path)
    if not rows:
        raise WorkflowError("No records to write")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    except OSError as exc:
        raise WorkflowError(f"Cannot write {path}: {exc}") from exc


def append_log(path: Path | None, event: dict[str, Any]) -> None:
    if path is None:
        return
    event = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **event}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError as exc:
        raise WorkflowError(f"Cannot write log {path}: {exc}") from exc


def required_text(row: dict[str, Any], field: str, where: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise WorkflowError(f"{where}: missing {field}")
    return value.strip()


def reviewed(row: dict[str, Any], field: str, where: str) -> bool:
    value = row.get(field)
    if value not in ("approved", "rejected"):
        raise WorkflowError(f"{where}: set {field} to approved or rejected before continuing")
    return value == "approved"


def unique_ids(rows: list[dict[str, Any]], field: str) -> None:
    values = [required_text(row, field, f"row {i}") for i, row in enumerate(rows, 1)]
    if len(values) != len(set(values)):
        raise WorkflowError(f"Duplicate {field}")
