"""Seeded train/val/test allocation shared by the question-level and anchor-level splits."""

from __future__ import annotations

import random

from ..constants import SPLITS
from .io import WorkflowError


def validate_ratios(ratios: tuple[float, float, float]) -> None:
    if any(not 0 < ratio < 1 for ratio in ratios) or abs(sum(ratios) - 1) > 1e-9:
        raise WorkflowError("Split ratios must be positive fractions summing to 1")


def partition_keys(keys, ratios: tuple[float, float, float], seed: int) -> dict[str, list[str]]:
    """Shuffle the sorted keys with the seed and cut them into train/val/test.

    One key is reserved for each split; the rest are divided by largest remainder.
    Callers check that there are at least three keys and that the ratios are valid.
    """
    shuffled = sorted(keys)
    random.Random(seed).shuffle(shuffled)
    remainder = len(shuffled) - 3
    quotas = [remainder * ratio for ratio in ratios]
    counts = [1 + int(quota) for quota in quotas]
    for index in sorted(range(3), key=lambda i: (quotas[i] % 1, -i), reverse=True)[:remainder - sum(map(int, quotas))]:
        counts[index] += 1
    parts: dict[str, list[str]] = {}
    offset = 0
    for split, count in zip(SPLITS, counts):
        parts[split] = shuffled[offset:offset + count]
        offset += count
    return parts
