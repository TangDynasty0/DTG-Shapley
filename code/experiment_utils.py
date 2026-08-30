"""Persistence, normalization, and synthetic-tail helpers."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dataset_loader import ExperimentDataset

SYNTHETIC_LOW_MIN = 1
SYNTHETIC_LOW_MAX = 2
def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(path)

def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

def augment_with_persistent_low_fields(
    dataset: ExperimentDataset,
    count: int,
    seed: int,
    cache_dir: Path,
) -> tuple[ExperimentDataset, Path | None]:
    if count == 0:
        return dataset, None
    if count < 0:
        raise ValueError("artificial_low_count must be non-negative.")

    sample_count = int(dataset.data_combinations.shape[1])
    cache_path = cache_dir / (
        f"{dataset.name}__samples{sample_count}__low{count}__seed{seed}.npz"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        with np.load(cache_path) as archive:
            artificial = np.asarray(archive["values"])
        if artificial.shape != (count, sample_count):
            raise ValueError(f"Invalid persistent low-field cache: {cache_path}")
    else:
        rng = np.random.default_rng(seed)
        artificial = rng.integers(
            SYNTHETIC_LOW_MIN,
            SYNTHETIC_LOW_MAX + 1,
            size=(count, sample_count),
            dtype=np.int8,
        )
        temporary = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, values=artificial)
        temporary.replace(cache_path)

    names = [f"synthetic_low_{index + 1}" for index in range(count)]
    preprocessing = dict(dataset.preprocessing)
    preprocessing["artificial_low_fields"] = {
        "count": count,
        "seed": seed,
        "range": [SYNTHETIC_LOW_MIN, SYNTHETIC_LOW_MAX],
        "independent_of_target": True,
        "cache_file": str(cache_path),
    }
    return replace(
        dataset,
        feature_names=[*dataset.feature_names, *names],
        data_combinations=np.vstack((dataset.data_combinations, artificial)),
        preprocessing=preprocessing,
    ), cache_path

def normalize(values: dict[str, float]) -> dict[str, float]:
    total = sum(values.values())
    if not values:
        return {}
    if abs(total) <= 1e-15:
        return {name: 1.0 / len(values) for name in values}
    normalized = {name: value / total for name, value in values.items()}
    last = next(reversed(normalized))
    normalized[last] += 1.0 - sum(normalized.values())
    return normalized

def _spearman(first: Sequence[float], second: Sequence[float]) -> float:
    def ranks(values: Sequence[float]) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        order = np.argsort(array, kind="mergesort")
        result = np.empty(len(array), dtype=float)
        position = 0
        while position < len(array):
            end = position + 1
            while end < len(array) and array[order[end]] == array[order[position]]:
                end += 1
            result[order[position:end]] = (position + end - 1) / 2.0
            position = end
        return result

    if len(first) < 2:
        return 1.0
    first_ranks = ranks(first)
    second_ranks = ranks(second)
    if np.std(first_ranks) == 0 or np.std(second_ranks) == 0:
        return 1.0 if np.allclose(first_ranks, second_ranks) else 0.0
    first_mean = float(np.mean(first_ranks))
    second_mean = float(np.mean(second_ranks))
    numerator = sum(
        (float(left) - first_mean) * (float(right) - second_mean)
        for left, right in zip(first_ranks, second_ranks)
    )
    first_sum_squares = sum(
        (float(value) - first_mean) ** 2 for value in first_ranks
    )
    second_sum_squares = sum(
        (float(value) - second_mean) ** 2 for value in second_ranks
    )
    denominator = math.sqrt(first_sum_squares * second_sum_squares)
    return numerator / denominator if denominator > 0.0 else 0.0
