"""CSV dataset loaders and value-function construction used by the paper."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np

from information_theory import HoldoutNaiveBayesInformationQuantifier
from value_adapter import DataValueEvaluatorFunction
from value_quantification import DataValueEvaluator, InformationTheoryQuantifier


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATASET_CSV_DIR = PACKAGE_ROOT / "dataset" / "csv"
STRUCTURED_TEXT_DEFAULT_SUBSET_COUNTS = {
    "internet_advertisements": 200,
    "cnae_9": 200,
    "farm_ads": 200,
    "spambase": 0,
}


@dataclass(frozen=True)
class ExperimentDataset:
    name: str
    feature_names: List[str]
    data_combinations: np.ndarray
    model_output: np.ndarray
    preprocessing: dict = field(default_factory=dict)


DatasetLoader = Callable[[], ExperimentDataset]


def _read_dense_csv(
    file_name: str,
    expected_feature_count: int,
    *,
    integer_features: bool,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    path = DATASET_CSV_DIR / file_name
    with path.open("r", encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream))
    expected_numeric_columns = expected_feature_count + 2
    if len(header) not in {expected_numeric_columns, expected_numeric_columns + 1}:
        raise ValueError(f"Unexpected CSV width for {path}: {len(header)}")
    if header[0] != "sample_id" or header[expected_feature_count + 1] != "target":
        raise ValueError(f"Unexpected CSV schema for {path}")

    raw = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=range(expected_numeric_columns),
        dtype=float,
        ndmin=2,
    )
    if raw.shape[1] != expected_numeric_columns:
        raise ValueError(f"Unexpected matrix shape for {path}: {raw.shape}")
    sample_ids = raw[:, 0].astype(int)
    if not np.array_equal(sample_ids, np.arange(raw.shape[0])):
        raise ValueError(f"Nonsequential sample_id values in {path}")

    data = raw[:, 1 : expected_feature_count + 1].T
    if integer_features:
        if not np.all(data == np.rint(data)):
            raise ValueError(f"Expected integer analysis values in {path}")
        data = np.rint(data).astype(int)
    target = raw[:, expected_feature_count + 1]
    if np.all(target == np.rint(target)):
        target = np.rint(target).astype(int)
    return header[1 : expected_feature_count + 1], data, target


def _plain_dataset(
    name: str,
    file_name: str,
    feature_count: int,
) -> ExperimentDataset:
    feature_names, data, target = _read_dense_csv(
        file_name,
        feature_count,
        integer_features=False,
    )
    return ExperimentDataset(
        name=name,
        feature_names=feature_names,
        data_combinations=data,
        model_output=target.reshape(1, -1),
        preprocessing={
            "source": f"dataset/csv/{file_name}",
            "representation": "human-and-machine-readable CSV",
        },
    )


def load_iris_dataset() -> ExperimentDataset:
    return _plain_dataset("iris", "iris.csv", 4)


def load_wine_dataset() -> ExperimentDataset:
    return _plain_dataset("wine", "wine.csv", 13)


def _select_feature_indices(
    feature_count: int,
    subset_count: int,
    subset_seed: int,
) -> np.ndarray:
    if subset_count < 0:
        raise ValueError("feature_subset_count must be non-negative.")
    if subset_count == 0 or subset_count >= feature_count:
        return np.arange(feature_count, dtype=int)
    return np.sort(
        np.random.default_rng(subset_seed).choice(
            feature_count,
            size=subset_count,
            replace=False,
        )
    )


def _structured_text_dataset(
    name: str,
    feature_names: Sequence[str],
    data: np.ndarray,
    target: np.ndarray,
    *,
    original_feature_count: int,
    selected_indices: Sequence[int],
    subset_seed: int,
    source: str,
    extra_metadata: dict | None = None,
) -> ExperimentDataset:
    metadata = {
        "source": source,
        "representation": "human-and-machine-readable CSV",
        "original_feature_count": int(original_feature_count),
        "selected_feature_count": len(feature_names),
        "selected_original_indices_zero_based": [
            int(index) for index in selected_indices
        ],
        "feature_subset_seed": int(subset_seed),
        "value_quantifier": "HoldoutNaiveBayesInformationQuantifier",
        "holdout_test_fraction": 0.2,
        "holdout_random_seed": 20260824,
    }
    metadata.update(extra_metadata or {})
    return ExperimentDataset(
        name=name,
        feature_names=list(feature_names),
        data_combinations=np.asarray(data),
        model_output=np.asarray(target).reshape(1, -1),
        preprocessing=metadata,
    )


def _load_selected_dense_dataset(
    *,
    name: str,
    file_name: str,
    feature_count: int,
    feature_subset_count: int,
    feature_subset_seed: int,
    extra_metadata: dict | None = None,
) -> ExperimentDataset:
    selected = _select_feature_indices(
        feature_count, feature_subset_count, feature_subset_seed
    )
    all_names, all_data, target = _read_dense_csv(
        file_name,
        feature_count,
        integer_features=True,
    )
    return _structured_text_dataset(
        name,
        [all_names[int(index)] for index in selected],
        all_data[selected],
        target,
        original_feature_count=feature_count,
        selected_indices=selected,
        subset_seed=feature_subset_seed,
        source=f"dataset/csv/{file_name}",
        extra_metadata=extra_metadata,
    )


def load_internet_advertisements_dataset(
    feature_subset_count: int = 200,
    feature_subset_seed: int = 11,
) -> ExperimentDataset:
    return _load_selected_dense_dataset(
        name="internet_advertisements",
        file_name="internet_advertisements.csv",
        feature_count=1_558,
        feature_subset_count=feature_subset_count,
        feature_subset_seed=feature_subset_seed,
        extra_metadata={
            "continuous_geometry_features": 3,
            "remaining_features": "binary URL and text indicators",
            "continuous_discretization": "10 quantile bins",
        },
    )


def load_cnae_9_dataset(
    feature_subset_count: int = 200,
    feature_subset_seed: int = 11,
) -> ExperimentDataset:
    return _load_selected_dense_dataset(
        name="cnae_9",
        file_name="cnae_9.csv",
        feature_count=856,
        feature_subset_count=feature_subset_count,
        feature_subset_seed=feature_subset_seed,
        extra_metadata={"documented_zero_fraction": 0.9922},
    )


def load_farm_ads_dataset(
    feature_subset_count: int = 200,
    feature_subset_seed: int = 11,
) -> ExperimentDataset:
    feature_count = 54_877
    selected = _select_feature_indices(
        feature_count, feature_subset_count, feature_subset_seed
    )
    sample_path = DATASET_CSV_DIR / "farm_ads_samples.csv"
    sample_rows = np.loadtxt(
        sample_path,
        delimiter=",",
        skiprows=1,
        dtype=int,
        ndmin=2,
    )
    if sample_rows.shape[1] != 2 or not np.array_equal(
        sample_rows[:, 0], np.arange(sample_rows.shape[0])
    ):
        raise ValueError(f"Unexpected Farm Ads sample table: {sample_path}")

    index_lookup = {
        int(original_index) + 1: local_index
        for local_index, original_index in enumerate(selected)
    }
    data = np.zeros((len(selected), sample_rows.shape[0]), dtype=np.float32)
    feature_path = DATASET_CSV_DIR / "farm_ads_features.csv"
    with feature_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            local_index = index_lookup.get(int(row["feature_index_one_based"]))
            if local_index is not None:
                data[local_index, int(row["sample_id"])] = float(row["value"])
    if np.all(data == np.rint(data)):
        data = np.rint(data).astype(int)

    return _structured_text_dataset(
        "farm_ads",
        [f"word_feature_{index + 1}" for index in selected],
        data,
        sample_rows[:, 1],
        original_feature_count=feature_count,
        selected_indices=selected,
        subset_seed=feature_subset_seed,
        source="dataset/csv/farm_ads_samples.csv and farm_ads_features.csv",
        extra_metadata={
            "sparse_format": "CSV long table",
            "text_preprocessing": "official stemming and stop-word removal",
        },
    )


def load_spambase_dataset(
    feature_subset_count: int = 0,
    feature_subset_seed: int = 11,
) -> ExperimentDataset:
    return _load_selected_dense_dataset(
        name="spambase",
        file_name="spambase.csv",
        feature_count=57,
        feature_subset_count=feature_subset_count,
        feature_subset_seed=feature_subset_seed,
        extra_metadata={"continuous_discretization": "10 quantile bins"},
    )


def build_value_function(
    dataset: ExperimentDataset,
    cache: bool = True,
    quantifier_name: str = "dataset_default",
) -> DataValueEvaluatorFunction:
    if quantifier_name not in {
        "dataset_default",
        "holdout_naive_bayes",
        "mutual_information",
    }:
        raise ValueError(f"Unsupported value quantifier: {quantifier_name}")
    use_holdout_naive_bayes = (
        quantifier_name == "holdout_naive_bayes"
        or (
            quantifier_name == "dataset_default"
            and (
                dataset.name == "toll_evasion"
                or dataset.preprocessing.get("value_quantifier")
                == "HoldoutNaiveBayesInformationQuantifier"
            )
        )
    )
    if use_holdout_naive_bayes:
        quantifier = HoldoutNaiveBayesInformationQuantifier(
            dataset.data_combinations,
            dataset.model_output,
            test_fraction=0.2,
            random_seed=20260824,
            alpha=1.0,
        )
    else:
        quantifier = InformationTheoryQuantifier()
    data_value_evaluator = DataValueEvaluator(
        quantifier,
        cache=cache,
    )
    return DataValueEvaluatorFunction(
        data_value_evaluator=data_value_evaluator,
        model_output=dataset.model_output,
        data_combinations=dataset.data_combinations,
        feature_names=dataset.feature_names,
    )


def load_experiment_dataset(
    dataset_name: str,
    *,
    feature_subset_count: int = 0,
    feature_subset_seed: int = 11,
    **_ignored: object,
) -> ExperimentDataset:
    """Load one of the public CSV datasets used in the paper."""
    if dataset_name == "iris":
        return load_iris_dataset()
    if dataset_name == "wine":
        return load_wine_dataset()
    structured = {
        "internet_advertisements": load_internet_advertisements_dataset,
        "cnae_9": load_cnae_9_dataset,
        "farm_ads": load_farm_ads_dataset,
        "spambase": load_spambase_dataset,
    }
    if dataset_name not in structured:
        raise ValueError(f"Unsupported public dataset: {dataset_name}")
    count = feature_subset_count or STRUCTURED_TEXT_DEFAULT_SUBSET_COUNTS[dataset_name]
    return structured[dataset_name](
        feature_subset_count=count,
        feature_subset_seed=feature_subset_seed,
    )
