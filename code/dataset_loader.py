"""Dataset loaders and value-function construction used by the paper."""

from __future__ import annotations

import csv
import urllib.request
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Callable, List, Sequence

import numpy as np
from sklearn import datasets

from information_theory import HoldoutNaiveBayesInformationQuantifier
from value_adapter import DataValueEvaluatorFunction
from value_quantification import DataValueEvaluator, InformationTheoryQuantifier


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATASET_CACHE_DIR = PACKAGE_ROOT / "dataset" / "uci"
UCI_NUMERIC_BINS = 10
UCI_DATASETS = {
    "internet_advertisements": {
        "archive_name": "internet_advertisements.zip",
        "url": "https://archive.ics.uci.edu/static/public/51/internet+advertisements.zip",
    },
    "cnae_9": {
        "archive_name": "cnae_9.zip",
        "url": "https://archive.ics.uci.edu/static/public/233/cnae+9.zip",
    },
    "farm_ads": {
        "archive_name": "farm_ads.zip",
        "url": "https://archive.ics.uci.edu/static/public/218/farm+ads.zip",
    },
    "spambase": {
        "archive_name": "spambase.zip",
        "url": "https://archive.ics.uci.edu/static/public/94/spambase.zip",
    },
}
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
def _normalize_feature_names(feature_names: Sequence[str]) -> List[str]:
    return [
        name.replace(" (cm)", "")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        for name in feature_names
    ]

def _make_sklearn_dataset(name: str, loader: Callable[..., object]) -> ExperimentDataset:
    raw_data = loader(return_X_y=False)
    feature_names = _normalize_feature_names(raw_data.feature_names)  # type: ignore[attr-defined]

    return ExperimentDataset(
        name=name,
        feature_names=feature_names,
        # Framework convention: rows are features, columns are samples.
        data_combinations=np.asarray(raw_data.data).T,  # type: ignore[attr-defined]
        model_output=np.asarray(raw_data.target).reshape(1, -1),  # type: ignore[attr-defined]
    )

def load_iris_dataset() -> ExperimentDataset:
    return _make_sklearn_dataset("iris", datasets.load_iris)

def load_wine_dataset() -> ExperimentDataset:
    return _make_sklearn_dataset("wine", datasets.load_wine)

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

def _parse_selected_svmlight_rows(
    rows: Sequence[str],
    selected_indices: np.ndarray,
    *,
    labels_in_rows: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    index_lookup = {
        int(original_index) + 1: local_index
        for local_index, original_index in enumerate(selected_indices)
    }
    matrix = np.zeros((len(selected_indices), len(rows)), dtype=np.float32)
    labels: list[float] = []
    for sample_index, row in enumerate(rows):
        tokens = row.strip().split()
        if not tokens:
            raise ValueError("Sparse dataset contains an empty row.")
        start = 0
        if labels_in_rows:
            labels.append(float(tokens[0]))
            start = 1
        for token in tokens[start:]:
            if ":" not in token:
                continue
            raw_index, raw_value = token.split(":", 1)
            local_index = index_lookup.get(int(raw_index))
            if local_index is not None:
                matrix[local_index, sample_index] = float(raw_value)
    if np.all(matrix == np.rint(matrix)):
        matrix = np.rint(matrix).astype(int)
    label_array = np.asarray(labels) if labels_in_rows else None
    return matrix, label_array

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
        "representation": "single_table",
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

def load_internet_advertisements_dataset(
    feature_subset_count: int = 200,
    feature_subset_seed: int = 11,
) -> ExperimentDataset:
    feature_count = 1_558
    selected = _select_feature_indices(
        feature_count, feature_subset_count, feature_subset_seed
    )
    archive_path = _get_uci_archive_path("internet_advertisements")
    with zipfile.ZipFile(archive_path) as archive:
        member = _find_archive_member(archive, "ad.data")
        rows = list(
            csv.reader(
                archive.read(member).decode("utf-8", errors="replace").splitlines()
            )
        )
    if not rows or any(len(row) != feature_count + 1 for row in rows):
        raise ValueError("Internet Advertisements has an unexpected row width.")
    data = np.zeros((len(selected), len(rows)), dtype=int)
    for local_index, original_index in enumerate(selected):
        values = [row[int(original_index)].strip() for row in rows]
        numeric = _as_numeric_column(values)
        if numeric is None:
            data[local_index] = _encode_categorical_column(values)
        elif int(original_index) < 3:
            data[local_index] = _discretize_numeric_column(numeric)
        else:
            data[local_index] = np.where(
                np.isfinite(numeric), numeric, 2
            ).astype(int)
    target = _encode_categorical_column([row[-1].strip() for row in rows])
    return _structured_text_dataset(
        "internet_advertisements",
        [f"ad_feature_{index + 1}" for index in selected],
        data,
        target,
        original_feature_count=feature_count,
        selected_indices=selected,
        subset_seed=feature_subset_seed,
        source="UCI Internet Advertisements ad.data",
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
    feature_count = 856
    selected = _select_feature_indices(
        feature_count, feature_subset_count, feature_subset_seed
    )
    archive_path = _get_uci_archive_path("cnae_9")
    with zipfile.ZipFile(archive_path) as archive:
        member = _find_archive_member(archive, "CNAE-9.data")
        raw = np.loadtxt(BytesIO(archive.read(member)), delimiter=",", dtype=int)
    if raw.ndim != 2 or raw.shape[1] != feature_count + 1:
        raise ValueError("CNAE-9 has an unexpected matrix shape.")
    return _structured_text_dataset(
        "cnae_9",
        [f"word_frequency_{index + 1}" for index in selected],
        raw[:, selected + 1].T,
        raw[:, 0],
        original_feature_count=feature_count,
        selected_indices=selected,
        subset_seed=feature_subset_seed,
        source="UCI CNAE-9 preprocessed word-frequency table",
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
    archive_path = _get_uci_archive_path("farm_ads")
    with zipfile.ZipFile(archive_path) as archive:
        member = _find_archive_member(archive, "farm-ads-vect")
        rows = archive.read(member).decode("utf-8").splitlines()
    data, labels = _parse_selected_svmlight_rows(
        rows, selected, labels_in_rows=True
    )
    if labels is None:
        raise ValueError("Farm Ads labels were not parsed.")
    return _structured_text_dataset(
        "farm_ads",
        [f"word_feature_{index + 1}" for index in selected],
        data,
        (labels > 0).astype(int),
        original_feature_count=feature_count,
        selected_indices=selected,
        subset_seed=feature_subset_seed,
        source="UCI Farm Ads SVMlight sparse vectors",
        extra_metadata={
            "sparse_format": "SVMlight",
            "text_preprocessing": "official stemming and stop-word removal",
        },
    )

def load_spambase_dataset(
    feature_subset_count: int = 0,
    feature_subset_seed: int = 11,
) -> ExperimentDataset:
    feature_count = 57
    selected = _select_feature_indices(
        feature_count, feature_subset_count, feature_subset_seed
    )
    archive_path = _get_uci_archive_path("spambase")
    with zipfile.ZipFile(archive_path) as archive:
        member = _find_archive_member(archive, "spambase.data")
        raw = np.loadtxt(BytesIO(archive.read(member)), delimiter=",", dtype=float)
    if raw.ndim != 2 or raw.shape[1] != feature_count + 1:
        raise ValueError("Spambase has an unexpected matrix shape.")
    discrete = np.vstack(
        [_discretize_numeric_column(raw[:, index]) for index in selected]
    )
    return _structured_text_dataset(
        "spambase",
        [f"spam_feature_{index + 1}" for index in selected],
        discrete,
        raw[:, -1].astype(int),
        original_feature_count=feature_count,
        selected_indices=selected,
        subset_seed=feature_subset_seed,
        source="UCI Spambase word/character frequency table",
        extra_metadata={"continuous_discretization": "10 quantile bins"},
    )

def _get_uci_archive_path(dataset_name: str) -> Path:
    """Download a UCI archive once and return its local cache path."""
    metadata = UCI_DATASETS[dataset_name]
    DATASET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = DATASET_CACHE_DIR / metadata["archive_name"]
    if archive_path.exists():
        return archive_path

    temporary_path = archive_path.with_suffix(f"{archive_path.suffix}.tmp")
    request = urllib.request.Request(
        metadata["url"],
        headers={"User-Agent": "greedySHAP-research/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            temporary_path.write_bytes(response.read())
        temporary_path.replace(archive_path)
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Unable to download UCI dataset '{dataset_name}' from "
            f"{metadata['url']}. Check the network connection and retry."
        ) from error

    return archive_path

def _find_archive_member(archive: zipfile.ZipFile, suffix: str) -> str:
    matches = [
        name
        for name in archive.namelist()
        if not name.startswith("__MACOSX/")
        and name.lower().endswith(suffix.lower())
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one archive member ending with '{suffix}', found {matches}."
        )
    return matches[0]

def _as_numeric_column(values: Sequence[str]) -> np.ndarray | None:
    numeric_values = []
    for value in values:
        stripped_value = value.strip()
        if stripped_value in {"", "?"}:
            numeric_values.append(np.nan)
            continue
        try:
            numeric_values.append(float(stripped_value))
        except ValueError:
            return None
    return np.asarray(numeric_values, dtype=float)

def _discretize_numeric_column(
    values: np.ndarray,
    n_bins: int = UCI_NUMERIC_BINS,
) -> np.ndarray:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.zeros(values.shape[0], dtype=int)

    filled_values = np.where(
        np.isfinite(values),
        values,
        np.median(finite_values),
    )
    quantile_edges = np.quantile(
        filled_values,
        np.linspace(0.0, 1.0, n_bins + 1)[1:-1],
    )
    unique_edges = np.unique(quantile_edges)
    return np.searchsorted(unique_edges, filled_values, side="right").astype(int)

def _encode_categorical_column(values: Sequence[str]) -> np.ndarray:
    normalized_values = np.asarray(
        [
            "__missing__" if value.strip() in {"", "?"} else value.strip()
            for value in values
        ],
        dtype=str,
    )
    _, encoded_values = np.unique(normalized_values, return_inverse=True)
    return encoded_values.astype(int)

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
    """Load one of the public datasets used in the paper."""
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
