"""Adapter from dataset value quantifiers to the Shapley interface."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

from shapley_core import ValueFunction
from value_quantification import DataValueEvaluator

class DataValueEvaluatorFunction(ValueFunction):
    """Wraps a DataValueEvaluator so it can be used by ContributionEvaluator."""

    def __init__(
        self,
        data_value_evaluator: DataValueEvaluator,
        model_output: np.ndarray,
        data_combinations: np.ndarray,
        feature_names: Optional[Sequence[str]] = None,
    ):
        self.data_value_evaluator = data_value_evaluator
        self.model_output = np.asarray(model_output)
        if self.model_output.ndim == 1:
            self.model_output = self.model_output.reshape(1, -1)
        self.data_combinations = data_combinations
        self.feature_names = list(feature_names) if feature_names is not None else None

    def _subset_matrix(self, subset: Sequence[Any]) -> np.ndarray:
        if self.data_combinations.ndim != 2:
            raise ValueError("data_combinations must be a 2D matrix")

        indices: List[int] = []
        for item in subset:
            if isinstance(item, int):
                indices.append(int(item))
            elif isinstance(item, str):
                if self.feature_names is None:
                    raise ValueError("feature_names are required when subset items are strings")
                indices.append(self.feature_names.index(item))
            else:
                indices.append(int(item))

        if len(indices) == 0:
            return np.empty((self.data_combinations.shape[0], 0))

        # data_combinations is organized as features x samples (4 x n)
        return self.data_combinations[indices, :]

    def _combo_name(self, subset: Sequence[Any]) -> str:
        if len(subset) == 0:
            return "empty_subset"

        names: List[str] = []
        for item in subset:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, int):
                if self.feature_names is not None:
                    names.append(self.feature_names[item])
                else:
                    names.append(str(item))
            else:
                names.append(str(item))

        return "+".join(names)

    def evaluate(self, subset: Sequence[Any]) -> float:
        """Compute the value of a subset using DataValueEvaluator.

        The subset is expected to specify selected feature indices or names.
        """
        sorted_subset = sorted(subset)
        subset_matrix = self._subset_matrix(sorted_subset)
        combo_name = self._combo_name(sorted_subset)
        combo_values = self.data_value_evaluator.evaluate_combination(
            subset_matrix,
            self.model_output,
            combo_name,
        )
        return sum(combo_values.values())
