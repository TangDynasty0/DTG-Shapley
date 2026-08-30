"""Comparison estimators used in the DTG-Shapley paper."""

from __future__ import annotations

import itertools
import logging
import math
import random
from typing import Dict, List, Optional, Sequence

import numpy as np

from shapley_core import ContributionResult, ValueFunction

class SampleContributionEvaluator:
    """Estimate Shapley contributions for a collection of data points."""

    def __init__(
        self,
        data: Sequence[Any],
        value_function: ValueFunction,
        n_samples: int = 100,
        random_seed: Optional[int] = None,
    ):
        self.data = list(data)
        self.value_function = value_function
        self.n_samples = n_samples
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)
        self._subset_value_cache: Dict[tuple[int, ...], float] = {}

    def estimate_shapley(self) -> ContributionResult:
        """Approximate Shapley values using random permutations.

        The algorithm samples random permutations of the data indices and
        accumulates marginal contributions for each item.
        """
        data_count = len(self.data)
        if self.n_samples <= 0:
            raise ValueError("n_samples must be greater than zero.")

        shapley_values: Dict[int, float] = {i: 0.0 for i in range(data_count)}

        for _ in range(self.n_samples):
            permutation = self._sample_random_permutation(data_count)
            marginal_scores = self._compute_marginal_contributions(permutation)
            logging.info(f"Permutation: {permutation}, Marginal Contributions: {marginal_scores}")
            for index, score in marginal_scores.items():
                shapley_values[index] += score

        for index in shapley_values:
            shapley_values[index] /= max(1, self.n_samples)

        return ContributionResult(
            shapley_values=shapley_values,
            sample_count=self.n_samples,
            data_count=data_count,
            diagnostics={"value_evaluations": len(self._subset_value_cache)},
        )

    def _sample_random_permutation(self, data_count: int) -> List[int]:
        permutation = list(range(data_count))
        self.rng.shuffle(permutation)
        return permutation

    def _compute_marginal_contributions(self, permutation: Sequence[int]) -> Dict[int, float]:
        marginal_contributions: Dict[int, float] = {}
        current_subset_indices: List[int] = []
        current_value = self._evaluate_subset(current_subset_indices)

        for item_index in permutation:
            next_subset = current_subset_indices + [item_index]
            next_value = self._evaluate_subset(next_subset)
            marginal_contributions[item_index] = next_value - current_value
            current_value = next_value
            current_subset_indices = next_subset

        return marginal_contributions

    def _evaluate_subset(self, indices: Sequence[int]) -> float:
        key = tuple(sorted(indices))
        if key not in self._subset_value_cache:
            subset = [self.data[i] for i in key]
            self._subset_value_cache[key] = self.value_function.evaluate(subset)
        return self._subset_value_cache[key]

class AntitheticSampleContributionEvaluator(SampleContributionEvaluator):
    """Permutation Monte Carlo with paired reverse permutations.

    ``n_samples`` is the total number of paths, preserving the same nominal
    marginal-sample budget as ordinary permutation Monte Carlo.
    """

    def estimate_shapley(self) -> ContributionResult:
        data_count = len(self.data)
        if self.n_samples <= 0:
            raise ValueError("n_samples must be greater than zero.")

        shapley_values = {index: 0.0 for index in range(data_count)}
        completed_paths = 0
        paired_paths = 0
        while completed_paths < self.n_samples:
            permutation = self._sample_random_permutation(data_count)
            permutations = [permutation]
            if completed_paths + 1 < self.n_samples:
                permutations.append(list(reversed(permutation)))
                paired_paths += 2
            for path in permutations:
                marginal_scores = self._compute_marginal_contributions(path)
                for index, score in marginal_scores.items():
                    shapley_values[index] += score
            completed_paths += len(permutations)

        for index in shapley_values:
            shapley_values[index] /= self.n_samples

        return ContributionResult(
            shapley_values=shapley_values,
            sample_count=self.n_samples,
            data_count=data_count,
            diagnostics={
                "value_evaluations": len(self._subset_value_cache),
                "sampling_strategy": "antithetic_permutation_mc",
                "completed_permutation_paths": completed_paths,
                "paired_reverse_paths": paired_paths,
            },
        )

class ExactShapleyEvaluator:
    """Compute exact Shapley values by enumerating every subset."""

    def __init__(
        self,
        data: Sequence[Any],
        value_function: ValueFunction,
    ):
        self.data = list(data)
        self.value_function = value_function
        self.data_count = len(self.data)
        self._subset_value_cache: Dict[tuple[int, ...], float] = {}

    def estimate_shapley(self) -> ContributionResult:
        """Compute exact Shapley values for all data points."""
        data_count = self.data_count
        if data_count == 0:
            return ContributionResult(shapley_values={}, sample_count=0, data_count=0)

        if data_count > 20:
            raise ValueError(
                "Exact Shapley computation is only feasible for small datasets (20 items or fewer)."
            )

        shapley_values: Dict[int, float] = {i: 0.0 for i in range(data_count)}
        total_factorial = math.factorial(data_count)

        for subset_size in range(data_count):
            weight = (
                math.factorial(subset_size)
                * math.factorial(data_count - subset_size - 1)
                / total_factorial
            )
            for subset in itertools.combinations(range(data_count), subset_size):
                subset_value = self._evaluate_subset(subset)
                subset_set = set(subset)

                for item_index in range(data_count):
                    if item_index in subset_set:
                        continue

                    next_subset = (*subset, item_index)
                    next_value = self._evaluate_subset(next_subset)
                    shapley_values[item_index] += weight * (next_value - subset_value)

        return ContributionResult(
            shapley_values=shapley_values,
            sample_count=0,
            data_count=data_count,
            diagnostics={"value_evaluations": len(self._subset_value_cache)},
        )

    def _evaluate_subset(self, indices: Sequence[int]) -> float:
        key = tuple(sorted(indices))
        if key in self._subset_value_cache:
            return self._subset_value_cache[key]

        subset = [self.data[i] for i in indices]
        value = self.value_function.evaluate(subset)
        self._subset_value_cache[key] = value
        return value

class TMCShapleyEvaluator:
    """Estimate Shapley values with Truncated Monte Carlo permutations.

    This implements Algorithm 1 from Ghorbani and Zou (ICML 2019). Each
    iteration scans a uniformly sampled permutation. Once the current
    coalition value is within ``truncation_tolerance`` of the full-coalition
    value, the remaining marginal contributions in that permutation are set
    to zero.
    """

    def __init__(
        self,
        data: Sequence[Any],
        value_function: ValueFunction,
        n_samples: int = 100,
        random_seed: Optional[int] = None,
        truncation_tolerance: float = 0.01,
        truncation_tolerance_ratio: Optional[float] = None,
    ):
        if n_samples <= 0:
            raise ValueError("n_samples must be greater than zero.")
        if truncation_tolerance < 0:
            raise ValueError("truncation_tolerance must be non-negative.")
        if (
            truncation_tolerance_ratio is not None
            and truncation_tolerance_ratio < 0
        ):
            raise ValueError(
                "truncation_tolerance_ratio must be non-negative."
            )

        self.data = list(data)
        self.value_function = value_function
        self.data_count = len(self.data)
        self.n_samples = n_samples
        self.random_seed = random_seed
        self.truncation_tolerance = float(truncation_tolerance)
        self.truncation_tolerance_ratio = (
            None
            if truncation_tolerance_ratio is None
            else float(truncation_tolerance_ratio)
        )
        self.rng = random.Random(random_seed)
        self._subset_value_cache: Dict[tuple[int, ...], float] = {}

    def estimate_shapley(self) -> ContributionResult:
        """Estimate all contributions using a fixed number of TMC iterations."""
        if self.data_count == 0:
            return ContributionResult(shapley_values={}, sample_count=0, data_count=0)

        contributions: Dict[int, float] = {
            index: 0.0 for index in range(self.data_count)
        }
        full_value = self._evaluate_subset(range(self.data_count))
        empty_value = self._evaluate_subset(())
        truncated_permutations = 0
        evaluated_marginals = 0
        observation_counts = {
            index: 0 for index in range(self.data_count)
        }
        effective_tolerance = self.truncation_tolerance
        if self.truncation_tolerance_ratio is not None:
            effective_tolerance = (
                self.truncation_tolerance_ratio
                * abs(full_value - empty_value)
            )

        for _ in range(self.n_samples):
            permutation = list(range(self.data_count))
            self.rng.shuffle(permutation)
            current_indices: List[int] = []
            current_value = empty_value

            for item_index in permutation:
                if abs(full_value - current_value) <= effective_tolerance:
                    truncated_permutations += 1
                    break

                current_indices.append(item_index)
                next_value = self._evaluate_subset(current_indices)
                contributions[item_index] += next_value - current_value
                observation_counts[item_index] += 1
                current_value = next_value
                evaluated_marginals += 1

        for item_index in contributions:
            contributions[item_index] /= self.n_samples

        return ContributionResult(
            shapley_values=contributions,
            sample_count=self.n_samples,
            data_count=self.data_count,
            diagnostics={
                "value_evaluations": len(self._subset_value_cache),
                "evaluated_marginals": evaluated_marginals,
                "total_marginal_samples": evaluated_marginals,
                "formal_observation_counts": observation_counts,
                "total_observation_counts": observation_counts,
                "truncated_permutations": truncated_permutations,
                "truncation_rate": truncated_permutations / self.n_samples,
                "effective_truncation_tolerance": effective_tolerance,
                "truncation_tolerance_ratio": self.truncation_tolerance_ratio,
                "coalition_evaluation_requests": evaluated_marginals + 2,
            },
        )

    def _evaluate_subset(self, indices: Sequence[int]) -> float:
        key = tuple(sorted(indices))
        if key not in self._subset_value_cache:
            subset = [self.data[i] for i in key]
            self._subset_value_cache[key] = self.value_function.evaluate(subset)
        return self._subset_value_cache[key]

class StratifiedSamplingShapleyEvaluator:
    """Estimate Shapley values by stratifying marginal contributions by size.

    For each item, coalitions not containing that item are partitioned into
    ``n`` strata according to coalition size. The estimator samples uniformly
    within every stratum and averages the stratum means with equal weight.
    ``n_samples`` is the total marginal-contribution budget per item.
    """

    def __init__(
        self,
        data: Sequence[Any],
        value_function: ValueFunction,
        n_samples: int = 100,
        random_seed: Optional[int] = None,
    ):
        if n_samples <= 0:
            raise ValueError("n_samples must be greater than zero.")

        self.data = list(data)
        self.value_function = value_function
        self.data_count = len(self.data)
        if self.data_count > 0 and n_samples < self.data_count:
            raise ValueError(
                "n_samples must be at least the number of data items so every "
                "coalition-size stratum receives a sample."
            )

        self.n_samples = n_samples
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)
        self._subset_value_cache: Dict[tuple[int, ...], float] = {}

    def estimate_shapley(self) -> ContributionResult:
        """Estimate every item's contribution from coalition-size strata."""
        if self.data_count == 0:
            return ContributionResult(shapley_values={}, sample_count=0, data_count=0)

        stratum_sample_counts = self._allocate_samples_to_strata()
        contributions: Dict[int, float] = {}

        for item_index in range(self.data_count):
            other_indices = [
                index for index in range(self.data_count) if index != item_index
            ]
            stratum_means: List[float] = []

            for coalition_size, stratum_count in enumerate(stratum_sample_counts):
                marginal_total = 0.0
                for _ in range(stratum_count):
                    coalition = tuple(
                        sorted(self.rng.sample(other_indices, coalition_size))
                    )
                    coalition_with_item = tuple(sorted((*coalition, item_index)))
                    marginal_total += (
                        self._evaluate_subset(coalition_with_item)
                        - self._evaluate_subset(coalition)
                    )
                stratum_means.append(marginal_total / stratum_count)

            contributions[item_index] = sum(stratum_means) / self.data_count

        full_value = self._evaluate_subset(range(self.data_count))
        empty_value = self._evaluate_subset(())
        efficiency_residual = float(
            sum(contributions.values()) - (full_value - empty_value)
        )

        return ContributionResult(
            shapley_values=contributions,
            sample_count=self.n_samples,
            data_count=self.data_count,
            diagnostics={
                "value_evaluations": len(self._subset_value_cache),
                "marginal_samples_per_item": self.n_samples,
                "total_marginal_samples": self.n_samples * self.data_count,
                "coalition_evaluation_requests": (
                    2 * self.n_samples * self.data_count + 2
                ),
                "efficiency_residual": efficiency_residual,
                "stratum_sample_counts": {
                    str(size): count
                    for size, count in enumerate(stratum_sample_counts)
                },
            },
        )

    def _allocate_samples_to_strata(self) -> List[int]:
        base_count, remainder = divmod(self.n_samples, self.data_count)
        return [
            base_count + (1 if size < remainder else 0)
            for size in range(self.data_count)
        ]

    def _evaluate_subset(self, indices: Sequence[int]) -> float:
        key = tuple(sorted(indices))
        if key not in self._subset_value_cache:
            subset = [self.data[i] for i in key]
            self._subset_value_cache[key] = self.value_function.evaluate(subset)
        return self._subset_value_cache[key]

class KernelSHAPEvaluator:
    """Estimate Shapley values using Shapley-kernel weighted regression.

    Empty- and full-coalition constraints are enforced analytically. When the
    requested budget covers all non-trivial coalitions, the regression uses
    the complete coalition set. Otherwise, coalition sizes are sampled evenly
    and regression weights correct that sampling distribution back to the
    Shapley kernel.
    """

    _MAX_EXHAUSTIVE_COALITIONS = 100_000

    def __init__(
        self,
        data: Sequence[Any],
        value_function: ValueFunction,
        n_samples: int = 100,
        random_seed: Optional[int] = None,
    ):
        if n_samples <= 0:
            raise ValueError("n_samples must be greater than zero.")

        self.data = list(data)
        self.value_function = value_function
        self.data_count = len(self.data)
        if self.data_count > 1 and n_samples < self.data_count - 1:
            raise ValueError(
                "n_samples must be at least data_count - 1 so every non-trivial "
                "coalition-size stratum receives a sample."
            )

        self.n_samples = n_samples
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)
        self._subset_value_cache: Dict[tuple[int, ...], float] = {}

    def estimate_shapley(self) -> ContributionResult:
        """Fit the constrained Shapley-kernel regression model."""
        if self.data_count == 0:
            return ContributionResult(shapley_values={}, sample_count=0, data_count=0)

        empty_value = self._evaluate_subset(())
        full_value = self._evaluate_subset(range(self.data_count))
        total_contribution = full_value - empty_value

        if self.data_count == 1:
            return ContributionResult(
                shapley_values={0: total_contribution},
                sample_count=0,
                data_count=1,
                diagnostics={
                    "value_evaluations": len(self._subset_value_cache),
                    "sampling_mode": "single_item_exact",
                    "coalition_samples": 0,
                    "regression_rank": 0,
                    "efficiency_residual": 0.0,
                },
            )

        coalitions, weights, sampling_mode = self._sample_coalitions()
        masks = np.zeros((len(coalitions), self.data_count), dtype=float)
        coalition_values = np.empty(len(coalitions), dtype=float)

        for row_index, coalition in enumerate(coalitions):
            masks[row_index, list(coalition)] = 1.0
            coalition_values[row_index] = self._evaluate_subset(coalition)

        last_column = masks[:, -1]
        design = masks[:, :-1] - last_column[:, np.newaxis]
        target = coalition_values - empty_value - last_column * total_contribution
        sqrt_weights = np.sqrt(np.asarray(weights, dtype=float))
        weighted_design = design * sqrt_weights[:, np.newaxis]
        weighted_target = target * sqrt_weights
        regression_rank = int(np.linalg.matrix_rank(weighted_design))

        if regression_rank < self.data_count - 1:
            raise ValueError(
                "KernelSHAP regression is rank deficient. Increase n_samples or "
                "use a different random_seed."
            )

        coefficients, residuals, _, singular_values = np.linalg.lstsq(
            weighted_design,
            weighted_target,
            rcond=None,
        )
        final_coefficient = total_contribution - float(np.sum(coefficients))
        all_coefficients = np.append(coefficients, final_coefficient)
        contributions = {
            index: float(value) for index, value in enumerate(all_coefficients)
        }
        efficiency_residual = float(
            sum(contributions.values()) - total_contribution
        )

        return ContributionResult(
            shapley_values=contributions,
            sample_count=len(coalitions),
            data_count=self.data_count,
            diagnostics={
                "value_evaluations": len(self._subset_value_cache),
                "requested_coalition_samples": self.n_samples,
                "coalition_samples": len(coalitions),
                "coalition_evaluation_requests": len(coalitions) + 2,
                "sampling_mode": sampling_mode,
                "regression_rank": regression_rank,
                "regression_residual_sum": (
                    float(residuals[0]) if residuals.size else 0.0
                ),
                "smallest_singular_value": float(singular_values[-1]),
                "efficiency_residual": efficiency_residual,
            },
        )

    def _sample_coalitions(
        self,
    ) -> tuple[List[tuple[int, ...]], List[float], str]:
        non_trivial_count = (1 << self.data_count) - 2
        if (
            self.n_samples >= non_trivial_count
            and non_trivial_count <= self._MAX_EXHAUSTIVE_COALITIONS
        ):
            coalitions: List[tuple[int, ...]] = []
            weights: List[float] = []
            for coalition_size in range(1, self.data_count):
                kernel_weight = self._shapley_kernel_weight(coalition_size)
                for coalition in itertools.combinations(
                    range(self.data_count), coalition_size
                ):
                    coalitions.append(coalition)
                    weights.append(kernel_weight)
            return coalitions, weights, "exhaustive"

        stratum_count = self.data_count - 1
        base_count, remainder = divmod(self.n_samples, stratum_count)
        coalitions = []
        weights = []

        for coalition_size in range(1, self.data_count):
            samples_in_stratum = base_count + (
                1 if coalition_size <= remainder else 0
            )
            importance_weight = 1.0 / (
                samples_in_stratum
                * coalition_size
                * (self.data_count - coalition_size)
            )
            for _ in range(samples_in_stratum):
                coalition = tuple(
                    sorted(
                        self.rng.sample(
                            range(self.data_count),
                            coalition_size,
                        )
                    )
                )
                coalitions.append(coalition)
                weights.append(importance_weight)

        return coalitions, weights, "stratified_kernel_sampling"

    def _shapley_kernel_weight(self, coalition_size: int) -> float:
        return (self.data_count - 1) / (
            math.comb(self.data_count, coalition_size)
            * coalition_size
            * (self.data_count - coalition_size)
        )

    def _evaluate_subset(self, indices: Sequence[int]) -> float:
        key = tuple(sorted(indices))
        if key not in self._subset_value_cache:
            subset = [self.data[i] for i in key]
            self._subset_value_cache[key] = self.value_function.evaluate(subset)
        return self._subset_value_cache[key]
