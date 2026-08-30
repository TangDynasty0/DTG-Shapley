"""Public DTG-Shapley implementation.

Only the paper's final algorithm is exposed as ``DTGShapleyEvaluator``.
The private base contains shared sampling and audit helpers and is not a
separate public algorithm version.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence

from shapley_core import ContributionResult, ValueFunction

class _DTGImplementationBase:
    """Dynamic Tail Grouping estimator for threshold-aware Shapley sampling.

    Screening repeatedly updates high, uncertain, and low groups from sampled
    permutation marginals. Low members remain eligible for individual and
    coalition audits so context-dependent contributors can be promoted. After
    grouping stabilizes, an independent coalition-size-stratified formal stage
    estimates only high and uncertain members while retaining classical
    Shapley as the estimand.

    Low status is defined per member, not by the low group's total value.
    Individual low-member values are deliberately approximate and receive the
    efficiency residual after estimating the selected high group. The focused
    stage stops after n_samples per selected member. If that would not reduce
    requests relative to n_samples full MC paths, the estimator falls back.
    """

    def __init__(
        self,
        data: Sequence[Any],
        value_function: ValueFunction,
        n_samples: int = 100,
        contribution_threshold: float = 0.02,
        threshold_margin: float = 0.005,
        confidence_z: float = 1.96,
        initial_samples: int = 8,
        check_interval: int = 5,
        stability_checks: int = 2,
        min_observations: int = 6,
        max_screening_budget_ratio: float = 0.2,
        low_audit_rate: float = 0.15,
        coalition_audits_per_check: int = 2,
        interaction_tolerance: float = 0.02,
        interaction_hold_checks: int = 2,
        min_low_count: int = 2,
        min_focused_savings_ratio: float = 0.05,
        enable_dynamic_regrouping: bool = True,
        enable_coalition_audit: bool = True,
        random_seed: Optional[int] = None,
    ):
        if n_samples <= 0:
            raise ValueError("n_samples must be greater than zero.")
        if not 0.0 < contribution_threshold < 1.0:
            raise ValueError("contribution_threshold must be in (0, 1).")
        if not 0.0 <= threshold_margin < contribution_threshold:
            raise ValueError(
                "threshold_margin must be non-negative and smaller than "
                "contribution_threshold."
            )
        if not math.isfinite(confidence_z) or confidence_z < 0.0:
            raise ValueError("confidence_z must be finite and non-negative.")
        for name, value in (
            ("initial_samples", initial_samples),
            ("check_interval", check_interval),
            ("stability_checks", stability_checks),
            ("min_observations", min_observations),
            ("interaction_hold_checks", interaction_hold_checks),
            ("min_low_count", min_low_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not 0.0 < max_screening_budget_ratio < 1.0:
            raise ValueError("max_screening_budget_ratio must be in (0, 1).")
        if not 0.0 <= low_audit_rate <= 1.0:
            raise ValueError("low_audit_rate must be in [0, 1].")
        if (
            isinstance(coalition_audits_per_check, bool)
            or not isinstance(coalition_audits_per_check, int)
            or coalition_audits_per_check < 0
        ):
            raise ValueError(
                "coalition_audits_per_check must be a non-negative integer."
            )
        if not math.isfinite(interaction_tolerance) or interaction_tolerance < 0.0:
            raise ValueError("interaction_tolerance must be finite and non-negative.")
        if not 0.0 <= min_focused_savings_ratio < 1.0:
            raise ValueError("min_focused_savings_ratio must be in [0, 1).")

        self.data = list(data)
        self.value_function = value_function
        self.data_count = len(self.data)
        self.n_samples = int(n_samples)
        self.contribution_threshold = float(contribution_threshold)
        self.threshold_margin = float(threshold_margin)
        self.confidence_z = float(confidence_z)
        self.initial_samples = int(initial_samples)
        self.check_interval = int(check_interval)
        self.stability_checks = int(stability_checks)
        self.min_observations = int(min_observations)
        self.max_screening_budget_ratio = float(max_screening_budget_ratio)
        self.low_audit_rate = float(low_audit_rate)
        self.coalition_audits_per_check = coalition_audits_per_check
        self.interaction_tolerance = float(interaction_tolerance)
        self.interaction_hold_checks = interaction_hold_checks
        self.min_low_count = min_low_count
        self.min_focused_savings_ratio = float(min_focused_savings_ratio)
        self.enable_dynamic_regrouping = bool(enable_dynamic_regrouping)
        self.enable_coalition_audit = bool(enable_coalition_audit)
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)
        self._subset_value_cache: Dict[tuple[int, ...], float] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._evaluation_requests = 0

    @staticmethod
    def _sample_variance(values: Sequence[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((value - mean) ** 2 for value in values) / (len(values) - 1)

    @classmethod
    def _standard_error(cls, values: Sequence[float]) -> float:
        if len(values) < 2:
            return math.inf
        return math.sqrt(max(0.0, cls._sample_variance(values)) / len(values))

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _uniform_permutation(self) -> List[int]:
        permutation = list(range(self.data_count))
        self.rng.shuffle(permutation)
        return permutation

    def _full_path_marginals(
        self,
        permutation: Sequence[int],
        empty_value: float,
    ) -> Dict[int, float]:
        marginals = {index: 0.0 for index in range(self.data_count)}
        selected: List[int] = []
        current_value = empty_value
        for item_index in permutation:
            selected.append(item_index)
            next_value = self._evaluate_subset(selected)
            marginals[item_index] = next_value - current_value
            current_value = next_value
        return marginals

    def _direct_marginals(
        self,
        permutation: Sequence[int],
        selected_indices: Sequence[int],
    ) -> Dict[int, float]:
        positions = {index: position for position, index in enumerate(permutation)}
        marginals: Dict[int, float] = {}
        for item_index in selected_indices:
            predecessor = permutation[: positions[item_index]]
            before = self._evaluate_subset(predecessor)
            after = self._evaluate_subset((*predecessor, item_index))
            marginals[item_index] = after - before
        return marginals

    def _direct_marginal_at_size(
        self,
        item_index: int,
        predecessor_size: int,
    ) -> float:
        others = [
            index for index in range(self.data_count) if index != item_index
        ]
        predecessor = self.rng.sample(others, predecessor_size)
        before = self._evaluate_subset(predecessor)
        after = self._evaluate_subset((*predecessor, item_index))
        return after - before

    def _stratified_formal_observations(
        self,
        selected_indices: Sequence[int],
    ) -> tuple[
        Dict[int, List[float]],
        Dict[int, Dict[int, List[float]]],
        str,
    ]:
        observations = {index: [] for index in range(self.data_count)}
        strata = {
            index: {size: [] for size in range(self.data_count)}
            for index in selected_indices
        }
        strategy = (
            "coalition_size_stratified"
            if self.n_samples >= self.data_count
            else "uniform_size_direct"
        )
        for item_index in selected_indices:
            sizes: List[int] = []
            if strategy == "coalition_size_stratified":
                while len(sizes) < self.n_samples:
                    cycle = list(range(self.data_count))
                    self.rng.shuffle(cycle)
                    sizes.extend(cycle)
                sizes = sizes[: self.n_samples]
            else:
                sizes = [
                    self.rng.randrange(self.data_count)
                    for _ in range(self.n_samples)
                ]
            for predecessor_size in sizes:
                marginal = self._direct_marginal_at_size(
                    item_index,
                    predecessor_size,
                )
                observations[item_index].append(marginal)
                strata[item_index][predecessor_size].append(marginal)
        return observations, strata, strategy

    def _stratified_estimate(
        self,
        values: Sequence[float],
        strata: Dict[int, List[float]],
    ) -> tuple[float, float]:
        populated = [samples for samples in strata.values() if samples]
        if len(populated) != self.data_count:
            return self._mean(values), self._standard_error(values)
        estimate = sum(self._mean(samples) for samples in populated) / self.data_count
        variance = sum(
            self._sample_variance(samples) / len(samples)
            for samples in populated
        ) / (self.data_count**2)
        return estimate, math.sqrt(max(0.0, variance))

    @staticmethod
    def _allocate_with_individual_cap(
        indices: Sequence[int],
        residual: float,
        weights: Dict[int, float],
        cap: float,
    ) -> tuple[Dict[int, float], bool]:
        if not indices:
            return {}, abs(residual) > 1e-12
        if residual <= 0.0:
            weight_sum = sum(weights.values())
            if weight_sum <= 0.0:
                return {
                    index: residual / len(indices) for index in indices
                }, False
            return {
                index: residual * weights[index] / weight_sum
                for index in indices
            }, False

        cap_violation = residual > cap * len(indices) + 1e-12
        if cap_violation:
            weight_sum = sum(weights.values())
            if weight_sum <= 0.0:
                return {
                    index: residual / len(indices) for index in indices
                }, True
            return {
                index: residual * weights[index] / weight_sum
                for index in indices
            }, True

        allocation = {index: 0.0 for index in indices}
        remaining = set(indices)
        remaining_value = residual
        while remaining:
            weight_sum = sum(weights[index] for index in remaining)
            if weight_sum <= 0.0:
                proposals = {
                    index: remaining_value / len(remaining)
                    for index in remaining
                }
            else:
                proposals = {
                    index: remaining_value * weights[index] / weight_sum
                    for index in remaining
                }
            capped = [
                index for index, value in proposals.items() if value > cap
            ]
            if not capped:
                for index, value in proposals.items():
                    allocation[index] += value
                break
            for index in capped:
                allocation[index] = cap
                remaining_value -= cap
                remaining.remove(index)
        return allocation, False

    def _classify(
        self,
        observations: Dict[int, List[float]],
        total_value: float,
        check_index: int,
        hold_until: Dict[int, int],
    ) -> tuple[List[int], List[int], List[int], Dict[int, Dict[str, float]]]:
        low_boundary = (self.contribution_threshold - self.threshold_margin) * total_value
        high_boundary = (self.contribution_threshold + self.threshold_margin) * total_value
        high: List[int] = []
        uncertain: List[int] = []
        low: List[int] = []
        statistics: Dict[int, Dict[str, float]] = {}
        for index in range(self.data_count):
            values = observations[index]
            mean = self._mean(values)
            standard_error = self._standard_error(values)
            radius = self.confidence_z * standard_error
            lower = mean - radius
            upper = mean + radius
            statistics[index] = {
                "mean": mean,
                "standard_error": standard_error,
                "lower": lower,
                "upper": upper,
                "observations": float(len(values)),
            }
            if len(values) < self.min_observations or not math.isfinite(radius):
                uncertain.append(index)
            elif lower >= high_boundary:
                high.append(index)
            elif upper <= low_boundary and hold_until.get(index, 0) < check_index:
                low.append(index)
            else:
                uncertain.append(index)
        return high, uncertain, low, statistics

    def _coalition_audits(
        self,
        low_group: Sequence[int],
        observations: Dict[int, List[float]],
        total_value: float,
        available_budget: int,
    ) -> tuple[List[Dict[str, Any]], set[int]]:
        if (
            not self.enable_coalition_audit
            or
            self.coalition_audits_per_check == 0
            or len(low_group) == 0
            or available_budget < 2
        ):
            return [], set()
        audits: List[Dict[str, Any]] = []
        suspects: set[int] = set()
        maximum_audits = min(
            self.coalition_audits_per_check,
            available_budget // 2,
        )
        low_list = list(low_group)
        for audit_index in range(maximum_audits):
            if audit_index == 0 or len(low_list) <= 2:
                audited = list(low_list)
            else:
                size = self.rng.randint(1, min(4, len(low_list)))
                audited = self.rng.sample(low_list, size)
            audited_set = set(audited)
            outside = [
                index for index in range(self.data_count) if index not in audited_set
            ]
            self.rng.shuffle(outside)
            context_size = self.rng.randint(0, len(outside))
            context = outside[:context_size]
            before = self._evaluate_subset(context)
            after = self._evaluate_subset((*context, *audited))
            observed = after - before
            predicted = sum(self._mean(observations[index]) for index in audited)
            residual = observed - predicted
            triggered = residual > self.interaction_tolerance * total_value
            if triggered:
                suspects.update(audited)
            audits.append(
                {
                    "members": list(audited),
                    "context_size": context_size,
                    "observed_group_marginal": observed,
                    "predicted_member_sum": predicted,
                    "interaction_residual": residual,
                    "triggered": triggered,
                }
            )
        return audits, suspects

    @staticmethod
    def _append_observations(
        observations: Dict[int, List[float]],
        marginals: Dict[int, float],
    ) -> None:
        for index, value in marginals.items():
            observations[index].append(value)

    def estimate_shapley(self) -> ContributionResult:
        if self.data_count == 0:
            return ContributionResult(shapley_values={}, sample_count=0, data_count=0)

        empty_value = self._evaluate_subset(())
        full_value = self._evaluate_subset(range(self.data_count))
        total_value = full_value - empty_value
        total_budget = self.n_samples * self.data_count
        screening_cap = min(
            total_budget,
            max(
                self.data_count,
                int(total_budget * self.max_screening_budget_ratio),
            ),
        )
        screening_start = self._evaluation_requests
        observations = {index: [] for index in range(self.data_count)}
        history: List[Dict[str, Any]] = []
        audit_history: List[Dict[str, Any]] = []
        hold_until: Dict[int, int] = {}

        if total_value <= 0.0:
            screening_cap = 0
        initial_count = min(
            self.initial_samples,
            screening_cap // max(1, self.data_count),
        )
        for _ in range(initial_count):
            marginals = self._full_path_marginals(
                self._uniform_permutation(), empty_value
            )
            self._append_observations(observations, marginals)
        initial_full_observations = {
            index: list(values) for index, values in observations.items()
        }

        check_index = 1
        high, uncertain, low, statistics = self._classify(
            observations, max(total_value, 1e-12), check_index, hold_until
        )
        previous_signature: Optional[tuple[tuple[int, ...], ...]] = None
        stable_count = 0
        dynamic_rounds = 0

        while total_value > 0.0:
            screening_used = self._evaluation_requests - screening_start
            available = screening_cap - screening_used
            audits, suspects = self._coalition_audits(
                low, observations, total_value, available
            )
            for audit in audits:
                audit_history.append({"check": check_index, **audit})
            if suspects:
                for index in suspects:
                    hold_until[index] = check_index + self.interaction_hold_checks
                low = [index for index in low if index not in suspects]
                uncertain = sorted(set((*uncertain, *suspects)))

            signature = (tuple(sorted(high)), tuple(sorted(uncertain)), tuple(sorted(low)))
            if signature == previous_signature and not suspects:
                stable_count += 1
            else:
                stable_count = 0
            previous_signature = signature
            history.append(
                {
                    "check": check_index,
                    "screening_requests": self._evaluation_requests - screening_start,
                    "high": list(high),
                    "uncertain": list(uncertain),
                    "low": list(low),
                    "interaction_suspects": sorted(suspects),
                    "statistics": {
                        str(index): dict(values)
                        for index, values in statistics.items()
                    },
                }
            )
            if not self.enable_dynamic_regrouping:
                break
            if stable_count >= self.stability_checks:
                break

            completed_rounds = 0
            for _ in range(self.check_interval):
                active = sorted(set((*high, *uncertain)))
                audited_low = [
                    index for index in low if self.rng.random() < self.low_audit_rate
                ]
                if low and not audited_low and self.low_audit_rate > 0.0:
                    audited_low = [self.rng.choice(low)]
                selected = sorted(set((*active, *audited_low)))
                required = 2 * len(selected)
                screening_used = self._evaluation_requests - screening_start
                if required == 0 or screening_used + required > screening_cap:
                    break
                marginals = self._direct_marginals(
                    self._uniform_permutation(), selected
                )
                self._append_observations(observations, marginals)
                dynamic_rounds += 1
                completed_rounds += 1
            if completed_rounds == 0:
                break
            check_index += 1
            high, uncertain, low, statistics = self._classify(
                observations, total_value, check_index, hold_until
            )

        selected_high = sorted(set((*high, *uncertain)))
        certified_low = sorted(low)
        screening_requests = self._evaluation_requests - screening_start
        focused_formal_requests = 2 * len(selected_high) * self.n_samples
        projected_focused_requests = screening_requests + focused_formal_requests
        focused_is_economical = (
            len(certified_low) >= self.min_low_count
            and len(selected_high) > 0
            and projected_focused_requests
            <= total_budget * (1.0 - self.min_focused_savings_ratio)
        )
        mode = "focused_high_stratified" if focused_is_economical else "uniform_mc_fallback"
        formal_start = self._evaluation_requests
        formal_observations = {index: [] for index in range(self.data_count)}
        formal_strata: Dict[int, Dict[int, List[float]]] = {}
        formal_sampling_strategy = "uniform_permutation"
        low_allocation_cap = (
            self.contribution_threshold - self.threshold_margin
        ) * total_value
        low_cap_violation = False

        if mode == "focused_high_stratified":
            (
                formal_observations,
                formal_strata,
                formal_sampling_strategy,
            ) = self._stratified_formal_observations(selected_high)
            estimates = {
                index: self._stratified_estimate(
                    formal_observations[index],
                    formal_strata[index],
                )
                for index in selected_high
            }
            high_values = {index: value[0] for index, value in estimates.items()}
            residual = total_value - sum(high_values.values())
            positive_weights = {
                index: max(0.0, self._mean(observations[index]))
                for index in certified_low
            }
            weight_sum = sum(positive_weights.values())
            if certified_low and weight_sum <= 0.0:
                low_weights = {index: 1.0 for index in certified_low}
            elif certified_low:
                low_weights = dict(positive_weights)
            else:
                low_weights = {}
            low_values, low_cap_violation = self._allocate_with_individual_cap(
                certified_low,
                residual,
                low_weights,
                low_allocation_cap,
            )
            allocated_sum = sum(low_values.values())
            low_weights = {
                index: (
                    low_values[index] / allocated_sum
                    if abs(allocated_sum) > 1e-12
                    else 1.0 / len(certified_low)
                )
                for index in certified_low
            }
            contributions = {
                **high_values,
                **low_values,
            }
            standard_errors = {
                index: estimates[index][1]
                for index in selected_high
            }
            sample_count = self.n_samples
            fallback_reason = None
        else:
            formal_observations = {
                index: list(values)
                for index, values in initial_full_observations.items()
            }
            additional_count = max(0, self.n_samples - initial_count)
            for _ in range(additional_count):
                marginals = self._full_path_marginals(
                    self._uniform_permutation(), empty_value
                )
                self._append_observations(formal_observations, marginals)
            contributions = {
                index: self._mean(formal_observations[index])
                for index in range(self.data_count)
            }
            standard_errors = {
                index: self._standard_error(formal_observations[index])
                for index in range(self.data_count)
            }
            low_weights = {}
            sample_count = self.n_samples
            fallback_reason = (
                "insufficient_certified_low_members"
                if len(certified_low) < self.min_low_count
                else "focused_sampling_not_cost_effective"
            )

        formal_requests = self._evaluation_requests - formal_start
        formal_observation_counts = {
            str(index): len(values)
            for index, values in formal_observations.items()
        }
        reused_initial_count = (
            initial_count if mode == "uniform_mc_fallback" else 0
        )
        total_observation_counts = {
            str(index): (
                len(observations[index])
                + len(formal_observations[index])
                - reused_initial_count
            )
            for index in range(self.data_count)
        }
        total_marginal_samples = sum(total_observation_counts.values())
        baseline_marginal_samples = self.n_samples * self.data_count
        cache_requests = self._cache_hits + self._cache_misses
        return ContributionResult(
            shapley_values=contributions,
            sample_count=sample_count,
            data_count=self.data_count,
            diagnostics={
                "algorithm": "private_shared_helpers",
                "mode": mode,
                "fallback_reason": fallback_reason,
                "estimation_target": "threshold_selective_classical_shapley",
                "low_contribution_definition": (
                    "individual_normalized_shapley_below_threshold"
                ),
                "contribution_threshold": self.contribution_threshold,
                "threshold_margin": self.threshold_margin,
                "confidence_z": self.confidence_z,
                "enable_dynamic_regrouping": self.enable_dynamic_regrouping,
                "enable_coalition_audit": self.enable_coalition_audit,
                "high_values_unbiased_formal_stage": (
                    mode == "focused_high_stratified"
                ),
                "low_values_are_approximate": (
                    mode == "focused_high_stratified"
                ),
                "formal_sampling_strategy": formal_sampling_strategy,
                "target_samples_per_high_member": self.n_samples,
                "total_budget_units": total_budget,
                "budget_unit": "ordinary_mc_coalition_request_baseline",
                "screening_budget_cap": screening_cap,
                "screening_budget_units": screening_requests,
                "formal_budget_units": formal_requests,
                "unused_budget_units": total_budget - screening_requests - formal_requests,
                "projected_focused_requests": projected_focused_requests,
                "request_reduction_vs_mc": (
                    1.0
                    - (screening_requests + formal_requests) / total_budget
                    if total_budget
                    else 0.0
                ),
                "initial_full_permutations": initial_count,
                "dynamic_screening_rounds": dynamic_rounds,
                "group_checks": len(history),
                "group_history": history,
                "coalition_audit_history": audit_history,
                "coalition_audit_count": len(audit_history),
                "triggered_coalition_audits": sum(
                    bool(audit["triggered"]) for audit in audit_history
                ),
                "high_group_positions": selected_high,
                "uncertain_group_positions": uncertain,
                "low_group_positions": certified_low,
                "high_group": [self.data[index] for index in selected_high],
                "uncertain_group": [self.data[index] for index in uncertain],
                "low_group": [self.data[index] for index in certified_low],
                "high_group_count": len(selected_high),
                "low_group_count": len(certified_low),
                "focused_samples": (
                    self.n_samples if mode == "focused_high_stratified" else 0
                ),
                "fallback_samples": (
                    sample_count if mode != "focused_high_stratified" else 0
                ),
                "formal_observation_counts": formal_observation_counts,
                "total_observation_counts": total_observation_counts,
                "total_marginal_samples": total_marginal_samples,
                "baseline_mc_marginal_samples": baseline_marginal_samples,
                "marginal_sample_reduction_vs_mc": (
                    1.0 - total_marginal_samples / baseline_marginal_samples
                    if baseline_marginal_samples
                    else 0.0
                ),
                "formal_stratum_sample_counts": {
                    str(index): {
                        str(size): len(samples)
                        for size, samples in sizes.items()
                    }
                    for index, sizes in formal_strata.items()
                },
                "low_allocation_individual_cap": low_allocation_cap,
                "low_allocation_cap_violation": low_cap_violation,
                "certified_low_max_estimated_value": max(
                    (contributions[index] for index in certified_low),
                    default=None,
                ),
                "certified_low_outputs_below_threshold": all(
                    contributions[index]
                    < self.contribution_threshold * total_value + 1e-12
                    for index in certified_low
                ),
                "low_allocation_weights": {
                    str(index): value for index, value in low_weights.items()
                },
                "screening_estimates": {
                    str(index): self._mean(values)
                    for index, values in observations.items()
                },
                "screening_observation_counts": {
                    str(index): len(values) for index, values in observations.items()
                },
                "standard_errors": {
                    str(index): value for index, value in standard_errors.items()
                },
                "value_evaluations": self._cache_misses,
                "value_function_calls": self._cache_misses,
                "coalition_evaluation_requests": self._evaluation_requests,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "cache_hit_rate": (
                    self._cache_hits / cache_requests if cache_requests else 0.0
                ),
                "cache_size": len(self._subset_value_cache),
                "efficiency_residual": sum(contributions.values()) - total_value,
            },
        )

    def _evaluate_subset(self, indices: Sequence[int]) -> float:
        self._evaluation_requests += 1
        key = tuple(sorted(set(indices)))
        if key in self._subset_value_cache:
            self._cache_hits += 1
            return self._subset_value_cache[key]
        subset = [self.data[index] for index in key]
        value = float(self.value_function.evaluate(subset))
        self._subset_value_cache[key] = value
        self._cache_misses += 1
        return value

class DTGShapleyEvaluator(_DTGImplementationBase):
    """Reusable, threshold-aware probability sampler for classical Shapley.

    The public estimator uses a fixed number of reusable sampling paths with a fixed number
    of reusable sampling paths. Inclusion probabilities are chosen from past
    observations only. Selected marginals receive Horvitz-Thompson correction,
    so every pre-fallback path remains part of the final estimate. A sampled
    permutation is paired with its reverse whenever the path budget permits.
    """

    _THRESHOLD_MODES = frozenset({"fixed", "relative", "hybrid"})

    def __init__(
        self,
        data: Sequence[Any],
        value_function: ValueFunction,
        n_samples: int = 100,
        threshold_mode: str = "relative",
        contribution_threshold: float = 0.04,
        relative_threshold_factor: float = 0.5,
        threshold_margin_ratio: float = 0.2,
        confidence_z: float = 1.96,
        initial_samples: int = 8,
        check_interval: int = 4,
        min_observations: int = 6,
        min_context_observations: int = 1,
        minimum_inclusion_probability: float = 0.10,
        min_adaptive_savings_ratio: float = 0.05,
        use_antithetic_sampling: bool = True,
        coalition_audits_per_check: int = 2,
        interaction_tolerance: float = 0.02,
        interaction_hold_checks: int = 2,
        enable_coalition_audit: bool = True,
        random_seed: Optional[int] = None,
    ):
        if threshold_mode not in self._THRESHOLD_MODES:
            choices = ", ".join(sorted(self._THRESHOLD_MODES))
            raise ValueError(f"threshold_mode must be one of: {choices}.")
        if not math.isfinite(relative_threshold_factor) or not (
            0.0 < relative_threshold_factor <= 1.0
        ):
            raise ValueError("relative_threshold_factor must be in (0, 1].")
        if not math.isfinite(threshold_margin_ratio) or not (
            0.0 <= threshold_margin_ratio < 1.0
        ):
            raise ValueError("threshold_margin_ratio must be in [0, 1).")
        if (
            isinstance(min_context_observations, bool)
            or not isinstance(min_context_observations, int)
            or min_context_observations < 0
        ):
            raise ValueError(
                "min_context_observations must be a non-negative integer."
            )
        if not math.isfinite(minimum_inclusion_probability) or not (
            0.0 < minimum_inclusion_probability <= 1.0
        ):
            raise ValueError(
                "minimum_inclusion_probability must be in (0, 1]."
            )
        if not math.isfinite(min_adaptive_savings_ratio) or not (
            0.0 <= min_adaptive_savings_ratio < 1.0
        ):
            raise ValueError("min_adaptive_savings_ratio must be in [0, 1).")

        # Reuse shared coalition-cache, marginal, and interaction-audit helpers.
        super().__init__(
            data=data,
            value_function=value_function,
            n_samples=n_samples,
            contribution_threshold=contribution_threshold,
            threshold_margin=min(0.005, contribution_threshold / 2.0),
            confidence_z=confidence_z,
            initial_samples=initial_samples,
            check_interval=check_interval,
            min_observations=min_observations,
            coalition_audits_per_check=coalition_audits_per_check,
            interaction_tolerance=interaction_tolerance,
            interaction_hold_checks=interaction_hold_checks,
            enable_coalition_audit=enable_coalition_audit,
            random_seed=random_seed,
        )
        self.threshold_mode = threshold_mode
        self.relative_threshold_factor = float(relative_threshold_factor)
        self.threshold_margin_ratio = float(threshold_margin_ratio)
        self.min_context_observations = min_context_observations
        self.minimum_inclusion_probability = float(
            minimum_inclusion_probability
        )
        self.min_adaptive_savings_ratio = float(min_adaptive_savings_ratio)
        self.use_antithetic_sampling = bool(use_antithetic_sampling)

    def _effective_threshold(self) -> float:
        relative = self.relative_threshold_factor / max(1, self.data_count)
        if self.threshold_mode == "fixed":
            return self.contribution_threshold
        if self.threshold_mode == "relative":
            return relative
        return max(self.contribution_threshold, relative)

    def _context_stratum(self, position: int) -> int:
        if self.data_count <= 1:
            return 1
        return min(2, (3 * position) // self.data_count)

    def _classify_probability_samples(
        self,
        corrected_sums: Dict[int, float],
        round_observations: Dict[int, List[float]],
        raw_observations: Dict[int, List[float]],
        context_counts: Dict[int, List[int]],
        completed_paths: int,
        total_value: float,
        check_index: int,
        hold_until: Dict[int, int],
    ) -> tuple[List[int], List[int], List[int], Dict[int, Dict[str, float]]]:
        threshold = self._effective_threshold()
        low_boundary = threshold * (1.0 - self.threshold_margin_ratio) * total_value
        high_boundary = threshold * (1.0 + self.threshold_margin_ratio) * total_value
        high: List[int] = []
        uncertain: List[int] = []
        low: List[int] = []
        statistics: Dict[int, Dict[str, float]] = {}
        for index in range(self.data_count):
            mean = (
                corrected_sums[index] / completed_paths
                if completed_paths > 0
                else 0.0
            )
            standard_error = self._standard_error(round_observations[index])
            radius = self.confidence_z * standard_error
            lower = mean - radius
            upper = mean + radius
            has_context_coverage = all(
                count >= self.min_context_observations
                for count in context_counts[index]
            )
            statistics[index] = {
                "mean": mean,
                "standard_error": standard_error,
                "lower": lower,
                "upper": upper,
                "observed_marginals": float(len(raw_observations[index])),
                "corrected_rounds": float(len(round_observations[index])),
                "early_context_observations": float(context_counts[index][0]),
                "middle_context_observations": float(context_counts[index][1]),
                "late_context_observations": float(context_counts[index][2]),
                "has_context_coverage": float(has_context_coverage),
            }
            if (
                len(raw_observations[index]) < self.min_observations
                or not math.isfinite(radius)
            ):
                uncertain.append(index)
            elif lower >= high_boundary:
                high.append(index)
            elif (
                upper <= low_boundary
                and has_context_coverage
                and hold_until.get(index, 0) < check_index
            ):
                low.append(index)
            else:
                uncertain.append(index)
        return high, uncertain, low, statistics

    def _inclusion_probabilities(
        self,
        high: Sequence[int],
        uncertain: Sequence[int],
        low: Sequence[int],
        statistics: Dict[int, Dict[str, float]],
        total_value: float,
    ) -> Dict[int, float]:
        high_set = set(high)
        low_set = set(low)
        threshold_value = self._effective_threshold() * total_value
        probabilities: Dict[int, float] = {}
        for index in range(self.data_count):
            if index in high_set:
                probability = 1.0
            elif index in low_set:
                probability = self.minimum_inclusion_probability
            else:
                mean = statistics.get(index, {}).get("mean", 0.0)
                standard_error = statistics.get(index, {}).get(
                    "standard_error", math.inf
                )
                if not math.isfinite(standard_error) or standard_error <= 0.0:
                    probability_high = 0.5 if mean < threshold_value else 1.0
                else:
                    z_score = (mean - threshold_value) / standard_error
                    probability_high = 0.5 * (
                        1.0 + math.erf(z_score / math.sqrt(2.0))
                    )
                ambiguity = 4.0 * probability_high * (1.0 - probability_high)
                priority = max(probability_high, ambiguity)
                probability = self.minimum_inclusion_probability + (
                    1.0 - self.minimum_inclusion_probability
                ) * priority
            probabilities[index] = min(1.0, max(
                self.minimum_inclusion_probability,
                probability,
            ))
        return probabilities

    def estimate_shapley(self) -> ContributionResult:
        if self.data_count == 0:
            return ContributionResult(shapley_values={}, sample_count=0, data_count=0)

        empty_value = self._evaluate_subset(())
        full_value = self._evaluate_subset(range(self.data_count))
        total_value = full_value - empty_value
        sampling_start = self._evaluation_requests
        corrected_sums = {index: 0.0 for index in range(self.data_count)}
        corrected_rounds = {index: [] for index in range(self.data_count)}
        raw_observations = {index: [] for index in range(self.data_count)}
        context_counts = {index: [0, 0, 0] for index in range(self.data_count)}
        inclusion_counts = {index: 0 for index in range(self.data_count)}
        probability_sums = {index: 0.0 for index in range(self.data_count)}
        history: List[Dict[str, Any]] = []
        audit_history: List[Dict[str, Any]] = []
        hold_until: Dict[int, int] = {}
        completed_paths = 0
        check_index = 0
        high: List[int] = []
        uncertain = list(range(self.data_count))
        low: List[int] = []
        statistics: Dict[int, Dict[str, float]] = {}
        adaptive_rounds = 0
        full_path_rounds = 0
        natural_fallback_rounds = 0

        while completed_paths < self.n_samples:
            paths_this_round = min(
                2 if self.use_antithetic_sampling else 1,
                self.n_samples - completed_paths,
            )
            force_initial = completed_paths < self.initial_samples
            if force_initial or total_value <= 0.0:
                probabilities = {index: 1.0 for index in range(self.data_count)}
                use_full_path = True
            else:
                probabilities = self._inclusion_probabilities(
                    high,
                    uncertain,
                    low,
                    statistics,
                    total_value,
                )
                expected_direct_cost = 2.0 * sum(probabilities.values())
                full_path_limit = self.data_count * (
                    1.0 - self.min_adaptive_savings_ratio
                )
                use_full_path = expected_direct_cost >= full_path_limit
                if use_full_path:
                    probabilities = {
                        index: 1.0 for index in range(self.data_count)
                    }
                    natural_fallback_rounds += 1

            permutation = self._uniform_permutation()
            permutations = [permutation]
            if paths_this_round == 2:
                permutations.append(list(reversed(permutation)))
            if use_full_path:
                selected = list(range(self.data_count))
                path_marginals = [
                    self._full_path_marginals(path, empty_value)
                    for path in permutations
                ]
                full_path_rounds += 1
            else:
                selected = [
                    index
                    for index in range(self.data_count)
                    if self.rng.random() < probabilities[index]
                ]
                path_marginals = [
                    self._direct_marginals(path, selected)
                    for path in permutations
                ]
                adaptive_rounds += 1

            positions_by_path = [
                {index: position for position, index in enumerate(path)}
                for path in permutations
            ]
            for index in range(self.data_count):
                probability_sums[index] += probabilities[index]
                if index in selected:
                    inclusion_counts[index] += paths_this_round
                    corrected_values = []
                    for path_index, marginals in enumerate(path_marginals):
                        marginal = marginals[index]
                        raw_observations[index].append(marginal)
                        context = self._context_stratum(
                            positions_by_path[path_index][index]
                        )
                        context_counts[index][context] += 1
                        corrected = marginal / probabilities[index]
                        corrected_sums[index] += corrected
                        corrected_values.append(corrected)
                    corrected_rounds[index].append(self._mean(corrected_values))
                else:
                    corrected_rounds[index].append(0.0)

            completed_paths += paths_this_round
            should_check = (
                completed_paths >= self.initial_samples
                and (
                    not history
                    or completed_paths % self.check_interval == 0
                    or completed_paths == self.n_samples
                )
            )
            if not should_check:
                continue
            check_index += 1
            high, uncertain, low, statistics = self._classify_probability_samples(
                corrected_sums,
                corrected_rounds,
                raw_observations,
                context_counts,
                completed_paths,
                max(total_value, 1e-12),
                check_index,
                hold_until,
            )
            available_audit_budget = max(
                0,
                self.n_samples * self.data_count
                - (self._evaluation_requests - sampling_start),
            )
            audits, suspects = self._coalition_audits(
                low,
                raw_observations,
                max(total_value, 1e-12),
                available_audit_budget,
            )
            for audit in audits:
                audit_history.append({"check": check_index, **audit})
            if suspects:
                for index in suspects:
                    hold_until[index] = check_index + self.interaction_hold_checks
                low = [index for index in low if index not in suspects]
                uncertain = sorted(set((*uncertain, *suspects)))
            next_probabilities = self._inclusion_probabilities(
                high,
                uncertain,
                low,
                statistics,
                max(total_value, 1e-12),
            )
            history.append(
                {
                    "check": check_index,
                    "completed_paths": completed_paths,
                    "high": list(high),
                    "uncertain": list(uncertain),
                    "low": list(low),
                    "interaction_suspects": sorted(suspects),
                    "next_inclusion_probabilities": {
                        str(index): value
                        for index, value in next_probabilities.items()
                    },
                    "statistics": {
                        str(index): dict(values)
                        for index, values in statistics.items()
                    },
                }
            )

        contributions = {
            index: corrected_sums[index] / self.n_samples
            for index in range(self.data_count)
        }
        standard_errors = {
            index: self._standard_error(corrected_rounds[index])
            for index in range(self.data_count)
        }
        selected_high = sorted(set((*high, *uncertain)))
        sampling_requests = self._evaluation_requests - sampling_start
        baseline_requests = self.n_samples * self.data_count
        if adaptive_rounds and natural_fallback_rounds:
            mode = "mixed_adaptive_and_natural_fallback"
        elif adaptive_rounds:
            mode = "adaptive_probability_sampling"
        else:
            mode = "uniform_mc_natural_fallback"
        cache_requests = self._cache_hits + self._cache_misses
        return ContributionResult(
            shapley_values=contributions,
            sample_count=self.n_samples,
            data_count=self.data_count,
            diagnostics={
                "algorithm": "dtg_shapley",
                "mode": mode,
                "estimation_target": "threshold_selective_classical_shapley",
                "estimator": "predictable_horvitz_thompson_antithetic",
                "all_sampling_paths_reused": True,
                "fixed_sampling_horizon": True,
                "predictable_inclusion_probabilities": True,
                "threshold_mode": self.threshold_mode,
                "fixed_contribution_threshold": self.contribution_threshold,
                "relative_threshold_factor": self.relative_threshold_factor,
                "effective_contribution_threshold": self._effective_threshold(),
                "threshold_margin_ratio": self.threshold_margin_ratio,
                "minimum_inclusion_probability": (
                    self.minimum_inclusion_probability
                ),
                "min_adaptive_savings_ratio": self.min_adaptive_savings_ratio,
                "paired_reverse_permutations": self.use_antithetic_sampling,
                "target_permutation_paths": self.n_samples,
                "completed_permutation_paths": completed_paths,
                "initial_full_paths": min(self.initial_samples, self.n_samples),
                "adaptive_rounds": adaptive_rounds,
                "full_path_rounds": full_path_rounds,
                "natural_fallback_rounds": natural_fallback_rounds,
                "group_checks": len(history),
                "group_history": history,
                "coalition_audit_history": audit_history,
                "coalition_audit_count": len(audit_history),
                "triggered_coalition_audits": sum(
                    bool(audit["triggered"]) for audit in audit_history
                ),
                "high_group_positions": selected_high,
                "certified_high_positions": list(high),
                "uncertain_group_positions": list(uncertain),
                "low_group_positions": list(low),
                "high_group_count": len(selected_high),
                "low_group_count": len(low),
                "total_observation_counts": {
                    str(index): len(raw_observations[index])
                    for index in range(self.data_count)
                },
                "corrected_round_counts": {
                    str(index): len(corrected_rounds[index])
                    for index in range(self.data_count)
                },
                "context_observation_counts": {
                    str(index): {
                        "early": context_counts[index][0],
                        "middle": context_counts[index][1],
                        "late": context_counts[index][2],
                    }
                    for index in range(self.data_count)
                },
                "inclusion_counts": {
                    str(index): inclusion_counts[index]
                    for index in range(self.data_count)
                },
                "mean_inclusion_probabilities": {
                    str(index): probability_sums[index]
                    / max(1, len(corrected_rounds[index]))
                    for index in range(self.data_count)
                },
                "screening_estimates": {
                    str(index): contributions[index]
                    for index in range(self.data_count)
                },
                "standard_errors": {
                    str(index): standard_errors[index]
                    for index in range(self.data_count)
                },
                "total_marginal_samples": sum(
                    len(values) for values in raw_observations.values()
                ),
                "baseline_mc_marginal_samples": baseline_requests,
                "marginal_sample_reduction_vs_mc": (
                    1.0
                    - sum(len(values) for values in raw_observations.values())
                    / baseline_requests
                    if baseline_requests
                    else 0.0
                ),
                "baseline_mc_coalition_requests": baseline_requests,
                "coalition_evaluation_requests": self._evaluation_requests,
                "sampling_and_audit_requests": sampling_requests,
                "request_reduction_vs_mc": (
                    1.0 - sampling_requests / baseline_requests
                    if baseline_requests
                    else 0.0
                ),
                "value_evaluations": self._cache_misses,
                "value_function_calls": self._cache_misses,
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "cache_hit_rate": (
                    self._cache_hits / cache_requests if cache_requests else 0.0
                ),
                "cache_size": len(self._subset_value_cache),
                "raw_efficiency_residual": sum(contributions.values()) - total_value,
                "low_values_are_approximate": False,
            },
        )
