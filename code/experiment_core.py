"""Shared case execution and result aggregation for paper experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from baselines import (
    AntitheticSampleContributionEvaluator,
    ExactShapleyEvaluator,
    KernelSHAPEvaluator,
    SampleContributionEvaluator,
    StratifiedSamplingShapleyEvaluator,
    TMCShapleyEvaluator,
)
from dataset_loader import build_value_function, load_experiment_dataset
from dtg_shapley import DTGShapleyEvaluator
from experiment_utils import (
    _atomic_json,
    _json_safe,
    _spearman,
    augment_with_persistent_low_fields,
    normalize,
)
from shapley_core import ValueFunction
@dataclass(frozen=True)
class ExperimentCase:
    experiment: str
    scenario: str
    dataset: str
    algorithm: str
    seed: int
    artificial_low_count: int
    equivalent_mc_paths: int
    contribution_threshold: float = 0.04
    threshold_margin: float = 0.005
    interaction_order: int = 0
    reference: bool = False
    enable_dynamic_regrouping: bool = True
    enable_coalition_audit: bool = True
    max_screening_budget_ratio: float = 0.2
    low_audit_rate: float = 0.15
    coalition_audits_per_check: int = 2
    implementation_version: str = "dtg_shapley_public"
    threshold_mode: str = "fixed"
    relative_threshold_factor: float = 0.5
    minimum_inclusion_probability: float = 0.10
    use_antithetic_sampling: bool = True
    feature_subset_count: int = 0
    feature_subset_seed: int = 11
    value_quantifier: str = "dataset_default"
    method_variant: str = "default"
    reference_group: str | None = None
    dtg_min_adaptive_savings_ratio: float = 0.05
    dtg_min_context_observations: int = 1
    tmc_truncation_tolerance_ratio: float = 0.01

    @property
    def condition_key(self) -> tuple[Any, ...]:
        return (
            self.scenario,
            self.dataset,
            self.artificial_low_count,
            self.reference_group if self.reference_group is not None else self.seed,
            self.contribution_threshold,
            self.interaction_order,
            self.feature_subset_count,
            self.feature_subset_seed,
            self.value_quantifier,
        )

    @property
    def case_id(self) -> str:
        payload_values = asdict(self)
        if self.algorithm != "dtg_shapley":
            for key in (
                "threshold_mode",
                "relative_threshold_factor",
                "minimum_inclusion_probability",
                "use_antithetic_sampling",
                "dtg_min_adaptive_savings_ratio",
                "dtg_min_context_observations",
            ):
                payload_values.pop(key, None)
        if self.algorithm != "tmc":
            payload_values.pop("tmc_truncation_tolerance_ratio", None)
        if self.feature_subset_count == 0 and self.feature_subset_seed == 11:
            payload_values.pop("feature_subset_count", None)
            payload_values.pop("feature_subset_seed", None)
        if self.value_quantifier == "dataset_default":
            payload_values.pop("value_quantifier", None)
        if self.dtg_min_adaptive_savings_ratio == 0.05:
            payload_values.pop("dtg_min_adaptive_savings_ratio", None)
        if self.dtg_min_context_observations == 1:
            payload_values.pop("dtg_min_context_observations", None)
        if self.method_variant == "default":
            payload_values.pop("method_variant", None)
        if self.reference_group is None:
            payload_values.pop("reference_group", None)
        payload = json.dumps(
            payload_values, sort_keys=True, separators=(",", ":")
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return (
            f"{self.experiment}__{self.scenario}__{self.dataset}"
            f"__low{self.artificial_low_count}__seed{self.seed}"
            f"__{self.algorithm}__b{self.equivalent_mc_paths}__{digest}"
        )

class SyntheticSourceGame(ValueFunction):
    """Additive source values plus unanimity-game interaction bonuses."""

    def __init__(
        self,
        additive_values: Sequence[float],
        interactions: dict[frozenset[int], float] | None = None,
    ):
        self.additive_values = list(additive_values)
        self.interactions = dict(interactions or {})

    def evaluate(self, subset: Sequence[Any]) -> float:
        coalition = frozenset(int(index) for index in subset)
        return float(
            sum(self.additive_values[index] for index in coalition)
            + sum(
                bonus
                for members, bonus in self.interactions.items()
                if members.issubset(coalition)
            )
        )

    def exact_shapley(self) -> dict[int, float]:
        values = {index: value for index, value in enumerate(self.additive_values)}
        for members, bonus in self.interactions.items():
            share = bonus / len(members)
            for index in members:
                values[index] += share
        return values

def build_synthetic_game(
    scenario: str,
    low_count: int,
    interaction_order: int,
) -> tuple[list[str], SyntheticSourceGame]:
    if scenario == "balanced":
        count = 12
        values = [1.0 / count] * count
        names = [f"ordinary_{index + 1}" for index in range(count)]
        return names, SyntheticSourceGame(values)

    if scenario == "balanced_interaction":
        count = 12
        pair_bonus = 0.03
        additive = [1.0 / count - pair_bonus] * count
        interactions = {
            frozenset((index, (index + 1) % count)): pair_bonus
            for index in range(count)
        }
        names = [f"ordinary_{index + 1}" for index in range(count)]
        return names, SyntheticSourceGame(additive, interactions)

    if scenario == "additive_tail":
        high_values = [0.34, 0.26, 0.20, 0.10]
        if low_count == 0:
            scale = 1.0 / sum(high_values)
            values = [value * scale for value in high_values]
        else:
            values = [*high_values, *([0.10 / low_count] * low_count)]
        names = [
            *(f"high_{index + 1}" for index in range(len(high_values))),
            *(f"low_{index + 1}" for index in range(low_count)),
        ]
        return names, SyntheticSourceGame(values)

    if scenario == "contextual_tail":
        if low_count < 4:
            raise ValueError("contextual_tail requires at least four low members.")
        high_additive = [0.26, 0.15, 0.13, 0.09]
        low_additive = [0.08 / low_count] * low_count
        interactions: dict[frozenset[int], float] = {
            frozenset((0, 1)): 0.16,
            frozenset((1, 2, 3)): 0.09,
        }
        for offset in range(low_count):
            interactions[frozenset((2, 4 + offset))] = 0.04 / low_count
        names = [
            *(f"high_{index + 1}" for index in range(4)),
            *(f"low_{index + 1}" for index in range(low_count)),
        ]
        return names, SyntheticSourceGame(
            [*high_additive, *low_additive],
            interactions,
        )

    if scenario.startswith("synergy_tail"):
        order = interaction_order or int(scenario.removeprefix("synergy_tail"))
        if low_count < order:
            raise ValueError("low_count must cover every interacting tail member.")
        high_values = [0.34, 0.24, 0.17]
        interaction_bonus = 0.06 * order
        noise_count = low_count - order
        noise_mass = 1.0 - sum(high_values) - interaction_bonus
        values = [
            *high_values,
            *([0.0] * order),
            *([noise_mass / noise_count] * noise_count if noise_count else []),
        ]
        interaction_members = frozenset(
            range(len(high_values), len(high_values) + order)
        )
        names = [
            *(f"high_{index + 1}" for index in range(len(high_values))),
            *(f"hidden_interaction_{index + 1}" for index in range(order)),
            *(f"low_{index + 1}" for index in range(noise_count)),
        ]
        return names, SyntheticSourceGame(
            values,
            {interaction_members: interaction_bonus},
        )

    raise ValueError(f"Unknown synthetic scenario: {scenario}")

def _named_diagnostics(
    diagnostics: dict[str, Any], names: Sequence[str]
) -> dict[str, Any]:
    converted = _json_safe(diagnostics)
    for key in ("high_group_positions", "uncertain_group_positions", "low_group_positions"):
        if key in converted:
            converted[key.removesuffix("_positions")] = [
                names[int(index)] for index in converted.get(key, [])
            ]
    for key in (
        "standard_errors",
        "screening_estimates",
        "screening_observation_counts",
        "formal_observation_counts",
        "total_observation_counts",
        "low_allocation_weights",
        "corrected_round_counts",
        "inclusion_counts",
        "mean_inclusion_probabilities",
    ):
        converted[key] = {
            names[int(index)]: value
            for index, value in converted.get(key, {}).items()
        }
    converted["formal_stratum_sample_counts"] = {
        names[int(index)]: counts
        for index, counts in converted.get(
            "formal_stratum_sample_counts", {}
        ).items()
    }
    converted["context_observation_counts"] = {
        names[int(index)]: counts
        for index, counts in converted.get(
            "context_observation_counts", {}
        ).items()
    }
    return converted

def run_case(case: ExperimentCase, cache_dir: Path) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    try:
        cache_path: Path | None = None
        if case.dataset == "synthetic":
            feature_names, value_function = build_synthetic_game(
                case.scenario,
                case.artificial_low_count,
                case.interaction_order,
            )
            dataset_metadata = {
                "name": case.scenario,
                "sample_count": None,
                "feature_count": len(feature_names),
                "feature_names": feature_names,
                "preprocessing": {"type": "analytic_cooperative_game"},
                "artificial_field_cache": None,
            }
        else:
            dataset = load_experiment_dataset(
                case.dataset,
                adult_income_discretize=True,
                bank_marketing_discretize=True,
                more_date=0,
                feature_subset_count=case.feature_subset_count,
                feature_subset_seed=case.feature_subset_seed,
            )
            dataset, cache_path = augment_with_persistent_low_fields(
                dataset,
                case.artificial_low_count,
                case.seed,
                cache_dir,
            )
            feature_names = list(dataset.feature_names)
            value_function = build_value_function(
                dataset,
                cache=True,
                quantifier_name=case.value_quantifier,
            )
            dataset_metadata = {
                "name": dataset.name,
                "sample_count": int(dataset.data_combinations.shape[1]),
                "feature_count": len(feature_names),
                "feature_names": feature_names,
                "preprocessing": _json_safe(dataset.preprocessing),
                "value_quantifier": case.value_quantifier,
                "artificial_field_cache": str(cache_path) if cache_path else None,
            }

        indices = list(range(len(feature_names)))
        if case.algorithm == "value_audit":
            empty_value = float(value_function.evaluate([]))
            full_value = float(value_function.evaluate(indices))
            singleton_values = {
                index: float(value_function.evaluate([index]) - empty_value)
                for index in indices
            }
            values_by_index = singleton_values
            full_increment = full_value - empty_value
            diagnostics = {
                "mode": "value_audit",
                "empty_value": empty_value,
                "full_value": full_value,
                "full_increment": full_increment,
                "positive_singleton_count": sum(
                    value > 0.0 for value in singleton_values.values()
                ),
                "decision": (
                    "eligible_for_shapley"
                    if full_increment > 0.0
                    else "skip_shapley_invalid_or_zero_full_value"
                ),
                "value_function_calls": len(indices) + 2,
                "coalition_evaluation_requests": len(indices) + 2,
            }
            sample_count = 0
        elif case.algorithm == "analytic_reference":
            if not isinstance(value_function, SyntheticSourceGame):
                raise ValueError("Analytic references require a synthetic game.")
            values_by_index = value_function.exact_shapley()
            diagnostics: dict[str, Any] = {
                "mode": "analytic_reference",
                "value_function_calls": 0,
                "coalition_evaluation_requests": 0,
            }
            sample_count = 0
        else:
            common = {"data": indices, "value_function": value_function}
            if case.algorithm == "exact":
                evaluator = ExactShapleyEvaluator(**common)
            elif case.algorithm == "mc" or case.algorithm.startswith(
                "mc_reference"
            ):
                evaluator = SampleContributionEvaluator(
                    **common,
                    n_samples=case.equivalent_mc_paths,
                    random_seed=(
                        case.seed + 1_000_003
                        if case.algorithm.startswith("mc_reference")
                        else case.seed
                    ),
                )
            elif case.algorithm == "antithetic_mc":
                evaluator = AntitheticSampleContributionEvaluator(
                    **common,
                    n_samples=case.equivalent_mc_paths,
                    random_seed=case.seed,
                )
            elif case.algorithm == "stratified":
                per_item_samples = max(
                    len(indices),
                    case.equivalent_mc_paths // 2,
                )
                evaluator = StratifiedSamplingShapleyEvaluator(
                    **common,
                    n_samples=per_item_samples,
                    random_seed=case.seed,
                )
            elif case.algorithm == "tmc":
                evaluator = TMCShapleyEvaluator(
                    **common,
                    n_samples=case.equivalent_mc_paths,
                    random_seed=case.seed,
                    truncation_tolerance_ratio=(
                        case.tmc_truncation_tolerance_ratio
                    ),
                )
            elif case.algorithm == "kernel_shap":
                kernel_budget = max(
                    len(indices) - 1,
                    len(indices) * case.equivalent_mc_paths,
                )
                evaluator = KernelSHAPEvaluator(
                    **common,
                    n_samples=kernel_budget,
                    random_seed=case.seed,
                )
            elif case.algorithm == "dtg_shapley":
                evaluator = DTGShapleyEvaluator(
                    **common,
                    n_samples=case.equivalent_mc_paths,
                    threshold_mode=case.threshold_mode,
                    contribution_threshold=case.contribution_threshold,
                    relative_threshold_factor=(
                        case.relative_threshold_factor
                    ),
                    minimum_inclusion_probability=(
                        case.minimum_inclusion_probability
                    ),
                    min_adaptive_savings_ratio=(
                        case.dtg_min_adaptive_savings_ratio
                    ),
                    min_context_observations=(
                        case.dtg_min_context_observations
                    ),
                    use_antithetic_sampling=case.use_antithetic_sampling,
                    coalition_audits_per_check=(
                        case.coalition_audits_per_check
                    ),
                    enable_coalition_audit=case.enable_coalition_audit,
                    random_seed=case.seed,
                )
            else:
                raise ValueError(f"Unsupported algorithm: {case.algorithm}")
            result = evaluator.estimate_shapley()
            values_by_index = result.shapley_values
            diagnostics = _named_diagnostics(result.diagnostics, feature_names)
            sample_count = result.sample_count
            if (
                case.algorithm in {"mc", "antithetic_mc"}
                or case.algorithm.startswith("mc_reference")
            ):
                per_member = {
                    name: case.equivalent_mc_paths for name in feature_names
                }
                diagnostics.update(
                    {
                        "formal_sampling_strategy": "uniform_permutation",
                        "formal_observation_counts": per_member,
                        "total_observation_counts": per_member,
                        "total_marginal_samples": (
                            len(feature_names) * case.equivalent_mc_paths
                        ),
                        "baseline_mc_marginal_samples": (
                            len(feature_names) * case.equivalent_mc_paths
                        ),
                        "marginal_sample_reduction_vs_mc": 0.0,
                        "request_reduction_vs_mc": 0.0,
                    }
                )
            elif case.algorithm == "stratified":
                per_member = {
                    name: per_item_samples for name in feature_names
                }
                diagnostics.update(
                    {
                        "formal_sampling_strategy": "coalition_size_stratified",
                        "formal_observation_counts": per_member,
                        "total_observation_counts": per_member,
                        "total_marginal_samples": (
                            len(feature_names) * per_item_samples
                        ),
                        "baseline_mc_marginal_samples": (
                            len(feature_names) * case.equivalent_mc_paths
                        ),
                        "marginal_sample_reduction_vs_mc": (
                            1.0
                            - per_item_samples / case.equivalent_mc_paths
                        ),
                    }
                )
            elif case.algorithm == "tmc":
                diagnostics.update(
                    {
                        "formal_sampling_strategy": (
                            "truncated_uniform_permutation"
                        ),
                        "baseline_mc_marginal_samples": (
                            len(feature_names) * case.equivalent_mc_paths
                        ),
                        "marginal_sample_reduction_vs_mc": (
                            1.0
                            - diagnostics.get("evaluated_marginals", 0)
                            / (
                                len(feature_names)
                                * case.equivalent_mc_paths
                            )
                        ),
                    }
                )
            elif case.algorithm == "kernel_shap":
                diagnostics.update(
                    {
                        "formal_sampling_strategy": (
                            "shapley_kernel_weighted_regression"
                        ),
                        "kernel_nominal_value_budget": kernel_budget,
                        "baseline_mc_marginal_samples": (
                            len(feature_names) * case.equivalent_mc_paths
                        ),
                    }
                )

        data_value_evaluator = getattr(
            value_function, "data_value_evaluator", None
        )
        actual_value_function_calls = (
            int(data_value_evaluator.quantification_calls)
            if data_value_evaluator is not None
            else int(
                diagnostics.get(
                    "value_function_calls",
                    diagnostics.get("value_evaluations", 0),
                )
            )
        )
        value_function_evaluation_requests = (
            int(data_value_evaluator.evaluation_requests)
            if data_value_evaluator is not None
            else int(
                diagnostics.get(
                    "coalition_evaluation_requests",
                    actual_value_function_calls,
                )
            )
        )
        diagnostics["actual_value_function_calls"] = (
            actual_value_function_calls
        )
        diagnostics["value_function_evaluation_requests"] = (
            value_function_evaluation_requests
        )
        values = {
            feature_names[index]: float(value)
            for index, value in values_by_index.items()
        }
        selected_low = set(diagnostics.get("low_group", []))
        synthetic_names = {
            name
            for name in feature_names
            if name.startswith("synthetic_low_") or name.startswith("low_")
        }
        captured = len(selected_low & synthetic_names)
        candidate_low_screening_ratio = (
            captured / len(synthetic_names) if synthetic_names else None
        )
        return {
            "schema_version": 1,
            "status": "success",
            "case_id": case.case_id,
            "case": _json_safe(asdict(case)),
            "dataset": dataset_metadata,
            "metrics": {
                "started_at": started_at,
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "runtime_seconds": time.perf_counter() - started,
                "sample_count": sample_count,
                "actual_value_function_calls": actual_value_function_calls,
                "value_function_evaluation_requests": (
                    value_function_evaluation_requests
                ),
                "known_low_capture_ratio": (
                    candidate_low_screening_ratio
                    if case.dataset == "synthetic"
                    else None
                ),
                "candidate_low_screening_ratio": (
                    candidate_low_screening_ratio
                ),
                "diagnostics": diagnostics,
            },
            "result": {
                "shapley_values": values,
                "normalized_shapley_values": normalize(values),
                "total_value": sum(values.values()),
            },
        }
    except Exception as error:
        return {
            "schema_version": 1,
            "status": "failed",
            "case_id": case.case_id,
            "case": _json_safe(asdict(case)),
            "metrics": {
                "started_at": started_at,
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "runtime_seconds": time.perf_counter() - started,
            },
            "error": {"type": type(error).__name__, "message": str(error)},
        }

def _load_results(raw_dir: Path) -> list[dict[str, Any]]:
    output = []
    for path in sorted(raw_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as stream:
            output.append(json.load(stream))
    return output

def _result_condition_key(result: dict[str, Any]) -> tuple[Any, ...]:
    case = result["case"]
    return (
        case["scenario"],
        case["dataset"],
        case["artificial_low_count"],
        case.get("reference_group")
        if case.get("reference_group") is not None
        else case["seed"],
        case["contribution_threshold"],
        case["interaction_order"],
        case.get("feature_subset_count", 0),
        case.get("feature_subset_seed", 11),
        case.get("value_quantifier", "dataset_default"),
    )

def summarize(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    successful = [result for result in results if result.get("status") == "success"]
    references = {
        _result_condition_key(result): result
        for result in successful
        if result["case"].get("reference")
    }
    rows: list[dict[str, Any]] = []
    for result in successful:
        case = result["case"]
        diagnostics = result["metrics"].get("diagnostics", {})
        threshold_mode = case.get("threshold_mode", "fixed")
        relative_threshold = case.get(
            "relative_threshold_factor", 0.5
        ) / max(1, len(result["dataset"]["feature_names"]))
        configured_threshold = case["contribution_threshold"]
        if threshold_mode == "relative":
            configured_threshold = relative_threshold
        elif threshold_mode == "hybrid":
            configured_threshold = max(configured_threshold, relative_threshold)
        effective_threshold = diagnostics.get(
            "effective_contribution_threshold",
            configured_threshold,
        )
        reference = references.get(_result_condition_key(result))
        names = result["dataset"]["feature_names"]
        normalized = result["result"]["normalized_shapley_values"]
        predicted_high = set(diagnostics.get("high_group", []))
        if not predicted_high:
            predicted_high = {
                name
                for name in names
                if normalized[name] >= effective_threshold
            }
        if "low_group" in diagnostics:
            predicted_low = set(diagnostics.get("low_group", []))
        else:
            predicted_low = {
                name
                for name in names
                if normalized[name] < effective_threshold
            }
        baseline_requests = case["equivalent_mc_paths"] * len(names)
        coalition_requests = diagnostics.get(
            "coalition_evaluation_requests",
            baseline_requests,
        )
        row: dict[str, Any] = {
            "case_id": result["case_id"],
            "experiment": case["experiment"],
            "scenario": case["scenario"],
            "dataset": case["dataset"],
            "algorithm": case["algorithm"],
            "value_quantifier": case.get(
                "value_quantifier", "dataset_default"
            ),
            "method_variant": case.get("method_variant", "default"),
            "reference_group": case.get("reference_group"),
            "seed": case["seed"],
            "artificial_low_count": case["artificial_low_count"],
            "interaction_order": case["interaction_order"],
            "equivalent_mc_paths": case["equivalent_mc_paths"],
            "feature_subset_count": case.get("feature_subset_count", 0),
            "feature_subset_seed": case.get("feature_subset_seed", 11),
            "contribution_threshold": case["contribution_threshold"],
            "threshold_mode": case.get("threshold_mode", "fixed"),
            "relative_threshold_factor": case.get(
                "relative_threshold_factor", 0.5
            ),
            "effective_contribution_threshold": effective_threshold,
            "runtime_seconds": result["metrics"]["runtime_seconds"],
            "mode": diagnostics.get("mode"),
            "high_group_count": diagnostics.get("high_group_count"),
            "low_group_count": diagnostics.get("low_group_count"),
            "group_checks": diagnostics.get("group_checks"),
            "triggered_coalition_audits": diagnostics.get(
                "triggered_coalition_audits"
            ),
            "coalition_evaluation_requests": coalition_requests,
            "unique_value_evaluations": diagnostics.get(
                "value_evaluations",
                diagnostics.get("value_function_calls"),
            ),
            "actual_value_function_calls": result["metrics"].get(
                "actual_value_function_calls",
                diagnostics.get("actual_value_function_calls"),
            ),
            "total_marginal_samples": diagnostics.get(
                "total_marginal_samples"
            ),
            "marginal_sample_reduction_vs_mc": diagnostics.get(
                "marginal_sample_reduction_vs_mc"
            ),
            "request_reduction_vs_mc": diagnostics.get(
                "request_reduction_vs_mc",
                (
                    1.0 - coalition_requests / baseline_requests
                    if baseline_requests
                    else None
                ),
            ),
            "known_low_capture_ratio": result["metrics"].get(
                "known_low_capture_ratio"
            ),
            "candidate_low_screening_ratio": result["metrics"].get(
                "candidate_low_screening_ratio"
            ),
            "reference_case_id": reference["case_id"] if reference else None,
            "reference_algorithm": (
                reference["case"]["algorithm"] if reference else None
            ),
            "all_nmae": None,
            "all_rmse": None,
            "spearman": None,
            "high_nmae": None,
            "high_rmse": None,
            "high_recall": None,
            "high_precision": None,
            "low_recall": None,
            "low_precision": None,
            "tail_total_abs_error": None,
            "true_high_count": None,
            "true_low_count": None,
            "mean_true_high_samples": None,
            "mean_true_low_samples": None,
            "low_to_high_sampling_ratio": None,
            "artificial_reference_low_rate": None,
            "verified_artificial_low_recall": None,
            "max_estimated_certified_low_contribution": (
                max((normalized[name] for name in predicted_low), default=None)
            ),
        }
        if reference is not None:
            expected = reference["result"]["normalized_shapley_values"]
            errors = [normalized[name] - expected[name] for name in names]
            true_high = {
                name
                for name in names
                if expected[name] >= effective_threshold
            }
            true_low = set(names) - true_high
            high_errors = [normalized[name] - expected[name] for name in true_high]
            intersection = true_high & predicted_high
            low_intersection = true_low & predicted_low
            observation_counts = diagnostics.get(
                "total_observation_counts", {}
            )
            high_sample_counts = [
                float(observation_counts[name])
                for name in true_high
                if name in observation_counts
            ]
            low_sample_counts = [
                float(observation_counts[name])
                for name in true_low
                if name in observation_counts
            ]
            mean_high_samples = (
                statistics.fmean(high_sample_counts)
                if high_sample_counts
                else None
            )
            mean_low_samples = (
                statistics.fmean(low_sample_counts)
                if low_sample_counts
                else None
            )
            artificial_names = {
                name
                for name in names
                if name.startswith("synthetic_low_")
                or name.startswith("low_")
            }
            verified_artificial_low = artificial_names & true_low
            row.update(
                {
                    "all_nmae": sum(abs(value) for value in errors) / len(errors),
                    "all_rmse": math.sqrt(
                        sum(value * value for value in errors) / len(errors)
                    ),
                    "spearman": _spearman(
                        [normalized[name] for name in names],
                        [expected[name] for name in names],
                    ),
                    "high_nmae": (
                        sum(abs(value) for value in high_errors) / len(high_errors)
                        if high_errors
                        else None
                    ),
                    "high_rmse": (
                        math.sqrt(
                            sum(value * value for value in high_errors)
                            / len(high_errors)
                        )
                        if high_errors
                        else None
                    ),
                    "high_recall": (
                        len(intersection) / len(true_high) if true_high else 1.0
                    ),
                    "high_precision": (
                        len(intersection) / len(predicted_high)
                        if predicted_high
                        else (1.0 if not true_high else 0.0)
                    ),
                    "low_recall": (
                        len(low_intersection) / len(true_low)
                        if true_low
                        else 1.0
                    ),
                    "low_precision": (
                        len(low_intersection) / len(predicted_low)
                        if predicted_low
                        else (1.0 if not true_low else 0.0)
                    ),
                    "tail_total_abs_error": abs(
                        sum(normalized[name] for name in true_low)
                        - sum(expected[name] for name in true_low)
                    ),
                    "true_high_count": len(true_high),
                    "true_low_count": len(true_low),
                    "mean_true_high_samples": mean_high_samples,
                    "mean_true_low_samples": mean_low_samples,
                    "low_to_high_sampling_ratio": (
                        mean_low_samples / mean_high_samples
                        if mean_low_samples is not None
                        and mean_high_samples not in (None, 0.0)
                        else None
                    ),
                    "artificial_reference_low_rate": (
                        len(verified_artificial_low) / len(artificial_names)
                        if artificial_names
                        else None
                    ),
                    "verified_artificial_low_recall": (
                        len(verified_artificial_low & predicted_low)
                        / len(verified_artificial_low)
                        if verified_artificial_low
                        else None
                    ),
                }
            )
        rows.append(row)
    return rows

def aggregate(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["experiment"],
            row["scenario"],
            row["dataset"],
            row["algorithm"],
            row["value_quantifier"],
            row["method_variant"],
            row["artificial_low_count"],
            row["interaction_order"],
            row["equivalent_mc_paths"],
            row["contribution_threshold"],
            row["threshold_mode"],
            row["relative_threshold_factor"],
            row["effective_contribution_threshold"],
            row["feature_subset_count"],
            row["feature_subset_seed"],
        )
        groups.setdefault(key, []).append(row)
    output = []
    labels = (
        "experiment",
        "scenario",
        "dataset",
        "algorithm",
        "value_quantifier",
        "method_variant",
        "artificial_low_count",
        "interaction_order",
        "equivalent_mc_paths",
        "contribution_threshold",
        "threshold_mode",
        "relative_threshold_factor",
        "effective_contribution_threshold",
        "feature_subset_count",
        "feature_subset_seed",
    )
    metrics = (
        "runtime_seconds",
        "coalition_evaluation_requests",
        "unique_value_evaluations",
        "actual_value_function_calls",
        "total_marginal_samples",
        "marginal_sample_reduction_vs_mc",
        "request_reduction_vs_mc",
        "all_nmae",
        "all_rmse",
        "spearman",
        "high_nmae",
        "high_rmse",
        "high_recall",
        "high_precision",
        "low_recall",
        "low_precision",
        "tail_total_abs_error",
        "known_low_capture_ratio",
        "candidate_low_screening_ratio",
        "artificial_reference_low_rate",
        "verified_artificial_low_recall",
        "mean_true_high_samples",
        "mean_true_low_samples",
        "low_to_high_sampling_ratio",
        "max_estimated_certified_low_contribution",
        "high_group_count",
        "low_group_count",
        "group_checks",
        "triggered_coalition_audits",
    )
    for key, members in groups.items():
        item = dict(zip(labels, key))
        item["replicates"] = len(members)
        for metric in metrics:
            values = [
                float(row[metric])
                for row in members
                if row.get(metric) is not None
            ]
            item[f"{metric}_mean"] = statistics.fmean(values) if values else None
            item[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        item["focused_selection_rate"] = sum(
            row.get("mode") == "focused_high_stratified" for row in members
        ) / len(members)
        item["adaptive_sampling_rate"] = sum(
            row.get("mode") in {
                "adaptive_probability_sampling",
                "mixed_adaptive_and_natural_fallback",
            }
            for row in members
        ) / len(members)
        output.append(item)
    return output

def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def write_chart(aggregates: Sequence[dict[str, Any]], path: Path) -> None:
    points = [
        row
        for row in aggregates
        if row.get("high_nmae_mean") is not None
        and row.get("coalition_evaluation_requests_mean") is not None
        and row["algorithm"] not in {"analytic_reference", "exact"}
        and not row["algorithm"].startswith("mc_reference")
    ]
    if not points:
        return
    width, height = 1180, 620
    colors = {
        "mc": "#2563eb",
        "antithetic_mc": "#0891b2",
        "stratified": "#7c3aed",
        "dtg_static": "#d97706",
        "dtg_no_audit": "#db2777",
        "dtg_shapley": "#059669",
        "dtg_shapley": "#dc2626",
        "tmc": "#ea580c",
        "kernel_shap": "#475569",
    }
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<style>text{font-family:Arial,sans-serif;letter-spacing:0;fill:#20242b}.axis{stroke:#59636f}.grid{stroke:#dce1e7}.label{font-size:13px}.title{font-size:21px;font-weight:700}</style>',
        '<text x="70" y="35" class="title">DTG-Shapley: high-field accuracy and sampling reduction</text>',
    ]
    panels = (
        (65, 70, 520, 460, "Contextual low-tail scaling", True),
        (640, 70, 500, 460, "Request reduction vs high-field error", False),
    )
    for left, top, panel_width, panel_height, title, tail_only in panels:
        selected = [
            row
            for row in points
            if (row["experiment"] == "e2_contextual_tail_scaling") == tail_only
        ]
        if not selected:
            continue
        x_values = [
            float(row["artificial_low_count"])
            if tail_only
            else float(row.get("request_reduction_vs_mc_mean") or 0.0)
            for row in selected
        ]
        y_values = [float(row["high_nmae_mean"]) for row in selected]
        min_x, max_x = min(x_values), max(x_values)
        max_y = max(y_values) or 1.0
        x_span = max(max_x - min_x, 1.0)
        svg.extend(
            [
                f'<text x="{left}" y="{top-14}" class="label" font-weight="700">{title}</text>',
                f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+panel_height}" class="axis"/>',
                f'<line x1="{left}" y1="{top+panel_height}" x2="{left+panel_width}" y2="{top+panel_height}" class="axis"/>',
            ]
        )
        for step in range(5):
            y = top + panel_height * step / 4
            svg.append(
                f'<line x1="{left}" y1="{y}" x2="{left+panel_width}" y2="{y}" class="grid"/>'
            )
        for row, x_value, y_value in zip(selected, x_values, y_values):
            x = left + panel_width * (x_value - min_x) / x_span
            y = top + panel_height * (1.0 - y_value / max_y)
            color = colors.get(row["algorithm"], "#4b5563")
            label = (
                f"{row['algorithm']} variant={row.get('method_variant', 'default')} "
                f"low={row['artificial_low_count']} "
                f"samples={row['equivalent_mc_paths']} highNMAE={y_value:.6f} "
                f"requestReduction={float(row.get('request_reduction_vs_mc_mean') or 0.0):.3f}"
            )
            svg.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" fill-opacity="0.78"><title>{label}</title></circle>'
            )
        x_label = (
            "Individually low-contribution members"
            if tail_only
            else "Request reduction relative to MC"
        )
        svg.append(
            f'<text x="{left+panel_width/2}" y="{top+panel_height+35}" text-anchor="middle" class="label">{x_label}</text>'
        )
    svg.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg), encoding="utf-8")

def refresh_outputs(run_dir: Path) -> tuple[int, int]:
    results = _load_results(run_dir / "raw")
    rows = summarize(results)
    aggregates = aggregate(rows)
    _atomic_json(run_dir / "summaries" / "summary.json", rows)
    _atomic_json(run_dir / "summaries" / "aggregate.json", aggregates)
    _write_csv(run_dir / "summaries" / "summary.csv", rows)
    _write_csv(run_dir / "summaries" / "aggregate.csv", aggregates)
    write_chart(aggregates, run_dir / "charts" / "comparison.svg")
    return len(results), sum(result.get("status") == "success" for result in results)
