from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR))

from dtg_shapley import DTGShapleyEvaluator
from dataset_loader import build_value_function, load_experiment_dataset
from shapley_core import ValueFunction


class AdditiveGame(ValueFunction):
    def evaluate(self, subset):
        weights = {0: 0.60, 1: 0.25, 2: 0.10, 3: 0.03, 4: 0.02}
        return sum(weights[index] for index in subset)


class DTGSmokeTest(unittest.TestCase):
    def test_public_estimator_runs(self):
        evaluator = DTGShapleyEvaluator(
            data=list(range(5)),
            value_function=AdditiveGame(),
            n_samples=20,
            initial_samples=4,
            check_interval=2,
            random_seed=11,
        )
        result = evaluator.estimate_shapley()
        self.assertEqual(result.sample_count, 20)
        self.assertEqual(result.data_count, 5)
        self.assertEqual(result.diagnostics["algorithm"], "dtg_shapley")
        self.assertTrue(all(math.isfinite(value) for value in result.shapley_values.values()))
        self.assertEqual(max(result.shapley_values, key=result.shapley_values.get), 0)
        self.assertEqual(result.diagnostics["completed_permutation_paths"], 20)
        self.assertLess(
            result.diagnostics["value_function_calls"],
            result.diagnostics["baseline_mc_marginal_samples"],
        )

    def test_iris_value_functions(self):
        dataset = load_experiment_dataset("iris")
        for quantifier in ("holdout_naive_bayes", "mutual_information"):
            value_function = build_value_function(
                dataset,
                cache=True,
                quantifier_name=quantifier,
            )
            result = DTGShapleyEvaluator(
                data=list(range(len(dataset.feature_names))),
                value_function=value_function,
                n_samples=8,
                initial_samples=4,
                check_interval=2,
                random_seed=11,
            ).estimate_shapley()
            self.assertEqual(result.data_count, 4)
            self.assertTrue(
                all(math.isfinite(value) for value in result.shapley_values.values())
            )

    def test_packaged_uci_archive(self):
        dataset = load_experiment_dataset(
            "cnae_9",
            feature_subset_count=10,
            feature_subset_seed=11,
        )
        self.assertEqual(dataset.data_combinations.shape, (10, 1080))
        value_function = build_value_function(
            dataset,
            cache=True,
            quantifier_name="mutual_information",
        )
        self.assertTrue(math.isfinite(value_function.evaluate([0])))


if __name__ == "__main__":
    unittest.main()
