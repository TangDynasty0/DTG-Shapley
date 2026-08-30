"""Information-theory based quantifier implementation.

This module provides `InformationTheoryQuantifier` which implements
`ValueQuantifier.quantify` using information-theoretic metrics.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List

import numpy as np

from collections import Counter

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


class InformationTheoryQuantifier:
    def quantify(self, input_data: np.ndarray, output: np.ndarray) -> float:
        """
        Calculate information-theoretic value:

            v(S) = I(X_S; Y)

        Parameters
        ----------
        input_data : np.ndarray
            Shape can be:
                (m, n): m features, n samples
                (1, n): 1 feature, n samples
                (n,):   1 feature, n samples

        output : np.ndarray
            Shape can be:
                (n,)
                (1, n)
                (n, 1)

        Returns
        -------
        float
            Joint mutual information I(X_S; Y), in bits.
            If input_data is empty, return 0.0.
        """

        # 若输入数据为空，说明当前组合中没有任何特征，价值定义为 0
        if input_data is None:
            return 0.0

        input_array = np.asarray(input_data)

        if input_array.size == 0:
            return 0.0

        x = self._prepare_input_data(input_array)
        y = self._prepare_output(output)

        if x.shape[0] == 0 or x.shape[1] == 0:
            return 0.0

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Sample size mismatch: input_data has {x.shape[0]} samples, "
                f"but output has {y.shape[0]} samples."
            )

        logging.info("Prepared input shape: %s", x.shape)
        logging.info("Prepared output shape: %s", y.shape)

        x_discrete = self._discretize_if_needed(x)

        if self._is_continuous(y):
            y_discrete = self._discretize_if_needed(y.reshape(-1, 1)).reshape(-1)
        else:
            y_discrete = y

        return self._calculate_mutual_information(x_discrete, y_discrete)

    def _prepare_input_data(self, input_data: np.ndarray) -> np.ndarray:
        """
        Convert input_data to shape:

            (n_samples, n_features)

        Original convention:
            input_data shape = (n_features, n_samples)

        Special cases:
            (1, n) -> (n, 1)
            (n,)   -> (n, 1)
        """

        x = np.asarray(input_data)

        if x.ndim == 1:
            # Treat 1D input as one feature over n samples
            return x.reshape(-1, 1)

        if x.ndim != 2:
            raise ValueError(
                "input_data must be a 1D or 2D array. "
                "Expected shape: (n_features, n_samples) or (n_samples,)."
            )

        # Your convention: rows are features, columns are samples
        # So transpose to: rows are samples, columns are features
        return x.T

    def _prepare_output(self, output: np.ndarray) -> np.ndarray:
        """
        Convert output to shape:

            (n_samples,)
        """

        y = np.asarray(output)

        if y.ndim == 1:
            return y

        if y.ndim == 2:
            if 1 in y.shape:
                return y.reshape(-1)

        raise ValueError(
            "output must be a 1D array or a 2D array with one dimension equal to 1, "
            "such as (n,), (1, n), or (n, 1)."
        )

    def _calculate_entropy(self, data: np.ndarray) -> float:
        """
        Calculate Shannon entropy:

            H(X) = - sum p(x) log2 p(x)

        Parameters
        ----------
        data : np.ndarray
            1D array:
                one random variable

            2D array:
                joint random variable, where each row is one sample

        Returns
        -------
        float
            Entropy in bits.
        """

        data = np.asarray(data)

        if data.ndim == 1:
            samples = data.tolist()
        elif data.ndim == 2:
            samples = [tuple(row) for row in data]
        else:
            raise ValueError("data must be 1D or 2D.")

        total = len(samples)
        counter = Counter(samples)

        entropy = 0.0

        for count in counter.values():
            p = count / total
            entropy -= p * np.log2(p)

        return entropy

    def _calculate_mutual_information(
        self,
        input_data: np.ndarray,
        output: np.ndarray
    ) -> float:
        """
        Calculate joint mutual information:

            I(X_S; Y) = H(X_S) + H(Y) - H(X_S, Y)

        Parameters
        ----------
        input_data : np.ndarray
            shape = (n_samples, n_features)

        output : np.ndarray
            shape = (n_samples,)

        Returns
        -------
        float
            Mutual information in bits.
        """

        x = np.asarray(input_data)
        y = np.asarray(output).reshape(-1)

        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                "input_data and output must have the same number of samples."
            )

        h_x = self._calculate_entropy(x)
        h_y = self._calculate_entropy(y)

        joint_xy = np.column_stack([x, y])
        h_xy = self._calculate_entropy(joint_xy)

        mi = h_x + h_y - h_xy

        # Avoid tiny negative values caused by floating-point errors
        return max(0.0, mi)

    def _discretize_if_needed(
        self,
        data: np.ndarray,
        n_bins: int = 10,
        strategy: str = "quantile"
    ) -> np.ndarray:
        """
        Discretize continuous variables.

        If data is not floating-point, it is treated as discrete and returned directly.
        """

        data = np.asarray(data)

        if not self._is_continuous(data):
            return data

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_samples = data.shape[0]

        # Avoid n_bins > n_samples
        actual_bins = min(n_bins, n_samples)

        discretizer = KBinsDiscretizer(
            n_bins=actual_bins,
            encode="ordinal",
            strategy=strategy # type: ignore
        )

        # KBins already removes duplicate or numerically indistinguishable
        # quantile edges. The warning is expected for repeated-valued features
        # and becomes extremely noisy during thousands of coalition calls.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Bins whose width are too small .* are removed\..*",
                category=UserWarning,
                module=r"sklearn\.preprocessing\._discretization",
            )
            warnings.filterwarnings(
                "ignore",
                message=r"The current default behavior, quantile_method=.*",
                category=FutureWarning,
                module=r"sklearn\.preprocessing\._discretization",
            )
            return discretizer.fit_transform(data).astype(int)

    def _is_continuous(self, data: np.ndarray) -> bool:
        """
        Determine whether data should be treated as continuous.

        Current rule:
            float dtype -> continuous
            int / str / object -> discrete
        """

        return np.issubdtype(np.asarray(data).dtype, np.floating)


class HoldoutNaiveBayesInformationQuantifier:
    """Held-out predictive information for high-dimensional discrete data.

    Coalition value is validation log-loss reduction, in bits, relative to a
    class-prior-only model. Categorical Naive Bayes likelihood terms are
    precomputed per feature, making repeated Shapley coalition scoring cheap.
    """

    def __init__(
        self,
        reference_data: np.ndarray,
        reference_output: np.ndarray,
        test_fraction: float = 0.2,
        random_seed: int = 20260824,
        alpha: float = 1.0,
    ):
        if not 0.0 < test_fraction < 1.0:
            raise ValueError("test_fraction must be in (0, 1).")
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("alpha must be finite and greater than zero.")
        data = np.asarray(reference_data)
        if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] < 4:
            raise ValueError(
                "reference_data must have shape (features, samples) with at "
                "least one feature and four samples."
            )
        output = np.asarray(reference_output).reshape(-1)
        if output.shape[0] != data.shape[1]:
            raise ValueError("reference_data and reference_output sizes differ.")
        classes, encoded_output = np.unique(output, return_inverse=True)
        if len(classes) < 2:
            raise ValueError("reference_output must contain at least two classes.")

        self.reference_sample_count = data.shape[1]
        self.test_fraction = float(test_fraction)
        self.random_seed = int(random_seed)
        self.alpha = float(alpha)
        self.classes = classes
        train_indices, test_indices = self._stratified_split(
            encoded_output, test_fraction, random_seed
        )
        self.train_count = len(train_indices)
        self.test_count = len(test_indices)
        self._test_output = encoded_output[test_indices]
        class_count = len(classes)
        train_class_counts = np.bincount(
            encoded_output[train_indices], minlength=class_count
        ).astype(float)
        prior = (train_class_counts + alpha) / (
            len(train_indices) + alpha * class_count
        )
        self._base_log_joint = np.broadcast_to(
            np.log(prior), (len(test_indices), class_count)
        ).copy()
        self._baseline_loss = self._log_loss_bits(self._base_log_joint)
        self._feature_terms: Dict[tuple[str, bytes], np.ndarray] = {}

        for row in data:
            key = self._feature_key(row)
            if key in self._feature_terms:
                # Duplicate and constant columns are distinct Shapley members,
                # but their Naive Bayes likelihood term is identical.
                continue
            _, encoded_feature = np.unique(row, return_inverse=True)
            category_count = int(encoded_feature.max()) + 1
            terms = np.empty((len(test_indices), class_count), dtype=float)
            for class_index in range(class_count):
                class_train = train_indices[
                    encoded_output[train_indices] == class_index
                ]
                counts = np.bincount(
                    encoded_feature[class_train], minlength=category_count
                ).astype(float)
                probabilities = (counts + alpha) / (
                    len(class_train) + alpha * category_count
                )
                terms[:, class_index] = np.log(
                    probabilities[encoded_feature[test_indices]]
                )
            self._feature_terms[key] = terms

    @staticmethod
    def _stratified_split(
        output: np.ndarray,
        test_fraction: float,
        random_seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(random_seed)
        train: List[int] = []
        test: List[int] = []
        for class_index in np.unique(output):
            indices = np.flatnonzero(output == class_index)
            rng.shuffle(indices)
            test_count = max(1, int(round(len(indices) * test_fraction)))
            if test_count >= len(indices):
                test_count = len(indices) - 1
            test.extend(indices[:test_count].tolist())
            train.extend(indices[test_count:].tolist())
        return np.asarray(sorted(train), dtype=int), np.asarray(sorted(test), dtype=int)

    @staticmethod
    def _feature_key(values: np.ndarray) -> tuple[str, bytes]:
        contiguous = np.ascontiguousarray(values)
        return contiguous.dtype.str, contiguous.tobytes()

    def _log_loss_bits(self, log_joint: np.ndarray) -> float:
        maximum = np.max(log_joint, axis=1, keepdims=True)
        log_normalizer = maximum[:, 0] + np.log(
            np.exp(log_joint - maximum).sum(axis=1)
        )
        true_log_probability = (
            log_joint[np.arange(self.test_count), self._test_output]
            - log_normalizer
        )
        return float(-np.mean(true_log_probability) / np.log(2.0))

    def quantify(self, input_data: Any, output: Any) -> float:
        data = np.asarray(input_data)
        if data.size == 0:
            return 0.0
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.ndim != 2 or data.shape[1] != self.reference_sample_count:
            raise ValueError(
                "input_data must contain unchanged rows from reference_data."
            )
        log_joint = self._base_log_joint.copy()
        for row in data:
            try:
                log_joint += self._feature_terms[self._feature_key(row)]
            except KeyError as error:
                raise ValueError(
                    "input_data contains a feature not present in reference_data."
                ) from error
        predictive_information = self._baseline_loss - self._log_loss_bits(log_joint)
        return max(0.0, float(predictive_information))



# class InformationTheoryQuantifier(ValueQuantifier):
#     """Quantify value using information theory metrics."""

#     def quantify(self, input_data: np.ndarray, output: np.ndarray) -> float:
#         """Calculate information-theoretic value (e.g., mutual information)."""
#         # TODO: Replace with a real mutual information / entropy calculation.
#         logging.info("InformationTheoryQuantifier.quantify called")
#         logging.debug("input_data shape: %s", getattr(input_data, 'shape', None))
#         logging.debug("output shape: %s", getattr(output, 'shape', None))
#         return 1.0

#     def _calculate_entropy(self, data: np.ndarray) -> float:
#         """Calculate Shannon entropy."""
#         # Placeholder
#         return 1.0

#     def _calculate_mutual_information(self, input_data: np.ndarray, output: np.ndarray) -> float:
#         """Calculate mutual information between input data and model output."""
#         # Placeholder
#         return 1.0
