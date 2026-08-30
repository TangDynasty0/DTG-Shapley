from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np

"""
Value Quantification Module

This module defines the value-quantification interface and the data-value
evaluator used by the information-theoretic implementations.
"""



class ValueQuantifier(ABC):
    """Abstract base class for data value quantification methods."""
    
    @abstractmethod
    def quantify(self, input_data: Any, output: Any) -> float:
        """
        Quantify the value of a model input combination relative to the output.
        
        Args:
            input_data: Model input features or input combination
            output: Model output or target values
            
        Returns:
            float: Quantified value score
        """
        pass

# Pull concrete implementations into separate modules to keep
# `value_quantification.py` focused on the abstract API.
from information_theory import InformationTheoryQuantifier


class TestQuantifier(ValueQuantifier):
    """A simple test quantifier for framework validation."""

    def __init__(self, fixed_value: float = 1.0):
        """Initialize with a fixed score for testing."""
        self.fixed_value = fixed_value

    def quantify(self, input_data: Any, output: Any) -> float:
        """Return a fixed score to verify framework behavior."""
        return float(self.fixed_value)


class DataValueEvaluator:
    """Main class for evaluating data combination values."""
    
    def __init__(self, quantifier: ValueQuantifier, cache: bool = True):
        """
        Initialize evaluator with a single quantification method.
        
        Args:
            quantifier: A ValueQuantifier instance.
            cache: Whether combination values should be cached.
        """
        if not isinstance(cache, bool):
            raise TypeError("cache must be a boolean value.")
        self.quantifier = quantifier
        self.cache = cache
        self.combo_value: Dict[str, float] = {}
        self.evaluation_requests = 0
        self.quantification_calls = 0
        self.cache_hits = 0
    
    def evaluate_combination(self, data_combinations: np.ndarray, 
                           model_output: np.ndarray, combo_name: str) -> Dict[str, float]:
        """
        Evaluate value of one or more data combinations represented as a matrix.
        
        Args:
            data_combinations: Matrix of shape (n_features, m_samples), where
                each row corresponds to a feature.
            model_output: Output vector of shape (1, m_samples).
            combo_name: Name of the combination to evaluate.
            
        Returns:
            Dict mapping combination names to their quantified values
        """
        if not isinstance(data_combinations, np.ndarray):
            data_combinations = np.asarray(data_combinations)
        if not isinstance(model_output, np.ndarray):
            model_output = np.asarray(model_output)
        if model_output.ndim == 1:
            model_output = model_output.reshape(1, -1)

        results = {
            combo_name: self._get_combo_value(
                combo_name,
                data_combinations,
                model_output,
            )
        }
        return results
    
    def _get_combo_value(self, combo_name: str, data: Any, model_output: Any) -> float:
        """Return cached value if available; otherwise compute and cache it."""
        self.evaluation_requests += 1
        if self.cache and combo_name in self.combo_value:
            self.cache_hits += 1
            return self.combo_value[combo_name]

        self.quantification_calls += 1
        value = self.quantify_combo(combo_name, data, model_output)
        if self.cache:
            self.combo_value[combo_name] = value
        return value

    def quantify_combo(self, combo_name: str, data: np.ndarray, model_output: np.ndarray) -> float:
        """Compute a single combination's value using the configured quantifier."""
        return self.quantifier.quantify(data, model_output)
