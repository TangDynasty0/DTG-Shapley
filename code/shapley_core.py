"""Shared public interfaces for Shapley estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

class ValueFunction(ABC):
    """Abstract base class for data value functions."""

    @abstractmethod
    def evaluate(self, subset: Sequence[Any]) -> float:
        """Compute the value of a subset of data items.

        Args:
            subset: A sequence representing the selected data points.

        Returns:
            A float score for the subset.
        """
        raise NotImplementedError

@dataclass
class ContributionResult:
    shapley_values: Dict[int, float]
    sample_count: int
    data_count: int
    diagnostics: Dict[str, Any] = field(default_factory=dict)
