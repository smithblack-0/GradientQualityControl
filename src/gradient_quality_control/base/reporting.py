"""
ReportingSubsystem implementation.

Generates statistics and vital statistics reports by querying StateManagementSubsystem.
"""

from numbers import Number
from typing import Any, Dict, List, Literal, Union

import torch

from .state_management import StateManagementSubsystem


def _to_python_number(
    value: Union[Number, torch.Tensor],
) -> Number:
    """Convert tensor or Number to Python number."""
    if isinstance(value, torch.Tensor):
        return value.item()
    return value


class ReportingSubsystem:
    """
    Stateless facade for generating statistics reports.

    Queries StateManagementSubsystem and formats state for reporting.
    """

    def __init__(
        self,
        state_manager: StateManagementSubsystem,
    ):
        """
        Initialize reporting subsystem.

        Args:
            state_manager: StateManagementSubsystem for querying state

        Post-conditions:
            Creates stateless facade with no internal state
        """
        self._state_manager = state_manager

    def aggregate_numeric_list(
        self,
        values: List[Union[Number, torch.Tensor]],
        behavior: Literal["mean", "max", "min"],
    ) -> Number:
        """
        Aggregate a list of numeric values using specified strategy.

        Args:
            values: List of numeric values (Number or scalar tensors)
            behavior: Aggregation strategy - "mean", "max", or "min"

        Returns:
            Aggregated numeric value (Python number)
        """
        # Convert all values to Python numbers
        numeric_values = [_to_python_number(v) for v in values]

        if behavior == "mean":
            return sum(numeric_values) / len(numeric_values)
        elif behavior == "max":
            return max(numeric_values)
        elif behavior == "min":
            return min(numeric_values)
        else:
            raise ValueError(f"Invalid aggregation behavior: {behavior}")

    def statistics(
        self,
        behavior: Literal["vital", "verbose"] = "verbose",
        aggregate_behavior: Literal["mean", "max", "min"] = "mean",
    ) -> Dict[str, Any]:
        """
        Generate statistics dictionary from available state.

        Args:
            behavior: "verbose" includes all state, "vital" includes only vital and optimizer state
            aggregate_behavior: How to aggregate multi-group hyperparameters when values differ

        Returns:
            Dictionary of statistics
        """
        result = {}

        # Get all available state
        state_list = self._state_manager.show_state()

        # Filter based on behavior
        if behavior == "vital":
            # Include only vital and optimizer state
            state_list = [
                (name, flag) for name, flag in state_list if flag in ("vital", "optimizer")
            ]
        # verbose includes everything, no filtering needed

        # Process each state entry
        for name, flag in state_list:
            try:
                value = self._state_manager.get_state(name)

                # Handle lists
                if isinstance(value, list):
                    # Check if all values in list are equal
                    if len(value) > 0 and all(v == value[0] for v in value):
                        # All equal: use scalar value (convert if tensor)
                        scalar = value[0]
                        if isinstance(scalar, torch.Tensor):
                            result[name] = _to_python_number(scalar)
                        else:
                            result[name] = scalar
                    else:
                        # Not all equal: aggregate and add * suffix
                        aggregated = self.aggregate_numeric_list(value, aggregate_behavior)
                        result[name + "*"] = aggregated
                # Handle tensors - convert to Python numbers
                elif isinstance(value, torch.Tensor):
                    result[name] = _to_python_number(value)
                # Handle numbers - include as-is
                elif isinstance(value, Number):
                    result[name] = value
                # Handle strings - only for vital/optional state, not optimizer params
                elif isinstance(value, str) and flag in ("vital", "optional"):
                    result[name] = value
                # All other types (None, objects, etc.) are omitted
            except Exception:
                # Skip if retrieval or processing fails
                pass

        return result

    def vital_statistics(
        self,
        aggregate_behavior: Literal["mean", "max", "min"] = "mean",
    ) -> Dict[str, Any]:
        """
        Generate curated vital statistics for real-time monitoring.

        Alias to statistics(behavior="vital").

        Args:
            aggregate_behavior: How to aggregate multi-group optimizer parameters when values differ

        Returns:
            Dictionary of vital statistics
        """
        return self.statistics(behavior="vital", aggregate_behavior=aggregate_behavior)
