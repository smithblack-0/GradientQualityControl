"""
StateManagementSubsystem implementation.

Manages all state for optimizer wrappers, including wrapper-specific state
and optimizer parameter group extensions.
"""

from numbers import Number
from typing import Any, Dict, List, Literal, Tuple

import torch
import torch_schedule_anything as tsa


def _is_scalar_numeric(value: Any) -> bool:
    """Check if value is a scalar numeric type or scalar tensor."""
    if isinstance(value, Number):
        return True
    if isinstance(value, torch.Tensor):
        return value.ndim == 0  # Scalar tensor
    return False


class StateManagementSubsystem:
    """
    Manages state for optimizer wrappers.

    Provides unified interface for wrapper state and optimizer parameter access.
    Handles serialization and ScheduleAnything integration.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
    ):
        """
        Initialize state management subsystem.

        Args:
            optimizer: PyTorch optimizer to manage state for
        """
        self.optimizer = optimizer
        self.wrapper_states: Dict[str, Dict[str, Any]] = {}  # {name: {"value": ..., "flag": ...}}

    def get_state(
        self,
        name: str,
    ) -> Any:
        """
        Retrieve state value by name.

        Args:
            name: State variable name

        Returns:
            Value from wrapper_states (without metadata) or list of values from optimizer
            param_groups

        Raises:
            KeyError: If name not found in either wrapper_states or optimizer param_groups
            TypeError: If optimizer param is non-numeric or non-scalar
        """
        # Check wrapper_states first
        if name in self.wrapper_states:
            return self.wrapper_states[name]["value"]

        # Check optimizer param_groups using ScheduleAnything utility
        regrouped = tsa.get_param_groups_regrouped_by_key(self.optimizer, name)

        # If no param groups returned, parameter doesn't exist
        if not regrouped:
            raise KeyError(f"State '{name}' not found")

        # Extract values and validate they're scalar numeric
        values = []
        for value, params, group in regrouped:
            # Validate that value is numeric or scalar tensor
            if not _is_scalar_numeric(value):
                raise TypeError(
                    f"Optimizer parameter '{name}' is not a scalar numeric type or scalar tensor"
                )
            values.append(value)

        return values

    def set_state(
        self,
        name: str,
        value: Any,
        flag: Literal["vital", "optional", "optimizer"],
    ) -> None:
        """
        Set state value with specified flag.

        Args:
            name: State variable name
            value: Value to store
            flag: One of "vital", "optional", "optimizer"

        Raises:
            RuntimeError: If flag changes for existing name, or if name collides with
            optimizer param
        """
        if flag in ("vital", "optional"):
            # Check for collision with optimizer params
            if len(self.optimizer.param_groups) > 0 and name in self.optimizer.param_groups[0]:
                # Verify it's in all groups (if in first group, likely in all)
                in_all_groups = all(name in group for group in self.optimizer.param_groups)
                if in_all_groups:
                    raise RuntimeError(
                        f"Cannot set wrapper state '{name}': name collides with optimizer parameter"
                    )

            # Check if name exists with different flag
            if name in self.wrapper_states:
                existing_flag = self.wrapper_states[name]["flag"]
                if existing_flag != flag:
                    raise RuntimeError(
                        f"Cannot change flag for '{name}' from '{existing_flag}' to '{flag}'"
                    )

            # Store with metadata
            self.wrapper_states[name] = {"value": value, "flag": flag}

        elif flag == "optimizer":
            # Check if parameter already exists in optimizer
            if len(self.optimizer.param_groups) > 0 and name in self.optimizer.param_groups[0]:
                raise RuntimeError(
                    f"Cannot extend optimizer with '{name}': parameter already exists"
                )

            # Convert scalar tensors to float for tsa compatibility
            if isinstance(value, torch.Tensor):
                if value.ndim == 0:  # Scalar tensor
                    value = value.item()
                else:
                    raise TypeError(
                        f"Optimizer parameter '{name}' must be scalar, not tensor of "
                        f"shape {value.shape}"
                    )

            # Extend optimizer param_groups using ScheduleAnything
            tsa.extend_optimizer(self.optimizer, name, value, overwrite_values=False)

    def show_state(self) -> List[Tuple[str, str]]:
        """
        List all available state lookups.

        Returns:
            List of (name, flag) tuples where flag is "vital", "optional", or "optimizer"
        """
        result = []

        # Add wrapper states
        for name, metadata in self.wrapper_states.items():
            result.append((name, metadata["flag"]))

        # Add optimizer params that are shared across all groups
        if len(self.optimizer.param_groups) == 0:
            return result

        # Find parameters shared across all groups
        first_group = self.optimizer.param_groups[0]
        for param_name in first_group.keys():
            # Skip 'params' key (contains actual parameter tensors)
            if param_name == "params":
                continue

            # Check if present in all groups
            in_all_groups = all(param_name in group for group in self.optimizer.param_groups)
            if not in_all_groups:
                continue

            # Check if value is numeric or scalar tensor
            value = first_group[param_name]
            if not _is_scalar_numeric(value):
                continue

            # Add to result
            result.append((param_name, "optimizer"))

        return result

    def state_dict(self) -> Dict[str, Any]:
        """
        Create state dictionary for serialization.

        Returns:
            Dictionary with "wrapper_states" and "optimizer_states" keys
        """
        return {
            "wrapper_states": self.wrapper_states,
            "optimizer_states": self.optimizer.state_dict(),
        }

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
    ) -> None:
        """
        Restore state from state dictionary.

        Args:
            state_dict: Dictionary from state_dict()
        """
        # Restore wrapper states
        self.wrapper_states = state_dict["wrapper_states"]

        # Restore optimizer states
        self.optimizer.load_state_dict(state_dict["optimizer_states"])
