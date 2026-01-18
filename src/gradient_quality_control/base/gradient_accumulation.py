"""
GradientAccumulationStepSubsystem implementation.

Manages gradient accumulation mechanics and optimizer stepping.
"""

from typing import Optional

import torch

from ..optimizer_utils import compute_grad_norm_from_optimizer
from .state_management import StateManagementSubsystem


class GradientAccumulationStepSubsystem:
    """
    Manages gradient accumulation and optimizer stepping.

    Handles batch counting, enforces accumulation bounds, averages gradients
    before stepping, and maintains step counters.
    """

    def __init__(
        self,
        state_manager: StateManagementSubsystem,
        optimizer: torch.optim.Optimizer,
        max_draws: int = 64,
    ):
        """
        Initialize gradient accumulation subsystem.

        Args:
            state_manager: StateManagementSubsystem for persisting counters
            optimizer: PyTorch optimizer to wrap
            max_draws: Maximum batches allowed to accumulate before forced step

        Post-conditions:
            - Initializes vital state: num_batches=0, num_steps=0, last_num_draws=None,
              last_grad_norm=None
            - Initializes optional state: num_draws=0
        """
        self._state_manager = state_manager
        self._optimizer = optimizer
        self._max_draws = max_draws

        # Initialize state
        state_manager.set_state("num_batches", 0, "vital")
        state_manager.set_state("num_steps", 0, "vital")
        state_manager.set_state("last_num_draws", None, "vital")
        state_manager.set_state("last_grad_norm", None, "vital")
        state_manager.set_state("num_draws", 0, "optional")

    @property
    def num_batches(self) -> int:
        """Total batches processed since creation."""
        return self._state_manager.get_state("num_batches")

    @property
    def num_steps(self) -> int:
        """Total optimizer steps taken."""
        return self._state_manager.get_state("num_steps")

    @property
    def num_draws(self) -> int:
        """Batches accumulated since last step."""
        return self._state_manager.get_state("num_draws")

    @property
    def max_draws(self) -> int:
        """maximum allowed draws"""
        return self._max_draws

    @property
    def last_num_draws(self) -> Optional[int]:
        """Batch count from most recent step. None before first step."""
        return self._state_manager.get_state("last_num_draws")

    @property
    def last_grad_norm(self) -> Optional[float]:
        """Gradient norm from most recent step. None before first step."""
        return self._state_manager.get_state("last_grad_norm")

    def batch_received(self) -> None:
        """
        Called when a batch is processed.

        Updates counters and enforces accumulation bounds.

        Raises:
            RuntimeError: If num_draws >= max_draws
        """
        current_draws = self._state_manager.get_state("num_draws")

        # Check accumulation bound
        if current_draws >= self._max_draws:
            raise RuntimeError(
                f"Maximum accumulation exceeded: num_draws={current_draws} >= "
                f"max_draws={self._max_draws}"
            )

        # Increment counters
        current_batches = self._state_manager.get_state("num_batches")
        self._state_manager.set_state("num_batches", current_batches + 1, "vital")
        self._state_manager.set_state("num_draws", current_draws + 1, "optional")

    def take_optimizer_step(self) -> None:
        """
        Average gradients, step optimizer, update counters.

        The only valid way to step the wrapped optimizer.

        Raises:
            RuntimeError: If num_draws == 0 (cannot step without batches)
        """
        current_draws = self._state_manager.get_state("num_draws")

        # Check that we have batches to step
        if current_draws == 0:
            raise RuntimeError("Cannot step optimizer: num_draws is 0 (no batches accumulated)")

        # Average gradients: divide by num_draws
        for group in self._optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.div_(current_draws)

        # Compute gradient norm
        grad_norm = compute_grad_norm_from_optimizer(self._optimizer)

        # Step the optimizer
        self._optimizer.step()

        # Zero gradients
        self._optimizer.zero_grad()

        # Update state
        current_steps = self._state_manager.get_state("num_steps")
        self._state_manager.set_state("last_num_draws", current_draws, "vital")
        self._state_manager.set_state("last_grad_norm", grad_norm, "vital")
        self._state_manager.set_state("num_steps", current_steps + 1, "vital")
        self._state_manager.set_state("num_draws", 0, "optional")
