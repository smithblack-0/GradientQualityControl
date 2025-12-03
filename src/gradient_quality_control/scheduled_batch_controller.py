"""
Scheduled Batch Controller for fixed gradient accumulation based on a schedule.
"""

from typing import Any, Callable, Dict, Optional

import torch

from .base import AbstractOptimizerWrapper


class OptimizerWrapperSBC(AbstractOptimizerWrapper):
    """
    Scheduled Batch Controller (SBC)

    This wrapper accumulates a fixed number of batches based on a schedule.
    The scheduler sets the target logical batch size, and SBC accumulates
    until reaching that size (rounded to nearest multiple of physical batch size).

    Useful for static gradient accumulation or scheduled batch size increases.
    WARNING: Attached schedulers set target batch size instead.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The underlying optimizer to wrap
    physical_batch_size : int
        The batch size of each microbatch
    max_batch_draws : int, optional (default: 64)
        Maximum accumulation steps before forcing an optimizer step
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        physical_batch_size: int,
        max_batch_draws: int = 64,
    ):
        super().__init__(optimizer)

        self.physical_batch_size = physical_batch_size
        self.max_draws = max_batch_draws

        # Scheduler controls target logical batch size via 'lr'
        self.param_groups = [{"lr": float(1.0)}]

        # Note: self.parameters is constructed in base class.

    @property
    def target_logical_batch_size(self) -> float:
        """Target logical batch size from scheduler."""
        return self.param_groups[0]["lr"]

    @property
    def target_draws(self) -> int:
        """Number of draws needed to reach target logical batch size."""
        raw = self.target_logical_batch_size / self.physical_batch_size
        return max(1, round(raw))

    def step(
        self,
        closure: Optional[Callable[[], Any]] = None,
    ) -> bool:
        """
        Accumulate batches until reaching target logical batch size.

        Parameters
        ----------
        closure : callable, optional
            Optional closure for optimizers like LBFGS. If provided and it
            returns a value, the result will be cached in `last_optimizer_result`.

        Returns
        -------
        bool
            True if the optimizer stepped, False if still accumulating.
        """
        # Check if we should step after this draw
        will_step = self.num_draws >= self.target_draws or self.num_draws >= self.max_draws

        if will_step:
            self._take_optimizer_step(closure)

        self._take_batch_step()
        return will_step

    def statistics(self) -> Dict[str, Any]:
        """Return runtime statistics."""
        statistics = self._get_base_statistics()
        statistics["target_draws"] = self.target_draws
        statistics["target_logical_batch_size"] = self.target_logical_batch_size
        statistics["physical_batch_size"] = self.physical_batch_size
        return statistics

    def zero_grad(self):
        raise NotImplementedError(
            "ScheduledBatchController automatically clears gradients in step(). "
            "Do not call zero_grad() manually."
        )

    def __repr__(self):
        return (
            f"<OptimizerWrapperSBC("
            f"physical={self.physical_batch_size}, "
            f"target_draws={self.target_draws}, "
            f"steps={self.num_steps}, "
            f"core={type(self.optimizer).__name__})>"
        )
