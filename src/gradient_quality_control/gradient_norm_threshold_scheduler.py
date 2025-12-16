"""
The Gradient Norm Threshold Scheduler is designed as an optimizer
wrapper which performs gradient accumulation, and only steps
the optimizer when the norm is below a threshold. It autotunes
logical batch sizes and provides some significant gradient noise
cleaning too.
"""

from typing import Any, Callable, Dict, Optional

import torch

from .base import AbstractOptimizerWrapper
from .optim_utils import (optimizer_extend,
                          optimizer_get_collection,
                          optimizer_get_raw_grad_norms,
                          arbitrary_schedule_factory)

class OptimizerWrapperGNTS(AbstractOptimizerWrapper):
    """
    Gradient Norm Threshold Scheduler (GNTS)

    This wrapper performs adaptive gradient accumulation, a Gradient Quality Control
    activity, based on the magnitude of the aggregated gradient. Rather than stepping the
    underlying optimizer after every batch, GNTS accumulates gradients
    until the total gradient norm falls below a configurable threshold.
    At that point, the wrapper triggers an optimizer step and resets
    accumulation.

    The system will setup the "gradient_norm_threshold" field in the underlying optimizer
    for all parameter group, with 1.0 by default, then check if the gradient norm for
    each group is above or below the threshold before starting. This means a simple
    way to adjust the importannce of groups is to ensure "gradient_norm_threshold" is
    defined in optimizer.param_groups before getting here.

    Important:
        - **Make sure to understand a higher threshold is MORE permissive, not less when tweaking per group thresholds**
        - If you use multiple param groups, the system waits to step until all param
          groups meet their threshold.
        - It is HIGHLY recommended to use the constructor utility "make_gnts_optimizer" instead since it returns both
            the schedule and optimizer needed.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_batch_draws: int = 64,
        group_mode: str = "L2"
    ):
        """
        Initialize the GNTS wrapper.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The underlying optimizer that GNTS will manage. GNTS defers
            stepping this optimizer until its control condition is met.
        max_batch_draws : int, optional (default: 64)
            The maximum number of consecutive batches GNTS will draw
            before forcing an optimizer step, regardless of the gradient
            norm condition.
        group
        """

        # Extend the optimizer to contain a gradient_norm_threshold target, if it does not already exist.
        # Then initialize max draws, and the underlying base class, finishing initialization.
        optimizer = optimizer_extend(optimizer, "gradient_norm_threshold", 1.0)
        self.max_draws = max_batch_draws
        super().__init__(optimizer)

    def step(
        self,
        closure: Optional[Callable[[], Any]] = None,
    ) -> bool:
        """
        Conditionally step the optimizer based on gradient quality.

        Accumulates gradients across batches until the mean gradient norm
        falls below the threshold (or max_draws is reached), then steps
        the optimizer. Call this once per batch as you would with a normal
        optimizer. Do not use zero_grad yourself.

        Parameters
        ----------
        closure : callable, optional
            Optional closure for optimizers like LBFGS. If provided and it
            returns a value, the result will be cached in `last_optimizer_result`.

        Returns
        -------
        bool
            True if the optimizer stepped, False if still accumulating gradients.
        """
        optimizer_needs_step = True

        # Get threshold and norms. Remember to account for the
        # fact we are summing, and need to divide to get the mean
        thresholds = optimizer_get_collection(self.optimizer, "gradient_norm_threshold")
        norms = optimizer_get_raw_grad_norms(self.optimizer)
        norms = [norm/self.max_draws for norm in norms]

        # Check all thresholds are satisfied.
        for threshold, norm in zip(thresholds, norms):
            if norm > threshold:
                optimizer_needs_step = False

        # Handle stepping
        optimizer_takes_step = False
        if optimizer_needs_step or self.num_draws >= self.max_draws:
            self._take_optimizer_step(closure)
            optimizer_takes_step = True
        self._take_batch_step()
        return optimizer_takes_step

    def statistics(self) -> Dict[str, Any]:
        """
        Return simple runtime statistics for inspection or logging.

        Returns
        -------
        dict
            A dictionary containing:
            - "batches": Total number of minibatches processed.
            - "steps": Total number of optimizer updates taken.
            - "norm_threshold": The current gradient-norm threshold.
            - "num_draws": Number of samples accumulated since
              the last optimizer step.
        """
        statistics = self._get_base_statistics()
        statistics["norm_threshold"] = self.norm_threshold
        return statistics

    def zero_grad(self):
        raise NotImplementedError("You should not use .zero_grad with this wrapper.")

    def __repr__(self):
        return (
            f"<OptimizerWrapperGNTS: "
            f"batches={self.num_batches}, "
            f"steps={self.num_steps}, "
            f"core={type(self.optimizer).__name__}>"
        )

