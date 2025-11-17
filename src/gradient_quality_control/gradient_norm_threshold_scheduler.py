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


class OptimizerWrapperGNTS(AbstractOptimizerWrapper):
    """
    Gradient Norm Threshold Scheduler (GNTS)

    This wrapper performs adaptive gradient accumulation, a Gradient Quality Control
    activity, based on the magnitude of the aggregated gradient. Rather than stepping the
    underlying optimizer after every batch, GNTS accumulates gradients
    until the total gradient norm falls below a configurable threshold.
    At that point, the wrapper triggers an optimizer step and resets
    accumulation.

    The threshold is controlled externally through the scheduler system:
    schedulers may adjust the value stored in the parameter group's `lr`
    field, which GNTS interprets as the current gradient-norm target
    rather than a learning rate.

    Users should not call `.zero_grad()` directly when using this wrapper,
    as gradient clearing is managed internally.
    """

    @property
    def norm_threshold(self):
        return self.param_groups[0]["lr"]

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_batch_draws: int = 64,
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
        """
        super().__init__(optimizer)

        # Schedulers will modify this thinking it is a learning rate,
        # but we instead interpret it as the gradient norm threshold
        self.param_groups = [{"lr": 1.0}]

        # Other kinds of initialization.
        self.max_draws = max_batch_draws

        # Note: self.parameters is constructed in base class.

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
        optimizer_was_stepped = False
        grads = [p.grad for p in self.parameters]
        current_norm = torch.nn.utils.get_total_norm(grads) / self.num_draws
        if current_norm < self.norm_threshold or self.num_draws >= self.max_draws:
            self._take_optimizer_step(closure)
            optimizer_was_stepped = True
        self._take_batch_step()
        return optimizer_was_stepped

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
