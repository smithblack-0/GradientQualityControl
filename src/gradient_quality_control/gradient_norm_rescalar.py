"""
The Gradient Norm Scaler performs direct gradient rescaling to achieve
a target gradient norm, either globally or per-parameter independently.
"""

from typing import Any, Callable, Dict, Literal, Optional

import torch

from .base import AbstractOptimizerWrapper


class OptimizerWrapperGNR(AbstractOptimizerWrapper):
    """
    Gradient Norm Rescaler (GNR)

    This wrapper rescales gradients to match a target norm specified by an
    attached scheduler. Unlike GNTS which accumulates batches, GNS directly
    scales gradients on every step.

    Two scaling modes are available:
    - 'global': Scales all gradients uniformly to achieve target total norm
    - 'independent': Scales each parameter's gradient independently to target norm

    The target norm is controlled via the scheduler system, which modifies
    the 'lr' field in param_groups (interpreted as target norm, not learning rate).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: Literal["global", "independent"] = "global",
    ):
        """
        Initialization of the Gradient Norm Rescaler
        optimizer wrapper.

        :param optimizer: The optimizer to wrap
        :param mode:  {'global', 'independent'}, optional (default: 'global')
        Scaling strategy:
        - 'global': All gradients scaled by same factor to match target total norm
        - 'independent': Each parameter gradient scaled to target norm separately

        """
        super().__init__(optimizer)

        if mode not in ("global", "independent"):
            raise ValueError(f"mode must be 'global' or 'independent', got {mode}")

        self.mode = mode

        # Scheduler controls this (interprets as target norm)
        self.param_groups = [{"lr": 1.0}]

        # Note: self.parameters is constructed in base class.

    @property
    def target_norm(self) -> float:
        """Current target norm from scheduler."""
        return self.param_groups[0]["lr"]

    def step(
        self,
        closure: Optional[Callable[[], Any]] = None,
    ) -> bool:
        """
        Scale gradients to target norm and step the optimizer.

        Automatically handles gradient clearing after optimizer step.

        Parameters
        ----------
        closure : callable, optional
            Optional closure for optimizers like LBFGS. If the closure returns
            a value, it will be cached in `last_optimizer_result`.

        Returns
        -------
        bool
            Whether an optimizer step was taken. This is always true in this
            wrapper variant, but the return is standardized across all wrapper
            classes
        """

        if self.mode == "global":
            grads = [p.grad for p in self.parameters]
            norm = torch.nn.utils.get_total_norm(grads) + 1e-12

        # Main scaling loop
        for parameter in self.parameters:
            if parameter.grad is None:
                continue

            # Handle indepedent vs global rescaling
            if self.mode == "global":
                parameter.grad *= self.target_norm / norm
            else:  # independent
                norm = parameter.grad.norm() + 1e-12
                parameter.grad *= self.target_norm / norm

        # Step underlying optimizer
        self._take_optimizer_step(closure)
        self._take_batch_step()
        return True

    def statistics(self) -> Dict[str, Any]:
        """
        Return runtime statistics.

        Returns
        -------
        dict
            Statistics including total steps and current target norm.
        """
        statistics = self._get_base_statistics()
        statistics["target_norm"] = self.target_norm
        statistics["mode"] = self.mode
        return statistics

    def zero_grad(self):
        """Not needed - gradients cleared automatically in step()."""
        raise NotImplementedError(
            "GNS automatically clears gradients in step(). " "Do not call zero_grad() manually."
        )

    def __repr__(self):
        return (
            f"<OptimizerWrapperGNR(mode={self.mode}, "
            f"target_norm={self.target_norm:.3f}, "
            f"batches={self.num_batches}, "
            f"steps={self.num_steps}, "
            f"core={type(self.optimizer).__name__})>"
        )
