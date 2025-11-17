"""
Gradient Noise Scale controller for adaptive gradient accumulation.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch import nn

from .base import AbstractOptimizerWrapper


class OptimizerWrapperGNS(AbstractOptimizerWrapper):
    """
    Gradient Noise Scale (GNS) Controller

    This wrapper uses the Gradient Noise Scale heuristic from McCandish et al.
    to determine when to step. GNS estimates the ratio of gradient variance to
    squared gradient magnitude, measuring noise in gradient estimates.

    The controller steps when: estimated_GNS < num_draws * noise_multiplier

    This criterion balances diminishing returns of noise reduction against the
    linear cost of additional samples. Gradient norm hooks are automatically
    attached to all parameters to track per-microbatch norms.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The underlying optimizer to wrap
    noise_multiplier : float, optional (default: 1.0)
        Cost-benefit tradeoff parameter for the GNS criterion.
        Smaller decreases tolerated noise.
    max_batch_draws : int, optional (default: 64)
        Maximum accumulation steps before forcing an optimizer step
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        noise_multiplier: float = 1.0,
        max_batch_draws: int = 64,
    ):
        super().__init__(optimizer)

        self.noise_multiplier = noise_multiplier
        self.max_draws = max_batch_draws

        # Attach gradient norm hooks
        for param in self.parameters:
            self._attach_grad_norm_hook(param)

        # State tracking
        self.grad_norms = []  # Per-microbatch gradient norms
        self._cached_gns = None  # Cached GNS estimate

    @staticmethod
    def _attach_grad_norm_hook(parameter: nn.Parameter):
        """Attach a hook to track per-microbatch gradient norm."""
        if hasattr(parameter, "_has_grad_norm_hook"):
            return  # Already has hook

        def grad_norm_hook(grad_input: torch.Tensor, param=parameter) -> torch.Tensor:
            param.last_gradient_norm = grad_input.norm()
            return grad_input

        parameter.register_hook(grad_norm_hook)
        parameter._has_grad_norm_hook = True

    @staticmethod
    def _get_independent_grad_norms(parameters: List[nn.Parameter]) -> float:
        """Compute gradient norm from last backward pass."""
        norms = []
        for p in parameters:
            if hasattr(p, "last_gradient_norm"):
                norms.append(p.last_gradient_norm)

        if not norms:
            raise RuntimeError("No gradient norms found. Did backward() run?")

        return torch.stack(norms).norm().item()

    def step(
        self,
        closure: Optional[Callable[[], Any]] = None,
    ) -> bool:
        """
        Conditionally step the optimizer based on GNS criterion.

        Accumulates gradients and tracks per-microbatch gradient norms.
        Steps when estimated_GNS < num_draws * payoff_multiplier, or when
        max_draws is reached.

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
        # Get gradient norm for this microbatch.
        current_grad_norm = self._get_independent_grad_norms(self.parameters)
        self.grad_norms.append(current_grad_norm)

        # if there are not enough samples for a GNS, we are done
        # Else, compute it
        if len(self.grad_norms) < 2:
            self._cached_gns = None
            self._take_batch_step()
            return False
        else:
            self._cached_gns = self.compute_gns_estimate(self.grad_norms)

        # Check if we should step (computes and caches GNS)
        force_step = self.num_draws >= self.max_draws
        gns_criterion_met = self._cached_gns <= self.num_draws * self.noise_multiplier
        will_step_optimizer = gns_criterion_met or force_step

        # Step the optimizer itself
        if will_step_optimizer:
            self._take_optimizer_step(closure)
            self.grad_norms.clear()
            self._cached_gns = None  # Clear cache

        # Routine bookkeeping and return.
        self._take_batch_step()
        return will_step_optimizer

    @staticmethod
    def compute_gns_estimate(gradient_norms: List[float]) -> float:
        """
        Computes the gradient noise scale from the
        provided lists
        Parameters
        ---------
            gradient_norms : list of floats
        :return: The GNS scale
        """
        # Compute statistics
        norms_array = np.array(gradient_norms)
        variance = np.var(norms_array)
        mean_squared = np.mean(norms_array**2)

        # Compute and return estimated gns
        estimated_gns = variance / (mean_squared + 1e-8)
        return estimated_gns

    def statistics(self) -> Dict[str, Any]:
        """Return runtime statistics including cached GNS estimate."""
        statistics = self._get_base_statistics()
        statistics["noise_multiplier"] = self.noise_multiplier
        statistics["estimated_gns"] = self._cached_gns  # Use cached value
        return statistics

    def zero_grad(self):
        raise NotImplementedError(
            "GNScale automatically clears gradients in step(). " "Do not call zero_grad() manually."
        )

    def __repr__(self):
        return (
            f"<OptimizerWrapperGNS("
            f"noise_multiplier={self.noise_multiplier:.2f}, "
            f"steps={self.num_steps}, "
            f"core={type(self.optimizer).__name__})>"
        )
