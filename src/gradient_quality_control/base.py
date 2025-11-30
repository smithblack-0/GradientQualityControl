"""
The base class largely takes care of common bookkeeping
issues related to the fact that we may take many minibatch samples
 before stepping the optimizer. This includes ensuring we use
mean gradients for optimization. amd logging and reporting
of statistics.
"""

from typing import Any, Callable, Dict, Optional

import torch
from torch.optim import Optimizer


class AbstractOptimizerWrapper(Optimizer):
    """
    Abstract base class for GQC optimizer wrappers.
    This pretends to be an optimizer so torch can interface with
    it, and largely GQC mechanisms are implemented by overriding
    methods and then using super into the original  call.

    Args:
        optimizer: The underlying PyTorch optimizer to wrap

    Fields:
         - last optimizer result: cached result of last optimizer call
         - mean_last_grad_norm: mean last gradient norms from the last time
            the optimizer took a step. L2 norm.
         - num_batches: total number of batches processed
         - num_steps: total number of optimizer steps taken
         - num_draws: Number of optimizer draws within this step.

    """

    def __init__(
        self,
        optimizer: Optimizer,
    ):
        self.optimizer = optimizer
        self.last_optimizer_result = None
        self.num_batches = 1  # One batch if we can even invoke the optimizer
        self.num_steps = 0
        self.num_draws = 1  # One draw because we had to take one batch.
        self.last_grad_norm = None
        self.last_step_num_draws = None

        self.parameters = []
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                self.parameters.append(p)

    def _take_batch_step(self):
        self.num_batches += 1
        self.num_draws += 1

    def _take_optimizer_step(
        self,
        closure: Optional[Callable[[], Any]] = None,
    ):
        """Takes an optimizer step, caching the return"""

        # Catch issue
        if self.num_draws == 0:
            raise RuntimeError("Forgot to invoke take batch step after take optimizer step")

        # Convert gradients to mean form
        grads = []
        for param in self.parameters:
            if param.grad is None:
                continue
            param.grad.data /= self.num_draws
            grads.append(param.grad)

        # Take step.
        self.last_grad_norm = torch.nn.utils.get_total_norm(grads).item()
        self.last_optimizer_result = self.optimizer.step(closure)
        self.optimizer.zero_grad()
        self.last_step_num_draws = self.num_draws
        self.num_steps += 1
        self.num_draws = 0

    def _get_base_statistics(self) -> Dict[str, Any]:
        """gets the basic statistics dictionary"""
        return {
            "batches": self.num_batches,
            "steps": self.num_steps,
            "num_draws": self.num_draws,
            "last_mean_grad_norm": self.last_grad_norm,
            "last_step_num_draws": self.last_step_num_draws,
        }

    def __getattr__(
        self,
        name: str,
    ):
        """Route everything not explicitly set  to the underlying optimizer."""
        return getattr(self.optimizer, name)

    def state_dict(self) -> Dict[str, Any]:
        """Returns a functional statedict"""
        output = {
            "last_optimizer_result": self.last_optimizer_result,
            "last_step_num_draws": self.last_step_num_draws,
            "num_batches": self.num_batches,
            "num_steps": self.num_steps,
            "num_draws": self.num_draws,
            "last_mean_grad_norm": self.last_grad_norm,
            "optimizer": self.optimizer.state_dict(),
        }
        return output

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
    ):
        """Loads optimizer state dict and underlying training functionality"""
        self.last_optimizer_result = state_dict["last_optimizer_result"]
        self.last_step_num_draws = state_dict["last_step_num_draws"]
        self.num_batches = state_dict["num_batches"]
        self.num_steps = state_dict["num_steps"]
        self.num_draws = state_dict["num_draws"]
        self.last_grad_norm = state_dict["last_mean_grad_norm"]
        self.last_step_num_draws = state_dict["last_step_num_draws"]
        self.optimizer.load_state_dict(state_dict["optimizer"])
