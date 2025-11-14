"""
The base class largely takes care of common bookkeeping
issues related to the fact that we may take many minibatch samples
 before stepping the optimizer. This includes ensuring we use
mean gradients for optimization. amd logging and reporting
of statistics.
"""

from torch.optim import Optimizer
from torch import nn
from typing import Callable, Any, Optional, Dict, List


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
         - num_batches: total number of batches processed
         - num_steps: total number of optimizer steps taken
         - num_draws: Number of optimizer draws within this step.
    """

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.last_optimizer_result = None
        self.num_batches = 0
        self.num_steps = 0
        self.num_draws = 1

        self.parameters = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                self.parameters.append(p)


    def _take_batch_step(self):
        self.num_batches += 1
        self.num_draws +=1

    def _take_optimizer_step(self, closure: Optional[Callable[[], Any]] = None):
        """Takes an optimizer step, caching the return"""

        # Catch issue
        if self.num_draws == 0:
            raise RuntimeError("Forgot to invoke take batch step after take optimizer step")

        # Convert gradients to mean form
        for param in self.parameters:
            if param.grad is None:
                continue
            param.grad.data /= self.num_draws

        # Take step.
        self.last_optimizer_result = self.optimizer.step(closure)
        self.optimizer.zero_grad()
        self.num_steps += 1
        self.num_draws = 0

    def _get_base_statistics(self)->Dict[str, Any]:
        """gets the basic statistics dictionary"""
        return {"batches" : self.num_batches,
                "steps" : self.num_steps,
                "num_draws" : self.num_draws,}

    def __getattr__(self, name):
        """Route everything not explicitly set  to the underlying optimizer."""
        return getattr(self.optimizer, name)
