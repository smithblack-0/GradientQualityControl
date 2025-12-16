"""
The base class largely takes care of common bookkeeping
issues related to the fact that we may take many minibatch samples
 before stepping the optimizer. This includes ensuring we use
mean gradients for optimization, logging and reporting
of statistics.
"""

from typing import Any, Callable, Dict, Optional

from torch.optim import Optimizer

from .optim_utils.optimizer_utils import optimizer_multiply_gradients, optimizer_get_grad_norm

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

        # Finish initialization
        self._initialized = True

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

        # Handle grad norm conversion
        # and operations
        optimizer_multiply_gradients(self.optimizer, 1/self.num_draws)
        last_grad_norm = optimizer_get_grad_norm(self.optimizer)

        # Take step.
        self.last_grad_norm = last_grad_norm
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

    def __getattribute__(
        self,
        name: str,
    ) -> Any:
        """
        Forwards attribute access to the wrapped optimizer while preserving normal
        inheritance for this wrapper and its subclasses.

        This allows the wrapper to satisfy isinstance(obj, Optimizer) checks while
        transparently exposing the wrapped optimizer's interface to users. Achieved
        by walking the MRO chain until reaching Optimizer, using normal lookup for
        anything found before that point, and forwarding everything else.
        """
        # Walk MRO until we hit Optimizer, check if attribute is in any class dict
        for cls in type(self).__mro__:
            if cls is Optimizer:
                break
            if name in cls.__dict__:
                return object.__getattribute__(self, name)

        # Check instance attributes
        instance_dict = object.__getattribute__(self, "__dict__")
        if name in instance_dict:
            return instance_dict[name]

        # Not found in wrapper hierarchy or instance, forward to wrapped optimizer
        optimizer = instance_dict["optimizer"]
        return getattr(optimizer, name)

    def __setattr__(
        self,
        name: str,
        value: Any,
    ) -> None:
        """
        Forwards attribute assignment to the wrapped optimizer while allowing the
        wrapper to maintain its own state during and after initialization.

        This ensures attribute assignments behave as if made directly on the wrapped
        optimizer, maintaining transparency for users. Uses an _initialized flag to
        distinguish initialization from post-init usage, forwarding new attributes
        while preserving existing wrapper state.
        """
        instance_dict = object.__getattribute__(self, "__dict__")
        if "_initialized" not in instance_dict:
            # Still in __init__, set everything locally
            object.__setattr__(self, name, value)
        else:
            # After __init__
            if name in instance_dict:
                # Existing instance attribute, update locally
                object.__setattr__(self, name, value)
            else:
                # New attribute, forward to wrapped optimizer
                optimizer = instance_dict["optimizer"]
                setattr(optimizer, name, value)

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

