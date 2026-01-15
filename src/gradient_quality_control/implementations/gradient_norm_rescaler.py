"""
Gradient Norm Rescaler (GNR)

Maintains consistent gradient magnitudes by rescaling to a target norm before each step.
Enables isogradient training - uniform gradient scale throughout training.
"""

from typing import Literal, Optional, Tuple

import torch
import torch.distributed as dist
import torch_schedule_anything as tsa

from ..base import AbstractOptimizerWrapper
from ..optimizer_utils import compute_grad_norm_from_optimizer, multiply_optimizer_gradients


class OptimizerWrapperGNR(AbstractOptimizerWrapper):
    """
    Gradient Norm Rescaler (GNR)

    Controls gradient magnitude by rescaling to a target norm before each optimizer step.
    Provides isogradient training - consistent gradient magnitudes throughout training.

    Schedulable Parameters:
        - target_gradient_norm: Target L2 norm for gradient rescaling
        - lr: Learning rate (from optimizer)
        - weight_decay: Weight decay (from optimizer)

    Behavior:
        - Steps every batch (no gradient accumulation)
        - Gradients rescaled to target_gradient_norm before each step
        - Compatible with distributed training (replicated/sharded modes)
    """

    def read_grad_norm_metric(self) -> float:
        """Compute L2 gradient norm across all parameters."""
        return compute_grad_norm_from_optimizer(self.optimizer)

    @staticmethod
    def merge_sharded_grad_norm(grad_norm: float) -> float:
        """Averages across sharded models"""
        # RMS averaging used, as it is the appropriate reduction
        # strategy given the built-in torch averaging. Note this
        # formula breaks immediately if sharding is putting
        # uneven number of entries per device.

        # Formula: sqrt(sum(norm²) / world_size)

        # Setup
        world_size = dist.get_world_size()
        grad_norm_tensor = torch.tensor([grad_norm], dtype=torch.float32)

        # Enter squared space, merge, then exit.
        grad_norm_tensor = grad_norm_tensor**2
        dist.all_reduce(grad_norm_tensor, op=dist.ReduceOp.SUM)
        grad_norm_tensor = grad_norm_tensor / world_size
        grad_norm_tensor = torch.sqrt(grad_norm_tensor)

        # Get item, and back to normal python.
        return grad_norm_tensor.item()

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        distributed_mode: Optional[Literal["replicated", "sharded"]] = None,
    ):
        """
        Initialize GNR wrapper.

        Args:
            optimizer: Configured PyTorch optimizer to wrap
            distributed_mode: Distributed training mode ('replicated' for DDP, 'sharded' for FSDP)
        """
        # Call parent with max_draws=1 since GNR always steps
        super().__init__(optimizer, max_draws=1, distributed_mode=distributed_mode)

        # Set schedulable target gradient norm
        self._set_state("target_gradient_norm", 1.0, "optimizer")

        # Bind gradient norm metric for distributed support
        self._bind_metric(
            "grad_norm",
            metric_reader=self.read_grad_norm_metric,
            replicated_merger=lambda x: x,  # Replicated: passthrough
            sharded_merger=self.merge_sharded_grad_norm,  # Sharded: RMS aggregation
            normal_merger=lambda x: x,
        )

    def step(self) -> bool:
        """
        Rescale gradients to target norm and step optimizer.

        Returns:
            bool: True (always steps - GNR never accumulates)
        """
        self._batch_received()

        # Compute update multiplier
        norm = self._get_metric("grad_norm")
        threshold = self._get_state("target_gradient_norm", aggregate_behavior="mean")
        if norm > 0:
            rescale_multiplier = threshold / norm
        else:
            # Gradients of length 0 just do not step
            rescale_multiplier = 0.0

        # Apply, step
        multiply_optimizer_gradients(self.optimizer, rescale_multiplier)
        self._take_optimizer_step()
        return True


def make_gnr_with_cosine_annealing_schedule(
    optimizer: torch.optim.Optimizer,
    initial_norm: float,
    final_norm: float,
    num_training_steps: int,
    num_warmup_steps: int,
    distributed_mode: Optional[Literal["replicated", "sharded"]] = None,
) -> Tuple[OptimizerWrapperGNR, tsa.SynchronousSchedule]:
    """
    Create GNR wrapper with constant learning rate and annealing target norm.

    Schedule Configuration:
        - Learning rate: Warmup to constant
        - Target gradient norm: Cosine anneal from initial_norm to final_norm
        - Weight decay: Warmup then cosine anneal to zero

    Args:
        optimizer: Configured PyTorch optimizer
        initial_norm: Starting target gradient norm
        final_norm: Ending target gradient norm
        num_training_steps: Total training steps
        num_warmup_steps: Steps for warmup phase
        distributed_mode: Distributed training mode if applicable

    Returns:
        Tuple[OptimizerWrapperGNR, SynchronousSchedule]: Configured wrapper and schedule
    """
    # Create wrapper
    wrapper = OptimizerWrapperGNR(optimizer, distributed_mode=distributed_mode)

    # LR schedule: warmup to constant
    lr_schedule = tsa.constant_with_warmup(
        wrapper,
        warmup_to_value=1.0,  # Multiplier keeps initial LR
        num_warmup_steps=num_warmup_steps,
        schedule_target="lr",
    )

    # Target norm schedule: cosine anneal from initial to final
    norm_schedule = tsa.cosine_annealing_with_warmup(
        wrapper,
        warmup_to_value=initial_norm,
        anneal_to_value=final_norm,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        schedule_target="target_gradient_norm",
    )

    # Weight decay schedule: warmup then cosine anneal to zero
    wd_schedule = tsa.cosine_annealing_with_warmup(
        wrapper,
        warmup_to_value=1.0,  # Multiplier keeps initial weight_decay
        anneal_to_value=0.0,  # Anneal to zero
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        schedule_target="weight_decay",
    )

    # Combine schedules
    schedule = tsa.SynchronousSchedule([lr_schedule, norm_schedule, wd_schedule])

    return wrapper, schedule


def make_gnr_with_cosine_annealing_schedule_conventional_lr(
    optimizer: torch.optim.Optimizer,
    initial_norm: float,
    final_norm: float,
    num_training_steps: int,
    num_warmup_steps: int,
    distributed_mode: Optional[Literal["replicated", "sharded"]] = None,
) -> Tuple[OptimizerWrapperGNR, tsa.SynchronousSchedule]:
    """
    Create GNR wrapper with conventional learning rate annealing and annealing target norm.

    Schedule Configuration:
        - Learning rate: Warmup then cosine anneal to zero
        - Target gradient norm: Cosine anneal from initial_norm to final_norm
        - Weight decay: Not scheduled (constant)

    Args:
        optimizer: Configured PyTorch optimizer
        initial_norm: Starting target gradient norm
        final_norm: Ending target gradient norm
        num_training_steps: Total training steps
        num_warmup_steps: Steps for warmup phase
        distributed_mode: Distributed training mode if applicable

    Returns:
        Tuple[OptimizerWrapperGNR, SynchronousSchedule]: Configured wrapper and schedule
    """
    # Create wrapper
    wrapper = OptimizerWrapperGNR(optimizer, distributed_mode=distributed_mode)

    # LR schedule: warmup then cosine anneal to zero
    lr_schedule = tsa.cosine_annealing_with_warmup(
        wrapper,
        warmup_to_value=1.0,  # Multiplier keeps initial LR
        anneal_to_value=0.0,  # Anneal to zero
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        schedule_target="lr",
    )

    # Target norm schedule: cosine anneal from initial to final
    norm_schedule = tsa.cosine_annealing_with_warmup(
        wrapper,
        warmup_to_value=initial_norm,
        anneal_to_value=final_norm,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        schedule_target="target_gradient_norm",
    )

    # Combine schedules (no weight decay scheduling)
    schedule = tsa.SynchronousSchedule([lr_schedule, norm_schedule])

    return wrapper, schedule
