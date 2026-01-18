"""
Gradient Norm Threshold Scheduler (GNTS)

Steps optimizer when mean gradient norm falls below threshold.
Implements adaptive gradient accumulation based on gradient quality.
"""

from typing import Literal, Optional, Tuple

import torch
import torch.distributed as dist
import torch_schedule_anything as tsa

from ..base import AbstractOptimizerWrapper
from ..optimizer_utils import compute_grad_norm_from_optimizer


class OptimizerWrapperGNTS(AbstractOptimizerWrapper):
    """
    Gradient Norm Threshold Scheduler (GNTS)

    Steps optimizer when mean gradient norm falls below threshold.
    Implements adaptive gradient accumulation based on gradient quality.
    Externally, it is just an optimizer wrapper that does not allow
    the usages of zero_grad.

    Algorithm:
        mean_norm = L2_gradient_norm / num_draws
        Step if: mean_norm <= gradient_norm_threshold OR num_draws >= max_draws
    """

    # Metric handling functions.
    #
    # Read reads from the optimizer, while
    # merge is needed only when sharding.
    def read_grad_norm_metric(self) -> float:
        """Reads the gradient norm off the optimizer"""
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
        max_batch_draws: int = 64,
        distributed_mode: Optional[Literal["replicated", "sharded"]] = None,
    ):
        """
        Initialize GNTS wrapper.

        Args:
            optimizer: Configured PyTorch optimizer to wrap
            max_batch_draws: Maximum batches to accumulate before forcing step
            distributed_mode: Distributed training mode ('replicated' for DDP, 'sharded' for FSDP)
        """
        # Call parent
        super().__init__(optimizer, max_draws=max_batch_draws, distributed_mode=distributed_mode)

        # Set schedulable threshold
        self._set_state("gradient_norm_threshold", 1.0, "optimizer")

        # Bind gradient norm metric for distributed support
        self._bind_metric(
            "grad_norm",
            metric_reader=self.read_grad_norm_metric,
            replicated_merger=lambda x: x,
            sharded_merger=self.merge_sharded_grad_norm,
            normal_merger=lambda x: x,
        )

    def get_mean_grad_norm(self) -> float:
        """Gets the mean grad norm."""
        # The backend stepping system accumulates sum grad
        # norms, not mean grad norms, requiring compensation
        # to get the true mean rate.

        # Fortunately, it is possible to mathematically rearrange
        # a mean to sum then divide, rather than divide then
        # sum. We do exactly this.
        sum_grad_norm = self._get_metric("grad_norm")
        return sum_grad_norm / self.num_draws

    def step(self) -> bool:
        """
        Step optimizer based on gradient norm. Treated
        largely like a normal step.

        Returns:
            True if optimizer stepped, False if accumulating
        """
        self._batch_received()

        # Decide if the optimizer should step.

        should_step = False
        threshold = self._get_state("gradient_norm_threshold", aggregate_behavior="min")
        grad_norm = self.get_mean_grad_norm()
        should_step = should_step or grad_norm <= threshold
        should_step = should_step or self.num_draws >= self.max_draws

        # Handle optimizer steps.
        if should_step:
            self._take_optimizer_step()

        return should_step


def make_gnts_with_cosine_annealing_schedule(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    num_warmup_steps: int,
    initial_threshold: float = 0.95,
    final_threshold: float = 0.25,
    warmup_multiplier: float = 10,
    max_batch_draws: int = 64,
    distributed_mode: Optional[Literal["replicated", "sharded"]] = None,
) -> Tuple[OptimizerWrapperGNTS, tsa.SynchronousSchedule]:
    """
    Factory: GNTS with constant LR and cosine annealing threshold.

    Learning rate warms up to constant, threshold undergoes inverse warmup then
    cosine anneals, weight decay warms up then cosine anneals to zero.

    Args:
        optimizer: Configured PyTorch optimizer
        num_training_steps: Total training steps
        num_warmup_steps: Steps for warmup phase
        initial_threshold: Threshold value after inverse warmup completes
        final_threshold: Ending gradient norm threshold at end of training
        warmup_multiplier: Inverse warmup starts at initial_threshold * warmup_multiplier
        max_batch_draws: Maximum accumulation before forcing step
        distributed_mode: Distributed training mode if applicable

    Returns:
        Tuple of (wrapper, schedule)
    """
    # Create wrapper
    wrapper = OptimizerWrapperGNTS(
        optimizer, max_batch_draws=max_batch_draws, distributed_mode=distributed_mode
    )

    # LR schedule: warmup to constant
    lr_schedule = tsa.constant_with_warmup(
        wrapper,
        warmup_to_value=1.0,  # Multiplier keeps initial LR
        num_warmup_steps=num_warmup_steps,
        schedule_target="lr",
    )

    # Threshold schedule: inverse warmup then cosine anneal
    threshold_schedule = tsa.cosine_annealing_with_inverse_warmup(
        wrapper,
        warmup_to_value=initial_threshold,
        anneal_to_value=final_threshold,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        warmup_multiplier=warmup_multiplier,
        schedule_target="gradient_norm_threshold",
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
    schedule = tsa.SynchronousSchedule([lr_schedule, threshold_schedule, wd_schedule])

    return wrapper, schedule


def make_gnts_with_cosine_annealing_schedule_conventional_lr(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    num_warmup_steps: int,
    initial_threshold: float = 0.95,
    final_threshold: float = 0.25,
    warmup_multiplier: float = 10,
    max_batch_draws: int = 64,
    distributed_mode: Optional[Literal["replicated", "sharded"]] = None,
) -> Tuple[OptimizerWrapperGNTS, tsa.SynchronousSchedule]:
    """
    Factory: GNTS with cosine annealing LR and threshold.

    Learning rate warms up then cosine anneals to zero (conventional), threshold
    undergoes inverse warmup then cosine anneals. Weight decay not scheduled.

    Args:
        optimizer: Configured PyTorch optimizer
        num_training_steps: Total training steps
        num_warmup_steps: Steps for warmup phase
        initial_threshold: Threshold value after inverse warmup completes
        final_threshold: Ending gradient norm threshold at end of training
        warmup_multiplier: Inverse warmup starts at initial_threshold * warmup_multiplier
        max_batch_draws: Maximum accumulation before forcing step
        distributed_mode: Distributed training mode if applicable

    Returns:
        Tuple of (wrapper, schedule)
    """
    # Create wrapper
    wrapper = OptimizerWrapperGNTS(
        optimizer, max_batch_draws=max_batch_draws, distributed_mode=distributed_mode
    )

    # LR schedule: warmup then cosine anneal to zero
    lr_schedule = tsa.cosine_annealing_with_warmup(
        wrapper,
        warmup_to_value=1.0,  # Multiplier keeps initial LR
        anneal_to_value=0.0,  # Anneal to zero
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        schedule_target="lr",
    )

    # Threshold schedule: inverse warmup then cosine anneal
    threshold_schedule = tsa.cosine_annealing_with_inverse_warmup(
        wrapper,
        warmup_to_value=initial_threshold,
        anneal_to_value=final_threshold,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        warmup_multiplier=warmup_multiplier,
        schedule_target="gradient_norm_threshold",
    )

    # Combine schedules (no weight decay scheduling)
    schedule = tsa.SynchronousSchedule([lr_schedule, threshold_schedule])

    return wrapper, schedule
