"""
Scheduled Batch Controller (SBC) - Control optimizer for logical batch size scheduling.

Steps optimizer when accumulated batch size meets or exceeds a scheduled logical batch size.
Enables dynamic batch size scheduling without code changes.
"""

from typing import Literal, Optional, Tuple

import torch
import torch.distributed as dist
import torch_schedule_anything as tsa

from ..base.abstract_optimizer_wrapper import AbstractOptimizerWrapper


class OptimizerWrapperSBC(AbstractOptimizerWrapper):
    """
    Scheduled Batch Controller - dynamically controls effective batch size through gradient
    accumulation.

    Steps the optimizer when the accumulated batch size (num_draws * physical_batch_size)
    meets or exceeds a scheduled logical_batch_size target. This enables dynamic batch size
    scheduling without modifying training loop code.

    The logical_batch_size parameter is exposed to ScheduleAnything, allowing it to follow
    arbitrary schedules throughout training.

    Key features:
        - Schedulable logical batch size via torch_schedule_anything integration
        - Distributed training support (replicated/sharded modes)
        - Automatic gradient accumulation based on batch size ratio
        - MAX aggregation across parameter groups (conservative stepping)

    This is a concrete implementation of AbstractOptimizerWrapper following the post-0.9.0
    architecture. Use the factory functions make_sbc_with_polynomial_schedule() or
    make_sbc_with_polynomial_schedule_conventional_lr() for typical use cases.

    Example:
        >>> base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> sbc_optimizer, schedule = make_sbc_with_polynomial_schedule(
        ...     base_optimizer,
        ...     physical_batch_size=32,
        ...     initial_batch_size=128,
        ...     final_batch_size=2048,
        ...     num_training_steps=10000,
        ...     num_warmup_steps=1000
        ... )
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     loss.backward()
        ...     sbc_optimizer.step()
        ...     schedule.step()
    """

    @staticmethod
    def merge_replicated_batch_sizes(batch_size: int) -> int:
        """Sum batch sizes across replicated devices using all_reduce."""
        batch_size_tensor = torch.tensor([float(batch_size)])
        dist.all_reduce(batch_size_tensor, op=dist.ReduceOp.SUM)
        return int(batch_size_tensor.item())

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        physical_batch_size: int,
        max_batch_draws: int = 64,
        distributed_mode: Optional[Literal["replicated", "sharded"]] = None,
    ):
        """
        Initialize Scheduled Batch Controller optimizer.

        Args:
            optimizer: Base PyTorch optimizer
            physical_batch_size: Size of each microbatch processed per step() call
            max_batch_draws: Maximum gradient accumulation steps before forcing optimizer step
            distributed_mode: 'replicated' for DDP (sum batch sizes), 'sharded' for FSDP
                            (passthrough), None for single-device training

        Raises:
            ValueError: If physical_batch_size <= 0
        """

        # Basic validation followed by setup
        if physical_batch_size <= 0:
            raise ValueError("physical_batch_size must be positive")
        if max_batch_draws <= 0:
            raise ValueError("max_batch_draws must be positive")
        if distributed_mode not in ("replicated", "sharded") and distributed_mode is not None:
            raise ValueError("distributed_mode can only be 'replicated' or 'sharded', or None")
        super().__init__(optimizer, max_draws=max_batch_draws, distributed_mode=distributed_mode)

        # Storage of ideal batch size, and binding of metric management
        # Metric system will combine detected physical batch size across all devices
        # as relevant. Replicated cases add up all the batches across all devices,
        # while other cases are just one big batch.
        self._set_state("physical_batch_size", physical_batch_size, "optional")
        self._bind_metric(
            "effective_batch_size",
            metric_reader=lambda: self._get_state("physical_batch_size"),
            replicated_merger=self.merge_replicated_batch_sizes,
            sharded_merger=lambda x: x,
            normal_merger=lambda x: x,
        )

        # Setup the scheduling target.

        self._set_state("logical_batch_size", 1.0, "optimizer")

    def step(self) -> bool:
        """
        Step optimizer when accumulated batch size meets or exceeds logical_batch_size.

        Accumulates gradients until the effective accumulated batch size
        (num_draws * effective_physical_batch_size) meets or exceeds the current
        logical_batch_size target, or until max_batch_draws is reached.

        Returns:
            True if optimizer stepped, False if still accumulating gradients
        """
        self._batch_received()
        logical_batch_size_target = self._get_state("logical_batch_size", aggregate_behavior="max")
        effective_batch_size = self._get_metric("effective_batch_size")
        if (
            self.num_draws * effective_batch_size >= logical_batch_size_target
            or self.max_draws <= self.num_draws
        ):
            self._take_optimizer_step()
            return True
        return False


def make_sbc_with_polynomial_schedule(
    optimizer: torch.optim.Optimizer,
    physical_batch_size: int,
    initial_batch_size: int,
    final_batch_size: int,
    num_training_steps: int,
    num_warmup_steps: int,
    polynomial_power: float = 2.0,
    max_batch_draws: int = 64,
    distributed_mode: Optional[Literal["replicated", "sharded"]] = None,
) -> Tuple[OptimizerWrapperSBC, tsa.SynchronousSchedule]:
    """
    Factory for SBC with polynomial batch schedule and constant LR.

    Creates an SBC optimizer with schedules for:
    - Learning rate: Constant (maintains optimizer's initial lr)
    - Logical batch size: Polynomial growth from initial to final
    - Weight decay: Cosine annealing to zero

    Args:
        optimizer: Configured optimizer (uses existing lr and weight_decay)
        physical_batch_size: Size of each microbatch
        initial_batch_size: Starting logical batch size (after warmup)
        final_batch_size: Ending logical batch size
        num_training_steps: Total training steps
        num_warmup_steps: Steps for warmup phase
        polynomial_power: Exponent for polynomial curve (default: 2.0 for quadratic)
        max_batch_draws: Maximum accumulation before forcing step (default: 64)
        distributed_mode: 'replicated' or 'sharded' for distributed training

    Returns:
        Tuple of (optimizer, synchronous_schedule)
    """

    # Make the new optimizer itself.
    optimizer = OptimizerWrapperSBC(
        optimizer, physical_batch_size, max_batch_draws, distributed_mode
    )

    # Bind schedule, make composite schedule. Keep in mind we are setting relative multipliers.
    lr_schedule = tsa.constant_with_warmup(optimizer, 1.0, num_warmup_steps, schedule_target="lr")
    wd_schedule = tsa.cosine_annealing_with_warmup(
        optimizer, 1.0, 0.0, num_warmup_steps, num_training_steps, schedule_target="weight_decay"
    )
    batch_schedule = tsa.polynomial_schedule_with_warmup(
        optimizer,
        initial_batch_size,
        final_batch_size,
        num_warmup_steps,
        num_training_steps,
        polynomial_exponent=polynomial_power,
        schedule_target="logical_batch_size",
    )
    schedule = tsa.SynchronousSchedule([lr_schedule, wd_schedule, batch_schedule])

    # Return the resulting scheduler and optimizer.

    return optimizer, schedule


def make_sbc_with_polynomial_schedule_conventional_lr(
    optimizer: torch.optim.Optimizer,
    physical_batch_size: int,
    initial_batch_size: int,
    final_batch_size: int,
    num_training_steps: int,
    num_warmup_steps: int,
    polynomial_power: float = 2.0,
    max_batch_draws: int = 64,
    distributed_mode: Optional[Literal["replicated", "sharded"]] = None,
) -> Tuple[OptimizerWrapperSBC, tsa.SynchronousSchedule]:
    """
    Factory for SBC with polynomial batch schedule and annealing LR.

    Creates an SBC optimizer with schedules for:
    - Learning rate: Cosine annealing to zero
    - Logical batch size: Polynomial growth from initial to final
    - Weight decay: Constant (maintains optimizer's initial weight_decay)

    Args:
        optimizer: Configured optimizer (uses existing lr and weight_decay)
        physical_batch_size: Size of each microbatch
        initial_batch_size: Starting logical batch size (after warmup)
        final_batch_size: Ending logical batch size
        num_training_steps: Total training steps
        num_warmup_steps: Steps for warmup phase
        polynomial_power: Exponent for polynomial curve (default: 2.0 for quadratic)
        max_batch_draws: Maximum accumulation before forcing step (default: 64)
        distributed_mode: 'replicated' or 'sharded' for distributed training

    Returns:
        Tuple of (optimizer, synchronous_schedule)
    """
    # Make the new optimizer itself.
    optimizer = OptimizerWrapperSBC(
        optimizer, physical_batch_size, max_batch_draws, distributed_mode
    )

    # Bind schedule, make composite schedule. Keep in mind we are setting relative multipliers.
    lr_schedule = tsa.cosine_annealing_with_warmup(
        optimizer, 1.0, 0.0, num_warmup_steps, num_training_steps, schedule_target="lr"
    )
    wd_schedule = tsa.constant_schedule(optimizer, 1.0, "weight_decay")
    batch_schedule = tsa.polynomial_schedule_with_warmup(
        optimizer,
        initial_batch_size,
        final_batch_size,
        num_warmup_steps,
        num_training_steps,
        polynomial_exponent=polynomial_power,
        schedule_target="logical_batch_size",
    )
    schedule = tsa.SynchronousSchedule([lr_schedule, wd_schedule, batch_schedule])

    # Return the resulting scheduler and optimizer.

    return optimizer, schedule
