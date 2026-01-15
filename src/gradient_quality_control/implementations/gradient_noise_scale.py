"""
Gradient Noise Scale (GNS)

Adaptive batch sizing based on gradient noise-to-signal ratio.
Steps when gradient variance indicates sufficient signal quality.
"""

from typing import List, Literal, Optional, Tuple

import torch
import torch.distributed as dist
import torch_schedule_anything as tsa

from ..base import AbstractOptimizerWrapper
from ..optimizer_utils import get_last_grad_norm_from_optimizer, setup_norm_logging_in_optimizer


class OptimizerWrapperGNS(AbstractOptimizerWrapper):
    """
    Gradient Noise Scale (GNS)

    Controls optimizer stepping based on gradient noise-to-signal ratio.
    Accumulates gradients until noise scale indicates sufficient sample quality.

    Schedulable Parameters:
        - noise_tolerance: Maximum acceptable noise-to-signal ratio
        - lr: Learning rate (from optimizer)
        - weight_decay: Weight decay (from optimizer)

    Behavior:
        - Accumulates gradients across batches
        - Steps when GNS <= num_draws * noise_tolerance
        - Forces step at max_batch_draws
        - Compatible with distributed training (replicated/sharded modes)

    Based on McCandlish et al.'s gradient noise scale theory.
    """

    def read_grad_norm_metric(self) -> float:
        """Compute L2 gradient norm across all parameters."""
        # Note we used a specialized system that caches the
        # grad norm value as it goes by, so that we can correctly
        # measure the per batch grad norm rather than the overall
        # one.
        return get_last_grad_norm_from_optimizer(self.optimizer)

    @staticmethod
    def merge_common_metrics(metric: float) -> List[float]:
        """Adds a list to anything going through"""
        return [metric]

    @staticmethod
    def merge_independent_metrics(metric: float) -> List[float]:
        """Merges metrics that exist on separate devices.
        Returns the same list of metrics on all ranks.
        """
        world_size = dist.get_world_size()

        local = torch.tensor([metric], dtype=torch.float32)
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)

        return [t.item() for t in gathered]

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_batch_draws: int = 64,
        distributed_mode: Optional[Literal["replicated", "sharded"]] = None,
    ):
        """
        Initialize GNS wrapper.

        Args:
            optimizer: Configured PyTorch optimizer to wrap
            max_batch_draws: Maximum batches to accumulate before forcing step
            distributed_mode: Distributed training mode ('replicated' for DDP, 'sharded' for FSDP)
        """
        # Call parent
        super().__init__(optimizer, max_draws=max_batch_draws, distributed_mode=distributed_mode)

        # Set schedulable noise tolerance and history
        self._set_state("noise_tolerance", 1.0, "optimizer")
        self._set_state("history", [], "optional")

        # Setup last grad norm logging
        setup_norm_logging_in_optimizer(self.optimizer)

        # Bind gradient norm metric for distributed support
        self._bind_metric(
            "grad_norms",
            metric_reader=self.read_grad_norm_metric,
            replicated_merger=self.merge_independent_metrics,  # Gather all metrics
            sharded_merger=self.merge_common_metrics,  # RMS aggregation
            normal_merger=self.merge_common_metrics,
        )

    def update_state(self):
        """Updates the history"""
        history = self._get_state("history")
        metrics = self._get_metric("grad_norms")
        history += metrics

    def clear_history(self):
        """clears the history"""
        self._set_state("history", [], "optional")

    def compute_approximate_gns(self):
        """Computes the approximate GNS as in Mccandish"""
        history = self._get_state("history")
        history_tensor = torch.tensor(history, dtype=torch.float32)

        # Compute core statistics
        mean_squared = torch.mean(history_tensor**2)
        variance = torch.var(history_tensor)

        # Compute GNS
        return variance / (mean_squared + 1e-8)

    def step(self) -> bool:
        """
        Step optimizer when gradient noise scale meets tolerance threshold.

        Returns:
            bool: True if optimizer stepped, False if accumulating
        """
        self._batch_received()
        self.update_state()

        tolerance = self._get_state("noise_tolerance", aggregate_behavior="min")
        gns = self.compute_approximate_gns()
        if gns <= self.num_draws * tolerance or self.num_draws >= self.max_draws:
            self._take_optimizer_step()
            self.clear_history()
            return True
        return False


def make_gns_with_cosine_annealing_schedule(
    optimizer: torch.optim.Optimizer,
    initial_tolerance: float,
    final_tolerance: float,
    num_training_steps: int,
    num_warmup_steps: int,
    warmup_multiplier: int = 10,
    max_batch_draws: int = 64,
    distributed_mode: Optional[Literal["replicated", "sharded"]] = None,
) -> Tuple[OptimizerWrapperGNS, tsa.SynchronousSchedule]:
    """
    Create GNS wrapper with annealing learning rate and tolerance.

    Schedule Configuration:
        - Learning rate: Warmup then cosine anneal to zero
        - Noise tolerance: Inverse warmup then cosine anneal from initial_tolerance to
          final_tolerance

    Args:
        optimizer: Configured PyTorch optimizer
        initial_tolerance: Starting noise-to-signal tolerance
        final_tolerance: Ending noise-to-signal tolerance
        num_training_steps: Total training steps
        num_warmup_steps: Steps for warmup phase
        warmup_multiplier: Inverse warmup starts at initial_tolerance * warmup_multiplier
        max_batch_draws: Maximum accumulation before forcing step
        distributed_mode: Distributed training mode if applicable

    Returns:
        Tuple[OptimizerWrapperGNS, SynchronousSchedule]: Configured wrapper and schedule
    """
    # Create wrapper
    wrapper = OptimizerWrapperGNS(
        optimizer, max_batch_draws=max_batch_draws, distributed_mode=distributed_mode
    )

    # LR schedule: warmup then cosine anneal
    lr_schedule = tsa.cosine_annealing_with_warmup(
        wrapper,
        warmup_to_value=1.0,  # Multiplier keeps initial LR
        anneal_to_value=0.0,  # Anneal to zero
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        schedule_target="lr",
    )

    # Noise tolerance schedule: inverse warmup then cosine anneal
    tolerance_schedule = tsa.cosine_annealing_with_inverse_warmup(
        wrapper,
        warmup_to_value=initial_tolerance,
        anneal_to_value=final_tolerance,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        warmup_multiplier=warmup_multiplier,
        schedule_target="noise_tolerance",
    )

    # Combine schedules
    schedule = tsa.SynchronousSchedule([lr_schedule, tolerance_schedule])

    return wrapper, schedule


def make_gns_default(
    optimizer: torch.optim.Optimizer,
    tolerance: float,
    num_training_steps: int,
    num_warmup_steps: int,
    max_batch_draws: int = 64,
    distributed_mode: Optional[Literal["replicated", "sharded"]] = None,
) -> Tuple[OptimizerWrapperGNS, tsa.SynchronousSchedule]:
    """
    Create GNS wrapper with default schedule configuration.

    Schedule Configuration:
        - Learning rate: Warmup then cosine anneal to zero
        - Noise tolerance: Inverse warmup to tolerance, then constant

    Args:
        optimizer: Configured PyTorch optimizer
        tolerance: Noise-to-signal tolerance threshold (constant after inverse warmup)
        num_training_steps: Total training steps
        num_warmup_steps: Steps for inverse warmup phase
        max_batch_draws: Maximum accumulation before forcing step
        distributed_mode: Distributed training mode if applicable

    Returns:
        Tuple[OptimizerWrapperGNS, SynchronousSchedule]: Configured wrapper and schedule
    """
    # Create wrapper
    wrapper = OptimizerWrapperGNS(
        optimizer, max_batch_draws=max_batch_draws, distributed_mode=distributed_mode
    )

    # LR schedule: warmup then cosine anneal
    lr_schedule = tsa.cosine_annealing_with_warmup(
        wrapper,
        warmup_to_value=1.0,  # Multiplier keeps initial LR
        anneal_to_value=0.0,  # Anneal to zero
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        schedule_target="lr",
    )

    # Noise tolerance schedule: inverse warmup to constant
    tolerance_schedule = tsa.constant_with_inverse_warmup(
        wrapper,
        warmup_to_value=tolerance,
        num_warmup_steps=num_warmup_steps,
        schedule_target="noise_tolerance",
    )

    # Combine schedules
    schedule = tsa.SynchronousSchedule([lr_schedule, tolerance_schedule])

    return wrapper, schedule
