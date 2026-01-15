"""
Metric Hypothesis Test

Original test series control, able to measure the variance in a metric
and use it to decide when to step.
"""

from numbers import Number
from typing import List, Literal, Optional, Tuple

import scipy.stats as stats
import torch
import torch.distributed as dist
import torch_schedule_anything as tsa

from ..base.abstract_optimizer_wrapper import AbstractOptimizerWrapper


class OptimizerWrapperMHT(AbstractOptimizerWrapper):
    """
    Metric Hypothesis Test - Step when the variance gets low enough.

    Steps the optimizer when the tracked CI in the metric being sampled
    falls under a percent error to a given confidence level. A lower percent
    error, or higher confidence level, is more restrictive.

    Key features:
    - Schedulable logical batch size via torch_schedule_anything integration, and confidence_level
      plus percent_error_threshold schedule targets.
    - Distributed training support (replicated/sharded modes)
    - MEAN aggregation across parameter groups if relevant.

    Note: It is presumed sharded models compute the same metrics that are being monitored, usually
    loss.
    """

    @property
    def running_mean(self) -> float:
        """Gets the running mean from storage"""
        return self._get_state("running_mean")

    @property
    def update_beta(self) -> float:
        """Gets the running mean update beta"""
        return self._get_state("update_beta")

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
        update_beta: float = 0.01,
    ):
        """
        :param optimizer: The optimizer
        :param max_batch_draws: The maximum number of draws
        :param distributed_mode: The distributed mode
        :param update_beta: How rapidly to update the running average. Higher is more aggressive
        """

        super().__init__(optimizer, max_draws=max_batch_draws, distributed_mode=distributed_mode)

        # Setup history and running average

        self._set_state("running_mean", None, "vital")
        self._set_state("update_beta", update_beta, "optional")
        self._set_state("history", [], "optional")

        # Setup direct percent_error_threshold and confidence_level optimizer groups.
        # Keep in mind these will be bound to by schedules; we are not saying it is
        # these values permanently.

        self._set_state("confidence_level", 1.0, "optimizer")
        self._set_state("percent_error_threshold", 1.0, "optimizer")

        # Register metrics.
        self._bind_metric(
            "step_metrics",
            metric_reader=lambda x: x,
            replicated_merger=self.merge_independent_metrics,
            sharded_merger=self.merge_common_metrics,
            normal_merger=self.merge_common_metrics,
        )

    def update_state(self, metric: Number):
        """
        Updates the internal state of the system,
        principly the history and running average
        """
        # Standardize
        if not isinstance(metric, float):
            metric = float(metric)

        # Updating history
        history = self._get_state("history")
        all_metrics = self._get_metric("step_metrics", metric)
        history += all_metrics

        # Updating running average
        update = sum(all_metrics) / len(all_metrics)
        if self.running_mean is None:
            average = update
        else:
            average = self.running_mean * (1 - self.update_beta) + update * self.update_beta
        self._set_state("running_mean", average, "vital")

    def should_we_step(self, judgment_list: List[float]) -> bool:
        """
        Tests and determine if we should step.
        :param judgment_list: The input parameters that matter
        :return: A bool. True means step, false means no.
        """

        # Check there are at least two batches
        if len(judgment_list) < 2:
            return False

        # Check if we are at max
        if self.num_draws >= self.max_draws:
            return True

        # Fetch and aggregate schedules from param groups
        confidence_level = self._get_state("confidence_level", aggregate_behavior="mean")
        percent_error_threshold = self._get_state(
            "percent_error_threshold", aggregate_behavior="mean"
        )

        # Compute ci_low, ci_high for this case and compute mean.
        mean = sum(judgment_list) / len(judgment_list)
        ci_low, ci_high = stats.t.interval(
            confidence=confidence_level,
            df=len(judgment_list) - 1,
            loc=mean,
            scale=stats.sem(judgment_list) + 1e-12,
        )
        # maximum and minimum
        max_threshold = mean * (1 + percent_error_threshold)
        min_threshold = mean * (1 - percent_error_threshold)

        # Is the confidence interval lower than the percent minimum?
        if ci_low < min_threshold:
            return False

        # Is the confidence interval higher than the percent maximum?
        if ci_high > max_threshold:
            return False

        # We must be passing
        return True

    def step(self, metric: Number) -> bool:
        """
        Performs the update to the history and running average
        and decides when to step.
        :param metric: A relevant metric, in float form, like loss
        :return: Whether we stepped or not
        """

        # Standardize and update state
        self._batch_received()
        self.update_state(metric)

        # Setup and evaluate question
        judgement_set = [self.running_mean] + self._get_state("history")
        if self.should_we_step(judgement_set):
            self._take_optimizer_step()
            self._set_state("history", [], "optional")
            return True
        return False


def make_mht_with_warmup_schedule(
    optimizer: torch.optim.Optimizer,
    confidence_level: float,
    percent_error_threshold: float,
    num_training_steps: int,
    num_warmup_steps: int,
    max_batch_draws: int = 64,
    distributed_mode: Optional[Literal["replicated", "sharded"]] = None,
) -> Tuple[OptimizerWrapperMHT, tsa.SynchronousSchedule]:
    """
    Factory for MHT with warmup-to-constant statistical parameters.

    Creates an MHT optimizer with schedules for:
    - Learning rate: Warmup into cosine annealing to zero
    - Confidence level: Warmup to constant (allows early rapid steps)
    - Percent error threshold: Warmup to constant (allows early rapid steps)

    Args:
        optimizer: Configured optimizer (uses existing lr)
        confidence_level: Target statistical confidence (e.g., 0.95 for 95%)
        percent_error_threshold: Maximum acceptable confidence interval width
        num_training_steps: Total training steps
        num_warmup_steps: Steps for warmup phase
        max_batch_draws: Maximum accumulation (default: 64)
        distributed_mode: 'replicated' or 'sharded' for distributed training

    Returns:
        Tuple of (optimizer, synchronous_schedule)
    """
    # Create wrapper
    wrapper = OptimizerWrapperMHT(
        optimizer, max_batch_draws=max_batch_draws, distributed_mode=distributed_mode
    )

    # Learning rate: Warmup then cosine anneal to zero (using multipliers)
    lr_schedule = tsa.cosine_annealing_with_warmup(
        wrapper,
        warmup_to_value=1.0,  # Keep at optimizer's initial lr
        anneal_to_value=0.0,  # Anneal to zero
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        schedule_target="lr",
    )

    # Confidence level: Warmup to constant
    confidence_schedule = tsa.cosine_annealing_with_warmup(
        wrapper,
        warmup_to_value=confidence_level,
        anneal_to_value=confidence_level,  # No annealing, stays constant
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        schedule_target="confidence_level",
    )

    # Percent error threshold: Warmup to constant
    error_schedule = tsa.cosine_annealing_with_warmup(
        wrapper,
        warmup_to_value=percent_error_threshold,
        anneal_to_value=percent_error_threshold,  # No annealing, stays constant
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        schedule_target="percent_error_threshold",
    )

    sync = tsa.SynchronousSchedule([lr_schedule, confidence_schedule, error_schedule])

    return wrapper, sync
