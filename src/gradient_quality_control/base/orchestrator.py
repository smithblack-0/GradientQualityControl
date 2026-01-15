"""
OrchestratorMainSystem implementation.

Main facade coordinating all subsystems and exposing unified public and protected API.
"""

from typing import Any, Dict, List, Literal, Optional

import torch

from .distributed_metrics import DistributedMetricsManagementSubsystem
from .gradient_accumulation import GradientAccumulationStepSubsystem
from .optimizer_mocking import OptimizerMockingMixin
from .reporting import ReportingSubsystem
from .state_management import StateManagementSubsystem


class OrchestratorMainSystem(OptimizerMockingMixin):
    """
    Main orchestrator coordinating all subsystems.

    Provides unified public API for optimizer wrapper functionality.
    Uses OptimizerMockingMixin for optimizer transparency.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        state_manager: StateManagementSubsystem,
        distributed_metrics: DistributedMetricsManagementSubsystem,
        accumulation: GradientAccumulationStepSubsystem,
        reporting: ReportingSubsystem,
    ):
        """
        Initialize orchestrator with all subsystem dependencies.

        Args:
            optimizer: Wrapped PyTorch optimizer
            state_manager: StateManagementSubsystem instance
            distributed_metrics: DistributedMetricsManagementSubsystem instance
            accumulation: GradientAccumulationStepSubsystem instance
            reporting: ReportingSubsystem instance

        Post-conditions:
            All subsystems stored and initialization finalized
        """
        self._optimizer = optimizer
        self._state_manager = state_manager
        self._distributed_metrics = distributed_metrics
        self._accumulation = accumulation
        self._reporting = reporting

        self._finalize_initialization()

    # =========================================================================
    # Public Properties
    # =========================================================================

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Direct access to wrapped optimizer."""
        return self._optimizer

    @property
    def num_batches(self) -> int:
        """Total batches processed since wrapper creation."""
        return self._accumulation.num_batches

    @property
    def num_steps(self) -> int:
        """Total optimizer steps taken."""
        return self._accumulation.num_steps

    @property
    def num_draws(self) -> int:
        """Batches accumulated since last step."""
        return self._accumulation.num_draws

    @property
    def max_draws(self) -> int:
        """Maximum allowed draws."""
        return self._accumulation.max_draws

    @property
    def last_num_draws(self) -> Optional[int]:
        """Number of batches in most recent optimizer step. None before first step."""
        return self._accumulation.last_num_draws

    @property
    def last_grad_norm(self) -> Optional[float]:
        """L2 gradient norm from most recent optimizer step. None before first step."""
        return self._accumulation.last_grad_norm

    @property
    def valid_schedule_targets(self) -> List[str]:
        """
        List of all schedulable parameter names.

        Includes optimizer native parameters and wrapper-extended parameters.
        """
        state_list = self._state_manager.show_state()
        return [name for name, flag in state_list if flag == "optimizer"]

    @property
    def distributed_mode(self) -> Optional[Literal["replicated", "sharded"]]:
        """Configured distributed execution mode for metric aggregation."""
        return self._distributed_metrics.distributed_mode

    @property
    def device(self) -> torch.device:
        """Device the optimizer's parameters are on. Returns device of first parameter."""
        return self._optimizer.param_groups[0]["params"][0].device

    # =========================================================================
    # Public Methods
    # =========================================================================

    def step(
        self,
        *args,
        **kwargs,
    ) -> bool:
        """
        Abstract method for subclasses to implement control algorithm.

        Subclasses must implement this to make step/accumulate decision.

        Args:
            *args: Arbitrary positional arguments for subclass use
            **kwargs: Arbitrary keyword arguments for subclass use

        Returns:
            bool: True if optimizer step was taken, False if accumulated

        Raises:
            NotImplementedError: This is an abstract method
        """
        raise NotImplementedError(
            "step() must be implemented by subclass. "
            "See base_object_api.md for subclassing contract."
        )

    def zero_grad(self) -> None:
        """
        Intentionally disabled. Wrapper manages gradient zeroing internally.

        Raises:
            NotImplementedError: Always raises - wrapper manages gradients
        """
        raise NotImplementedError(
            "zero_grad() is disabled. Wrapper manages gradient zeroing internally."
        )

    def statistics(
        self,
        behavior: Literal["vital", "verbose"] = "verbose",
        aggregate_behavior: Literal["mean", "max", "min"] = "mean",
    ) -> Dict[str, Any]:
        """
        Returns complete or filtered statistics dictionary for logging and debugging.

        Args:
            behavior: "verbose" includes all state, "vital" includes only vital state
            aggregate_behavior: How to aggregate multi-group parameters when values differ

        Returns:
            Dictionary of statistics
        """
        return self._reporting.statistics(behavior, aggregate_behavior)

    def vital_statistics(
        self,
        aggregate_behavior: Literal["mean", "max", "min"] = "mean",
    ) -> Dict[str, Any]:
        """
        Returns curated vital statistics for real-time monitoring.

        Args:
            aggregate_behavior: How to aggregate multi-group parameters when values differ

        Returns:
            Dictionary of vital statistics
        """
        return self._reporting.vital_statistics(aggregate_behavior)

    def state_dict(self) -> Dict[str, Any]:
        """
        Serializes complete wrapper state for checkpointing.

        Returns:
            Dictionary containing wrapper and optimizer state
        """
        return self._state_manager.state_dict()

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
    ) -> None:
        """
        Restores wrapper state from checkpoint.

        Args:
            state_dict: State dictionary from previous state_dict() call
        """
        self._state_manager.load_state_dict(state_dict)

    # =========================================================================
    # Protected Methods (for subclass use)
    # =========================================================================

    def _set_state(
        self,
        name: str,
        value: Any,
        flag: Literal["vital", "optional", "optimizer"],
    ) -> None:
        """
        Store wrapper state and expose parameters to ScheduleAnything.

        For subclass implementation use only.

        Args:
            name: State variable name
            value: Value to store
            flag: Storage destination - "vital", "optional", or "optimizer"
        """
        self._state_manager.set_state(name, value, flag)

    def _get_state(
        self,
        name: str,
        aggregate_behavior: Optional[Literal["mean", "max", "min"]] = None,
    ) -> Any:
        """
        Retrieve state from wrapper or optimizer with optional aggregation.

        For subclass implementation use only.

        Args:
            name: State variable name
            aggregate_behavior: Optional aggregation strategy for list values

        Returns:
            State value, optionally aggregated
        """
        value = self._state_manager.get_state(name)

        if aggregate_behavior is None:
            return value

        # If aggregation requested and value is a list, aggregate it
        if isinstance(value, list):
            return self._reporting.aggregate_numeric_list(value, aggregate_behavior)

        # Otherwise return as-is
        return value

    def _bind_metric(
        self,
        name: str,
        metric_reader: Any,
        replicated_merger: Any,
        sharded_merger: Any,
        normal_merger: Any = lambda x: x,
    ) -> None:
        """
        Register metric and its distributed resolution rules.

        For subclass implementation use only.

        Args:
            name: Metric name
            metric_reader: Callable to read metric value
            replicated_merger: Merger for replicated distributed mode
            sharded_merger: Merger for sharded distributed mode
            normal_merger: Merger for single-device mode (default: passthrough)
        """
        self._distributed_metrics.bind_metric(
            name, metric_reader, replicated_merger, sharded_merger, normal_merger
        )

    def _get_metric(self, name: str, *args, **kwargs) -> Any:
        """
        Resolve metric value with distributed execution handling.

        For subclass implementation use only.

        Args:
            name: Metric name
            *args: Positional arguments forwarded to metric reader
            **kwargs: Keyword arguments forwarded to metric reader

        Returns:
            Metric value, potentially merged based on distributed mode
        """
        return self._distributed_metrics.get_metric(name, *args, **kwargs)

    def _batch_received(self) -> None:
        """
        Update counters when batch is processed.

        For subclass implementation use only. Call at start of step() implementation.
        """
        self._accumulation.batch_received()

    def _take_optimizer_step(self) -> None:
        """
        Average gradients and step optimizer.

        For subclass implementation use only. Call when control algorithm decides to step.
        """
        self._accumulation.take_optimizer_step()
