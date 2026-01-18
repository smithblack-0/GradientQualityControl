"""
DistributedMetricsManagementSubsystem implementation.

Manages metric binding and resolution under distributed execution modes.
"""

from typing import Any, Callable, Dict, Literal, Optional


class DistributedMetricsManagementSubsystem:
    """
    Manages metric resolution for distributed training.

    Binds metrics with mode-specific mergers and resolves them based on
    configured distributed execution mode (None, replicated, or sharded).
    """

    def __init__(
        self,
        distributed_state: Optional[Literal["replicated", "sharded"]],
    ):
        """
        Initialize distributed metrics subsystem.

        Args:
            distributed_state: Distributed execution mode - None, "replicated", or "sharded"

        Raises:
            ValueError: If distributed_state is not a valid mode
        """
        # Validate distributed_state
        if distributed_state not in (None, "replicated", "sharded"):
            raise ValueError(
                f"distributed_state must be None, 'replicated', or 'sharded', "
                f"got {distributed_state!r}"
            )

        self._distributed_mode = distributed_state
        self._metrics: Dict[str, Dict[str, Callable]] = {}

    @property
    def distributed_mode(self) -> Optional[Literal["replicated", "sharded"]]:
        """Get the configured distributed execution mode (immutable)."""
        return self._distributed_mode

    def bind_metric(
        self,
        name: str,
        metric_reader: Callable[..., Any],
        replicated_merger: Callable[[Any], Any],
        sharded_merger: Callable[[Any], Any],
        normal_merger: Callable[[Any], Any],
    ) -> None:
        """
        Register a metric with its resolution rules.

        Args:
            name: Metric name (must be unique)
            metric_reader: Callable that reads the metric value
            replicated_merger: Callable to merge metric in replicated mode
            sharded_merger: Callable to merge metric in sharded mode
            normal_merger: Callable to merge metric in normal mode

        Raises:
            RuntimeError: If name already registered
            TypeError: If any parameter is not callable
        """
        # Check if already registered
        if name in self._metrics:
            raise RuntimeError(f"Metric '{name}' is already registered")

        # Validate all callables
        if not callable(metric_reader):
            raise TypeError(f"metric_reader must be callable, got {type(metric_reader)}")
        if not callable(replicated_merger):
            raise TypeError(f"replicated_merger must be callable, got {type(replicated_merger)}")
        if not callable(sharded_merger):
            raise TypeError(f"sharded_merger must be callable, got {type(sharded_merger)}")
        if not callable(normal_merger):
            raise TypeError(f"normal_merger must be callable, got {type(normal_merger)}")

        # Register metric
        self._metrics[name] = {
            "reader": metric_reader,
            "replicated_merger": replicated_merger,
            "sharded_merger": sharded_merger,
            "normal_merger": normal_merger,
        }

    def get_metric(
        self,
        name: str,
        *args,
        **kwargs,
    ) -> Any:
        """
        Resolve a metric value based on distributed mode.

        Args:
            name: Metric name
            *args: Arguments to forward to metric_reader
            **kwargs: Keyword arguments to forward to metric_reader

        Returns:
            Merged metric value

        Raises:
            KeyError: If metric lookup fails
            RuntimeError: If metric read or merging fails
        """
        # Validate metric is registered
        if name not in self._metrics:
            raise KeyError(f"Metric lookup failed: metric '{name}' not registered")

        metric_spec = self._metrics[name]

        # Read metric value
        try:
            value = metric_spec["reader"](*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Metric read failed for '{name}': {e}") from e

        # Select merger based on distributed mode
        if self._distributed_mode is None:
            merger = metric_spec["normal_merger"]
            pathway = "single device pathway"
        elif self._distributed_mode == "replicated":
            merger = metric_spec["replicated_merger"]
            pathway = "replicated distributed pathway"
        elif self._distributed_mode == "sharded":
            merger = metric_spec["sharded_merger"]
            pathway = "sharded distributed pathway"
        else:
            # Should never reach here due to constructor validation
            raise RuntimeError(f"Invalid distributed mode: {self._distributed_mode}")

        # Apply merger
        try:
            return merger(value)
        except Exception as e:
            raise RuntimeError(f"Metric merging failed in {pathway} for '{name}': {e}") from e
