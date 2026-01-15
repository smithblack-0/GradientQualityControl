"""
AbstractOptimizerWrapper implementation.

User-facing class providing convenient constructor that automatically constructs
and wires all subsystems.
"""

from typing import Literal, Optional

import torch

from .distributed_metrics import DistributedMetricsManagementSubsystem
from .gradient_accumulation import GradientAccumulationStepSubsystem
from .orchestrator import OrchestratorMainSystem
from .reporting import ReportingSubsystem
from .state_management import StateManagementSubsystem


class AbstractOptimizerWrapper(OrchestratorMainSystem):
    """
    User-facing wrapper class that auto-constructs all subsystems.

    Primary entry point for users. Subclasses implement step() for specific
    control algorithms.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_draws: int = 64,
        distributed_mode: Optional[Literal["replicated", "sharded"]] = None,
    ):
        """
        Initialize wrapper with automatic subsystem construction.

        Args:
            optimizer: Configured PyTorch optimizer to wrap
            max_draws: Maximum batches that can accumulate before forcing step. Must be >= 1.
            distributed_mode: Distributed training mode - None (single device),
                            "replicated" (DDP), or "sharded" (FSDP)

        Raises:
            TypeError: If optimizer is not instance of torch.optim.Optimizer
            ValueError: If max_draws < 1
            ValueError: If distributed_mode not in valid values
            RuntimeError: If distributed execution detected but distributed_mode is None

        Post-conditions:
            All subsystems constructed and wired
        """
        # Validate optimizer type
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                f"optimizer must be instance of torch.optim.Optimizer, got {type(optimizer)}"
            )

        # Validate max_draws
        if max_draws < 1:
            raise ValueError(f"max_draws must be >= 1, got {max_draws}")

        # Validate distributed_mode
        if distributed_mode is not None and distributed_mode not in ("replicated", "sharded"):
            raise ValueError(
                f"distributed_mode must be None, 'replicated', or 'sharded', got {distributed_mode}"
            )

        # Check for distributed execution without distributed_mode
        if torch.distributed.is_initialized() and distributed_mode is None:
            raise RuntimeError(
                "Detected distributed execution (torch.distributed is initialized) but "
                "distributed_mode is None. "
                "Please specify distributed_mode='replicated' or 'sharded'."
            )

        # Construct subsystems
        state_manager = StateManagementSubsystem(optimizer)
        distributed_metrics = DistributedMetricsManagementSubsystem(distributed_mode)
        accumulation = GradientAccumulationStepSubsystem(state_manager, optimizer, max_draws)
        reporting = ReportingSubsystem(state_manager)

        # Initialize parent orchestrator
        super().__init__(optimizer, state_manager, distributed_metrics, accumulation, reporting)
