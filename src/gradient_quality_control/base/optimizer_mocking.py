"""
OptimizerMockingMixin implementation.

Provides attribute forwarding mechanics to make wrapper transparently duck-type
as wrapped optimizer while blocking direct state mutation after initialization.
"""

from typing import Any

from torch.optim import Optimizer


class OptimizerMockingMixin:
    """
    Mixin providing transparent attribute forwarding to wrapped optimizer.

    Allows wrapper to duck-type as optimizer while maintaining separate interface.
    Blocks direct attribute mutation after initialization is complete.
    """

    def _finalize_initialization(self) -> None:
        """
        Mark end of construction phase.

        Must be called by orchestrator at end of __init__.

        Post-conditions:
            Sets _initialized flag to True
        """
        object.__setattr__(self, "_initialized", True)

    def __getattribute__(
        self,
        name: str,
    ) -> Any:
        """
        Forward attribute access to wrapped optimizer while preserving wrapper's interface.

        Lookup order:
        1. Check wrapper class hierarchy (up to Optimizer) for class attributes
        2. Check instance __dict__ for instance attributes
        3. Forward to wrapped optimizer

        Args:
            name: Attribute name to retrieve

        Returns:
            Attribute value from wrapper or optimizer
        """
        # Get instance dict to check for instance attributes
        instance_dict = object.__getattribute__(self, "__dict__")

        # Walk MRO until reaching Optimizer class
        for cls in object.__getattribute__(self, "__class__").__mro__:
            # Stop before Optimizer (don't check Optimizer's __dict__)
            if cls is Optimizer:
                break

            # Check if name is in this class's __dict__
            if name in cls.__dict__:
                # Use normal object lookup
                return object.__getattribute__(self, name)

        # Check instance __dict__
        if name in instance_dict:
            return instance_dict[name]

        # Not found in wrapper hierarchy or instance - forward to optimizer
        optimizer = instance_dict["_optimizer"]
        return getattr(optimizer, name)

    def __setattr__(
        self,
        name: str,
        value: Any,
    ) -> None:
        """
        Forward attribute assignment to wrapped optimizer while allowing initialization.

        During initialization (_initialized not set):
            Sets attributes locally on wrapper instance

        After initialization (_initialized is True):
            - Raises RuntimeError if name collides with wrapper's interface
            - Otherwise forwards to wrapped optimizer

        Args:
            name: Attribute name to set
            value: Value to assign

        Raises:
            RuntimeError: If attempting to set wrapper attribute after initialization
        """
        # Get instance dict
        instance_dict = object.__getattribute__(self, "__dict__")

        # Check if we're still in __init__
        if "_initialized" not in instance_dict:
            # During initialization - set locally
            object.__setattr__(self, name, value)
            return

        # After initialization - check for collisions and forward

        # Check if name is in instance dict (instance attribute collision)
        if name in instance_dict:
            raise RuntimeError(f"Cannot set attribute '{name}': collides with wrapper interface")

        # Walk MRO until Optimizer to check for class attribute collisions
        for cls in object.__getattribute__(self, "__class__").__mro__:
            # Stop before Optimizer
            if cls is Optimizer:
                break

            # Check if name in this class's __dict__
            if name in cls.__dict__:
                raise RuntimeError(
                    f"Cannot set attribute '{name}': collides with wrapper interface"
                )

        # No collision - forward to optimizer
        optimizer = instance_dict["_optimizer"]
        setattr(optimizer, name, value)
