"""
The arbitrary schedule subsystem is designed to extend torch
schedule systems to be able to schedule any feature such as
weight decay, not just learning rate, using only the torch
scheduling subsystem.

Basically, we create and present relevant schedules with a
'fake' optimizer where setting the learning rates of the
optimizer is actually setting the relevant feature of
the real optimizer by proxy. For instance, if "weight_decay"
is being targetted by proxy, setting the learning rate to 0.1
in the proxy sets weight decay to 0.1 in the main optimizer.

This is done by means of UserDict indirection.
"""

import warnings
from numbers import Number
from collections import UserDict
from typing import Optional, Any, Callable, Dict, List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import StateDict

from .optimizer_utils import optimizer_extend

# ================================================================================
# Error Control. The ability to manipulate the error context this system
# runs within
# ================================================================================

THROW_ERROR_ON_DESYNC = True

def throw_errors_on_desync(flag: bool):
    """
    State editor to turn throw on desync on and off.
    """
    global THROW_ERROR_ON_DESYNC
    THROW_ERROR_ON_DESYNC= flag

# ===============================================================================================
# Proxy system. The proxy dict will update the backend on the actual target whenever
# a schedule tries to set to it
# ===============================================================================================

class ProxyDictByLR(UserDict):
    """
    A specialized proxy dictionary class,
    this will extract only the targeted
    feature which can then be set to by
    schedulers. Setting to it then updates
    the main dictionary too, using the
    appropriate term instead.

    Additionally, schedulers like to set their
    own variables, particularly 'initial_lr'.
    To accomodate this, we setup when needed
    a 'scheduler_namespace' feature that
    allows anything reading from this to modify
    and talk to something it believes is in
    the main dictionary, when in fact they
    are isolated


    """
    def __init__(self,
                 key_to_proxy: str,
                 dictionary: Dict[str, Any],
        ):
        # Create the proxy
        if key_to_proxy not in dictionary:
            raise KeyError(f"Proxy key named '{key_to_proxy}' not found in dictionary")
        proxy_dictionary = {"lr" : dictionary[key_to_proxy]}
        super().__init__(proxy_dictionary)

        # Backend
        self.dictionary = dictionary
        self.proxy_key = key_to_proxy
        self.existing_keys = list(dictionary.keys())
        self._initialize = True

    def __setitem__(self,
                    key: str,
                    value: Any,
                    ) -> None:
        """
        Sets an item in the dictionary, presumably
        the learning rate from the schedule
        :param key: Should be learning rate
        :param value: The value to set it to
        :raises: KeyError if not learning rate
        """
        # initialization
        if not hasattr(self, "_initialize"):
            return super().__setitem__(key, value)

        # Main logic.
        if key != 'lr' and key in self.existing_keys:
            raise KeyError("key is not lr when set by schedule and originally existed. Not allowed.")
        if self['lr'] != self.dictionary[self.proxy_key]:
            if THROW_ERROR_ON_DESYNC:
                msg = ("Danger: Proxy and backend have become desynced. This is a strong sign you setup your schedules"
                       "incorrectly. \n"
                       "To disabled this error, use throw_errors_on_desync(False)")
                raise RuntimeError(msg)
            else:
                msg = ("Warning: Someone has been modifying the backend optimizer state without going through the proxy."
                       "Are you sure you hooked up your schedulers correctly?")
                warnings.warn(msg)

        self.dictionary[self.proxy_key] = value
        super().__setitem__(key, value)

# =========================================================================================
# Main adapter and factory classes. These basically pretend to be a lightweight optimizer.
# In normal use, the user never even sees the arbitrary schedule adapter.
# =========================================================================================

class ArbitraryScheduleAdapter(Optimizer):
    """
    A specialized class used for attaching a schedule
    to an arbitrary optimizer characteristic, such as
    "norm_threshold" or "weight_decay" which is compatible
    with torch schedulers. Schedules which set to this
    optimizer class will actually set to the target
    proxy instead.

    This is not actually an optimizer,
    and attempts to use it as one will tend to throw;
    instead, it is an optimizer stub that schedules
    know how to use.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        target_name: str,
        default_value: Optional[float] = None,
    ) -> None:
        """
        Args:
            optimizer: The optimizer to subsume
            target_name: The dictionary feature name to proxy as 'lr' instead
            default_value: Optional. If not provided, the feature must exist ahead of 
                time.
        """
        self.optimizer = optimizer_extend(optimizer, target_name, default_value)
        self.param_groups = [ProxyDictByLR(target_name, param_dict) for param_dict in self.optimizer.param_groups]
        self.feature_name = target_name

    # This is a stub.
    def step(self, closure: Optional[Callable[[], float]] = None):
        raise NotImplementedError("This is a stub. Please construct through arbitrary_schedule_factory instead")
    def load_state_dict(self, state_dict: StateDict) -> None:
        raise NotImplementedError("This is a stub. Please construct through arbitrary_schedule_factory instead")
    def state_dict(self)->Dict[str, Any]:
        raise NotImplementedError("This is a stub. Please construct through arbitrary_schedule_factory instead")

def arbitrary_schedule_factory(
    feature_name: str,
    optimizer: Optimizer,
    schedule_factory: Callable[[Optimizer], _LRScheduler],
    default_value: Optional[float] = None,
) -> _LRScheduler:
    """
    Factory function to create an ArbitrarySchedule for scheduling arbitrary optimizer parameters.

    Args:
        feature_name: The optimizer parameter to schedule (e.g., "weight_decay", "momentum")
        optimizer: The optimizer to bind the schedule to.
        schedule_factory: A callable that takes an optimizer and returns a scheduler
        default_value: Default value for the parameter if it doesn't exist in optimizer

    Returns:
        _LRScheduler that schedules the specified parameter
    """
    adapter = ArbitraryScheduleAdapter(optimizer, feature_name, default_value)
    schedule = schedule_factory(adapter)
    return schedule

class SynchronousSchedule(_LRScheduler):
    """
    With the introduction of Arbitrary Schedules, it is now
    possible to have schedules wherein more than one schedule
    is defined and active on the same time, just on different
    optimizer hyperparameters.

    To handle this, we define the SynchronousSchedule which
    can be loaded with a collection of such schedules and step
    them in lockstep.

    It does not handle features other than epoch.
    """
    @property
    def schedule_names(self)->List[str]:
        return list(self.schedules.keys())

    def __init__(self,
                 schedules: List[_LRScheduler],
                 ):
        """
        Initializes the class
        :param schedules:
        """

        assigned_schedules = {}
        for schedule in schedules:
            if isinstance(schedule.optimizer, ArbitraryScheduleAdapter):
                name = schedule.optimizer.feature_name
            else:
                name = "lr"
            if name in assigned_schedules:
                raise RuntimeError(f"There is already a schedule named '{name}'")
            assigned_schedules[name] = schedule
        self.schedules = assigned_schedules

    def get_last_schedule(self,
                          name: str,
                          )->List[Number]:
        """
        Gets a collection of the arbitrary last schedule values
        :param name: The name of the values, a schedule.
        :return: The returned values
        """
        return self.schedules[name].get_last_lr()

    def step(self, epoch: Optional[int] = None):
        """Steps the schedules"""
        for schedule in self.schedules.values():
            schedule.step(epoch)

    def state_dict(self) -> Dict[str, Any]:
        """Save the state dict of the schedule"""
        output = {}
        for name, schedule in self.schedules.items():
            output[name] = schedule.state_dict()
        return output
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the state dict of the schedule"""
        for name, individual_state in state_dict.items():
            self.schedules[name].load_state_dict(state_dict)

    def get_last_lr(self) -> list[float]:
        """gets the last learning rate"""
        return self.get_last_schedule("lr")

def arbitrary_schedule_factory(
    feature_name: str,
    optimizer: Optimizer,
    schedule_factory: Callable[[Optimizer], _LRScheduler],
    default_value: Optional[float] = None,
) -> _LRScheduler:
    """
    Factory function to create an ArbitrarySchedule for scheduling arbitrary optimizer parameters.

    Args:
        feature_name: The optimizer parameter to schedule (e.g., "weight_decay", "momentum")
        optimizer: The optimizer to bind the schedule to.
        schedule_factory: A callable that takes an optimizer and returns a scheduler
        default_value: Default value for the parameter if it doesn't exist in optimizer

    Returns:
        _LRScheduler that schedules the specified parameter
    """
    adapter = ArbitraryScheduleAdapter(optimizer, feature_name, default_value)
    schedule = schedule_factory(adapter)
    return schedule
