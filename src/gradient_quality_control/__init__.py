from importlib.metadata import PackageNotFoundError, version

from .base import AbstractOptimizerWrapper
from .gradient_noise_scale import OptimizerWrapperGNS
from .gradient_norm_rescalar import OptimizerWrapperGNR
from .gradient_norm_threshold_scheduler import OptimizerWrapperGNTS
from .metric_hypothesis_test import OptimizerWrapperMHT
from .scheduled_batch_controller import OptimizerWrapperSBC
from .scheduling_utils import (
    get_curved_batch_schedule,
    get_direct_cosine_annealing_with_warmup,
    get_norm_threshold_cosine_annealing_with_warmup,
    get_quadratic_batch_schedule,
)

__all__ = [
    "AbstractOptimizerWrapper",
    "OptimizerWrapperGNS",
    "OptimizerWrapperGNR",
    "OptimizerWrapperMHT",
    "OptimizerWrapperSBC",
    "OptimizerWrapperGNTS",
    "get_quadratic_batch_schedule",
    "get_direct_cosine_annealing_with_warmup",
    "get_norm_threshold_cosine_annealing_with_warmup",
    "get_curved_batch_schedule",
    "__version__",
]

try:
    __version__ = version("torch-gqc")
except PackageNotFoundError:
    pass
