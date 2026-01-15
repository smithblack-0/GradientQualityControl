from importlib.metadata import PackageNotFoundError, version

from .base import AbstractOptimizerWrapper
from .implementations.gradient_noise_scale import (
    OptimizerWrapperGNS,
    make_gns_default,
    make_gns_with_cosine_annealing_schedule,
)
from .implementations.gradient_norm_rescaler import (
    OptimizerWrapperGNR,
    make_gnr_with_cosine_annealing_schedule,
    make_gnr_with_cosine_annealing_schedule_conventional_lr,
)
from .implementations.gradient_norm_threshold_scheduler import (
    OptimizerWrapperGNTS,
    make_gnts_with_cosine_annealing_schedule,
    make_gnts_with_cosine_annealing_schedule_conventional_lr,
)
from .implementations.schedule_batch_controller import (
    OptimizerWrapperSBC,
    make_sbc_with_polynomial_schedule,
    make_sbc_with_polynomial_schedule_conventional_lr,
)

# from .scheduled_batch_controller import OptimizerWrapperSBC

# from gradient_noise_scale import OptimizerWrapperGNS
# from gradient_norm_rescalar import OptimizerWrapperGNR
# from metric_hypothesis_test import OptimizerWrapperMHT


__all__ = [
    "AbstractOptimizerWrapper",
    # SBC imports
    "OptimizerWrapperSBC",
    "make_sbc_with_polynomial_schedule",
    "make_sbc_with_polynomial_schedule_conventional_lr",
    # GNTS imports
    "OptimizerWrapperGNTS",
    "make_gnts_with_cosine_annealing_schedule",
    "make_gnts_with_cosine_annealing_schedule_conventional_lr",
    # GNR imports
    "OptimizerWrapperGNR",
    "make_gnr_with_cosine_annealing_schedule",
    "make_gnr_with_cosine_annealing_schedule_conventional_lr",
    # GNS imports
    "OptimizerWrapperGNS",
    "make_gns_with_cosine_annealing_schedule",
    "make_gns_default",
    "__version__",
]

try:
    __version__ = version("torch-gqc")
except PackageNotFoundError:
    pass
