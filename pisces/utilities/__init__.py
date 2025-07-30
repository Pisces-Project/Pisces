"""General-purpose utility functions and configuration for the Pisces astrophysics library.

This module aggregates common functionality used throughout Pisces, including:

- Global configuration settings (:mod:`utilities.config`).
- Logging (:mod:`utilities.logging`).
- Numerical integration routines (:mod:`utilities.math_ops`).
- Fundamental physics utilities (e.g., mean molecular weights) (:mod:`utilities.physics`).
- Symbolic manipulation routines (:mod:`utilities.symbols`).

"""

__all__ = [
    "pisces_config",
    "pisces_logger",
    "__RNG__",
    "unyt_yaml",
]

# Import pisces configuration components. At the public level,
# we only import the configuration object itself.
from .config import pisces_config

# Import the yaml configuration manager.
from .io_tools import unyt_yaml

# Import the logging configuration and setup. At the public level,
# we only include the actual logger.
from .log import pisces_logger

# Import the RNG object.
from .rng import __RNG__

# We do not explicitly import any of the mathematics operations or
# the symbolic operations here. They must be retrieved directly from
# the lower level modules.
