"""Mathematics utilities for Pisces.

This module provides support for various mathematical operations
which are used throughout the Pisces astrophysics library. This includes
statistical methods to sample particles, integration and interpolation methods, etc.
"""

__all__ = ["sample_from_pdf", "sample_from_cdf", "integrate", "integrate_toinf", "integrate_mass", "random_fields"]

# Import integration modules.
# Import the random fields module.
from . import random_fields
from .integration import integrate, integrate_mass, integrate_toinf

# Import sampling modules.
from .sampling import sample_from_cdf, sample_from_pdf
