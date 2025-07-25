"""Physics subpackage for Pisces.

This subpackage contains physical modeling tools and computational routines
used in astrophysical simulations, modeling, and analysis. It provides
implementations for physical conversions, equilibrium diagnostics,
distribution function models, and other domain-specific calculations.

Purpose
-------
The :mod:`physics` module aims to provide reusable, numerically robust implementations
of widely used physical operations relevant to structure formation, galaxy dynamics,
and dark matter modeling. These tools are unit-aware (via `unyt`) and compatible
with Pisces modeling conventions.
"""

from . import virialization
from .conversions import (
    compute_mean_molecular_weight,
    compute_mean_molecular_weight_per_electron,
)

__all__ = [
    "compute_mean_molecular_weight",
    "compute_mean_molecular_weight_per_electron",
    "virialization",
]
