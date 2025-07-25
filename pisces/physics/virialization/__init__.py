"""Tools and algorithms for solving the collisionless Boltzmann equation.

This module contains a number of algorithms and tools for computing velocity
distributions for collisionless elements of models so that they can be
correctly converted into particle datasets.
"""

__all__ = [
    "compute_relative_potential",
    "compute_relative_energy",
    "sample_eddington_velocities",
    "compute_eddington_distribution",
]

from .eddington import (
    compute_eddington_distribution,
    compute_relative_energy,
    compute_relative_potential,
    sample_eddington_velocities,
)
