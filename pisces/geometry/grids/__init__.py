"""
Support for simple structured grids in Pisces.

This module provides classes and functions to create and manage structured grids in
Pisces. Structured grids are essential for representing spatial domains in simulations and analyses.
All models in :mod:`~pisces.models` are defined using structured grids as their coordinate framework.
"""

__all__ = ["GenericGrid", "load_grid_from_hdf5_group", "load_grid"]

from .core import GenericGrid
from .utils import load_grid, load_grid_from_hdf5_group
