"""
Support for simple structured grids in Pisces.

This module provides classes and functions to create and manage structured grids in
Pisces. Structured grids are essential for representing spatial domains in simulations and analyses.
All models in :mod:`models` are defined using structured grids as their coordinate framework.
"""

__all__ = ["GenericGrid"]

from .core import GenericGrid
