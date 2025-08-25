"""
Extension classes and utilities for astrophysical simulations.

This module defines infrastructure for creating and managing *initial conditions*
(ICs) for use with a variety of astrophysical simulation codes.
It provides a core class (:class:`~pisces.extensions.simulation.core.initial_conditions.InitialConditions`)
that can load, manipulate, and export IC datasets in formats compatible with popular simulation frameworks.
"""

__all__ = ["InitialConditions", "Gadget4Frontend", "gadget"]

from . import gadget
from .core import InitialConditions
from .gadget import Gadget4Frontend
