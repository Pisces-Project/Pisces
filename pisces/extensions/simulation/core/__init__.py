"""
Core infrastructure for simulation extensions in Pisces.

This module provides the base structures from which simulation frontends and initial
condition generators can be built. It defines abstract classes and interfaces that
facilitate the creation of simulation setups compatible with various astrophysical
simulation codes.

.. note::

    In general, this module is only intended to be used internally by other
    simulation extension modules. Users looking to run simulations should typically
    interact with the higher-level frontends provided in specific simulation code
    submodules (e.g., :mod:`~pisces.extensions.simulation.gadget`).
"""

__all__ = [
    "InitialConditions",
    "InitialConditions1DCartesian",
    "InitialConditions1DSpherical",
    "InitialConditions2DCartesian",
    "InitialConditions3DCartesian",
    "InitialConditionsCartesian",
    "SimulationFrontend",
    "GadgetLikeFrontend",
]

# Import all the types of initial conditions.
# Import the frontends
from .frontends import GadgetLikeFrontend, SimulationFrontend
from .initial_conditions import (
    InitialConditions,
    InitialConditions1DCartesian,
    InitialConditions1DSpherical,
    InitialConditions2DCartesian,
    InitialConditions3DCartesian,
    InitialConditionsCartesian,
)
