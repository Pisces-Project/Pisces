"""
Core infrastructure for Pisces simulation extensions.

This module contains the foundational classes and logic used to support
hydrodynamics and N-body simulation workflows within Pisces. It currently
provides the :class:`~pisces.extensions.simulation.core.initial_conditions.InitialConditions` class for
creating, loading, and managing initial condition (IC) datasets.
"""

__all__ = ["InitialConditions"]

from .initial_conditions import InitialConditions
