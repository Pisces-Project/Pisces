"""
Support for the Gadget SPH code and its descendants.

This module provides frontend support to initialize Gadget-2 simulations
and manage their initial conditions. It includes classes and utilities
for loading, manipulating, and exporting initial conditions (ICs)
for Gadget-based simulations, including Gadget-2 and Gadget-3.

For more information on Gadget, see the `official documentation <https://wwwmpa.mpa-garching.mpg.de/gadget/>`_.
"""

__all__ = ["Gadget2Frontend"]

# Provide access to the frontend.
from .frontends import Gadget2Frontend
