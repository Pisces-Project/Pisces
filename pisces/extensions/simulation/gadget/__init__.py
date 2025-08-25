"""
Support for the Gadget SPH code and its descendants.

This module provides support for generating initial conditions for the
Gadget-4 simulation code and its variants.

For more information on Gadget, see the `official documentation <https://wwwmpa.mpa-garching.mpg.de/gadget4/>`_.
"""

__all__ = ["Gadget4Frontend", "GadgetParticleDataset"]

# Provide access to the frontend.
from .frontends import Gadget4Frontend
from .particles import GadgetParticleDataset
