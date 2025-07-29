"""Astrophysical models of stellar systems.

This module contains the core classes for various types of stellar models including
simple polytropic models, isothermal models, and more complex models that
include stellar evolution and other physical processes. Each model has its own
class with some standard conventions on storage and behavior. Models are
organized by type.

For details on the nature of the different models in this module, refer
to the documentation: :ref:`stars_overview`.

"""

__all__ = [
    "PolytropicStarModel",
]

from .polytropes import PolytropicStarModel
