"""
Extension classes for running astrophysical simulations using Pisces models.

Pisces models are designed to be exportable to many different simulation / analysis toolkits to
enable advanced workflows and extensibility. One such use case is to run full astrophysical
simulations using the models defined in Pisces. This package contains extension classes that
facilitate running simulations using different simulation codes, such as Gadget and Arepo.
These classes handle the conversion of Pisces models into the appropriate initial conditions
formats required by the simulation codes, as well as any necessary configuration settings.

The process of running hydrodynamical simulations is somewhat complex and involves many tunable
parameters. As such, the simulations extension package is a fairly deep package with multiple
submodules. To read more about how things work before jumping in and using these classes, please refer to
the documentation page on running simulations with Pisces: :ref:`simulations`.
"""

__all__ = [
    "gadget",
    "frontends",
    "initial_conditions",
]

# Import the relevant frontend modules for the different codes.
from . import gadget

# Make accessing the frontends easier by exposing them at the package level.
from .core import frontends, initial_conditions
