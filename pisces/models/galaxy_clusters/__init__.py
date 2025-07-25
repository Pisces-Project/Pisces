"""Galaxy cluster models.

This module contains the core classes for galaxy cluster models in the Pisces project. Models include
various features ranging from simple models with spherical symmetry and perfect hydrostatic equilibrium to
more complex models that include anisotropic velocity distributions, non-thermal pressure support, and
non-spherical geometries. Each model is designed to be flexible and extensible, allowing for a wide range of
astrophysical scenarios to be simulated and analyzed.

For details on the nature of the different models in this module, refer
to the documentation: :ref:`galaxy_clusters_overview`.
"""

__all__ = ["SphericalGalaxyClusterModel", "MagnetizedSphericalGalaxyClusterModel"]

# Directly import the relevant model classes.
from .spherical import (
    MagnetizedSphericalGalaxyClusterModel,
    SphericalGalaxyClusterModel,
)
