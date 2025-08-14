"""Astrophysical models for the Pisces project.

This module contains the core classes for various types of astrophysical models
from stellar systems to dark matter halos and galaxy clusters. Each model has
its own class with some standard conventions on storage and behavior. Models are
organized by type.
"""

__all__ = [
    "galaxy_clusters",
    "stars",
    "galaxies",
    "load_model",
    "inspect_model_metadata",
    "inspect_model_grid",
    "inspect_model_fields",
    "inspect_model_coordinate_system",
]

# Import the top level modules for each of the different
# model types.
from . import galaxies, galaxy_clusters, stars
from .core.utils import (
    inspect_model_coordinate_system,
    inspect_model_fields,
    inspect_model_grid,
    inspect_model_metadata,
    load_model,
)
