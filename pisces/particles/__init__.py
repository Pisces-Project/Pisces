"""Support for particle datasets in Pisces.

This module provides functionality for handling particle datasets, including
sampling particles, preparing SPH initial conditions, etc.
"""

__all__ = ["ParticleDataset", "concatenate_particles"]
from .base import ParticleDataset
from .utils import concatenate_particles
