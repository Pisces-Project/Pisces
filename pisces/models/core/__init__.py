"""Core structures and base classes for Pisces models.

This module provides the foundational classes and utilities for defining and managing
astrophysical models within the Pisces framework. It includes base classes for models,
as well as utilities for model management and interaction.
"""

__all__ = ["BaseHook", "_HookTools", "ParticleGenerationHook", "ModelConfig", "BaseModel", "load_model"]

from .base import BaseModel
from .hooks import BaseHook, ParticleGenerationHook, _HookTools
from .utils import ModelConfig, load_model
