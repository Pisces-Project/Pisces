"""Default registry container module."""

from ._generic import Registry

__default_profile_registry__ = Registry()
"""
The default registry for storing profiles.
"""
__default_coordinate_registry__ = Registry()
"""
The default registry for coordinate systems.
"""
__default_grid_registry__ = Registry()
"""
The default registry for grid types.
"""
__default_model_registry__ = Registry()
"""
The default registry for models.
"""
