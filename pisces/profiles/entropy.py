"""Entropy profiles for use in Pisces models.

The :mod:`~profiles.entropy` module provides a number of built-in profiles to model entropy in astrophysical systems,
such as galaxy clusters, hot gas halos, and stellar environments.
"""

from abc import ABC
from typing import Any

from .base import BaseSphericalRadialProfile


class BaseSphericalEntropyProfile(BaseSphericalRadialProfile, ABC):
    """Base class for spherical entropy profiles.

    This class provides a foundation for implementing various entropy profiles in spherical coordinates.
    It inherits from `BaseSphericalRadialProfile` and defines the basic structure and methods for entropy profiles.
    """

    # @@ CLASS ATTRIBUTES (INVARIANT) @@ #
    _IS_ABC = True

    # @@ CLASS ATTRIBUTES @@ #
    AXES: list[str] = ["r"]
    DEFAULT_PARAMETERS: dict[str, Any] = {}
