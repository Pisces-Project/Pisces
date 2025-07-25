"""
Utilities and registries for testing profiles.
"""

from pisces import profiles
from pisces.profiles.base import BaseProfile

# ---------------------------------- #
# Registry creation and management   #
# ---------------------------------- #
# These are just registry lists of profile classes of different types. They are
# present here so that we have an easy to setting up tests to run on all profiles of
# a specific type. These can be dynamically generated or written by hand.
__ALL_PROFILES__ = {
    cls_name: cls
    for cls_name, cls in profiles.__dict__.items()
    if isinstance(cls, type) and issubclass(cls, BaseProfile) and not getattr(cls, "__IS_ABSTRACT__", True)
}
