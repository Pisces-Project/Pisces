"""Generic structures.

Generic structures and functions which are used in many places
throughout the code base but are insufficient in their complexity
or specialization to warrant independent modules.
"""

from abc import ABCMeta
from typing import Protocol, TypeVar, runtime_checkable


# ====================================== #
# Registry Classes and Infrastructure    #
# ====================================== #
# These are classes and various additional infrastructure for
# creating registries of classes which can be used when looking up
# classes to load an object from disk.
@runtime_checkable
class _HasRegistryAttributes(Protocol):
    __IS_ABSTRACT__: bool
    __DEFAULT_REGISTRY__: "Registry"


_T = TypeVar("_T", bound="_HasRegistryAttributes")


class Registry:
    """Simple Registry class that wraps a dictionary with helpful methods."""

    def __init__(self, name="Registry"):
        self._registry = {}
        self._name = name

    def register(self, key, value, *, overwrite=False):
        """
        Register a new key-value pair.

        Parameters
        ----------
        key: str
         Hashable key to register
        value: Any
            Value to store
        overwrite: bool
            If False, raises an error if key exists
        """
        if not overwrite and key in self._registry:
            raise KeyError(f"Key '{key}' is already registered in {self._name}.")
        self._registry[key] = value

    def remove(self, key):
        """Remove a key from the registry."""
        if key not in self._registry:
            raise KeyError(f"Key '{key}' not found in {self._name}.")
        del self._registry[key]

    def get(self, key, default=None):
        """Retrieve a value by key. Returns default if not found."""
        return self._registry.get(key, default)

    def has(self, key):
        """Check if a key exists in the registry."""
        return key in self._registry

    def clear(self):
        """Clear the registry."""
        self._registry.clear()

    def keys(self):
        return self._registry.keys()

    def values(self):
        return self._registry.values()

    def items(self):
        return self._registry.items()

    def __contains__(self, key):
        return self.has(key)

    def __getitem__(self, key):
        return self._registry[key]

    def __setitem__(self, key, value):
        self.register(key, value, overwrite=True)

    def __delitem__(self, key):
        self.remove(key)

    def __len__(self):
        return len(self._registry)

    def __iter__(self):
        return iter(self._registry)

    def __repr__(self):
        return f"<{self._name} with {len(self._registry)} items: {list(self._registry.keys())}>"


class RegistryMeta(ABCMeta):
    """Metaclass capable of registering classes to a default registry."""

    def __new__(mcs, name, bases, namespace, **kwargs):
        # Create the class object using the base metaclass
        cls_object: _T = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Register the class and return.
        mcs.__register_class__(cls_object)
        return cls_object

    @staticmethod
    def __register_class__(cls_object: _T) -> _T:
        # If the class is not abstract, register it to the default registry
        if (not cls_object.__IS_ABSTRACT__) and (cls_object.__name__ not in cls_object.__DEFAULT_REGISTRY__):
            try:
                cls_object.__DEFAULT_REGISTRY__.register(cls_object.__name__, cls_object)
            except Exception as exp:
                raise TypeError(f"Failed to register class {cls_object.__name__}: {exp}") from exp

        return cls_object
