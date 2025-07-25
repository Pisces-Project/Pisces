"""Generic structures.

Generic structures and functions which are used in many places
throughout the code base but are insufficient in their complexity
or specialization to warrant independent modules.
"""


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
            key: Hashable key to register
            value: Value to store
            overwrite (bool): If False, raises an error if key exists
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
