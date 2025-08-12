"""Pisces global configuration system.

This module provides a hierarchical, YAML-backed configuration manager for
Pisces. Configurations can be accessed and updated via dot-separated keys
(e.g., ``pisces_config['system.appearance.disable_progress_bars'] = False``) and
changes are persisted to disk if autosave is enabled.

Configuration files are located using the following precedence:

    1. Environment variable ``$PISCES_CONFIG`` (if set)
    2. Local project file ``.piscesrc`` in the current working directory
    3. User-specific config at ``~/.config/pisces/config.yaml``
    4. Package default config distributed with Pisces

Use :attr:`pisces_config` to access the active configuration. It behaves like a
nested dictionary with automatic loading and saving.
"""

import os
from collections.abc import MutableMapping
from pathlib import Path

from platformdirs import user_config_dir

from .io_tools import unyt_yaml


# --------------------------------- #
# Configuration Manager             #
# --------------------------------- #
# Functions for configuring the environment at the MPI
# and XSPEC levels.
class ConfigManager(MutableMapping):
    """Hierarchical configuration manager with dot-separated keys and optional autosave.

    Stores configuration data as nested dictionaries, backed by a YAML file.
    Allows dot-separated key access for nested structures.

    Parameters
    ----------
    path: str or `Path`
        Path to the YAML configuration file.
    autosave: bool
        If True, automatically save changes to disk. Defaults to True.

    """

    __YAML__ = unyt_yaml
    """ The YAML manager."""

    def __init__(self, path: str | Path, autosave: bool = True):
        self._path = Path(path).expanduser().resolve()
        self._autosave = autosave
        self._data = self._load()

    def _load(self) -> dict:
        """Load configuration data from the YAML file."""
        if not self._path.exists():
            return {}
        with open(self._path) as f:
            return self.__YAML__.load(f) or {}

    def _save(self) -> None:
        """Save configuration data to the YAML file."""
        with open(self._path, "w") as f:
            self.__YAML__.dump(self._data, f)

    def _traverse(self, key: str, create_missing: bool = False):
        """Navigate nested dictionaries using dot-separated keys.

        Parameters
        ----------
        key : str
            Dot-separated key (e.g., "database.host").
        create_missing : bool, optional
            If True, create intermediate dictionaries as needed.

        Returns
        -------
        tuple
            A tuple (parent dictionary, final key).

        Raises
        ------
        KeyError
            If a key is missing and `create_missing` is False.

        """
        keys = key.split(".")
        node = self._data
        for k in keys[:-1]:
            if k not in node:
                if create_missing:
                    node[k] = {}
                else:
                    raise KeyError(f"'{k}' not found in config.")
            node = node[k]
        return node, keys[-1]

    def __getitem__(self, key: str):
        """Retrieve a value from the configuration using a dot-separated key.

        Parameters
        ----------
        key : str
            The dot-separated key identifying the configuration value.

        Returns
        -------
        Any
            The corresponding value from the configuration.

        Raises
        ------
        KeyError
            If the specified key does not exist.

        """
        node, final_key = self._traverse(key)
        return node[final_key]

    def __setitem__(self, key: str, value):
        """Set a configuration value using a dot-separated key.

        Parameters
        ----------
        key : str
            The dot-separated key identifying the configuration value.
        value : Any
            The value to assign.

        Notes
        -----
        If `autosave` is enabled, the configuration will be written to disk after setting.

        """
        node, final_key = self._traverse(key, create_missing=True)
        node[final_key] = value
        if self._autosave:
            self._save()

    def __delitem__(self, key: str):
        """Delete a configuration value using a dot-separated key.

        Parameters
        ----------
        key : str
            The dot-separated key identifying the configuration value to delete.

        Raises
        ------
        KeyError
            If the specified key does not exist.

        Notes
        -----
        If `autosave` is enabled, the change is written to disk immediately.

        """
        node, final_key = self._traverse(key)
        del node[final_key]
        if self._autosave:
            self._save()

    def __iter__(self):
        """Return an iterator over the top-level keys in the configuration.

        Returns
        -------
        Iterator[str]
            An iterator over the top-level keys in the root configuration dictionary.

        """
        return iter(self._data)

    def __len__(self) -> int:
        """Return the number of top-level keys in the configuration.

        Returns
        -------
        int
            Number of top-level entries in the configuration.

        """
        return len(self._data)

    def __repr__(self) -> str:
        """Return a string representation of the configuration manager.

        Returns
        -------
        str
            A string showing the path to the config file and current in-memory state.

        """
        return f"<ConfigManager path={self._path} data={self._data}>"

    def to_dict(self) -> dict:
        """Return the full configuration data as a dictionary."""
        return self._data

    def update(self, E=None, **F) -> None:
        """Update the configuration with another dictionary."""
        self._data.update(E=E, **F)
        if self._autosave:
            self._save()


# Cache to avoid reloading
__PCONFIG__ = None


def get_config() -> ConfigManager:
    """Return global Pisces configuration following precedence."""
    # Seek out a global configuration in the
    # name space.
    global __PCONFIG__
    if __PCONFIG__ is not None:
        return __PCONFIG__

    # We've failed to identify an existing __PCONFIG__
    # configuration. We'll need to see out candidates.
    candidates = []

    # 1. Environment override
    # 2. Project-local file
    # 3. User-global config
    # 4. Package defaults
    env_path = os.environ.get("PISCES_CONFIG")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.append(Path.cwd() / ".piscesrc")
    user_path = Path(user_config_dir("pisces")) / "config.yaml"
    candidates.append(user_path)
    default_path = Path(__file__).parents[1] / "bin" / "config.yaml"
    candidates.append(default_path)

    # Find the first existing config
    for path in candidates:
        if path.exists():
            __PCONFIG__ = ConfigManager(path)
            break
    else:
        raise OSError(f"Missing default configuration file at {default_path}.\nWasPisces install corrupted?")

    return __PCONFIG__


pisces_config = get_config()
