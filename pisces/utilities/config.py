""" Configuration setup and access for Pisces.
"""
import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import Union

import yaml
from platformdirs import user_config_dir


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

    def __init__(self, path: Union[str, Path], autosave: bool = True):
        self._path = Path(path).expanduser().resolve()
        self._autosave = autosave
        self._data = self._load()

    def _load(self) -> dict:
        """Load configuration data from the YAML file."""
        if not self._path.exists():
            return {}
        with open(self._path) as f:
            return yaml.safe_load(f) or {}

    def _save(self) -> None:
        """Save configuration data to the YAML file."""
        with open(self._path, "w") as f:
            yaml.safe_dump(self._data, f, default_flow_style=False)

    def _traverse(self, key: str, create_missing: bool = False):
        """Navigate nested dictionaries using dot-separated keys.

        Args:
            key (str): Dot-separated key (e.g., "database.host").
            create_missing (bool): If True, create intermediate dictionaries as needed.

        Returns:
            tuple: (parent dictionary, final key)

        Raises:
            KeyError: If a key is missing and create_missing is False.
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
        node, final_key = self._traverse(key)
        return node[final_key]

    def __setitem__(self, key: str, value):
        node, final_key = self._traverse(key, create_missing=True)
        node[final_key] = value
        if self._autosave:
            self._save()

    def __delitem__(self, key: str):
        node, final_key = self._traverse(key)
        del node[final_key]
        if self._autosave:
            self._save()

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"<ConfigManager path={self._path} data={self._data}>"

    def to_dict(self) -> dict:
        """Return the full configuration data as a dictionary."""
        return self._data


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
        raise OSError(
            f"Missing default configuration file at {default_path}.\nWas"
            "Pisces install corrupted?"
        )

    return __PCONFIG__


pisces_config = get_config()
