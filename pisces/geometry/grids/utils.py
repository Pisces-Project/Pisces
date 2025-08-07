"""Utilities for working with grids in Pisces."""

from pathlib import Path
from typing import TYPE_CHECKING, Union

import h5py

from pisces._registries import __default_grid_registry__

if TYPE_CHECKING:
    from pisces._generic import Registry
    from pisces.geometry.grids.core import Grid


def load_grid_from_hdf5_group(group: h5py.Group, registry: Registry = None) -> "Grid":
    """
    Load a grid instance from an HDF5 group using the provided registry.

    Parameters
    ----------
    group : h5py.Group
        HDF5 group containing the saved grid data.
    registry : Registry, optional
        Registry mapping grid class names to their corresponding classes.
        If None, the default grid registry will be used.

    Returns
    -------
    Grid
        An instance of the appropriate grid subclass.

    Raises
    ------
    ValueError
        If the group is missing required metadata or the class is unregistered.
    """
    registry = registry or __default_grid_registry__

    class_name = group.attrs.get("CLASS_NAME")
    if class_name is None:
        raise ValueError("Missing 'CLASS_NAME' attribute in HDF5 group.")

    if class_name not in registry:
        raise ValueError(f"Unknown grid class '{class_name}'. Is it registered?")

    cls = registry[class_name]
    return cls._load_grid_from_hdf5_group(group)


def load_grid(filename: Union[str, Path], group_path: str, registry: Registry = None):
    """
    Load a grid instance from an HDF5 file.

    Parameters
    ----------
    filename : str or Path
        Path to the HDF5 file.
    group_path : str
        Path within the HDF5 file to the group containing the grid.
    registry : Registry, optional
        Registry mapping grid class names to their corresponding classes.
        If None, the default grid registry will be used.

    Returns
    -------
    Grid
        A grid instance loaded from the file.
    """
    import h5py

    with h5py.File(filename, "r") as f:
        if group_path not in f:
            raise KeyError(f"HDF5 file does not contain group '{group_path}'.")

        group = f[group_path]
        return load_grid_from_hdf5_group(group, registry=registry)
