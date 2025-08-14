"""Utility tools for accessing Pisces model configuration.

This module provides infrastructure for binding model classes to their
corresponding configuration entries in the Pisces global config registry.
It is used internally by Pisces to expose user-defined or default configuration
options at the class level without hardcoding values.

"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import h5py

from pisces._registries import __default_model_registry__
from pisces.utilities.config import pisces_config

if TYPE_CHECKING:
    from pisces._generic import Registry
    from pisces.models.core.base import BaseModel


class ModelConfig:
    """Descriptor for accessing the registered configuration of a model class.

    This descriptor provides dynamic, read-only access to the model’s
    configuration as stored in the global `pisces_config` registry. It is intended
    to be used as a class-level attribute within model classes.

    Usage
    -----
    Define a `config` attribute using this descriptor:

    .. code-block:: python

        class MyModel(BaseModel):
            config = ModelConfig()

    Then access it via the class or an instance:

    .. code-block:: python

        MyModel.config  # Access class-level configuration
        my_model.config  # Also works via instance

    The corresponding entry in the configuration registry should be:

    .. code-block:: yaml

        models:
          MyModel:
            param1: ...
            param2: ...

    Raises
    ------
    AttributeError
        If the configuration entry for the model class is missing.

    Returns
    -------
    dict
        A dictionary of configuration options for the model.

    """

    def __get__(self, instance: object, owner: type) -> dict:
        """Fetch the model configuration."""
        class_name = owner.__name__
        try:
            return pisces_config[f"models.{class_name}"]
        except KeyError as err:
            raise AttributeError(
                f"Configuration entry not found for model '{class_name}'. "
                f"Expected key 'models.{class_name}' in `pisces_config`."
            ) from err


# --------------------------------- #
# Utility Loaders                   #
# --------------------------------- #
def load_model(
    path: Union[str, Path],
    registry: Optional["Registry"] = None,
) -> "BaseModel":
    """
    Load a :class:`~models.core.base.BaseModel` from an HDF5 file.

    The file is loaded by accessing the model metadata specifying the
    class name of the model. The class is then resolved using the provided
    ``registry``.

    Parameters
    ----------
    path: str or ~pathlib.Path
        Path to the HDF5 file containing the model.
    registry: Registry, optional
        The registry class in which to look up the model class. If one is not provided,
        then the default registry is used. This should be sufficient in almost all cases
        as all the native models in Pisces are automatically registered here.

    Returns
    -------
    ~models.core.base.BaseModel
        An instance of the resolved model class.

    Raises
    ------
    FileNotFoundError
        If the path does not exist or is not a file.
    ValueError
        If the file is missing the "__model_class__" attribute.
    LookupError
        If the model class name is not present in the registry.
    TypeError
        If the resolved registry entry is not a BaseModel subclass.
    """
    # Convert the path to a proper path object and then
    # ensure that the path actually is a file and exists.
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No file found at: {p}")

    # Read the class name attribute, tolerating bytes/str and optional JSON encoding
    with h5py.File(p, "r") as f:
        try:
            model_class_name = json.loads(f.attrs["__model_class__"])
        except KeyError as e:
            raise ValueError("Missing '__model_class__' attribute in HDF5 file.") from e

    # Load the model class from the registry.
    reg = registry if registry is not None else __default_model_registry__

    try:
        model_cls = reg[model_class_name]
    except KeyError as e:
        raise LookupError(f"Model class '{model_class_name}' not found in registry.") from e

    return model_cls(p)


def inspect_model_metadata(
    path: Union[str, Path],
    registry: Optional["Registry"] = None,
) -> dict:
    """
    Inspect a model file's metadata without fully loading the model.

    This function opens an HDF5 model file, reads the ``__model_class__`` attribute
    to determine the associated model class, and uses that class's
    ``metadata_serializer`` to parse the file's ``.attrs`` into a metadata
    dictionary.

    Parameters
    ----------
    path : str or ~pathlib.Path
        Path to the HDF5 file containing the model.
    registry : Registry, optional
        Registry to use for resolving the model class. If omitted, the default
        Pisces model registry will be used.

    Returns
    -------
    dict
        A dictionary of metadata for the model as parsed by its
        ``metadata_serializer``.

    Raises
    ------
    FileNotFoundError
        If the file does not exist or is not a file.
    ValueError
        If the file is missing the ``__model_class__`` attribute.
    LookupError
        If the model class name is not found in the registry.
    AttributeError
        If the resolved class does not define ``metadata_serializer``.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No file found at: {p}")

    with h5py.File(p, "r") as f:
        # --- Get model class name ---
        try:
            model_class_name = json.loads(f.attrs["__model_class__"])
        except KeyError as e:
            raise ValueError("Missing '__model_class__' attribute in HDF5 file.") from e

        # --- Resolve model class ---
        reg = registry if registry is not None else __default_model_registry__
        try:
            model_cls = reg[model_class_name]
        except KeyError as e:
            raise LookupError(f"Model class '{model_class_name}' not found in registry.") from e

        # --- Ensure it has a metadata serializer ---
        if not hasattr(model_cls, "metadata_serializer"):
            raise AttributeError(f"Model class '{model_cls.__name__}' does not define a 'metadata_serializer'.")

        # --- Deserialize metadata from attrs ---
        serializer = model_cls.metadata_serializer
        metadata = serializer.deserialize_dict(dict(f.attrs))

    return metadata


def inspect_model_fields(
    path: Union[str, Path],
) -> dict:
    """
    Inspect the names and shapes of fields in a model file without fully loading the model.

    This function opens an HDF5 model file, navigates to the top-level ``FIELDS`` group,
    and extracts the dataset names along with their shapes. It does not load the actual
    field data into memory.

    Parameters
    ----------
    path : str or ~pathlib.Path
        Path to the HDF5 file containing the model.

    Returns
    -------
    dict
        A mapping of field names (strings) to shapes (tuples of ints) for each
        dataset in the ``FIELDS`` group.

    Raises
    ------
    FileNotFoundError
        If the file does not exist or is not a file.
    KeyError
        If the file does not contain a top-level ``FIELDS`` group.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No file found at: {p}")

    with h5py.File(p, "r") as f:
        if "FIELDS" not in f:
            raise KeyError("No 'FIELDS' group found in the model file.")

        fields_group = f["FIELDS"]
        field_info = {}
        for field_name, dataset in fields_group.items():
            if isinstance(dataset, h5py.Dataset):
                field_info[field_name] = tuple(dataset.shape)
            else:
                # Could be a subgroup containing component datasets
                field_info[field_name] = {
                    comp_name: tuple(comp.shape)
                    for comp_name, comp in dataset.items()
                    if isinstance(comp, h5py.Dataset)
                }

    return field_info


def inspect_model_grid(
    path: Union[str, Path],
    registry: Optional["Registry"] = None,
):
    """
    Load the grid from a model file without fully instantiating the model.

    This function opens the model's HDF5 file, navigates to the ``/GRID`` group,
    and uses :func:`~pisces.models.core.grid.load_grid` to create and return
    the grid object directly.

    Parameters
    ----------
    path : str or ~pathlib.Path
        Path to the HDF5 file containing the model.
    registry : Registry, optional
        Registry mapping grid class names to their corresponding classes.
        If None, the default Pisces grid registry will be used.

    Returns
    -------
    ~pisces.models.core.grid.Grid
        The grid instance loaded from the model file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist or is not a file.
    KeyError
        If the file does not contain a ``/GRID`` group.
    """
    from pisces._registries import __default_grid_registry__
    from pisces.geometry.grids.utils import load_grid

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No file found at: {p}")

    reg = registry if registry is not None else __default_grid_registry__
    return load_grid(p, "/GRID", registry=reg)


def inspect_model_coordinate_system(
    path: Union[str, Path],
    registry: Optional["Registry"] = None,
):
    """
    Inspect the coordinate system of a model without fully instantiating the model.

    This function:

      1. Opens the model's HDF5 file.
      2. Loads the ``/GRID`` group using :func:`~pisces.models.core.grid.load_grid`.
      3. Returns the grid's :attr:`coordinate_system` object.

    Parameters
    ----------
    path : str or ~pathlib.Path
        Path to the HDF5 file containing the model.
    registry : Registry, optional
        Registry mapping grid class names to their corresponding classes.
        If None, the default Pisces grid registry will be used.

    Returns
    -------
    ~pisces.geometry.coordinate_systems.base.CoordinateSystem
        The coordinate system instance associated with the model's grid.

    Raises
    ------
    FileNotFoundError
        If the file does not exist or is not a file.
    KeyError
        If the file does not contain a ``/GRID`` group.
    AttributeError
        If the loaded grid does not define a ``coordinate_system`` attribute.
    """
    from pisces._registries import __default_grid_registry__
    from pisces.geometry.grids.utils import load_grid

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No file found at: {p}")

    reg = registry if registry is not None else __default_grid_registry__
    grid = load_grid(p, "/GRID", registry=reg)

    if not hasattr(grid, "coordinate_system"):
        raise AttributeError(f"Grid loaded from '{p}' has no 'coordinate_system' attribute.")

    return grid.coordinate_system
