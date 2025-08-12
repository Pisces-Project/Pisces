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
