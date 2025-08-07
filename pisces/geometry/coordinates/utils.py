"""Utilities for working with Pisces coordinate systems."""

import json
from typing import TYPE_CHECKING

from pisces._registries import __default_coordinate_registry__

if TYPE_CHECKING:
    from pisces._generic import Registry

    from .base import CoordinateSystem


def load_coordinate_dict(data: dict, registry: "Registry" = None) -> "CoordinateSystem":
    """
    Load a coordinate system from a dictionary using the registry.

    Parameters
    ----------
    data : dict
        A dictionary with the following structure:

        .. code-block:: python

            {
                "class": "<CoordinateSystemClassName>",
                "parameters": {
                    ...
                },  # Optional keyword arguments
            }

    registry : Registry, optional
        The registry to use to resolve the coordinate system class.
        Defaults to the Pisces default registry.

    Returns
    -------
    CoordinateSystem
        An instance of the requested coordinate system.

    Raises
    ------
    KeyError
        If the class name is missing or not found in the registry.
    TypeError
        If instantiation with the given parameters fails.
    """
    if "class" not in data:
        raise KeyError("Missing required key 'class' in input dictionary.")

    cls_name = data["class"]
    parameters = data.get("parameters", {})
    registry = registry or __default_coordinate_registry__

    try:
        cls = registry.get(cls_name)
    except KeyError as e:
        raise KeyError(f"Coordinate system '{cls_name}' not found in registry.") from e

    try:
        return cls(**parameters)
    except Exception as e:
        raise TypeError(f"Failed to instantiate '{cls_name}' with parameters {parameters}: {e}") from e


def load_coordinate_string(json_string: str, registry: "Registry" = None) -> "CoordinateSystem":
    """
    Load a coordinate system from a JSON string.

    Parameters
    ----------
    json_string : str
        A JSON string with the structure:

        .. code-block:: json

            {
                "class": "SphericalCoordinateSystem",
                "parameters": { ... }
            }

    registry : Registry, optional
        Registry to resolve the coordinate system class.

    Returns
    -------
    CoordinateSystem
        An instance of the requested coordinate system.

    Raises
    ------
    ValueError
        If the JSON is invalid.
    See load_coordinate_dict for additional exceptions.
    """
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    return load_coordinate_dict(data, registry=registry)


def write_coordinate_system_to_dict(cs: "CoordinateSystem") -> dict:
    """
    Serialize a coordinate system to a dictionary.

    Parameters
    ----------
    cs : CoordinateSystem

    Returns
    -------
    dict
        A dictionary with keys "class" and "parameters".
    """
    return {"class": cs.__class__.__name__, "parameters": cs.parameters}


def write_coordinate_system_to_json(cs: "CoordinateSystem", indent: int = None) -> str:
    """
    Serialize a coordinate system to a JSON string.

    Parameters
    ----------
    cs : CoordinateSystem
    indent : int, optional
        Indent level for pretty-printed JSON.

    Returns
    -------
    str
    """
    return json.dumps(write_coordinate_system_to_dict(cs), indent=indent)


def list_registered_coordinate_systems(registry: "Registry" = None) -> list[str]:
    """
    List all registered coordinate system class names.

    Parameters
    ----------
    registry : Registry, optional

    Returns
    -------
    list of str
    """
    registry = registry or __default_coordinate_registry__
    return sorted(registry.keys())


def get_coordinate_system_class(name: str, registry: "Registry" = None) -> type:
    """
    Retrieve a coordinate system class from the registry.

    Parameters
    ----------
    name : str
        Class name
    registry : Registry, optional

    Returns
    -------
    type
        Coordinate system class
    """
    registry = registry or __default_coordinate_registry__
    return registry.get(name)


def is_coordinate_system_instance(obj: object) -> bool:
    """
    Check if an object is an instance of CoordinateSystem.

    Parameters
    ----------
    obj : object

    Returns
    -------
    bool
    """
    from .base import CoordinateSystem

    return isinstance(obj, CoordinateSystem)


def convert_coords_between_systems(source: "CoordinateSystem", target: "CoordinateSystem", *coords) -> tuple:
    """
    Convert coordinates from one system to another.

    Parameters
    ----------
    source : CoordinateSystem
        The source coordinate system.
    target : CoordinateSystem
        The target coordinate system.
    coords : tuple of array-like
        Coordinates in the source system.

    Returns
    -------
    tuple of array-like
        Coordinates in the target system.
    """
    return target.convert_coords_from(source, *coords)
