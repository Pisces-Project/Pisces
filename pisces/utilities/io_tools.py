"""IO utilities for Pisces.

This module contains a number of helpful IO operations for Pisces which are used
frequently in various parts of the project.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Union

import unyt

if TYPE_CHECKING:
    from ruamel.yaml import YAML
    from ruamel.yaml.constructor import Constructor
    from ruamel.yaml.nodes import Node
    from ruamel.yaml.representer import Representer


# ----------------------------------------------- #
# HDF5 File Management and Serialization Tools    #
# ----------------------------------------------- #
# This module provides a JSON-based serialization utility for storing complex
# Python objects as HDF5 attributes. It supports custom types like `unyt` arrays,
# quantities, and units, allowing them to be serialized to JSON strings that can
# be safely stored as attributes in HDF5 files.
class HDF5Serializer:
    """
    A JSON-based serialization utility for storing complex Python objects as HDF5 attributes.

    This class provides a standardized mechanism for serializing and
    deserializing Python objects — including extended types like `unyt` arrays,
    quantities, and units — to and from JSON strings that can be safely stored
    as attributes in HDF5 files.

    Custom types are registered via the `__REGISTRY__` class variable, which maps
    Python types to a tuple of:
        - A unique string tag identifying the type in serialized form,
        - A serialization function (returning a JSON-serializable dict),
        - A deserialization function (taking that dict and reconstructing the object).

    Built-in JSON-compatible types (e.g., int, float, str, list, dict) are serialized
    directly via `json.dumps`. All deserialized outputs are reconstructed either
    via the registry or returned as-is for base types.

    This class is intended to be subclassed or extended to support additional
    custom types.
    """

    # ----------------------------------------------- #
    # Serialization Methods                           #
    # ----------------------------------------------- #
    @staticmethod
    def serialize_unyt(o: Union[unyt.unyt_array, unyt.unyt_quantity]) -> dict:
        """
        Serialize a unyt object into a dictionary.

        Parameters
        ----------
        o : unyt.unyt_array or unyt.unyt_quantity
            The unyt object to serialize.

        Returns
        -------
        dict
            A dictionary representation of the unyt object.
        """
        if isinstance(o, unyt.unyt_array):
            return {"value": o.value.tolist(), "units": str(o.units)}
        elif isinstance(o, unyt.unyt_quantity):
            return {"value": o.value, "units": str(o.units)}
        else:
            raise TypeError(
                f"Unsupported type for serialization: {type(o)}. Expected unyt.unyt_array or unyt.unyt_quantity."
            )

    @staticmethod
    def serialize_unyt_unit(o: unyt.Unit) -> dict:
        """
        Serialize a unyt unit into a dictionary.

        Parameters
        ----------
        o : unyt.Unit
            The unyt unit to serialize.

        Returns
        -------
        dict
            A dictionary representation of the unyt unit.
        """
        return {"value": str(o)}

    # ----------------------------------------------- #
    # Deserialization Methods                         #
    # ----------------------------------------------- #
    @staticmethod
    def deserialize_unyt_unit(o: dict) -> unyt.Unit:
        """
        Deserialize a dictionary into a unyt unit.

        Parameters
        ----------
        o : dict
            The dictionary to deserialize.

        Returns
        -------
        unyt.Unit
            A unyt unit object.
        """
        return unyt.Unit(o["value"])

    @staticmethod
    def deserialize_unyt(o: dict) -> Union[unyt.unyt_array, unyt.unyt_quantity]:
        """
        Deserialize a dictionary into a unyt object.

        Parameters
        ----------
        o : dict
            The dictionary to deserialize.

        Returns
        -------
        unyt.unyt_array or unyt.unyt_quantity
            A unyt object.
        """
        if "units" not in o:
            raise ValueError("Missing 'units' key in the dictionary for deserialization.")

        units = unyt.Unit(o["units"])

        if "value" in o and isinstance(o["value"], list):
            return unyt.unyt_array(o["value"], units)
        elif "value" in o:
            return unyt.unyt_quantity(o["value"], units)
        else:
            raise ValueError("Missing 'value' key in the dictionary for deserialization.")

    # ----------------------------------------------- #
    # Class Variables / Registry                      #
    # ----------------------------------------------- #
    __REGISTRY__: dict[type, tuple[str, Callable[[Any], dict], Callable[[dict], Any]]] = {
        unyt.unyt_quantity: ("unyt_quantity", serialize_unyt, deserialize_unyt),
        unyt.unyt_array: ("unyt_array", serialize_unyt, deserialize_unyt),
        unyt.Unit: ("unyt_unit", serialize_unyt_unit, deserialize_unyt_unit),
    }

    @classmethod
    def serialize_data(cls, data: Any) -> str:
        """
        Serialize a Python object into a JSON string suitable for storing as an HDF5 attribute.

        This method first checks if the input object matches any of the types
        registered in the class's `__REGISTRY__`. If a match is found, the
        corresponding custom serializer is used to convert the object into a
        JSON-serializable dictionary, and a `"tag"` field is added to identify
        the object's type for deserialization.

        If no custom serializer is found, the method attempts to serialize the
        object directly using the default `json.dumps()` behavior. If this fails,
        an error is raised, indicating that the data requires a custom serializer.

        Parameters
        ----------
        data : Any
            The Python object to serialize.

        Returns
        -------
        str
            A JSON string representing the serialized object.

        Raises
        ------
        ValueError
            If the object cannot be serialized to JSON.
        """
        # Search for a match to the data type to see
        # if we have a custom serializer for it.
        for _type, (tag, serializer, _) in cls.__REGISTRY__.items():
            if isinstance(data, _type):
                # We found a serializer for this type. We just
                # dump from the serializer and return it.
                return json.dumps({"tag": tag, **serializer(data)})

        # We didn't find a custom serializer, so we now
        # need to just load the data from json. If this fails,
        # its because we failed to catch data that needed to be
        # serialized, so we raise an error.
        try:
            return json.dumps(data)
        except Exception as exp:
            raise ValueError(f"Failed to serialize data to JSON: {exp}") from exp

    @classmethod
    def deserialize_data(cls, data: str) -> Any:
        """
        Deserialize a JSON string into a Python object, using registered custom deserializers if necessary.

        This method expects a JSON-encoded string, which may either represent
        a basic JSON-compatible type (e.g., int, float, str, list, dict) or a
        custom-serialized object with an embedded `"tag"` field. If a `"tag"` is
        present, it is used to identify and invoke a registered deserializer for
        the corresponding object type.

        If no `"tag"` is present, the method assumes the data represents a
        standard JSON type and returns the parsed object directly.

        Parameters
        ----------
        data : str
            A JSON-encoded string representing the serialized object.

        Returns
        -------
        Any
            The deserialized Python object.

        Raises
        ------
        ValueError
            If the input is not valid JSON, or if a tag is provided but
            no matching deserializer is found in the registry.
        """
        # Begin by deserializing the JSON string for the data.
        # Everything should be a JSON string, so we can use
        # the json.loads() method to parse it. We'll still
        # need to deserialize after that.
        try:
            parsed = json.loads(data)
        except Exception as exp:
            raise ValueError(f"Failed to parse JSON string: {exp}") from exp

        # With the loaded json string, we need to check for a
        # tag and figure out if we have a custom deserializer
        # for the tag.
        if isinstance(parsed, dict) and "tag" in parsed:
            tag = parsed["tag"]

            for _type, (reg_tag, _, deserializer) in cls.__REGISTRY__.items():  # noqa: PERF102
                if tag == reg_tag:
                    # We have a matching tag, so we can
                    # deserialize the data using the registered deserializer.
                    return deserializer(parsed)

            # There is no matching deserializer for the tag, so
            # we need to raise an error.
            raise ValueError(f"Unrecognized serialization tag: {tag}")

        return parsed  # base types: int, float, list, dict, etc.

    @classmethod
    def serialize_dict(cls, data: dict) -> dict:
        """
        Serialize a dictionary, converting any unyt objects to JSON-compatible formats.

        Parameters
        ----------
        data : dict
            The dictionary to serialize.

        Returns
        -------
        dict
            A new dictionary with unyt objects serialized.
        """
        return {k: cls.serialize_data(v) for k, v in data.items()}

    @classmethod
    def deserialize_dict(cls, data: dict) -> dict:
        """
        Deserialize a dictionary, converting JSON-compatible formats back to unyt objects.

        Parameters
        ----------
        data : dict
            The dictionary to deserialize.

        Returns
        -------
        dict
            A new dictionary with unyt objects deserialized.
        """
        return {k: cls.deserialize_data(v) for k, v in data.items()}


# ----------------------------------------------- #
# YAML Reader/Writer tools                        #
# ----------------------------------------------- #
class _YAMLHandler(ABC):
    """
    Abstract base class for defining custom YAML representers and constructors.

    Subclasses must define:
        - `__tag__`: a YAML tag string (e.g., "!unyt_quantity")
        - `__type__`: the Python type to associate with this handler
        - `to_yaml()`: a static method to serialize the Python object to a YAML node
        - `from_yaml()`: a static method to deserialize a YAML node to a Python object

    This class provides a consistent interface for registering type-specific
    (de)serialization logic with a `ruamel.yaml.YAML` instance.

    Example
    -------
    class MyTypeHandler(_YAMLHandler):
        __tag__ = "!my_type"
        __type__ = MyType

        @staticmethod
        def to_yaml(representer, obj):
            return representer.represent_mapping(MyTypeHandler.__tag__, {
                "x": obj.x,
                "y": obj.y
            })

        @staticmethod
        def from_yaml(loader, node):
            data = loader.construct_mapping(node, deep=True)
            return MyType(data["x"], data["y"])

    yaml = YAML()
    MyTypeHandler.register(yaml)
    """

    __tag__: str = None
    __type__: type = None

    @staticmethod
    @abstractmethod
    def to_yaml(representer: "Representer", obj: Any) -> "Node":
        """
        Convert a Python object to a YAML node.

        Parameters
        ----------
        representer : ruamel.yaml.representer.Representer
            The YAML representer to use for creating the node.
        obj : Any
            The Python object to serialize.

        Returns
        -------
        ruamel.yaml.nodes.Node
            A YAML node representing the serialized object.
        """
        pass

    @staticmethod
    @abstractmethod
    def from_yaml(loader: "Constructor", node: "Node") -> Any:
        """
        Convert a YAML node to a Python object.

        Parameters
        ----------
        loader : ruamel.yaml.constructor.Constructor
            The YAML loader to use for constructing the Python object.
        node : ruamel.yaml.nodes.Node
            The YAML node to deserialize.

        Returns
        -------
        Any
            The deserialized Python object.
        """
        pass

    @classmethod
    def register(cls, yaml: "YAML") -> None:
        """
        Register this handler's representer and constructor with a YAML instance.

        Parameters
        ----------
        yaml : ruamel.yaml.YAML
            The YAML instance to register the handler with.
        """
        if cls.__tag__ is None or cls.__type__ is None:
            raise ValueError(f"{cls.__name__} must define both __tag__ and __type__.")
        yaml.representer.add_representer(cls.__type__, cls.to_yaml)
        yaml.constructor.add_constructor(cls.__tag__, cls.from_yaml)


class UnytArrayHandler(_YAMLHandler):
    """Unyt array handler for YAML serialization/deserialization."""

    __tag__ = "!unyt_array"
    __type__ = unyt.unyt_array

    @staticmethod
    def to_yaml(representer, obj):
        return representer.represent_mapping(
            UnytArrayHandler.__tag__, {"value": obj.d.tolist(), "units": str(obj.units)}
        )

    @staticmethod
    def from_yaml(loader, node):
        data = loader.construct_mapping(node, deep=True)
        return unyt.unyt_array(data["value"], data["units"])


class UnytQuantityHandler(_YAMLHandler):
    """Unyt quantity handler for YAML serialization/deserialization."""

    __tag__ = "!unyt_quantity"
    __type__ = unyt.unyt_quantity

    @staticmethod
    def to_yaml(representer, obj):
        return representer.represent_mapping(
            UnytQuantityHandler.__tag__, {"value": obj.value.item(), "units": str(obj.units)}
        )

    @staticmethod
    def from_yaml(loader, node):
        data = loader.construct_mapping(node, deep=True)
        return unyt.unyt_quantity(data["value"], data["units"])


class UnytUnitHandler(_YAMLHandler):
    """Unyt unit handler for YAML serialization/deserialization."""

    __tag__ = "!unyt_unit"
    __type__ = unyt.Unit

    @staticmethod
    def to_yaml(representer, obj):
        return representer.represent_scalar(UnytUnitHandler.__tag__, str(obj))

    @staticmethod
    def from_yaml(loader, node):
        value = loader.construct_scalar(node)
        return unyt.Unit(value)


def get_unyt_compatible_yaml() -> "YAML":
    """
    Get a YAML instance configured for unyt compatibility.

    This function creates a `ruamel.yaml.YAML` instance and registers
    custom representers and constructors for unyt types.

    Returns
    -------
    ruamel.yaml.YAML
        A YAML instance with unyt support.
    """
    from ruamel.yaml import YAML

    yaml = YAML(typ="rt")
    UnytArrayHandler.register(yaml)
    UnytQuantityHandler.register(yaml)
    UnytUnitHandler.register(yaml)

    return yaml


def get_default_yaml() -> "YAML":
    """
    Get a default YAML instance without unyt support.

    This function creates a `ruamel.yaml.YAML` instance with no custom
    representers or constructors registered.

    Returns
    -------
    ruamel.yaml.YAML
        A default YAML instance.
    """
    from ruamel.yaml import YAML

    return YAML(typ="rt")


unyt_yaml = get_unyt_compatible_yaml()
"""~ruamel.yaml.YAML: A YAML instance configured for unyt compatibility."""
