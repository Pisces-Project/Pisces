"""IO utilities for Pisces.

This module contains a number of helpful IO operations for Pisces which are used
frequently in various parts of the project.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import unyt
from ruamel.yaml.comments import CommentedMap

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
    def _to_jsonable(cls, obj: Any, _path="root"):
        """Recursively convert `obj` into a JSON-serializable Python structure.

        Notes
        -----
        - This returns only JSON-native container/scalar
          types: dict, list, str, int, float, bool, None.
        - Custom types registered in `__REGISTRY__` are encoded
          as dicts with a `"tag"` plus payload.
        - Strings are treated as primitives (never iterated like sequences).
        - Actual JSON encoding (quoting/escaping) is done by `json.dumps` elsewhere.
        """
        # Start by processing matches to custom (registered) serialization types.
        # If the object is an instance of any registered type, serialize it and inject a "tag".
        for _type, (tag, serializer, _) in cls.__REGISTRY__.items():
            if isinstance(obj, _type):
                payload = serializer(obj)
                if not isinstance(payload, dict):
                    raise ValueError(f"{_path}: custom serializer for {type(obj)} must return a dict")
                # Merge the payload with a required "tag" to identify the type on load
                return {"tag": tag, **payload}

        # These are directly JSON-serializable and don't need any transformation.
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj

        # Convert numpy scalars to Python scalars, and numpy ndarrays to nested lists.
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # Now start handling instances where recursive checking becomes necessary.
        # HDF5 attribute keys must be strings, so enforce that here.
        if isinstance(obj, dict):
            out = {}
            for key, value in obj.items():
                if not isinstance(key, str):
                    raise TypeError(f"{_path}: HDF5 attribute keys must be strings; got key type {type(key)}")
                # Recurse into each value; add a breadcrumb path for helpful error messages
                out[key] = cls._to_jsonable(value, _path=f"{_path}.{key}")
            return out

        # Convert any non-string, non-bytes sequence to a plain JSON list.
        if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            return [cls._to_jsonable(v, _path=f"{_path}[{i}]") for i, v in enumerate(obj)]

        # Store small binary blobs as base64-encoded strings with a distinct tag for round-tripping.
        if isinstance(obj, (bytes, bytearray)):
            import base64

            b64 = base64.b64encode(bytes(obj)).decode("ascii")
            return {"tag": "bytes_b64", "value": b64}

        # Anything else can't be represented in JSON without a custom serializer.
        raise ValueError(f"{_path}: cannot serialize object of unsupported type {type(obj)}")

    @classmethod
    def _from_jsonable(cls, obj, _path="root"):
        """Recursively reconstruct Python objects from JSON-serializable structures.

        Mirrors `_to_jsonable`:
        - Dicts with a `"tag"` key are dispatched to a registered deserializer,
          or handled specially (e.g., `"bytes_b64"`).
        - Plain dicts/lists are recursively walked.
        - Primitives (None/bool/int/float/str) pass through unchanged.
        """
        # ---- 1) Tagged custom objects ------------------------------------------------------------
        # Objects serialized by registry serializers include a "tag" key.
        if isinstance(obj, dict) and "tag" in obj:
            tag = obj["tag"]

            # Try custom deserializers from the registry first
            for _type, (reg_tag, _, deserializer) in cls.__REGISTRY__.items():  # noqa: PERF102
                if tag == reg_tag:
                    # Pass the full dict to the registered deserializer
                    return deserializer(obj)

            # Built-in special tag for base64-encoded bytes
            if tag == "bytes_b64":
                import base64

                try:
                    return base64.b64decode(obj["value"])
                except Exception as exp:
                    raise ValueError(f"{_path}: failed to decode base64 bytes") from exp

            # Unknown tag
            raise ValueError(f"{_path}: unrecognized serialization tag '{tag}'")

        # ---- 2) Plain mapping → dict (recurse over values) --------------------------------------
        # Keys are expected to be strings (enforced on the serialize path).
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                # We do not enforce key type here; serialize path already did.
                out[k] = cls._from_jsonable(v, _path=f"{_path}.{k}")
            return out

        # ---- 3) JSON list → Python list (recurse) ------------------------------------------------
        if isinstance(obj, list):
            return [cls._from_jsonable(v, _path=f"{_path}[]") for v in obj]

        # ---- 4) Primitives (None/bool/int/float/str) --------------------------------------------
        # These are already JSON-native; return as-is.
        return obj

    # -------- public API (updated) --------
    @classmethod
    def serialize_data(cls, data: Any) -> str:
        """Recursively serialize `data` to a JSON string for HDF5 attributes."""
        try:
            jsonable = cls._to_jsonable(data)
            return json.dumps(jsonable)
        except Exception as exp:
            raise ValueError(f"Failed to serialize data (type={type(data)}): {exp}") from exp

    @classmethod
    def deserialize_data(cls, data: str) -> Any:
        """Recursively deserialize a JSON string produced by `serialize_data`."""
        try:
            parsed = json.loads(data)
        except Exception as exp:
            raise ValueError(f"Failed to parse JSON string: {exp}") from exp
        return cls._from_jsonable(parsed)

    @classmethod
    def serialize_dict(cls, data: dict) -> dict:
        """Recursively serialize a dict’s values (keys must be strings)."""
        if not isinstance(data, dict):
            raise TypeError("serialize_dict expects a dict")
        return {k: cls.serialize_data(v) for k, v in data.items()}

    @classmethod
    def deserialize_dict(cls, data: dict) -> dict:
        """Recursively deserialize a dict produced by `serialize_dict`."""
        if not isinstance(data, dict):
            raise TypeError("deserialize_dict expects a dict")
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
            data = CommentedMap()
            loader.construct_mapping(node, maptyp=data,  deep=True)
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
        yaml.representer.add_multi_representer(cls.__type__, cls.to_yaml)
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
        data = CommentedMap()
        loader.construct_mapping(node, maptyp=data, deep=True)
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
        data = CommentedMap()
        loader.construct_mapping(node, maptyp=data, deep=True)
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


class NumpyArrayHandler(_YAMLHandler):
    """NumPy array handler for YAML serialization/deserialization."""

    __tag__ = "!ndarray"
    __type__ = np.ndarray

    @staticmethod
    def to_yaml(representer, obj: np.ndarray):
        # We store both dtype and shape to ensure safe reconstruction
        return representer.represent_mapping(
            NumpyArrayHandler.__tag__, {"dtype": str(obj.dtype), "shape": obj.shape, "data": obj.tolist()}
        )

    @staticmethod
    def from_yaml(loader, node):
        data = CommentedMap()
        loader.construct_mapping(node, maptyp=data, deep=True)

        # Validate required fields
        if not all(k in data for k in ("dtype", "shape", "data")):
            raise ValueError(f"Invalid ndarray YAML mapping: {data}")

        arr = np.array(data["data"], dtype=np.dtype(data["dtype"]))

        # Optionally enforce shape
        if tuple(arr.shape) != tuple(data["shape"]):
            try:
                arr = arr.reshape(data["shape"])
            except Exception as e:
                raise ValueError(f"Shape mismatch when reconstructing ndarray: {e}") from e

        return arr


class PathHandler(_YAMLHandler):
    """Path handler for YAML serialization/deserialization."""

    __tag__ = "!path"
    __type__ = Path

    @staticmethod
    def to_yaml(representer, obj: Path):
        data = {"absolute": obj.is_absolute(), "parts": list(obj.parts)}
        if obj.drive:
            data["drive"] = obj.drive
        return representer.represent_mapping(PathHandler.__tag__, data)

    @staticmethod
    def from_yaml(loader, node):
        data = CommentedMap()
        loader.construct_mapping(node, maptyp=data, deep=True)
        parts = data.get("parts", [])
        path = Path(*parts)
        if data.get("absolute", False) and not path.is_absolute():
            path = path.resolve()
        return path


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
    PathHandler.register(yaml)
    NumpyArrayHandler.register(yaml)

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
