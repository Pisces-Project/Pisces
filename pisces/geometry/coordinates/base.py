"""Base classes for configuring coordinate systems in Pisces."""

import json
from abc import ABC, ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Union

from pisces._registries import __default_coordinate_registry__

if TYPE_CHECKING:
    from pisces._generic import Registry


class _CoordinateSystemMeta(ABCMeta):
    """
    Metaclass for all coordinate systems.

    Automatically registers non-abstract coordinate system classes into the default registry.
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        # Create the class object using the base metaclass
        cls_object: type[CoordinateSystem] = super().__new__(mcs, name, bases, namespace, **kwargs)

        # If the class is not abstract, register it to the default registry
        if not cls_object.__IS_ABSTRACT__:
            try:
                cls_object.__DEFAULT_REGISTRY__.register(cls_object.__name__, cls_object)
            except Exception as exp:
                raise TypeError(f"Failed to register coordinate system {cls_object.__name__}: {exp}") from exp

        return cls_object


class CoordinateSystem(ABC, metaclass=_CoordinateSystemMeta):
    """
    Base class for all Pisces coordinate systems.

    The :class:`CoordinateSystem` provides a common interface for
    defining and interacting with coordinate systems including converting
    from one coordinate system to another.
    """

    # =============================== #
    # CLASS FLAGS / CONFIG            #
    # =============================== #
    # CoordinateSystem flags are used to indicate to the metaclass whether
    # certain procedures should be executed on the class.
    __DEFAULT_REGISTRY__: "Registry" = __default_coordinate_registry__
    __IS_ABSTRACT__: bool = True

    # =============================== #
    # PARAMETERS                      #
    # =============================== #
    # This is where the coordinate system defines its
    # default parameters and their values.
    __PARAMETERS__: dict[str, Union[float, int]] = {}
    __NDIM__: int = 0

    # =============================== #
    # INITIALIZATION                  #
    # =============================== #
    def __init__(self, **parameters):
        """Initialize the coordinate system with validated parameters."""
        self._parameters = self._validate_parameters(**parameters)

    def _validate_parameters(self, **parameters) -> dict[str, Union[float, int]]:
        """Validate input parameters against the class defaults."""
        validated = self.__class__.__PARAMETERS__.copy()
        for key, value in parameters.items():
            if key not in validated:
                raise TypeError(f"Unknown parameter '{key}' in {self}")
            validated[key] = value
        return validated

    # =============================== #
    # DUNDER METHODS / PROPERTIES     #
    # =============================== #
    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v!r}" for k, v in self._parameters.items())
        return f"{self.__class__.__name__}({param_str})"

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and self._parameters == other._parameters

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, tuple(sorted(self._parameters.items()))))

    @property
    def parameters(self) -> dict[str, Union[float, int]]:
        """Return a copy of the coordinate system's parameters."""
        return self._parameters.copy()

    @property
    def ndim(self) -> int:
        """Return the number of dimensions of this coordinate system."""
        return self.__class__.__NDIM__

    def copy(self) -> "CoordinateSystem":
        """Return a shallow copy of the coordinate system."""
        return self.__class__(**self._parameters)

    @classmethod
    def default_parameters(cls) -> dict[str, Union[float, int]]:
        """Return a copy of the default parameters for this coordinate system."""
        return cls.__PARAMETERS__.copy()

    # =============================== #
    # COORDINATE CONVERSION           #
    # =============================== #
    @abstractmethod
    def convert_to_cartesian(self, *coordinates) -> tuple[Any, ...]:
        """
        Convert coordinates from this system to Cartesian.

        Parameters
        ----------
        coordinates : tuple of array-like
            The coordinates in this system.

        Returns
        -------
        tuple of array-like
            The corresponding Cartesian coordinates.
        """
        pass

    @abstractmethod
    def convert_from_cartesian(self, *coordinates) -> tuple[Any, ...]:
        """
        Convert coordinates from Cartesian to this system.

        Parameters
        ----------
        coordinates : tuple of array-like
            The Cartesian coordinates.

        Returns
        -------
        tuple of array-like
            The corresponding coordinates in this system.
        """
        pass

    def convert_coords_from(self, other: "CoordinateSystem", *coordinates) -> tuple[Any, ...]:
        """
        Convert coordinates from another coordinate system into this system.

        Parameters
        ----------
        other : CoordinateSystem
            The coordinate system from which the input coordinates originate.
        coordinates : tuple of array-like
            The coordinates in the `other` system.

        Returns
        -------
        tuple of array-like
            The coordinates expressed in this system.
        """
        cartesian = other.convert_to_cartesian(*coordinates)
        return self.convert_from_cartesian(*cartesian)

    def convert_coords_to(self, other: "CoordinateSystem", *coordinates) -> tuple[Any, ...]:
        """
        Convert coordinates from this system into another coordinate system.

        Parameters
        ----------
        other : CoordinateSystem
            The target coordinate system.
        coordinates : tuple of array-like
            The coordinates in this system.

        Returns
        -------
        tuple of array-like
            The coordinates expressed in the `other` system.
        """
        cartesian = self.convert_to_cartesian(*coordinates)
        return other.convert_from_cartesian(*cartesian)

    # =============================== #
    # SERIALIZATION                   #
    # =============================== #
    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the coordinate system, including its class name and parameters."""
        return {"class": self.__class__.__name__, "parameters": self._parameters}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CoordinateSystem":
        """
        Deserialize a coordinate system from a dictionary.

        Raises TypeError if the class name does not match.
        """
        if data.get("class") != cls.__name__:
            raise TypeError(f"Expected class '{cls.__name__}', got '{data.get('class')}'")
        return cls(**data.get("parameters", {}))

    def to_json_string(self) -> str:
        """Serialize the coordinate system to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json_string(cls, json_string: str) -> "CoordinateSystem":
        """Deserialize a coordinate system from a JSON string."""
        return cls.from_dict(json.loads(json_string))
