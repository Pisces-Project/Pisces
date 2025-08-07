"""
Base classes for grids in Pisces.

This module defines the abstract base class `Grid` for representing structured
and unstructured discretizations of coordinate systems in the Pisces geometry library.
"""

from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import h5py
import numpy as np
import unyt

from pisces._registries import __default_grid_registry__
from pisces.utilities.io_tools import HDF5Serializer

if TYPE_CHECKING:
    from pisces._generic import Registry
    from pisces.geometry.coordinates.base import CoordinateSystem


class _GridMeta(ABCMeta):
    """
    Metaclass for all grid types.

    Automatically registers non-abstract grid classes into the default registry.
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        # Create the class object using the base metaclass
        cls_object: type[Grid] = super().__new__(mcs, name, bases, namespace, **kwargs)

        # If the class is not abstract, register it to the default registry
        if not cls_object.__IS_ABSTRACT__:
            try:
                cls_object.__DEFAULT_REGISTRY__.register(cls_object.__name__, cls_object)
            except Exception as exp:
                raise TypeError(f"Failed to register grid class {cls_object.__name__}: {exp}") from exp

        return cls_object


class Grid(ABC):
    """
    Base class for all grids in Pisces.

    A grid represents a discretization of a coordinate system into
    a finite number of points or cells. This base class provides
    the common interface and functionality for all grid types, including
    structured and unstructured grids. For more details on grid structure,
    see the relevant documentation.
    """

    metadata_serializer: HDF5Serializer = HDF5Serializer
    """~pisces.utilities.io.HDF5Serializer: Serializer for model metadata.

    This serializer is used to read and write grid metadata to HDF5 files.
    """
    # ============================== #
    # CLASS FLAGS                    #
    # ============================== #
    __IS_ABSTRACT__: bool = True
    """Flag indicating that this is an abstract base class."""
    __DEFAULT_REGISTRY__: "Registry" = __default_grid_registry__
    """The default registry for grid types."""

    # ============================== #
    # INITIALIZATION                 #
    # ============================== #
    # These methods determine the behavior of the grid
    # vis-a-vis initialization. There are several methods placed
    # here which dictate specific elements of the initialization process.
    #
    # To subclass effectively, you will need to override
    # some (or all) of these methods to implement custom logic
    # for your grid type.

    def _configure_coordinate_system(self, coordinate_system: "CoordinateSystem"):
        """
        Configure the coordinate system's grid.

        This is the first step in the initialization of a grid and should
        simply set the ``self.__coordinate_system__`` attribute.
        """
        self.__coordinate_system__: CoordinateSystem = coordinate_system

    def _configure_axes_and_fills(self, axes: Union[str, tuple[str, ...]], fill_values: dict[str, Any] = None):
        """
        Configure the axes and fill values of the grid using axis labels.

        This allows users to select which axes of the coordinate system are
        "active" in the grid, and which are fixed using constant values.

        Sets
        ----
        __axes__ : tuple of str
            Names of the axes included in the grid. These must be valid
            axis labels from the coordinate system in question.

        __fill_values__ : dict of str, Any
            Fill values for the axes excluded from the grid.


        Parameters
        ----------
        axes : str or tuple of str
            The axes to include in the grid. This can be:

            - A tuple of axis names (e.g., ("r", "z")).
            - The string 'all' to include all axes from the coordinate system.

        fill_values : dict[str, Any], optional

            A dictionary mapping excluded axis names to constant fill values.

        Raises
        ------
        ValueError
            If axes are invalid or fill values are missing/misaligned.
        TypeError
            If inputs are the wrong types.
        """
        # Extract all of the axes from the coordinate
        # system so that we can validate the input.
        all_axes = self.__coordinate_system__.axes

        # Normalize input axes.
        if isinstance(axes, str):
            if axes.lower() == "all":
                axes_tuple = tuple(all_axes)
            else:
                raise ValueError(f"Invalid string for axes: {axes}. Must be 'all'.")
        elif isinstance(axes, (tuple, list)):
            if not all(isinstance(ax, str) for ax in axes):
                raise TypeError("All axis names must be strings.")
            axes_tuple = tuple(axes)
        else:
            raise TypeError(f"Axes must be a tuple of strings or 'all', got {type(axes)}.")

        # Validate axes are in the coordinate system
        invalid_axes = set(axes_tuple) - set(all_axes)
        if invalid_axes:
            raise ValueError(f"Unknown axis names {invalid_axes} not in coordinate system axes {all_axes}.")

        # Set the __axes__ attribute, ensuring consistent ordering
        # (sorted according to the order in all_axes).
        self.__axes__: tuple[str, ...] = tuple(sorted(set(axes_tuple), key=all_axes.index))

        # Determine fill axes (those not present in __axes__)
        expected_fill_axes = [ax for ax in all_axes if ax not in self.__axes__]

        # Validate and store fill values
        fill_values = fill_values or {}
        if not isinstance(fill_values, dict):
            raise TypeError("fill_values must be a dict mapping axis names (str) to values.")

        missing = set(expected_fill_axes) - set(fill_values.keys())
        extra = set(fill_values.keys()) - set(expected_fill_axes)
        if missing or extra:
            messages = []
            if missing:
                messages.append(f"Missing fill values for axes: {sorted(missing)}.")
            if extra:
                messages.append(f"Unexpected fill values for axes not being filled: {sorted(extra)}.")
            raise ValueError(" ".join(messages))

        # Set the __fill_values__ attribute.
        self.__fill_values__: dict[str, Any] = fill_values.copy()

    def _configure_units(self, units: Optional[dict[str, Union[str, unyt.Unit]]]):
        """
        Configure the units of the grid.

        This method sets the ``__units__`` attribute, a dictionary mapping each
        axis name to its corresponding unit.

        All axes, including both active and filled axes, must have a unit.
        If a unit is missing, it is filled with the dimensionless unit (i.e., "").

        Parameters
        ----------
        units : dict of str -> (str or unyt.Unit)
            A dictionary mapping axis names to unit strings or Unit objects.

        Raises
        ------
        ValueError
            If any axis in `units` is not in the coordinate system.
        TypeError
            If a unit is not a string or a `unyt.Unit` object.
        """
        from unyt import Unit

        units = units if units is not None else {}

        axis_names = self.__coordinate_system__.axes
        unit_dict: dict[str, unyt.Unit] = {}

        for axis in axis_names:
            value = units.get(axis, "")
            if isinstance(value, str):
                try:
                    unit_dict[axis] = Unit(value)
                except Exception as e:
                    raise ValueError(f"Invalid unit string for axis '{axis}': {value}") from e
            elif isinstance(value, Unit):
                unit_dict[axis] = value
            else:
                raise TypeError(f"Unit for axis '{axis}' must be a string or unyt.Unit, got {type(value)}.")

        self.__units__ = unit_dict

    @abstractmethod
    def _configure_grid_attributes(self, *args, **kwargs):
        """
        Configure grid-specific attributes.

        This method is called after the coordinate system, axes, fill values,
        and units have been configured. It has the following required responsibilities:

        - Set the ``__bbox__`` and ``__ddim__`` attributes so that we know what the
          bounding box and dimensionality of the grid are. These are each numpy arrays
          with shape (N, 2) and (N,) respectively, where N is the number of axes. The bounding
          box should be float valued with the first column being the minimum and the second
          column being the maximum for each axis. The dimensionality should be an integer
          array indicating the number of grid points along each axis.

          Only the active axes should appear in either of these attributes.

        Parameters
        ----------
        *args
            Positional arguments specific to the grid type.
        **kwargs
            Keyword arguments specific to the grid type.

        Raises
        ------
        NotImplementedError
            If not overridden in subclasses.
        """
        self.__bbox__: np.ndarray = None
        self.__ddim__: np.ndarray = None

        raise NotImplementedError(f"{self.__class__.__name__} must implement _configure_grid_attributes().")

    def __init__(
        self,
        coordinate_system: "CoordinateSystem",
        *args,
        axes: Union[str, tuple[str, ...]] = "all",
        fill_values: Optional[dict[str, Any]] = None,
        units: Optional[dict[str, Union[str, unyt.Unit]]] = None,
        **kwargs,
    ):
        # --- PRELIMINARY INITIALIZATION --- #
        # These steps in the initialization process are largely
        # about data coercion and just setting up the various
        # relevant attributes.

        # Begin by configuring the coordinate system. At this
        # stage, we call out to _configure_coordinate_system and
        # set the __coordinate_system__ attribute.
        self._configure_coordinate_system(coordinate_system)

        # With the coordinate system configured, we can now handle
        # the axes and the fill values.
        self._configure_axes_and_fills(
            axes,
            fill_values=fill_values,
        )

        # With the axes and the handles, we will want to
        # also have ``__axes_indices__`` and ``__axes_mask__``
        # as well as ``__fill_indices__`` and ``__fill_mask__``.
        self.__axes_indices__ = [self.__coordinate_system__.axes.index(ax) for ax in self.__axes__]
        self.__fill_indices__ = [
            self.__coordinate_system__.axes.index(ax)
            for ax in self.__coordinate_system__.axes
            if ax not in self.__axes__
        ]
        self.__axes_mask__ = np.zeros(self.__coordinate_system__.__NDIM__, dtype=bool)
        self.__axes_mask__[self.__axes_indices__] = True
        self.__fill_mask__ = ~self.__axes_mask__

        # Now we can handle the units. Once the units are in place,
        # we can proceed to configure the fill values so that we
        # don't retain any units.
        self._configure_units(units)

        for axis in self.__fill_values__:
            if hasattr(self.__fill_values__[axis], "units"):
                self.__fill_values__[axis] = self.__fill_values__[axis].to_value(self.__units__[axis])

        # --- GRID-SPECIFIC INITIALIZATION --- #
        # At this stage, we start performing configuration that
        # really depends on the nature of the underlying grid and
        # takes all of the args and kwargs.
        self._configure_grid_attributes(*args, **kwargs)

    # ============================== #
    # DUNDER METHODS                 #
    # ============================== #
    def __getitem__(self, key):
        """
        Get coordinates corresponding to the given grid index or mask.

        Parameters
        ----------
        key : slice, int, tuple, array-like, or boolean mask
            Indexing expression for selecting grid points.

        Returns
        -------
        np.ndarray

            If key selects a single point: coordinate tuple ``(x, y, ...)``.
            If key selects multiple points: array of shape ``(N, D)`` of coordinates.

        """
        # Handle ellipsis (e.g., grid[...]) so that
        # we can pass on to the other cases more easily.
        if key is Ellipsis:
            return self.get_meshgrid(indexing="ij")

        # Handle the case where the key is a masking array. This might either
        # be an array of indices or it might be a boolean mask. Either way,
        # we handle that here.
        if isinstance(key, np.ndarray):
            if key.dtype == bool:
                # Boolean mask
                if key.shape != self.shape:
                    raise ValueError(f"Boolean mask shape {key.shape} does not match grid shape {self.shape}")
                flat_coords = self.get_flat_coordinates()
                return flat_coords[key.ravel()]
            elif np.issubdtype(key.dtype, np.integer):
                # Index array
                mesh = self.get_meshgrid(indexing="ij")
                stacked = np.stack(mesh, axis=-1)
                return stacked[key]

        # Handle tuple indexing like grid[3, 5] or grid[:, 3]
        if isinstance(key, tuple):
            if all(isinstance(k, (int, slice)) for k in key):
                # Expand int -> slice to match grid shape
                slices = tuple(slice(k, k + 1) if isinstance(k, int) else k for k in key)
                coords = self.get_coordinates_slice(*slices)
                if all(isinstance(k, int) for k in key):
                    # All indices → return single coordinate tuple (x, y, ...)
                    return tuple(coord[0] for coord in coords)
                else:
                    # Mixed indexing → return meshgrid
                    return np.meshgrid(*coords, indexing="ij")
            else:
                raise TypeError(f"Unsupported tuple key types in {key}.")

        # Handle single int or slice
        if isinstance(key, (int, slice)):
            # Promote to full-dim tuple with rest as slices
            key = (key,) + (slice(None),) * (len(self.shape) - 1)
            return self.__getitem__(key)

        raise TypeError(f"Unsupported index type: {type(key)}")

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        axes = ", ".join(self.__axes__)
        shape = self.shape
        return f"<{cls_name} axes=({axes}), shape={shape}>"

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        return self.size

    def __iter__(self):
        """Iterate over all grid coordinate points."""
        for coord in self.get_flat_coordinates():
            yield tuple(coord)

    def __array__(self, dtype=None):
        array = np.stack(self.get_meshgrid(), axis=-1)
        return array.astype(dtype) if dtype is not None else array

    # ============================== #
    # PROPERTIES                     #
    # ============================== #
    @property
    def active_axes(self) -> tuple[str, ...]:
        """Tuple of axis names that are active in the grid."""
        return self.__axes__

    @property
    def fill_values(self) -> dict[str, Any]:
        """Dictionary of fill values for axes not active in the grid."""
        return self.__fill_values__.copy()

    @property
    def coordinate_system(self) -> "CoordinateSystem":
        """The coordinate system associated with this grid."""
        return self.__coordinate_system__

    @property
    def active_axes_mask(self) -> np.ndarray:
        """Boolean mask array indicating which axes are active in the grid."""
        return self.__axes_mask__.copy()

    @property
    def ndim_active(self) -> int:
        """Number of active axes in the grid."""
        return len(self.__axes__)

    @property
    def ndim_inactive(self) -> int:
        """Number of inactive (filled) axes in the grid."""
        return self.__coordinate_system__.__NDIM__ - len(self.__axes__)

    @property
    def ndim(self) -> int:
        """Total number of axes in the coordinate system."""
        return self.__coordinate_system__.__NDIM__

    @property
    def inactive_axes_mask(self) -> np.ndarray:
        """Boolean mask array indicating which axes are filled (not active) in the grid."""
        return self.__fill_mask__.copy()

    @property
    def units(self) -> dict[str, unyt.Unit]:
        """Dictionary mapping axis names to their units."""
        return self.__units__.copy()

    @property
    def bounding_box(self) -> np.ndarray:
        """
        The bounding box of the grid.

        Returns
        -------
        np.ndarray
            An array of shape (N, 2) where N is the number of active axes.
            Each row corresponds to an axis, with the first column being the
            minimum and the second column being the maximum for that axis.
        """
        return self.__bbox__.copy()

    @property
    def dimensions(self) -> np.ndarray:
        """
        The number of grid points along each active axis.

        Returns
        -------
        np.ndarray
            An array of shape (N,) where N is the number of active axes.
            Each entry corresponds to the number of grid points along that axis.
        """
        return self.__ddim__.copy()

    @property
    def shape(self) -> tuple[int, ...]:
        """
        The shape of the grid.

        Returns
        -------
        tuple of int
            A tuple representing the number of grid points along each active axis.
        """
        return tuple(self.__ddim__.tolist())

    @property
    def size(self) -> int:
        """
        The total number of grid points in the grid.

        Returns
        -------
        int
            The product of the number of grid points along each active axis.
        """
        return int(np.prod(self.__ddim__))

    # ============================== #
    # COORDINATE CONVERSION METHODS  #
    # ============================== #
    # These methods form the meat of the grid's functionality.
    # We need to be able to take a grid cell's indices and determine the
    # coordinate of the cell.
    @abstractmethod
    def _convert_slice_to_coordinates(self, axis: int, slc: slice) -> np.ndarray:
        """
        Convert a slice along a given axis into coordinate values.

        Parameters
        ----------
        axis : int
            The axis along which the slice is defined (e.g., 0 for x).
        slc : slice
            A slice object representing the grid indices to convert.

        Returns
        -------
        np.ndarray
            1D array of coordinates corresponding to the given slice.
        """
        raise NotImplementedError

    def get_axis_coordinate_array(self, axis: Union[int, str], slc: slice, units: bool = False) -> np.ndarray:
        """
        Return the coordinate values along a single axis for a given slice.

        Parameters
        ----------
        axis : int or str
            The index or name of the axis to extract coordinates for.
        slc : slice
            A slice object specifying the indices along the axis to extract.
        units: bool
            If ``True``, then the result will carry the values specified in the
            units of the grid (:attr:`units`). If ``False`` (default), the result
            will be a plain numpy array without units.

        Returns
        -------
        np.ndarray
            1D array of coordinate values for the selected slice along the axis.
        """
        if isinstance(axis, str):
            axis = self.__axes__.index(axis)

        out = self._convert_slice_to_coordinates(axis, slc)

        if units:
            return out * self.units[axis]
        else:
            return out

    def get_axis_array(self, axis: Union[int, str], units: bool = False) -> np.ndarray:
        """
        Return the full coordinate array for a single axis.

        Parameters
        ----------
        axis : int or str
            The index or name of the axis.
        units: bool
            If ``True``, then the result will carry the values specified in the
            units of the grid (:attr:`units`). If ``False`` (default), the result
            will be a plain numpy array without units.

        Returns
        -------
        np.ndarray
            1D array of coordinate values for the entire axis.
        """
        return self.get_axis_coordinate_array(axis, slice(None), units=units)

    def get_axis_arrays(
        self, axes: Optional[Sequence[Union[int, str]]] = None, units: bool = False
    ) -> tuple[np.ndarray, ...]:
        """
        Return 1D coordinate arrays for one or more axes.

        Parameters
        ----------
        axes : sequence of str or int, optional
            The axes for which to retrieve coordinate arrays. If not provided,
            all active axes are used.
        units: bool
            If ``True``, then the result will carry the values specified in the
            units of the grid (:attr:`units`). If ``False`` (default), the result
            will be a plain numpy array without units.

        Returns
        -------
        tuple of np.ndarray
            Tuple of 1D arrays, one for each axis specified.
        """
        axes = axes if axes is not None else self.__axes__
        return tuple(self.get_axis_array(ax, units=units) for ax in axes)

    def get_coordinates_slice(
        self, *slcs: slice, axes: Optional[Sequence[str]] = None, units: bool = False
    ) -> tuple[np.ndarray, ...]:
        """
        Return coordinate values for a sliced region of the grid.

        Parameters
        ----------
        slcs : slice(s)
            One slice per axis in the specified `axes`. The number of slices must
            match the number of active axes.
        axes : sequence of str, optional
            The axes along which the slices apply. If not provided, all active axes
            are assumed.
        units: bool
            If ``True``, then the result will carry the values specified in the
            units of the grid (:attr:`units`). If ``False`` (default), the result
            will be a plain numpy array without units.

        Returns
        -------
        tuple of np.ndarray
            Coordinate arrays (1D) for each axis in the full coordinate system.
            Axes not included in `axes` are filled with the constant fill values.
        """
        axes = axes if axes is not None else self.__axes__
        if len(slcs) != len(axes):
            raise ValueError(f"Expected {len(axes)} slices, got {len(slcs)}.")

        coords = []
        for ax_name in self.__coordinate_system__.__AXES__:
            if ax_name in axes:
                idx = axes.index(ax_name)
                coords.append(self.get_axis_coordinate_array(ax_name, slcs[idx], units=units))
            else:
                v = self.__fill_values__[ax_name]

                if units:
                    v = v * self.units[ax_name]
                else:
                    pass

                coords.append(v)

        return tuple(coords)

    def get_meshgrid_slice(
        self, *slcs: slice, axes: Optional[Sequence[str]] = None, indexing: str = "ij"
    ) -> tuple[np.ndarray, ...]:
        """
        Return a meshgrid of coordinates for a sliced region of the grid.

        Parameters
        ----------
        slcs : slice(s)
            One slice per axis in the specified `axes`. The number of slices must
            match the number of axes.
        axes : sequence of str, optional
            Axes for which to build the meshgrid. If not provided, all active axes
            are used.
        indexing : {"ij", "xy"}, default="ij"
            Indexing convention for meshgrid construction.

        Returns
        -------
        tuple of np.ndarray
            A tuple of N-dimensional arrays forming the meshgrid of coordinates,
            one array for each axis in the full coordinate system.
        """
        coords_1d = self.get_coordinates_slice(*slcs, axes=axes)
        return np.meshgrid(*coords_1d, indexing=indexing)

    def get_meshgrid(self, axes: Optional[Sequence[str]] = None, indexing: str = "ij") -> tuple[np.ndarray, ...]:
        """
        Return the full meshgrid of coordinates for the specified axes.

        Parameters
        ----------
        axes : sequence of str, optional
            The axes to include in the meshgrid. If not specified, all active
            axes are used.
        indexing : {"ij", "xy"}, default="ij"
            Indexing convention for meshgrid construction.

        Returns
        -------
        tuple of np.ndarray
            A tuple of N-dimensional arrays forming the full coordinate meshgrid.
        """
        axes = axes if axes is not None else self.__axes__
        slcs = (slice(None),) * len(axes)
        return self.get_meshgrid_slice(*slcs, axes=axes, indexing=indexing)

    def get_index_arrays(self) -> tuple[np.ndarray, ...]:
        """
        Return 1D arrays of grid indices for each active axis.

        Returns
        -------
        tuple of np.ndarray
            A tuple of 1D arrays, one for each active axis, containing index values.
        """
        return tuple(np.arange(n) for n in self.shape)

    def get_index_meshgrid(self, indexing: str = "ij") -> tuple[np.ndarray, ...]:
        """
        Return a meshgrid of index values for each active axis.

        Parameters
        ----------
        indexing : {"ij", "xy"}, default="ij"
            Indexing convention to use in constructing the meshgrid.

        Returns
        -------
        tuple of np.ndarray
            A tuple of N-dimensional arrays, each containing the index values
            along one axis of the grid.
        """
        return np.meshgrid(*self.get_index_arrays(), indexing=indexing)

    def get_flat_coordinates(self) -> np.ndarray:
        """
        Return a flattened array of coordinate vectors for all grid points.

        Returns
        -------
        np.ndarray
            An array of shape (N_points, N_dims), where N_points is the total
            number of grid points and N_dims is the number of dimensions in
            the coordinate system. Each row represents the coordinates of one point.
        """
        coords = self.get_meshgrid(indexing="ij")
        flat = [c.ravel() for c in coords]
        return np.stack(flat, axis=-1)  # shape (N_points, N_dims)

    def get_coordinate_dict(self, meshgrid: bool = False) -> dict[str, np.ndarray]:
        """
        Return coordinate arrays in dictionary form with axis names as keys.

        Parameters
        ----------
        meshgrid : bool, default=False
            If True, return meshgrid arrays. If False, return 1D arrays.

        Returns
        -------
        dict of str -> np.ndarray
            A dictionary mapping axis names to either 1D or meshgrid arrays
            of coordinates, depending on the value of `meshgrid`.
        """
        axes = self.__coordinate_system__.__AXES__
        coords = self.get_meshgrid() if meshgrid else self.get_axis_arrays()
        return dict(zip(axes, coords, strict=False))

    # ============================== #
    # GENERATOR METHODS              #
    # ============================== #
    # These are methods used to generate new arrays with shapes
    # matching the grid.
    def _create_like(self, fill_value: Any = 0, dtype: Optional[np.dtype] = None, element_shape=()) -> np.ndarray:
        """
        Create a numpy array with a shape consistent with that of the grid.

        Parameters
        ----------
        fill_value: any
            The value to fill the array with. Default is 0.
        dtype: np.dtype, optional
            The data type of the array. If None, defaults to float.
        element_shape: tuple, optional
            Additional shape to append to the grid shape. Default is (). This allows
            users to get scalar, vector, or tensor fields on the grid.

        Returns
        -------
        numpy.ndarray
            An array of shape `grid.shape + element_shape` filled with `fill_value`.
        """
        if dtype is None:
            dtype = float
        full_shape = self.shape + tuple(element_shape)
        arr = np.full(full_shape, fill_value, dtype=dtype)
        return arr

    def zeros_like(self, dtype: Optional[np.dtype] = None, element_shape=()) -> np.ndarray:
        """
        Create a zero-filled array with the same shape as the grid.

        Parameters
        ----------
        dtype : np.dtype, optional
            Desired data type. Defaults to float if not provided.
        element_shape : tuple, optional
            Extra shape appended to grid shape (e.g., for vector/tensor fields).

        Returns
        -------
        np.ndarray
            Array of shape `grid.shape + element_shape`, filled with zeros.
        """
        return self._create_like(0, dtype=dtype, element_shape=element_shape)

    def ones_like(self, dtype: Optional[np.dtype] = None, element_shape=()) -> np.ndarray:
        """
        Create a one-filled array with the same shape as the grid.

        Parameters
        ----------
        dtype : np.dtype, optional
            Desired data type. Defaults to float if not provided.
        element_shape : tuple, optional
            Extra shape appended to grid shape (e.g., for vector/tensor fields).

        Returns
        -------
        np.ndarray
            Array of shape `grid.shape + element_shape`, filled with ones.
        """
        return self._create_like(1, dtype=dtype, element_shape=element_shape)

    def full_like(self, fill_value: Any, dtype: Optional[np.dtype] = None, element_shape=()) -> np.ndarray:
        """
        Create a filled array with the same shape as the grid.

        Parameters
        ----------
        fill_value : Any
            The value to fill the array with.
        dtype : np.dtype, optional
            Desired data type. Defaults to float if not provided.
        element_shape : tuple, optional
            Extra shape appended to grid shape (e.g., for vector/tensor fields).

        Returns
        -------
        np.ndarray
            Array of shape `grid.shape + element_shape`, filled with `fill_value`.
        """
        return self._create_like(fill_value, dtype=dtype, element_shape=element_shape)

    def empty_like(self, dtype: Optional[np.dtype] = None, element_shape=()) -> np.ndarray:
        """
        Create an uninitialized array with the same shape as the grid.

        Parameters
        ----------
        dtype : np.dtype, optional
            Desired data type. Defaults to float if not provided.
        element_shape : tuple, optional
            Extra shape appended to grid shape (e.g., for vector/tensor fields).

        Returns
        -------
        np.ndarray
            Array of shape `grid.shape + element_shape`, uninitialized (contents arbitrary).
        """
        if dtype is None:
            dtype = float
        full_shape = self.shape + tuple(element_shape)
        return np.empty(full_shape, dtype=dtype)

    # ============================== #
    # IO METHODS                     #
    # ============================== #
    # These methods are used to serialize and deserialize
    # the grid. These are generally abstract methods to allow subclasses
    # to define a custom behavior for the IO methods.

    @abstractmethod
    def _save_grid_to_hdf5_group(self, group: h5py.Group):
        """
        Save this grid to an HDF5 group.

        This method forms the core of the HDF5 IO support for grids. It should
        be written specially for any given grid to ensure that all relevant data is
        saved and can be reloaded later.

        Grids should store any metadata in the group's metadata attributes and then
        make sure of datasets to store any more substantial data or arrays. Grids are required
        to include the ``CLASS_NAME`` attribute in the group to indicate the class of the grid
        being saved. This allows for recovery of the correct class later when loading.

        Parameters
        ----------
        group : h5py.Group
            The HDF5 group to which the grid should be saved.

        """
        ...

    @classmethod
    @abstractmethod
    def _load_grid_from_hdf5_group(cls, group: h5py.Group):
        """
        Load a grid from an HDF5 group.

        This method should be implemented by subclasses to read the relevant
        metadata and datasets from the provided HDF5 group and reconstruct
        an instance of the grid.

        Parameters
        ----------
        group : h5py.Group
            The HDF5 group from which to load the grid.

        Returns
        -------
        Grid
            An instance of the grid reconstructed from the HDF5 data.

        """
        ...

    def to_hdf5(self, filename: Union[str, Path], group: str = "grid", overwrite: bool = False):
        """
        Save the grid to an HDF5 file.

        Parameters
        ----------
        filename : str or Path
            Path to the HDF5 file.
        group : str, default = "grid"
            Name of the group within the file to store the grid.
        overwrite : bool, default = False
            Whether to overwrite the group if it already exists.

        Raises
        ------
        ValueError
            If the group already exists and `overwrite` is False.
        """
        filename = Path(filename)

        with h5py.File(filename, "a") as f:
            if group in f:
                if overwrite:
                    del f[group]
                else:
                    raise ValueError(f"HDF5 group '{group}' already exists in file '{filename}'.")

            g = f.create_group(group)
            self._save_grid_to_hdf5_group(g)

    @classmethod
    def from_hdf5(cls, filename: Union[str, Path], group: str = "grid") -> "Grid":
        """
        Load a grid from an HDF5 file.

        Parameters
        ----------
        filename : str or Path
            Path to the HDF5 file.
        group : str, default = "grid"
            Name of the group within the file to load from.

        Returns
        -------
        GenericGrid
            Reconstructed grid instance.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        KeyError
            If the group does not exist in the file.
        """
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f"HDF5 file not found: {filename}")

        with h5py.File(filename, "r") as f:
            if group not in f:
                raise KeyError(f"Group '{group}' not found in HDF5 file '{filename}'.")

            return cls._load_grid_from_hdf5_group(f[group])
