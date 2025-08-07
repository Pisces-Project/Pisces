"""
Core grid classes and utilities for Pisces geometry module.

This module defines the typical / classic grid structures that are relevant
to developers of models in Pisces. It includes classes for structured grids,
unstructured grids, and various grid utilities.
"""

from typing import TYPE_CHECKING, Any, Optional, Union

import h5py
import numpy as np
import unyt

from .base import Grid

if TYPE_CHECKING:
    from pisces.geometry.coordinates.base import CoordinateSystem


class GenericGrid(Grid):
    """
    A structured grid defined by 1D edge arrays along each axis.

    This class supports the representation of generic, structured grids using
    edge arrays that define cell boundaries along each active axis. Each cell's
    center is computed as the midpoint between its bounding edges. The grid
    supports flexible dimensionality, custom axis selection, unit specification,
    and missing-axis filling via fixed values.

    :class:`GenericGrid` is most appropriate when your coordinate axes are non-uniform
    or have arbitrary spacing, and when you need precise control over the placement
    of grid cells in physical space.
    """

    # ============================== #
    # INITIALIZATION                 #
    # ============================== #
    def _configure_grid_attributes(self, *edges, **kwargs):
        """
        Configure the grid attributes based on the provided edge arrays.

        This method is used to set the ``__ddim__`` and ``__bbox__`` attributes as
        usual, but it also sets the ``__edges__`` attribute, which contains a record of all
        of the edges. We also set the ``__centers__`` attribute, which contains the centers of each
        cell along each axis.
        """
        # Start by ensuring that we have the right number of edge
        # arrays and that they are all 1D, monotonic, and unit compatible.
        if len(edges) != len(self.__axes__):
            raise ValueError(
                f"Number of edge arrays ({len(edges)}) does not match number of active axes ({len(self.__axes__)})."
            )

        self.__edges__ = []

        for axis, edge_array in zip(self.__axes__, edges, strict=False):
            # Start with basic validation. We ensure that the edge
            # array get its units handled properly and then we coerce
            # to a numpy array for standardization.
            if hasattr(edge_array, "units"):
                try:
                    edge_array = edge_array.to_value(self.units[axis])
                except Exception as exp:
                    raise ValueError(f"Edge array for axis '{axis}' has incompatible units: {exp}") from exp

            edge_array = np.asarray(edge_array)

            # Now check the array for monotonicity and dimensionality.
            if edge_array.ndim != 1:
                raise ValueError(f"Edge array for axis '{axis}' must be 1D, but has shape {edge_array.shape}.")
            if edge_array.size < 2:
                raise ValueError(f"Edge array for axis '{axis}' must have at least two elements to define edges.")
            if not np.all(np.diff(edge_array) > 0):
                raise ValueError(f"Edge array for axis '{axis}' must be strictly monotonic increasing.")

            # Add the axis to the edges.
            self.__edges__.append(edge_array)

        # Construct the center arrays.
        self.__centers__ = [0.5 * (earr[:-1] + earr[1:]) for earr in self.__edges__]

        # With the centers and the edges defined, we can now define
        # the domain dimensions and the bounding box.
        self.__ddim__ = np.asarray([len(carr) for carr in self.__centers__])
        self.__bbox__ = np.asarray([(earr[0], earr[-1]) for earr in self.__edges__])

    def __init__(
        self,
        coordinate_system: "CoordinateSystem",
        *edges,
        axes: Union[str, tuple[str, ...]] = "all",
        fill_values: Optional[dict[str, Any]] = None,
        units: Optional[dict[str, Union[str, unyt.Unit]]] = None,
        **kwargs,
    ):
        """
        Initialize a generic grid from a set of edge arrays.

        Parameters
        ----------
        coordinate_system: CoordinateSystem
            The coordinate system in which the grid is defined. This coordinate
            system will determine the labels for the coordinate axes and the dimension
            of the grid.
        edges: numpy.ndarray
            1D arrays representing the edges of each grid cell along each axis. There must
            be as many edge arrays as there are active axes (see `axes` parameter) in the grid.
            For each axis, center each cell will be positioned between the adjacent edge.

            These arrays must be strictly monotonic. If an edges array contains units, they
            will be converted to the specified unit in the `units` parameter if possible. Otherwise
            an error will be raised.
        axes: str or list of str, optional
            The active axes of the grid. This defines which of the coordinate axes are actually
            incorporated into the grid and which are fixed. If ``"all"`` (default), then all of
            the coordinate axes are used in the grid. Otherwise, only those specified are used.

            If an axis is excluded, it must then be present in the `fill_values` parameter to ensure
            that the grid is fully defined.
        fill_values: dict, optional
            A dictionary mapping axis names (as strings) to fixed values for any axes that are
            not included in the grid (i.e., axes that are not in `axes` if `axes` is not ``"all"``).
            This ensures that the grid is fully defined in the coordinate system.

            If an axis is excluded from the grid but not provided in this dictionary, an error
            will be raised.
        units: dict, optional
            A dictionary mapping axis names (as strings) to units (either as strings or
            unyt.Unit objects) for each axis in the grid. If provided, the edge arrays
            will be converted to these units. If not provided, the units of the edge arrays
            will be used as-is.
        kwargs
        """
        # Pass everything through the base class initializer so that we can
        # configure the grid. We'll catch the edges via the _configure_grid_attributes
        # method.
        super().__init__(coordinate_system, *edges, axes=axes, fill_values=fill_values, units=units, **kwargs)

    # ============================== #
    # COORDINATE CONVERSION METHODS  #
    # ============================== #
    def _convert_slice_to_coordinates(self, axis: int, slc: slice) -> np.ndarray:
        return self.__centers__[axis][slc]

    # ============================== #
    # IO METHODS                     #
    # ============================== #
    def _save_grid_to_hdf5_group(self, group: h5py.Group):
        # Create and save the metadata into the attributes of the group. In
        # this case, we have to keep track of the coordinate system, the axes, the
        # fill values, and the units. The coordinate system can be changed to a dictionary
        # so that we have access to it.
        metadata = {
            "CLASS_NAME": self.__class__.__name__,
            "coordinate_system": self.coordinate_system.to_dict(),
            "axes": self.__axes__,
            "fill_values": self.fill_values,
            "units": {k: str(v) for k, v in self.units.items()},
        }

        for _key, _value in self.metadata_serializer.serialize_dict(metadata).items():
            group.attrs[_key] = _value

        # Create the edge datasets for each of the edges.
        for axis, edge_array in zip(self.__axes__, self.__edges__, strict=False):
            dset_name = f"edges_{axis}"
            if dset_name in group:
                raise ValueError(f"Dataset '{dset_name}' already exists in the HDF5 group.")
            _ = group.create_dataset(dset_name, data=edge_array)

    @classmethod
    def _load_grid_from_hdf5_group(cls, group: h5py.Group) -> "GenericGrid":
        """
        Load a GenericGrid instance from an HDF5 group.

        This method reconstructs a grid from its saved metadata and edge arrays.

        Parameters
        ----------
        group : h5py.Group
            HDF5 group containing the grid data, as saved by `_save_grid_to_hdf5_group`.

        Returns
        -------
        GenericGrid
            A new grid instance reconstructed from the HDF5 group.
        """
        from pisces.geometry.coordinates.utils import load_coordinate_dict

        # --- Deserialize metadata --- #
        raw_attrs = dict(group.attrs)
        metadata = cls.metadata_serializer.deserialize_dict(raw_attrs)

        # Check class name compatibility
        if metadata.get("CLASS_NAME", None) != cls.__name__:
            raise ValueError(
                f"Incompatible class in HDF5 group: expected {cls.__name__}, found {metadata.get('CLASS_NAME')}"
            )

        # Reconstruct coordinate system
        csys_dict = metadata["coordinate_system"]
        coordinate_system = load_coordinate_dict(csys_dict)

        axes = tuple(metadata["axes"])
        fill_values = metadata.get("fill_values", {})
        units = metadata.get("units", {})

        # --- Load edge arrays --- #
        edge_arrays = []
        for axis in axes:
            dset_name = f"edges_{axis}"
            if dset_name not in group:
                raise ValueError(f"Missing edge dataset '{dset_name}' in HDF5 group.")
            edge_array = group[dset_name][...]
            edge_arrays.append(edge_array)

        # --- Instantiate and return grid --- #
        return cls(
            coordinate_system,
            *edge_arrays,
            axes=axes,
            fill_values=fill_values,
            units=units,
        )
