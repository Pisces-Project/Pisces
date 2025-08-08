.. _grid_overview:

=============================
Grids in Pisces
=============================

Grids are the backbone of all spatial models in Pisces.

Every model in Pisces is defined on top of a grid from :mod:`~geometry.grids`, which
discretizes a coordinate system into a structured set of points or cells.
The grid defines *where* computations happen — providing the spatial structure for scalar,
vector, or tensor fields. Without a grid, you cannot evaluate a field, solve PDEs,
or build geometry-aware models.

Grids live within a :class:`~geometry.coordinates.base.CoordinateSystem`
(e.g., Cartesian, cylindrical, spherical) and define a
finite region — or **bounding box** — within that system, partitioned into discrete cells.
You can think of a grid as the scaffolding
that supports the geometry of your entire simulation or analysis pipeline.

Pisces provides a flexible system for defining and interacting with grids,
including serialization, indexing, unit support, and coordinate transformations.

.. note::

    Grids are defined in the :mod:`geometry.grids` module.

Grid Basics
-------------------

A Pisces grid always requires a :class:`~geometry.coordinates.base.CoordinateSystem` to define
the geometry in which it lives. This coordinate system determines:

- The number of spatial dimensions (e.g., 2D vs 3D)
- The axis names and their interpretation (e.g., ``["r", "theta", "phi"]`` for spherical)

The grid itself requires some prescription for how to discretize space into the grid.
This can be done a number of ways, depending on the grid type.

Building a Grid
^^^^^^^^^^^^^^^

Pisces provides multiple grid types tailored to different modeling needs.
While each grid class (e.g., :class:`~geometry.grids.core.GenericGrid`)
may accept different arguments, all grids share a common conceptual interface.

A key part of this interface is the distinction between **active** and **filled** axes:

- **Active axes** are those that vary across the grid. For example, in a 2D Cartesian grid,
  both ``x`` and ``y`` are typically active.
- **Filled axes** are those that remain constant throughout the grid. For instance, in a 3D
  spherical coordinate system, you might construct a 2D grid with ``r`` and ``theta`` as active,
  and ``phi`` fixed to a constant value via a fill.

When constructing a grid, you specify:

- ``axes`` — a sequence of axis names to be treated as active
- ``fill_values`` — a dictionary providing constant values for all filled axes
- ``units`` — optional dictionary mapping axis names to their units, specified as strings or
  :class:`unyt.Unit` objects

Here is an example of constructing a 2D Cartesian grid with **logarithmic spacing**
in both ``x`` and ``y`` directions:

.. code-block:: python

    import numpy as np
    from pisces.geometry.coordinates import Cartesian2DCoordinateSystem
    from pisces.geometry.grids.core import GenericGrid

    csys = Cartesian2DCoordinateSystem()

    x_edges = np.geomspace(1.0, 100.0, 101)  # 100 logarithmic cells in x
    y_edges = np.geomspace(1.0, 100.0, 101)  # 100 logarithmic cells in y

    grid = GenericGrid(
        csys,
        x_edges,
        y_edges,
        axes=("x", "y"),
        units={"x": "kpc", "y": "kpc"}
    )

This creates a 2D grid with shape ``(100, 100)``, using logarithmic spacing
and associated units. Internally, the grid computes cell centers from the edge arrays
and defines the spatial geometry accordingly.

.. note::

    Edge arrays must be strictly monotonic and 1D. For ``N`` cells along an axis,
    the corresponding edge array must have ``N + 1`` entries. Pisces grids use
    **cell-centered discretization**.

Grid Properties
^^^^^^^^^^^^^^^^

Although different grid classes in Pisces may vary in how they are constructed,
they all conform to a common, standardized interface once instantiated.
This unified API makes it easy to inspect grid structure, regardless of type.

Below is a list of the key properties exposed by all grids:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Description
   * - :attr:`~geometry.grids.base.Grid.coordinate_system`
     - The :class:`~geometry.coordinates.base.CoordinateSystem` instance associated with the grid,
       defining its geometric basis (e.g., Cartesian, spherical).
   * - :attr:`~geometry.grids.base.Grid.shape`
     - A tuple representing the number of cells along each **active axis**.
       This defines the grid’s dimensional shape.
   * - :attr:`~geometry.grids.base.Grid.ndim`
     - Total number of axes in the coordinate system (including both active and filled axes).
       Equivalent to ``len(coordinate_system.axes)``.
   * - :attr:`~geometry.grids.base.Grid.ndim_active`
     - Number of axes that vary across the grid (i.e., the active axes). Equal to ``len(shape)``.
   * - :attr:`~geometry.grids.base.Grid.ndim_inactive`
     - Number of axes that are **filled** — held constant at fixed values. This is ``ndim - ndim_active``.
   * - :attr:`~geometry.grids.base.Grid.active_axes`
     - Tuple of axis names (e.g., ``("r", "theta")``) that are active in the grid.
   * - :attr:`~geometry.grids.base.Grid.fill_values`
     - Dictionary mapping each **filled axis** name to its fixed scalar value (e.g., ``{"phi": 0.0}``).
   * - :attr:`~geometry.grids.base.Grid.units`
     - Dictionary mapping axis names to their physical units as :class:`unyt.Unit` objects.
       Includes both active and filled axes.
   * - :attr:`~geometry.grids.base.Grid.bounding_box`
     - A NumPy array of shape ``(N, 2)``, where ``N = ndim_active``.
       Each row gives ``[min, max]`` coordinates for an active axis, defining the spatial extent of the grid.

These properties allow introspection of grid geometry, discretization, and metadata
independent of how the grid was constructed.

Accessing Grid Coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^

Once constructed, a grid provides a complete, unit-aware description of its
spatial layout. Pisces offers a consistent set of tools for retrieving this
coordinate information in whatever form your workflow requires.

You can:

- Slice along a **single axis** to obtain 1D coordinate arrays.
- Generate **full meshgrids** of coordinates for the entire domain or for a subregion.
- Retrieve coordinates for **specific regions** or **individual points**.
- Map directly from **grid indices** to **physical locations**.
- Flatten the entire grid into a list of coordinate vectors for vectorized operations.

Grids are fully indexable using standard NumPy-style syntax, and also expose
a family of convenience methods for common coordinate operations. Most coordinate
outputs are returned as :class:`unyt.unyt_array` objects carrying the appropriate
physical unit for each axis. The only exceptions are methods that must return a
single stacked array spanning multiple axes (e.g., :meth:`get_flat_coordinates`
or the :meth:`__array__` protocol), which return magnitudes only for unit
consistency; in these cases you can consult :attr:`~geometry.grids.base.Grid.units`
to recover per-axis units.

Indexing into Grids
````````````````````````````````````````

Pisces grids support intuitive indexing using square brackets, similar to NumPy arrays.
This allows you to extract **physical coordinates** associated with one or more grid
points directly from the grid object.

All coordinate-returning index operations preserve **per-axis physical units**
(using :class:`unyt.unyt_array`) wherever possible. The only exceptions are cases
where results must be stacked into a single 2-D array containing heterogeneous
units (e.g., flattened coordinate lists), or the :meth:`__array__` protocol
(which always returns magnitudes for NumPy compatibility).

Several indexing patterns are supported:

.. code-block:: python

    grid[3, 5]             # Single point (tuple of unyt_quantities)
    grid[:, 5]             # Meshgrid slice at all x-values for fixed y=5 (tuple of unyt_arrays)
    grid[...]              # Full meshgrid (tuple of unyt_arrays), same as grid.get_meshgrid()
    grid[mask]             # Boolean mask → magnitudes, shape (N, ndim), see note
    grid[index_array]      # Integer index array → magnitudes, shape (..., ndim), see note

The type and shape of the result depend on the index structure:

- **Ellipsis (``...``)** returns the full meshgrid as a tuple of N-D
  :class:`unyt.unyt_array` objects, one per axis:

  .. code-block:: python

      xg, yg, zg = grid[...]  # each with shape == grid.shape, each carrying its axis unit

- **Tuple of integers** returns a tuple of scalar
  :class:`unyt.unyt_quantity` values:

  .. code-block:: python

      grid[3, 4]  # → (x : kpc, y : kpc)

- **Tuple containing slices** returns a tuple of
  :class:`unyt.unyt_array` meshgrids for the selected region:

  .. code-block:: python

      xg, yg = grid[:, 4]  # xg, yg each carry axis units

- **Boolean mask** must match ``grid.shape`` exactly. Returns a
  **NumPy array of magnitudes** with shape ``(N, ndim)``, where each column
  corresponds to an axis. Per-axis units are available from
  :attr:`~geometry.grids.base.Grid.units`.

- **Integer index arrays** are applied to the stacked meshgrid
  (shape ``grid.shape + (ndim,)``). Returns a **NumPy array of magnitudes**
  with shape determined by the index array. Per-axis units are available
  from :attr:`~geometry.grids.base.Grid.units`.

.. note::

   Indexing that returns a tuple of coordinate arrays or scalars will always
   include physical units. Indexing that must return a single N×D array
   (mask or integer array) returns **magnitudes only** for unit consistency,
   since heterogeneous units cannot be represented in a single array.

Single Axis Coordinates
````````````````````````````````````````

Grids provide dedicated methods for retrieving **1D coordinate arrays**
along individual axes. These methods always return results as
:class:`unyt.unyt_array` objects carrying the appropriate physical unit
for the axis in question:

- :meth:`~geometry.grids.base.Grid.get_axis_coordinate_array` —
  Return a 1D coordinate array for a **specified slice** along a single axis.
  Useful for extracting a subset of coordinates without building a full meshgrid.

- :meth:`~geometry.grids.base.Grid.get_axis_array` —
  Return the **full 1D coordinate array** for a single axis, covering the
  entire extent of that axis in the grid.

- :meth:`~geometry.grids.base.Grid.get_axis_arrays` —
  Return a **tuple** of full 1D coordinate arrays for one or more axes.
  If no axes are specified, all active axes are included.

Sliced Region Coordinates
````````````````````````````````````````

:meth:`~geometry.grids.base.Grid.get_coordinates_slice`
Return a **tuple of 1D** :class:`unyt.unyt_array` coordinate arrays for a
user-specified sliced region of the grid.

- Each active axis varies along the slice and returns a coordinate array
  in its own physical units.
- Each filled (inactive) axis returns its constant value as a length-1
  :class:`unyt.unyt_array` with the correct unit.

Coordinate Meshgrids
````````````````````````````````````````

:meth:`~geometry.grids.base.Grid.get_meshgrid` and
:meth:`~geometry.grids.base.Grid.get_meshgrid_slice`
Return a **tuple of multidimensional** :class:`unyt.unyt_array` objects,
one per axis, each carrying its corresponding physical unit.

- Active axes vary across the meshgrid according to the grid shape or
  specified slices.
- Filled axes appear as constant arrays, broadcast to match the output shape.

Index Meshgrids
````````````````````````````````````````

:meth:`~geometry.grids.base.Grid.get_index_arrays` and
:meth:`~geometry.grids.base.Grid.get_index_meshgrid`
Return **integer index arrays** (no units) describing the grid’s logical
structure rather than its physical coordinates.

Flattened Coordinates
````````````````````````````````````````

:meth:`~geometry.grids.base.Grid.get_flat_coordinates`
Return a **NumPy array of magnitudes** with shape ``(N_points, ndim)``:

- ``N_points`` is the total number of points in the grid.
- ``ndim`` is the total number of axes in the coordinate system.

Each column corresponds to one axis in the coordinate system. Because
heterogeneous units cannot be stored in a single numeric array, the
result contains magnitudes only; per-axis units are available from
:attr:`~geometry.grids.base.Grid.units`.

Coordinate Dictionaries
````````````````````````````````````````

:meth:`~geometry.grids.base.Grid.get_coordinate_dict`
Return a dictionary mapping each axis name to a coordinate array with
units preserved.

- If ``meshgrid=False`` (default), each value is a 1D
  :class:`unyt.unyt_array` covering that axis.
- If ``meshgrid=True``, each value is an N-D :class:`unyt.unyt_array`
  matching the shape of the full coordinate meshgrid.


Generating Field Arrays
^^^^^^^^^^^^^^^^^^^^^^^^

One useful feature of Pisces grids is the ability to generate arrays with the correct shape
and metadata to be fully compatible with a particular grid. This ensures that user-defined
fields align with the grid’s coordinate structure, boundary conditions, and dimensionality.

Pisces provides utility methods that allow users to initialize field arrays (or buffers)
directly from the grid object. These arrays can be used for scalar, vector, or tensor fields,
and will automatically match the grid’s internal shape — including ghost zones or chunking, if applicable.

Available methods include:

- :meth:`~geometry.grids.base.Grid.zeros_like` —
  Create a zero-filled array with the same shape as the grid.

- :meth:`~geometry.grids.base.Grid.ones_like` —
  Create an array filled with ones.

- :meth:`~geometry.grids.base.Grid.full_like` —
  Create an array filled with a constant value of your choice.

- :meth:`~geometry.grids.base.Grid.empty_like` —
  Create an uninitialized array (contents are arbitrary).

Each of these methods supports an optional ``element_shape`` argument. This is particularly useful
for generating **vector** or **tensor** fields. For example, a vector field on a 2D grid would use
``element_shape=(2,)``, while a rank-2 tensor would use ``element_shape=(2, 2)``.

You may also specify a desired data type using the ``dtype`` argument.

Example usage:

.. code-block:: python

    # Scalar field (default)
    field = grid.zeros_like()

    # Vector field (2 components at each grid point)
    velocity = grid.zeros_like(element_shape=(2,))

    # Tensor field (3x3 at each grid point)
    stress = grid.zeros_like(element_shape=(3, 3), dtype=np.float64)

Grid I/O
--------

Grids in Pisces can be easily saved to and loaded from HDF5 files using a standardized interface.

This allows for persistent storage of grid metadata, coordinates, and structure, making it easy
to serialize simulation setups or share grid definitions across workflows.

Saving a Grid
^^^^^^^^^^^^^

To save a grid to an HDF5 file, use the :meth:`~geometry.grids.base.Grid.to_hdf5` method:

.. code-block:: python

    grid.to_hdf5("my_grid.h5")

By default, this saves the grid into the group ``"grid"`` within the file. You can customize
the group name and control whether to overwrite existing groups:

.. code-block:: python

    grid.to_hdf5("output.h5", group="initial_conditions", overwrite=True)

What’s Stored?
^^^^^^^^^^^^^^

The saved HDF5 group contains:

- The class name and metadata required to reconstruct the grid
- The coordinate system definition (as a serialized dictionary)
- The edge arrays for each active axis
- Units and fill values
- Any additional subclass-specific data

This metadata is stored in HDF5 group attributes and datasets in a self-describing format.
All coordinate systems and grid classes must be registered with the Pisces registry system
to support reconstruction.

Loading a Grid
^^^^^^^^^^^^^^

To load a grid from an HDF5 file, use the class method:

.. code-block:: python

    from pisces.geometry.grids.core import load_grid

    grid = load_grid("my_grid.h5", group_path="grid")

This function automatically detects the saved grid class from the file
(using the ``CLASS_NAME`` attribute), loads the appropriate subclass, and restores all settings.

Alternatively, if you already know the grid type, you can use the ``from_hdf5`` method directly on the class:

.. code-block:: python

    from pisces.geometry.grids.core import GenericGrid

    grid = GenericGrid.from_hdf5("my_grid.h5")

This is equivalent but skips class inference. You must ensure that the saved grid matches
the class you're loading from.
