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

Once a grid has been constructed, Pisces provides a rich set of tools for accessing and manipulating
coordinate information. This includes methods for slicing along a single axis, generating meshgrids,
retrieving coordinate values for specific regions or the full grid, and mapping from indices to physical locations.

Grids are indexable using standard NumPy-style indexing, and expose a variety of convenience methods
for common coordinate operations.

Indexing into Grids
````````````````````````````````````````

Pisces grids support intuitive indexing using square brackets, similar to NumPy arrays.
This allows you to extract physical coordinates associated with one or more grid points
directly from the grid object.

Several indexing patterns are supported:

.. code-block:: python

    grid[3, 5]             # Returns the coordinate tuple at index (3, 5)
    grid[:, 5]             # Returns a meshgrid slice at all x-values for fixed y=5
    grid[...]              # Returns the full meshgrid (equivalent to grid.get_meshgrid())
    grid[mask]             # Boolean mask of shape == grid.shape → returns matching coordinates
    grid[index_array]      # Index array → returns coordinates from stacked meshgrid

The type and shape of the result depend on the structure of the index:

- **Ellipsis (`...`)** returns the full meshgrid as a tuple of N-dimensional arrays, one per axis:

  .. code-block:: python

      grid[...]  # → (xg, yg, zg) each of shape grid.shape

- **Tuple of integers** returns a single point's coordinates as a tuple:

  .. code-block:: python

      grid[3, 4]  # → (x, y)

- **Tuple containing slices** returns a meshgrid over the selected region:

  .. code-block:: python

      grid[:, 4]  # → meshgrid at all x for y = 4

- **Boolean mask** must match ``grid.shape`` exactly. Returns an array with one coordinate vector per `True` entry:

  .. code-block:: python

      mask = np.random.rand(*grid.shape) > 0.5
      coords = grid[mask]  # → shape (N, ndim)

- **Integer index arrays** are applied to the stacked meshgrid of shape ``grid.shape + (ndim,)``:

  .. code-block:: python

      idx = np.array([[0, 0], [2, 3]])
      coords = grid[idx]  # → shape (2, ndim)

.. note::

   All indexing operations return **physical coordinates**, including filled axes.
   These are not indices or data values—they represent spatial positions in the coordinate system.

In addition to indexing, there are also a number of different access patterns which are mediated through
methods of the grid class.

Single Axis Coordinates
````````````````````````````````````````

Grids expose several methods for retrieving **1D coordinate arrays** along individual axes.
These are useful when working with profiles, line cuts, or when building meshgrids manually.

- :meth:`~geometry.grids.base.Grid.get_axis_coordinate_array`
  Returns coordinate values along a **single axis** for a specified index range (slice).

- :meth:`~geometry.grids.base.Grid.get_axis_array`
  Returns the **full 1D coordinate array** for a single axis.

- :meth:`~geometry.grids.base.Grid.get_axis_arrays`
  Returns a tuple of **1D coordinate arrays** for multiple axes.

.. hint::

    If you pass ``units=True`` to any of these methods, the returned arrays will carry physical units
    as :class:`unyt.unyt_array` objects. This is useful for computations where unit consistency is important.

    .. code-block:: python

        grid.get_axis_array("x", units=True)
        # → returns a unyt array like [1.0 kpc, 2.0 kpc, ..., 10.0 kpc]

Sliced Region Coordinates
````````````````````````````````````````

To retrieve coordinate values for a **sliced region** of the grid, use the method:

- :meth:`~geometry.grids.base.Grid.get_coordinates_slice`

This method returns a **tuple of 1D coordinate arrays**, one for each axis in the grid's coordinate system.

Each array corresponds to either:

- The **sliced range** for an active axis (i.e., one that varies across the grid),
- Or the **fixed value** for a filled (inactive) axis.

Coordinate Meshgrids
````````````````````````````````````````

Pisces provides two methods for generating meshgrids of physical coordinates:

- :meth:`~geometry.grids.base.Grid.get_meshgrid`
- :meth:`~geometry.grids.base.Grid.get_meshgrid_slice`

Each method returns a tuple of multidimensional NumPy arrays—one per axis in the coordinate system.
These arrays contain the physical coordinates of every point in the grid or subregion, and can be used
to evaluate functions, define fields, or create visualizations.

To generate a meshgrid over the entire domain:

.. code-block:: python

    from pisces.geometry.coordinates import Cartesian2DCoordinateSystem
    from pisces.geometry.grids.core import GenericGrid
    import numpy as np

    csys = Cartesian2DCoordinateSystem()
    x = np.linspace(0, 10, 5)
    y = np.linspace(0, 20, 3)

    grid = GenericGrid(
        coordinate_system=csys,
        x, y,
        axes=["x", "y"],
        units={"x": "km", "y": "km"}
    )

    xg, yg = grid.get_meshgrid()

    print(xg.shape)  # (5, 3)
    print(yg.shape)  # (5, 3)

The default indexing convention is ``"ij"`` (matrix-style indexing). You can also specify ``indexing="xy"`` if needed.

To generate a meshgrid over just a portion of the grid, use :meth:`~geometry.grids.base.Grid.get_meshgrid_slice`:

.. code-block:: python

    xg, yg = grid.get_meshgrid_slice(slice(1, 4), slice(0, 2))

    print(xg.shape)  # (3, 2)
    print(yg.shape)  # (3, 2)

This returns only the coordinates within the specified slices along each axis.

If the grid was constructed with one or more *filled* axes (i.e., constant-valued dimensions),
those axes will still be represented in the output of ``get_meshgrid`` or ``get_meshgrid_slice``.
The coordinate array for a filled axis will be a constant array, broadcast to the full shape.

For example, if using a cylindrical grid with ``r`` and ``z`` as active axes and ``theta`` fixed:

.. code-block:: python

    rg, thetag, zg = grid.get_meshgrid()
    # thetag is a constant array filled with the fixed value

Index Meshgrids
````````````````````````````````````````

If you're interested in working with **grid indices** rather than physical coordinates, Pisces provides two
convenient methods:

- :meth:`~geometry.grids.base.Grid.get_index_arrays`
- :meth:`~geometry.grids.base.Grid.get_index_meshgrid`

These return arrays of integer indices that correspond to the grid’s structure.

To retrieve 1D index arrays for each axis:

.. code-block:: python

    i, j = grid.get_index_arrays()

To retrieve a full meshgrid of index values (with the same shape as the grid):

.. code-block:: python

    i, j = grid.get_index_meshgrid()

This is particularly useful for evaluating functions or assigning values based on location in the grid.

You can also specify the indexing convention (``"ij"`` or ``"xy"``) just like with coordinate meshgrids.

Flattened Coordinates
````````````````````````````````````````

To access the full grid as a **flattened list of coordinates**, use:

- :meth:`~geometry.grids.base.Grid.get_flat_coordinates`

This method returns a 2D NumPy array of shape ``(N_points, ndim)``, where:

- ``N_points`` is the total number of grid points (i.e., ``np.prod(grid.shape)``),
- ``ndim`` is the number of dimensions in the coordinate system.

Each row in the result corresponds to a point in the grid:

.. code-block:: python

    flat = grid.get_flat_coordinates()
    print(flat.shape)  # (N_points, ndim)

    print(flat[0])     # (x0, y0, z0)

This is ideal for vectorized computations, function evaluations, or exporting coordinates to other systems.

Coordinate Dictionaries
````````````````````````````````````````

To access coordinates in a **dictionary form**, mapping axis names to coordinate arrays, use:

- :meth:`~geometry.grids.base.Grid.get_coordinate_dict`

This is helpful when working with plotting libraries or modeling frameworks that expect named axes.

Example usage:

.. code-block:: python

    coords = grid.get_coordinate_dict()
    print(coords["x"])  # 1D array or meshgrid for the x-axis

By default, this returns 1D arrays for each axis. To get full meshgrid arrays:

.. code-block:: python

    mesh_coords = grid.get_coordinate_dict(meshgrid=True)
    print(mesh_coords["x"].shape)  # e.g., (Nx, Ny, ...)

These dictionaries ensure that the axis labels remain attached to their corresponding arrays, which improves
clarity when working with multi-dimensional data.

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
