.. _coordinate_overview:

=============================
Coordinate Systems in Pisces
=============================

Coordinate systems in Pisces define how points in space are represented
and transformed across different geometries. They form the foundational
layer for structured grids, enabling grids to locate themselves in physical
or abstract coordinate spaces with precision.

Pisces supports a variety of coordinate systems, each
designed to suit different geometrical symmetries and computational
requirements. These include:

- **Cartesian (1D, 2D, 3D)** — For standard Euclidean geometry.
- **Polar** — For 2D problems with circular symmetry.
- **Cylindrical** — For 3D problems with axial symmetry.
- **Spherical** — For 3D problems with spherical symmetry.

For a full list of available coordinate systems, see the :mod:`~geometry.coordinates` module.

Coordinate System Basics
------------------------

Each coordinate system in Pisces:

- Specifies its number of dimensions (e.g., 2D or 3D).
- Defines named axes (e.g., ``["r", "theta", "phi"]`` for spherical coordinates).
- Implements conversion methods to and from Cartesian coordinates.
- Can be used to convert coordinates between different systems.

This ensures that operations defined in one coordinate system
can be translated across systems without ambiguity, facilitating
numerical methods, plotting, and physical modeling.

Creating Coordinate Systems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Coordinate systems are Python classes. Some coordinate systems (like Cartesian) do not require parameters:

.. code-block:: python

    from pisces.geometry.coordinates import Cartesian3DCoordinateSystem

    csys = Cartesian3DCoordinateSystem()

Others may define parameters (e.g., scalings or curvature) which can be passed during initialization:

.. code-block:: python

    from pisces.geometry.coordinates import SomeCustomSystem

    csys = SomeCustomSystem(alpha=1.2, beta=3.4)

Pisces will validate the parameters and raise an error if an unknown or invalid parameter is provided.
Using the parameters, the class will set up its conversion systems so that all of the relevant methods work
correctly.

Performing Coordinate Transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each coordinate system provides methods for converting to and from Cartesian coordinates:

.. code-block:: python

    r, theta, phi = 1.0, 0.5, 1.2
    csys = SphericalCoordinateSystem()

    x, y, z = csys.convert_to_cartesian(r, theta, phi)
    r2, theta2, phi2 = csys.convert_from_cartesian(x, y, z)

To convert between two arbitrary coordinate systems, use:

.. code-block:: python

    from pisces.geometry.coordinates import Cartesian3DCoordinateSystem

    spherical = SphericalCoordinateSystem()
    cartesian = Cartesian3DCoordinateSystem()

    r, theta, phi = 2.0, 1.0, 0.5
    x, y, z = spherical.convert_coords_to(cartesian, r, theta, phi)

    # Or convert from another system into spherical:
    r2, theta2, phi2 = spherical.convert_coords_from(cartesian, x, y, z)

Coordinate System IO
---------------------

Coordinate systems can be serialized to dictionaries or JSON for storage, logging, or transmission:

.. code-block:: python

    csys = SphericalCoordinateSystem()
    data_dict = csys.to_dict()
    json_str = csys.to_json_string()

They can be reconstructed using the corresponding class method:

.. code-block:: python

    from pisces.geometry.coordinates import SphericalCoordinateSystem

    csys = SphericalCoordinateSystem.from_dict(data_dict)
    csys = SphericalCoordinateSystem.from_json_string(json_str)

These representations store the class name and parameter values, allowing full reconstruction
in other sessions or systems.
