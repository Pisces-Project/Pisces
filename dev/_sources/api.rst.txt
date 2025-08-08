.. _api:

API
===

This page provides the complete reference for all public classes, functions, and modules in the Pisces codebase.

Pisces is modular by design: components like profiles, fields, and models are constructed from interoperable
building blocks, enabling flexible and reusable modeling pipelines.

Components & Substructures
--------------------------

These modules define the lower-level components of Pisces, including radial profiles for density, temperature,
entropy, and more. These are often used directly or embedded in larger models.

.. autosummary::
    :caption: Profiles
    :toctree: _as_gen
    :recursive:
    :template: module.rst

    profiles

.. currentmodule:: pisces.geometry

.. autosummary::
    :caption: Geometry
    :toctree: _as_gen
    :recursive:
    :template: module.rst

    grids
    coordinates

Models
------

Pisces provides model generators for various astrophysical systems. These are higher-level tools built using
the component profiles and utilities defined in the codebase. Model generators typically produce structured
HDF5 datasets with precomputed physical fields.

.. currentmodule:: pisces

.. autosummary::
    :toctree: _as_gen
    :recursive:
    :template: module.rst

    models


Galaxy Cluster Models
''''''''''''''''''''''

Models for spherically symmetric galaxy clusters in hydrostatic equilibrium. These tools allow construction of
clusters from analytic density, temperature, or entropy profiles and compute derived quantities such as total
mass, gravitational potential, and sound speed.

.. currentmodule:: models.galaxy_clusters.spherical

.. autosummary::
    :nosignatures:

    SphericalGalaxyClusterModel
    MagnetizedSphericalGalaxyClusterModel


Stellar Models
''''''''''''''''''''''

Stellar model generators (e.g., for polytropes, ZAMS stars, or neutron stars) will be included here.

.. currentmodule:: models.stars.polytropes

.. autosummary::
    :nosignatures:

    PolytropicStarModel


Galaxy Models
''''''''''''''''''''''

Future models for galaxies (e.g., stellar+dark matter composites, rotating disks) will be listed here.

.. autosummary::
    :nosignatures:

    # Placeholder for future entries

.. note::

    Uh Oh! Looks like we haven't implemented any galaxy models yet. Want to help us
    get started? Check out the developer documentation for how to contribute: :ref:`developer_guide`.

Extension Modules & Data Representation
------------------------------------------------
These modules provide additional support / functionality for Pisces, including data representation
and extension modules for handling specific data formats or structures. This is where Pisces houses connections
to 3rd-party libraries / software.

.. currentmodule:: pisces

.. autosummary::
    :toctree: _as_gen
    :recursive:
    :template: module.rst

    particles

Mathematics & Physics
--------------------------

Depending on the model you're using and its assumptions, any number of physics or mathematical calculations may
occur, ranging from simple expressions to complex tensor manipulations. These modules contain specialized
subsystems for such computations.

.. autosummary::
    :toctree: _as_gen
    :recursive:
    :template: module.rst

    physics
    math_utils


Utilities
---------

General-purpose utilities used throughout Pisces, including physical constants, unit handling, and helper functions
for numerical or symbolic work.

.. autosummary::
    :toctree: _as_gen
    :recursive:
    :template: module.rst

    utilities
