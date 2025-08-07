"""
Template example of a pisces coordinate system.

This template should be copied and modified to create new coordinate systems
for Pisces. It provides a structured outline and documentation to guide
the implementation of new systems.
"""

from abc import ABC

from .base import CoordinateSystem


class _TemplateCoordinateSystem(CoordinateSystem, ABC):
    r"""
    Abstract template for Pisces coordinate systems.

    This class serves as a base for defining new coordinate systems
    in the Pisces framework. It outlines the required interface and provides
    structured documentation for subclass authors.

    Coordinate System Definition
    ----------------------------
    A coordinate system maps between native coordinates :math:`(u, v, w)`
    and standard Cartesian coordinates :math:`(x, y, z)` via explicit
    forward and inverse transformations.

    Let the system be defined by three coordinates:

    - :math:`u`: First coordinate (e.g., radial distance :math:`r`)
    - :math:`v`: Second coordinate (e.g., angle :math:`\theta`)
    - :math:`w`: Third coordinate (e.g., height or depth :math:`z`)

    These labels are placeholders; real systems (e.g., cylindrical, spherical)
    should override them with appropriate names and equations.

    Cartesian Conversion
    --------------------
    A concrete subclass must define how to convert to and from Cartesian space.

    **To Cartesian:**

    .. math::

        x = f_x(u, v, w) \\
        y = f_y(u, v, w) \\
        z = f_z(u, v, w)

    **From Cartesian:**

    .. math::

        u = f_u(x, y, z) \\
        v = f_v(x, y, z) \\
        w = f_w(x, y, z)

    These transformation equations should be compatible with
    vectorized NumPy arrays or broadcasting scalars.

    Subclassing Guidelines
    ----------------------
    When creating a new coordinate system, follow these steps:

    1. Set ``__IS_ABSTRACT__ = False`` to enable registration.
    2. Define ``__PARAMETERS__`` if your system needs any constants
       (e.g., eccentricity, curvature).
    3. Implement the following methods:
       - :meth:`convert_to_cartesian`
       - :meth:`convert_from_cartesian`
    4. Optionally override:
       - :meth:`__repr__` for informative string output.
       - :meth:`_validate_parameters` for custom checks.
    5. Add a docstring describing:
       - The coordinate meaning.
       - Conversion equations.
       - Applications (e.g., spherical symmetry).

    Example
    -------
    Here's a simplified cylindrical coordinate system:

    .. math::

        x = r \cos\theta \\
        y = r \sin\theta \\
        z = z

    .. math::

        r = \sqrt{x^2 + y^2} \\
        \theta = \arctan2(y, x) \\
        z = z

    Notes
    -----
    - All angles must be in radians.
    - Cartesian transformations must preserve shapes and support broadcasting.
    - This class is not registered and is meant purely as a development template.
    """

    __IS_ABSTRACT__ = True
    """Mark this false to ensure that it gets registered in the registry."""
    __PARAMETERS__ = {}
    """Add parameters here with their default values."""

    def convert_to_cartesian(self, *coords):
        """
        Convert native coordinates to Cartesian coordinates.

        Parameters
        ----------
        coords : tuple of array-like
            Coordinates in this coordinate system.

        Returns
        -------
        tuple of array-like
            Cartesian coordinates (x, y, z)

        Raises
        ------
        NotImplementedError
            Always — subclasses must override this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.convert_to_cartesian() must be implemented "
            "in subclasses to define how to map to Cartesian coordinates."
        )

    def convert_from_cartesian(self, *coords):
        """
        Convert Cartesian coordinates to this coordinate system.

        Parameters
        ----------
        coords : tuple of array-like
            Cartesian coordinates (x, y, z)

        Returns
        -------
        tuple of array-like
            Coordinates in this coordinate system

        Raises
        ------
        NotImplementedError
            Always — subclasses must override this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.convert_from_cartesian() must be implemented "
            "in subclasses to define how to map from Cartesian coordinates."
        )
