"""Built-in Pisces coordinate systems."""

import numpy as np

from .base import CoordinateSystem


class Cartesian1DCoordinateSystem(CoordinateSystem):
    r"""
    One-dimensional Cartesian coordinate system.

    Coordinates
    -----------
    - :math:`x`: Position along a 1D Euclidean axis.

    Conversion
    ----------
    This system is already Cartesian. Conversion functions return the input.

    .. math::

        x = x

    Notes
    -----
    This coordinate system is used in problems where only one spatial
    dimension is relevant.
    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {}

    def convert_to_cartesian(self, x):
        return x

    def convert_from_cartesian(self, x):
        return x


class Cartesian2DCoordinateSystem(CoordinateSystem):
    r"""
    Two-dimensional Cartesian coordinate system.

    Coordinates
    -----------
    - :math:`x`: Horizontal position.
    - :math:`y`: Vertical position.

    Conversion
    ----------
    This system is already Cartesian. Conversion functions return the input.

    .. math::

        x = x \\
        y = y

    Notes
    -----
    Cartesian 2D coordinates describe flat Euclidean planes with orthogonal axes.
    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {}

    def convert_to_cartesian(self, x, y):
        return x, y

    def convert_from_cartesian(self, x, y):
        return x, y


class Cartesian3DCoordinateSystem(CoordinateSystem):
    r"""
    Three-dimensional Cartesian coordinate system.

    Coordinates
    -----------
    - :math:`x`: Position along the x-axis.
    - :math:`y`: Position along the y-axis.
    - :math:`z`: Position along the z-axis.

    Conversion
    ----------
    This system is already Cartesian. Conversion functions return the input.

    .. math::

        x = x \\
        y = y \\
        z = z

    Notes
    -----
    Standard orthonormal coordinate system in 3D Euclidean space.
    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {}

    def convert_to_cartesian(self, x, y, z):
        return x, y, z

    def convert_from_cartesian(self, x, y, z):
        return x, y, z


class SphericalCoordinateSystem(CoordinateSystem):
    r"""
    Spherical coordinate system in three dimensions.

    Coordinates
    -----------
    - :math:`r`: Radial distance from the origin.
    - :math:`\theta`: Polar angle (radians) from the positive z-axis.
    - :math:`\phi`: Azimuthal angle (radians) from the positive x-axis.

    Conversion to Cartesian
    -----------------------
    .. math::

        x = r \sin\theta \cos\phi \\
        y = r \sin\theta \sin\phi \\
        z = r \cos\theta

    Conversion from Cartesian
    -------------------------
    .. math::

        r = \sqrt{x^2 + y^2 + z^2} \\
        \theta = \arccos\left(\frac{z}{r}\right) \\
        \phi = \arctan2(y, x)

    Notes
    -----
    This system is used for problems with spherical symmetry. All angles
    are expressed in radians.
    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {}

    def convert_to_cartesian(self, r, theta, phi):
        # theta: polar angle (from z-axis), phi: azimuthal angle (from x-axis)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    def convert_from_cartesian(self, x, y, z):
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return r, theta, phi


class CylindricalCoordinateSystem(CoordinateSystem):
    r"""
    Cylindrical coordinate system in three dimensions.

    Coordinates
    -----------
    - :math:`r`: Radial distance from the z-axis.
    - :math:`\theta`: Azimuthal angle (radians) from the x-axis.
    - :math:`z`: Height along the z-axis.

    Conversion to Cartesian
    -----------------------
    .. math::

        x = r \cos\theta \\
        y = r \sin\theta \\
        z = z

    Conversion from Cartesian
    -------------------------
    .. math::

        r = \sqrt{x^2 + y^2} \\
        \theta = \arctan2(y, x) \\
        z = z

    Notes
    -----
    Suitable for problems with axial or cylindrical symmetry.
    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {}

    def convert_to_cartesian(self, r, theta, z):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return (x, y, z)

    def convert_from_cartesian(self, x, y, z):
        r = np.hypot(x, y)
        theta = np.arctan2(y, x)
        return (r, theta, z)


class PolarCoordinatesSystem(CoordinateSystem):
    r"""
    Polar coordinate system in two dimensions.

    Coordinates
    -----------
    - :math:`r`: Radial distance from the origin.
    - :math:`\theta`: Angle (radians) from the positive x-axis.

    Conversion to Cartesian
    -----------------------
    .. math::

        x = r \cos\theta \\
        y = r \sin\theta

    Conversion from Cartesian
    -------------------------
    .. math::

        r = \sqrt{x^2 + y^2} \\
        \theta = \arctan2(y, x)

    Notes
    -----
    Commonly used for circular or rotational symmetry in 2D. Angles are in radians.
    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {}

    def convert_to_cartesian(self, r, theta):
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y

    def convert_from_cartesian(self, x, y):
        r = np.hypot(x, y)
        theta = np.arctan2(y, x)
        return r, theta
