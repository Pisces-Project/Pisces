"""Temperature profiles for use in Pisces models.

The :mod:`~profiles.temperature` module provides a number of built-in profiles to model density and surface density
profiles in various contexts, including for galaxies, clusters of galaxies, stars, etc.
"""

from abc import ABC
from typing import Any

import unyt

from .base import BaseSphericalRadialProfile


class BaseSphericalTemperatureProfile(BaseSphericalRadialProfile, ABC):
    """Base class for spherical temperature profiles.

    This class provides a foundation for implementing various temperature profiles in spherical coordinates.
    It inherits from `BaseSphericalRadialProfile` and defines the basic structure and methods for temperature profiles.
    """

    # @@ CLASS ATTRIBUTES (INVARIANT) @@ #
    _IS_ABC = True

    # @@ CLASS ATTRIBUTES @@ #
    AXES: list[str] = ["r"]
    DEFAULT_PARAMETERS: dict[str, Any] = {}


class IsothermalTemperatureProfile(BaseSphericalTemperatureProfile):
    r"""Isothermal Temperature Profile.

    A constant-temperature profile often used as a first approximation in galaxy and cluster modeling.

    .. math::

        T(r) = T_0

    where :math:`T_0` is a uniform temperature.

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``T_0``
             - :math:`T_0`
             - Constant temperature

    Example
    -------
    .. plot::

        >>> from pisces.profiles.temperature import (
        ...     IsothermalTemperatureProfile,
        ... )
        >>> import matplotlib.pyplot as plt
        >>> r = np.linspace(0, 10, 100)
        >>> profile = IsothermalTemperatureProfile(
        ...     T_0=1e7 * unyt.Unit("K")
        ... )
        >>> T = profile(r)
        >>> plt.plot(r, T)
        >>> plt.xlabel("r [kpc]")
        >>> plt.ylabel("Temperature [K]")
        >>> plt.grid()
        >>> plt.show()

    """

    __IS_ABSTRACT__ = False
    __VARIABLES__ = ["r"]
    __VARIABLES_LATEX__ = [r"r"]
    __PARAMETERS__ = {
        "T_0": unyt.unyt_quantity(1e7, "K"),
    }

    @classmethod
    def __function__(cls, r, T_0=1e7):
        return T_0

    @classmethod
    def __function_units__(cls, r, T_0=unyt.unyt_quantity(1, "K"), **kwargs):
        return T_0.units


class BetaModelTemperatureProfile(BaseSphericalTemperatureProfile):
    r"""Beta Model Temperature Profile.

    This profile models the temperature decline using a single β-model, widely used in
    galaxy clusters and X-ray gas distributions.

    .. math::

        T(r) = T_0 \left(1 + \left(\frac{r}{r_c}\right)^2\right)^{-\beta}

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``T_0``
             - :math:`T_0`
             - Central temperature
           * - ``r_c``
             - :math:`r_c`
             - Core radius
           * - ``beta``
             - :math:`\beta`
             - Slope parameter
    """

    __IS_ABSTRACT__ = False
    __VARIABLES__ = ["r"]
    __VARIABLES_LATEX__ = [r"r"]
    __PARAMETERS__ = {
        "T_0": unyt.unyt_quantity(5.0, "keV"),
        "r_c": unyt.unyt_quantity(500.0, "kpc"),
        "beta": unyt.unyt_quantity(0.7, ""),
    }

    @classmethod
    def __function__(cls, r, T_0, r_c, beta):
        return T_0 * (1 + (r / r_c) ** 2) ** (-beta)

    @classmethod
    def __function_units__(cls, r, T_0, **kwargs):
        return T_0.units


class DoubleBetaTemperatureProfile(BaseSphericalTemperatureProfile):
    r"""Double Beta Temperature Profile.

    This model combines two β-components to describe systems with two distinct thermal structures.

    .. math::

        T(r) = T_0 \left(1 + \left(\frac{r}{r_c}\right)^2\right)^{-\beta_1}
             + T_1 \left(1 + \left(\frac{r}{r_c}\right)^2\right)^{-\beta_2}

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``T_0``
             - :math:`T_0`
             - Temperature of first component
           * - ``T_1``
             - :math:`T_1`
             - Temperature of second component
           * - ``r_c``
             - :math:`r_c`
             - Core radius (shared)
           * - ``beta_1``
             - :math:`\beta_1`
             - Slope of first component
           * - ``beta_2``
             - :math:`\beta_2`
             - Slope of second component
    """

    __IS_ABSTRACT__ = False
    __VARIABLES__ = ["r"]
    __VARIABLES_LATEX__ = [r"r"]
    __PARAMETERS__ = {
        "T_0": unyt.unyt_quantity(5.0, "keV"),
        "T_1": unyt.unyt_quantity(3.0, "keV"),
        "r_c": unyt.unyt_quantity(500.0, "kpc"),
        "beta_1": unyt.unyt_quantity(0.7, ""),
        "beta_2": unyt.unyt_quantity(1.0, ""),
    }

    @classmethod
    def __function__(cls, r, T_0, T_1, r_c, beta_1, beta_2):
        return T_0 * (1 + (r / r_c) ** 2) ** (-beta_1) + T_1 * (1 + (r / r_c) ** 2) ** (-beta_2)

    @classmethod
    def __function_units__(cls, r, T_0, **kwargs):
        return T_0.units


class CoolingFlowTemperatureProfile(BaseSphericalTemperatureProfile):
    r"""Cooling Flow Temperature Profile.

    A power-law model for cooling flows in cluster cores.

    .. math::

        T(r) = T_0 \left( \frac{r}{r_c} \right)^{-a}

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``T_0``
             - :math:`T_0`
             - Central temperature
           * - ``r_c``
             - :math:`r_c`
             - Core radius
           * - ``a``
             - :math:`a`
             - Power-law slope
    """

    __IS_ABSTRACT__ = False
    __VARIABLES__ = ["r"]
    __VARIABLES_LATEX__ = [r"r"]
    __PARAMETERS__ = {
        "T_0": unyt.unyt_quantity(5.0, "keV"),
        "r_c": unyt.unyt_quantity(100.0, "kpc"),
        "a": unyt.unyt_quantity(0.8, ""),
    }

    @classmethod
    def __function__(cls, r, T_0, r_c, a):
        return T_0 * (r / r_c) ** (-a)

    @classmethod
    def __function_units__(cls, r, T_0, **kwargs):
        return T_0.units


class AM06TemperatureProfile(BaseSphericalTemperatureProfile):
    r"""Ascasibar & Markevitch (2006) Temperature Profile.

    Flexible 3-parameter model for cool-core clusters.

    .. math::

        T(r) = \frac{T_0}{1 + \frac{r}{a}} \cdot \frac{c + \frac{r}{a_c}}{1 + \frac{r}{a_c}}

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - ``T_0``
             - :math:`T_0`
             - Central temperature
           * - ``a``
             - :math:`a`
             - Transition scale radius
           * - ``a_c``
             - :math:`a_c`
             - Core radius
           * - ``c``
             - :math:`c`
             - Inner slope factor
    """

    __IS_ABSTRACT__ = False
    __VARIABLES__ = ["r"]
    __VARIABLES_LATEX__ = [r"r"]
    __PARAMETERS__ = {
        "T_0": unyt.unyt_quantity(4.0, "keV"),
        "a": unyt.unyt_quantity(300.0, "kpc"),
        "a_c": unyt.unyt_quantity(50.0, "kpc"),
        "c": unyt.unyt_quantity(0.2, ""),
    }

    @classmethod
    def __function__(cls, r, T_0, a, a_c, c):
        return T_0 / (1 + r / a) * (c + r / a_c) / (1 + r / a_c)

    @classmethod
    def __function_units__(cls, r, T_0, **kwargs):
        return T_0.units


class VikhlininTemperatureProfile(BaseSphericalTemperatureProfile):
    r"""Vikhlinin Temperature Profile.

    Empirical model that captures a peaked core, a power-law decline, and a central cooling component.

    .. math::

        T(r) = T_0 \left( \frac{r}{r_t} \right)^{-a}
                     \left[1 + \left( \frac{r}{r_t} \right)^b \right]^{-c/b}
                     \cdot \frac{\left(\left(\frac{r}{r_{\text{cool}}}\right)^{a_{\text{cool}}} +
                     \frac{T_{\text{min}}}{T_0}\right)}
                     {\left(\left(\frac{r}{r_{\text{cool}}}\right)^{a_{\text{cool}}} + 1\right)}

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``T_0``
             - :math:`T_0`
             - Central peak temperature
           * - ``r_t``
             - :math:`r_t`
             - Truncation radius
           * - ``a``
             - :math:`a`
             - Power-law inner slope
           * - ``b``
             - :math:`b`
             - Sharpness of transition
           * - ``c``
             - :math:`c`
             - Outer slope
           * - ``T_min``
             - :math:`T_{\text{min}}`
             - Minimum temperature in cooling region
           * - ``r_cool``
             - :math:`r_{\text{cool}}`
             - Radius of cooling region
           * - ``a_cool``
             - :math:`a_{\text{cool}}`
             - Slope of cooling core
    """

    __IS_ABSTRACT__ = False
    __VARIABLES__ = ["r"]
    __VARIABLES_LATEX__ = [r"r"]
    __PARAMETERS__ = {
        "T_0": unyt.unyt_quantity(5.0, "keV"),
        "r_t": unyt.unyt_quantity(100.0, "kpc"),
        "a": unyt.unyt_quantity(0.1, ""),
        "b": unyt.unyt_quantity(2.0, ""),
        "c": unyt.unyt_quantity(1.2, ""),
        "T_min": unyt.unyt_quantity(2.0, "keV"),
        "r_cool": unyt.unyt_quantity(10.0, "kpc"),
        "a_cool": unyt.unyt_quantity(0.2, ""),
    }

    @classmethod
    def __function__(cls, r, T_0, r_t, a, b, c, T_min, r_cool, a_cool):
        x = (r / r_cool) ** a_cool
        t1 = (r / r_t) ** (-a)
        t2 = (1 + (r / r_t) ** b) ** (-c / b)
        t3 = (x + T_min / T_0) / (x + 1)
        return T_0 * t1 * t2 * t3

    @classmethod
    def __function_units__(cls, r, T_0, **kwargs):
        return T_0.units
