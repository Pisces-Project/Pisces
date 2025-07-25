"""Density profiles for use in Pisces models.

The :mod:`~profiles.density` module provides a number of built-in profiles to model density and surface density
profiles in various contexts, including for galaxies, clusters of galaxies, stars, etc.
"""

from abc import ABC
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import sympy as sp
import unyt
from math_utils.integration import integrate_mass
from scipy.integrate import quad, quad_vec

from pisces.profiles.base import (
    BaseCylindricalDiskProfile,
    BaseProfile,
    BaseSphericalRadialProfile,
    derived_profile,
)
from pisces.utilities.config import pisces_config

# Type Hints
_UnitType = Union[str, unyt.Unit]
_UnitValue = Union[unyt.unyt_array, unyt.unyt_quantity]

if TYPE_CHECKING:
    from astropy.cosmology import Cosmology


# ======================================= #
# Radial Profiles (Spherical)             #
# ======================================= #
class BaseSphericalDensityProfile(BaseSphericalRadialProfile, ABC):
    r"""Abstract base class defining the shared behaviors of all spherically symmetric density profiles.

    This class provides a unified interface for constructing and analyzing
    spherical density profiles common in astrophysics, such as those describing
    dark matter halos, stellar distributions, or gas profiles.

    Several additional features are implemented to support both symbolic
    and numerical computations within a consistent, unit-aware framework.

    Features
    --------

    .. dropdown:: Derived Profiles

        The following symbolic derived profiles are automatically constructed and accessible
        via :meth:`~profiles.base.BaseProfile.get_derived_profile`:

        - ``derivative`` : Radial derivative of the density profile (from parent class).
        - ``enclosed_mass`` : The mass enclosed within radius :math:`r`.

          .. math::

              M(<r) = \int_0^r dr' 4\pi r'^2 \rho(r')

        - ``gravitational_field`` : Symbolic gravitational field at radius ``r``.

          .. math::

              \nabla \Phi = \frac{-GM(<r)}{r^2} = \frac{-G}{r^2} \int_0^r dr' 4\pi r'^2 \rho(r')

        - ``gravitational_potential`` : Symbolic gravitational potential at radius ``r``.

          .. math::

              \Phi = \frac{GM(<r)}{r} = \frac{G}{r} \int_0^r dr' 4\pi r'^2 \rho(r')


    .. dropdown:: Numerical Operations

        This class provides several numerical methods for common astrophysical computations:

        - **Gravitational Operations**:

          - :meth:`compute_circular_velocity` : The velocity of a circular orbit at
            a given radius.
          - :meth:`compute_escape_velocity` : The escape velocity as a function of
            the radius.

        - **Projection**:

          - :meth:`compute_surface_density` : Projected surface density profile.

        - **Mathematical Operations**:

          - :meth:`compute_total_mass` : Total mass (if the profile converges).
          - :meth:`compute_enclosed_mass` : Numerical enclosed mass profile.
          - :meth:`compute_fractional_mass_radius` : Radius enclosing a target mass fraction.

        - **Cosmology**:

          - :meth:`compute_cosmological_overdensity_profile` : Overdensity relative to critical density.
          - :meth:`compute_cosmological_overdensity_radius` : Radius enclosing a target overdensity.

        - **Lensing**:

          - :meth:`compute_deflection_angle` : Gravitational lensing deflection angle.
          - :meth:`compute_einstein_radius` : Einstein ring angular radius.
          - :meth:`compute_lensing_convergence` : Lensing convergence :math:`\kappa(R)`.

    Notes
    -----
    - This class is abstract; concrete subclasses must set ``__IS_ABSTRACT__ = False``.
    - All subclasses define ``__function__`` and ``__function_units__`` to specify
      the symbolic density expression and its units.
    - Units are handled consistently using :mod:`unyt`.
    - Cosmological methods require :mod:`astropy.cosmology` and a valid cosmology.
    - Derived profiles follow the standard symbolic setup of :class:`BaseProfile`.

    Examples
    --------
    **Subclassing Example**

    .. code-block:: python

        from pisces.profiles.density import (
            BaseSphericalDensityProfile,
        )
        import sympy as sp
        import unyt


        class SimplePowerLawDensityProfile(
            BaseSphericalDensityProfile
        ):
            __IS_ABSTRACT__ = False
            __PARAMETERS__ = {
                "rho_0": 1.0 * unyt.Msun / unyt.kpc**3
            }

            @classmethod
            def __function__(cls, r, rho_0=1.0):
                return rho_0 / r**2

            @classmethod
            def __function_units__(
                cls,
                r,
                rho_0=unyt_quantity("1 Msun/kpc**3"),
            ):
                r, rho_0 = unyt.Unit(r), unyt.Unit(rho_0)
                return rho_0 / r**2

    **Numerical Usage**

    .. code-block:: python

        profile = SimplePowerLawDensityProfile(
            rho_0=0.1 * unyt.Msun / unyt.kpc**3
        )

        # Density at radius 1 kpc
        rho = profile(1.0 * unyt.kpc)

        # Enclosed mass at 5 kpc
        M_enc = profile.compute_enclosed_mass(
            5 * unyt.kpc
        )

        # Surface density profile
        Sigma = profile.compute_surface_density(
            2.0 * unyt.kpc
        )

        # Gravitational potential as symbolic derived profile
        phi_profile = profile.get_derived_profile(
            "gravitational_potential"
        )
        phi_expr = phi_profile.__function__(
            sp.Symbol("r"), G=1.0, rho_0=1.0
        )

    See Also
    --------
    ~profiles.base.BaseSphericalRadialProfile, ~profiles.base.BaseProfile, NFWDensityProfile, HernquistDensityProfile

    """

    __IS_ABSTRACT__ = True

    # ------------------------------ #
    # Derived Profile Implementation #
    # ------------------------------ #
    @derived_profile("enclosed_mass")
    @classmethod
    def _enclosed_mass(cls):
        def _func(r, **params):
            rho = cls.__function__(r, **params)
            return 4 * sp.pi * sp.integrate(rho * r**2, (r, 0, r))

        def _unit_func(r, **param_units):
            rho_unit = cls.__function_units__(r, **param_units)
            return rho_unit * r**3

        return _func, _unit_func, ["r"], cls.__PARAMETERS__.copy()

    @derived_profile("gravitational_field")
    @classmethod
    def _gravitational_field(cls):
        def _func(r, **params):
            G = params.pop("G")
            rho = cls.__function__(r, **params)
            M = 4 * sp.pi * sp.integrate(rho * r**2, (r, 0, r))
            return G * M / r**2

        def _unit_func(r, **params):
            G = params.pop("G").units
            rho_unit = cls.__function_units__(r, **params)
            return G * rho_unit * r

        # Add G to the available parameters.
        parameters = cls.__PARAMETERS__.copy()
        # noinspection PyUnresolvedReferences
        parameters["G"] = unyt.physical_constants.gravitational_constant

        return _func, _unit_func, ["r"], parameters

    @derived_profile("gravitational_potential")
    @classmethod
    def _gravitational_potential(cls):
        def _func(r, **params):
            G = params.pop("G")
            rho = cls.__function__(r, **params)
            M = 4 * sp.pi * sp.integrate(rho * r**2, (r, 0, r))
            return -G * M / r

        def _unit_func(r, **param_units):
            G = param_units.pop("G").units
            rho_unit = cls.__function_units__(r, **param_units)
            return G * rho_unit * r**2

        # Add G to the available parameters.
        parameters = cls.__PARAMETERS__.copy()
        # noinspection PyUnresolvedReferences
        parameters["G"] = unyt.physical_constants.gravitational_constant

        return _func, _unit_func, ["r"], parameters

    # ------------------------------ #
    # Numerical Computations         #
    # ------------------------------ #
    def compute_surface_density(self, R: _UnitValue, units: _UnitType | None = None, **kwargs) -> _UnitValue:
        r"""Numerically compute the projected surface density at radius R from the origin.

        Mathematically, this is the operation

        .. math::

            \Sigma(R) = 2 \int_0^\infty \rho \left( \sqrt{R^2 + z^2} \right) dz,

        where :math:`z` refers to the displacement along the line of sight.

        For some value of :math:`R`, this method will perform the relevant integral
        of the profile and provide the calculated surface density.

        Parameters
        ----------
        R : ~unyt.array.unyt_array or ~unyt.array.unyt_quantity
            The projected radius from the origin in physical units with dimension length.
            `R` may be either an array of values or a scalar.
        units: ~unyt.unit_object.Unit or str, optional
            The units in which to return the calculated surface density. By
            default, the internal units are preserved.
        kwargs:
            Additional keyword arguments to pass on to :func:`~scipy.integrate.quad_vec`.

        Returns
        -------
        sigma : ~unyt.array.unyt_array or ~unyt.array.unyt_quantity
            Surface density at each `R`, with correct units.

        Example
        -------
        .. code-block:: python

            import numpy as np
            from pisces.profiles.density import (
                NFWDensityProfile,
            )

            # Create a density profile with characteristic parameters
            profile = NFWDensityProfile(rho_0=1.0, r_s=10.0)

            # Define projected radii (can be a scalar or array)
            R = np.linspace(0.1, 100.0, 50)  # kpc

            # Compute the surface density at each radius
            sigma = profile.compute_surface_density(
                R, units="Msun/pc**2"
            )

            print(sigma)

        """
        # Coerce the input R array into a valid unyt_array and
        # then extract the raw buffer and units.
        R_array = unyt.array.unyt_array(R)
        r_arr, r_units = R_array.d, R_array.units

        # Begin with the actual integration step before
        # concerning ourselves with the units. We define
        # the integrand in a unit-less form.
        def _integrand_(z, _rr):
            xi = np.sqrt(_rr**2 + z**2)
            return self.__call_no_units__(xi)

        # Perform the integration vis-a-vis the quad_vec function
        # from SciPy.
        result = 2 * quad_vec(_integrand_, 0, np.inf, args=(r_arr,), full_output=False, **kwargs)[0]

        # Determine the units. We have standard propagation
        # to get rho, and then we integrate through by z, which
        # should ARBITRARILY have the same units as _rr, so we
        # just multiply by a factor of the length unit.
        sd_unit = self.get_output_units(r_units) * r_units
        sd = result * sd_unit

        if units is not None:
            sd = sd.to(units)

        if np.isscalar(R):
            return sd[0]
        return sd

    def compute_enclosed_mass(self, r: _UnitValue, units: _UnitType | None = None, **kwargs) -> _UnitValue:
        r"""Numerically compute the enclosed mass :math:`M(r)` within radius :math:`r`.

        The enclosed mass is given by:

        .. math::

            M(r) = 4 \pi \int_0^r \rho(r') \, r'^2 \, dr',

        where :math:`\rho(r)` is the spherically symmetric density profile.

        Parameters
        ----------
        r : :class:`~unyt.unyt_quantity` or :class:`~unyt.unyt_array`
            Radius or array of radii at which to compute enclosed mass. Must carry length units (e.g., kpc, pc).
        units : str or :class:`~unyt.Unit`, optional
            Desired output units for mass. If not provided, output units follow from the profile parameters.
        **kwargs
            Additional arguments passed to :func:`scipy.integrate.quad`.

        Returns
        -------
        mass : :class:`~unyt.unyt_quantity` or :class:`~unyt.unyt_array`
            Enclosed mass at each radius with appropriate units.

        Notes
        -----
        - Assumes spherical symmetry.
        - Uses :func:`scipy.integrate.quad` for numerical integration.
        - Units are handled automatically using :mod:`unyt`.

        Examples
        --------
        Compute enclosed mass for a Hernquist profile:

        .. code-block:: python

            import numpy as np
            import unyt
            from pisces.profiles.density import (
                HernquistDensityProfile,
            )

            profile = HernquistDensityProfile(
                rho_0=0.05 * unyt.Msun / unyt.kpc**3,
                r_s=10 * unyt.kpc,
            )

            radii = np.logspace(-1, 2, 100) * unyt.kpc
            mass = profile.compute_enclosed_mass(
                radii, units="Msun"
            )

            print(mass)

        See Also
        --------
        :meth:`compute_total_mass`, :meth:`compute_surface_density`, :func:`scipy.integrate.quad`

        """
        r_array = unyt.array.unyt_array(r)
        r_values, r_unit = r_array.value, r_array.units

        # Units for density and mass
        rho_unit = self.get_output_units(r_unit)
        mass_unit = rho_unit * r_unit**3

        # Unitless density function for integration
        def _density_fn(r_):
            return self.__call_no_units__(r_)

        # Enclosed mass integral for each radius
        mass_values = integrate_mass(_density_fn, r_values, **kwargs)

        mass = mass_values * mass_unit

        if units is not None:
            mass = mass.to(units)

        if np.isscalar(r):
            return mass[0]
        return mass

    def compute_total_mass(self, units: _UnitType | None = None, **kwargs) -> unyt.unyt_quantity:
        r"""Numerically compute the total mass of the profile by integrating to infinity.

        .. math::

            M_{\mathrm{tot}} = 4 \pi \int_0^\infty \rho(r) \, r^2 \, dr

        If the profile has finite total mass, this method returns its value. Divergent profiles
        will raise a :class:`RuntimeError`.

        Parameters
        ----------
        units : str or :class:`~unyt.Unit`, optional
            Desired output units for mass. Defaults to units implied by the profile parameters.
        **kwargs
            Additional arguments passed to :func:`scipy.integrate.quad`.

        Returns
        -------
        mass : :class:`~unyt.unyt_quantity`
            Total mass with correct units.

        Raises
        ------
        RuntimeError
            If the integral fails to converge or the result is infinite/NaN.

        Examples
        --------

        .. code-block:: python

            import unyt
            from pisces.profiles.density import (
                PlummerDensityProfile,
            )

            profile = PlummerDensityProfile(
                M=1e11 * unyt.Msun, r_s=5 * unyt.kpc
            )
            total_mass = profile.compute_total_mass(
                units="Msun"
            )

            print(f"Total mass: {total_mass:.3e}")

        See Also
        --------
        :meth:`compute_enclosed_mass`, :func:`scipy.integrate.quad`

        """
        import warnings

        from scipy.integrate import IntegrationWarning

        # Use arbitrary length scale (units cancel internally)
        r_unit = unyt.Unit("kpc")
        rho_unit = self.get_output_units(r_unit)
        mass_unit = rho_unit * r_unit**3

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always", IntegrationWarning)

            def _density_fn(r_):
                return self(r_).value

            result, err = quad(lambda r_: _density_fn(r_) * 4 * np.pi * r_**2, 0, np.inf, **kwargs)[:2]

            # Check for divergence (nan/inf result)
            if not np.isfinite(result):
                raise RuntimeError("Total mass integral did not converge: result is infinite or NaN.")

            # Check for integration warnings (e.g., max subdivisions exceeded)
            for warn in warning_list:
                if issubclass(warn.category, IntegrationWarning):
                    raise RuntimeError(f"Total mass integral raised an IntegrationWarning: {warn.message}")

        # Handle the units.
        value = result * mass_unit

        if units is not None:
            return value.to(units)
        else:
            return value

    def compute_fractional_mass_radius(
        self,
        rmin: float | unyt.unyt_quantity,
        rmax: float | unyt.unyt_quantity,
        fraction: float = 0.5,
        units: _UnitType | None = "kpc",
        **kwargs,
    ):
        r"""Find the radius enclosing a given fraction of the total mass.

        Solves numerically for radius :math:`r_f` such that:

        .. math::

            M(r_f) = f \times M_{\mathrm{tot}}

        where :math:`f` is the desired mass fraction.

        Parameters
        ----------
        rmin : float or :class:`~unyt.unyt_quantity`
            Lower bound for the radius search interval.
        rmax : float or :class:`~unyt.unyt_quantity`
            Upper bound for the radius search interval.
        fraction : float, optional
            Mass fraction to enclose, between 0 and 1. Default is 0.5 (half-mass radius).
        units : str or :class:`~unyt.Unit`, optional
            Units for `rmin` and `rmax`. Default is ``kpc``.
        **kwargs
            Additional arguments passed to :meth:`compute_enclosed_mass` and :meth:`compute_total_mass`.

        Returns
        -------
        radius : :class:`~unyt.unyt_quantity`
            Radius enclosing the specified mass fraction.

        Raises
        ------
        ValueError
            If the mass fraction is outside (0, 1) or search bounds are invalid.

        Examples
        --------

        .. code-block:: python

            import unyt
            from pisces.profiles.density import (
                HernquistDensityProfile,
            )

            profile = HernquistDensityProfile(
                rho_0=0.05 * unyt.Msun / unyt.kpc**3,
                r_s=10 * unyt.kpc,
            )

            half_mass_radius = (
                profile.compute_fractional_mass_radius(
                    0.1, 100, fraction=0.5, units="kpc"
                )
            )
            print(f"Half-mass radius: {half_mass_radius:.2f}")

        See Also
        --------
        :meth:`compute_total_mass`, :meth:`compute_enclosed_mass`, :func:`scipy.optimize.brentq`

        """
        from scipy.optimize import brentq

        if not 0 < fraction < 1:
            raise ValueError("Fraction must be between 0 and 1.")

        # Establish unit system
        unit = unyt.Unit(units)

        # Handle rmin and rmax as unit-consistent floats
        rmin = unyt.array.unyt_quantity(rmin, unit).to_value(unit)
        rmax = unyt.array.unyt_quantity(rmax, unit).to_value(unit)

        if rmin <= 0 or rmax <= rmin:
            raise ValueError("Require 0 < rmin < rmax for valid search bounds.")

        # Compute total mass with error handling
        try:
            total_mass = self.compute_total_mass(units="g", **kwargs).value
        except Exception as e:
            raise ValueError(f"Total mass calculation did not converge: {e}") from e

        # Residual function for root finding
        def _residual(r_):
            m_enc = self.compute_enclosed_mass(r_ * unit, **kwargs).to_value("g")
            return m_enc - (fraction * total_mass)

        # Root finding
        r_solution = brentq(_residual, rmin, rmax)

        return r_solution * unit

    def compute_cosmological_overdensity_profile(
        self, z: float, R: _UnitValue, cosmology: Optional["Cosmology"] = None, **kwargs
    ):
        r"""Compute the spherical overdensity profile relative to the critical density at redshift :math:`z`.

        The overdensity is defined as:

        .. math::

            \Delta(R) = \frac{3 M(R)}{4 \pi R^3 \rho_{\mathrm{crit}}(z)}

        where :math:`M(R)` is the enclosed mass and :math:`\rho_{\mathrm{crit}}(z)` is the critical density.

        Parameters
        ----------
        z : float
            Redshift at which to compute the critical density.
        R : :class:`~unyt.unyt_quantity` or :class:`~unyt.unyt_array`
            Radius or array of radii where overdensity is computed (must carry length units).
        cosmology : ~astropy.cosmology.Cosmology, optional
            Cosmology used to compute :math:`\rho_{\mathrm{crit}}(z)`. Defaults to
            ``pisces_config['physics.default_cosmology']``.
        **kwargs
            Additional arguments passed to :meth:`compute_enclosed_mass`.

        Returns
        -------
        overdensity : :class:`numpy.ndarray`
            Dimensionless overdensity :math:`\Delta(R)` at each radius.

        Raises
        ------
        ValueError
            If cosmology cannot compute the critical density.

        Examples
        --------

        .. code-block:: python

            import unyt
            from astropy.cosmology import Planck18
            from pisces.profiles.density import (
                NFWDensityProfile,
            )

            profile = NFWDensityProfile(
                rho_0=0.05 * unyt.Msun / unyt.kpc**3,
                r_s=10 * unyt.kpc,
            )

            R = unyt.unyt_array([1, 10, 100], "kpc")
            delta = profile.compute_cosmological_overdensity_profile(
                z=0.5, R=R, cosmology=Planck18
            )

            print(delta)

        See Also
        --------
        :meth:`compute_enclosed_mass`
        :meth:`compute_cosmological_overdensity_radius`
        :class:`astropy.cosmology.Cosmology`

        """
        # Determine the cosmology and extract the critical
        # density from it. This requires accessing the configuration
        # and loading astropy cosmology.
        import astropy.cosmology as cosmo

        if cosmology is None:
            _default = pisces_config["physics.default_cosmology"]
            try:
                cosmology = getattr(cosmo, _default)
            except Exception as exp:
                raise ValueError(f"Default cosmology `{_default}` is not available in astropy cosmology.") from exp

        # Coerce the inputs so that we have the
        # relevant length units. We then propagate
        # to get the density units so that we have minimal
        # FPE issues.
        R_array = unyt.array.unyt_array(R)
        density_array = self(R_array)

        # Try to obtain the critical density from
        # the desired cosmology. We force to CGS units
        # so that we don't have to worry about unyt / astropy unit
        # operation parity.
        try:
            rho_crit = cosmology.critical_density(z).to_value(str(density_array.units))
        except Exception as exp:
            raise ValueError(f"Provided cosmology object failed to compute critical density: {exp}") from exp

        # Compute the mass and the relevant overdensities.
        rho_crit = rho_crit * density_array.units
        mass = self.compute_enclosed_mass(R_array, **kwargs)
        delta = (3 * mass) / (4 * np.pi * R_array**3 * rho_crit)

        return delta.d

    def compute_cosmological_overdensity_radius(
        self,
        z: float,
        delta_target: float,
        rmin: float | unyt.unyt_quantity,
        rmax: float | unyt.unyt_quantity,
        units: _UnitType | None = "kpc",
        cosmology: Optional["Cosmology"] = None,
        **kwargs,
    ):
        r"""Find the radius enclosing a target overdensity relative to the critical density at redshift :math:`z`.

        Numerically solves for radius :math:`r_{\Delta}` satisfying:

        .. math::

            \Delta(r_{\Delta}) = \Delta_{\mathrm{target}}

        Parameters
        ----------
        z : float
            Redshift at which to compute overdensity.
        delta_target : float
            Desired overdensity value (must be positive).
        rmin : float
            Lower bound for root-finding search interval, in units specified by ``units``.
        rmax : float
            Upper bound for root-finding search interval.
        units : str or :class:`~unyt.Unit`, optional
            Units for ``rmin`` and ``rmax``. Default is ``kpc``.
        cosmology : :class:`~astropy.cosmology.Cosmology`, optional
            Cosmology used to compute :math:`\rho_{\mathrm{crit}}(z)`. Defaults to ``pisces_config``.
        **kwargs
            Additional arguments passed to :meth:`compute_enclosed_mass`.

        Returns
        -------
        r_delta : :class:`~unyt.unyt_quantity`
            Radius enclosing the target overdensity.

        Raises
        ------
        ValueError
            If inputs are invalid or critical density calculation fails.

        Examples
        --------

        .. code-block:: python

            from astropy.cosmology import Planck18
            from pisces.profiles.density import (
                HernquistDensityProfile,
            )

            profile = HernquistDensityProfile(
                rho_0=0.05, r_s=10
            )

            r_delta = profile.compute_cosmological_overdensity_radius(
                z=0.3,
                delta_target=200,
                rmin=1,
                rmax=500,
                units="kpc",
                cosmology=Planck18,
            )

            print(f"r_200 = {r_delta:.2f}")

        See Also
        --------
        :meth:`compute_cosmological_overdensity_profile`, :meth:`compute_enclosed_mass`, :func:`scipy.optimize.brentq`

        """
        import astropy.cosmology as cosmo
        from scipy.optimize import brentq

        # Validate that the delta_target is legitimate and
        # coerce the units so that they behave nicely.
        if delta_target <= 0:
            raise ValueError("Overdensity target must be positive.")

        unit = unyt.Unit(units)
        rmin = unyt.array.unyt_quantity(rmin, unit).to_value(unit)
        rmax = unyt.array.unyt_quantity(rmax, unit).to_value(unit)

        if rmin <= 0 or rmax <= rmin:
            raise ValueError("Require 0 < rmin < rmax for valid search bounds.")

        # Resolve cosmology, fallback to pisces_config
        if cosmology is None:
            _default = pisces_config["physics.default_cosmology"]
            try:
                cosmology = getattr(cosmo, _default)
            except Exception as exp:
                raise ValueError(f"Default cosmology `{_default}` is not available in astropy.cosmology.") from exp

        # Get critical density with units consistent to profile density
        rho_unit = self.get_output_units(unit)
        try:
            rho_crit = cosmology.critical_density(z).to_value(str(rho_unit))
        except Exception as exp:
            raise ValueError(f"Failed to compute critical density with provided cosmology: {exp}") from exp

        # Residual function for root-finding
        rho_crit = rho_crit * rho_unit

        def _residual(r_):
            mass = self.compute_enclosed_mass(r_ * unit, **kwargs)
            delta_r = (3 * mass) / (4 * np.pi * (r_ * unit) ** 3 * rho_crit)
            return delta_r.to_value() - delta_target

        # Root finding
        r_solution = brentq(_residual, rmin, rmax)

        return r_solution * unit

    def compute_circular_velocity(
        self,
        r: _UnitValue,
        units: _UnitType | None = None,
        G: unyt.unyt_quantity | None = None,
        **kwargs,
    ) -> _UnitValue:
        r"""Compute the circular velocity at radius :math:`r`.

        For a test particle in circular orbit, the gravitational force provides the necessary
        centripetal acceleration. The circular velocity is given by:

        .. math::

            v_c(r) = \sqrt{ \frac{G M(r)}{r} },

        where :math:`M(r)` is the enclosed mass at radius :math:`r`.

        Parameters
        ----------
        r : ~unyt.array.unyt_quantity or ~unyt.array.unyt_array
            Radius or array of radii at which to compute the circular velocity (must carry length units).
        units : str or ~unyt.Unit, optional
            Desired output units for the circular velocity. By default, uses the internal
            units determined by `G` and the profile parameters.
        G : ~unyt.unyt_quantity, optional
            Gravitational constant to use. Defaults to :data:`~unyt.physical_constants.gravitational_constant`.
        kwargs : dict
            Additional keyword arguments passed to :meth:`compute_enclosed_mass`.

        Returns
        -------
        v_c : ~unyt.unyt_quantity or ~unyt.unyt_array
            Circular velocity at each input radius, with appropriate units.

        Example
        -------
        .. code-block:: python

            import numpy as np
            from pisces.profiles.density import (
                NFWDensityProfile,
            )

            profile = NFWDensityProfile(rho_0=1.0, r_s=10.0)

            r = np.linspace(0.1, 100.0, 50)  # kpc
            v_circ = profile.compute_circular_velocity(
                r, units="km/s"
            )

            print(v_circ)

        """
        # Extract G.
        G = G if G is not None else unyt.physical_constants.gravitational_constant

        # Enforce units on the radial array so
        # that we know its an unyt_array.
        r_array = unyt.array.unyt_array(r)
        m_enc = self.compute_enclosed_mass(r_array, **kwargs)

        # Now compute the profile.
        v_c = np.sqrt(G * m_enc / r_array)

        if units is None:
            return v_c
        else:
            return v_c.to(units)

    def compute_escape_velocity(
        self,
        r: _UnitValue,
        units: _UnitType | None = None,
        G: unyt.unyt_quantity | None = None,
        **kwargs,
    ) -> _UnitValue:
        r"""Compute the escape velocity at radius :math:`r`.

        The escape velocity is the minimum speed required for a test particle to
        escape to infinity from radius :math:`r` in a spherically symmetric gravitational potential:

        .. math::

            v_{\mathrm{esc}}(r) = \sqrt{ \frac{2 G M(r)}{r} },

        where :math:`M(r)` is the enclosed mass at radius :math:`r`.

        Parameters
        ----------
        r : ~unyt.array.unyt_quantity or ~unyt.array.unyt_array
            Radius or array of radii at which to compute the escape velocity (must carry length units).
        units : str or ~unyt.Unit, optional
            Desired output units for the escape velocity. By default,
            uses the internal units determined by `G` and the profile parameters.
        G : ~unyt.unyt_quantity, optional
            Gravitational constant to use. Defaults to :data:`~unyt.physical_constants.gravitational_constant`.
        kwargs : dict
            Additional keyword arguments passed to :meth:`compute_enclosed_mass`.

        Returns
        -------
        v_esc : ~unyt.unyt_quantity or ~unyt.unyt_array
            Escape velocity at each input radius, with appropriate units.

        Example
        -------
        .. code-block:: python

            import numpy as np
            from pisces.profiles.density import (
                HernquistDensityProfile,
            )

            profile = HernquistDensityProfile(
                rho_0=1.0, r_s=5.0
            )

            r = np.logspace(-1, 2, 50)  # kpc
            v_esc = profile.compute_escape_velocity(
                r, units="km/s"
            )

            print(v_esc)

        """
        G = G if G is not None else unyt.physical_constants.gravitational_constant

        r_array = unyt.array.unyt_array(r)
        m_enc = self.compute_enclosed_mass(r_array, **kwargs)

        v_esc = np.sqrt(2 * G * m_enc / r_array)

        if units is None:
            return v_esc
        else:
            return v_esc.to(units)

    def compute_deflection_angle(
        self,
        R: _UnitValue,
        mode: str = "angular",
        units: _UnitType | None = "arcsec",
        G: unyt.unyt_quantity | None = None,
        z_lens: float | None = None,
        z_source: float | None = None,
        D_l: _UnitValue | None = None,
        D_s: _UnitValue | None = None,
        D_ls: _UnitValue | None = None,
        cosmology: Optional["Cosmology"] = None,
        **kwargs,
    ) -> _UnitValue:
        r"""Compute the gravitational lensing deflection angle at projected radius :math:`R`.

        Supports both **physical** and **angular** deflection:

        - **Physical Deflection** (length units):

          .. math::

              \alpha_{\mathrm{phys}}(R) = \frac{4 G M(<R)}{c^2 R}

        - **Angular Deflection** (default, angular units):

          .. math::

              \alpha_{\mathrm{ang}}(R) = \alpha_{\mathrm{phys}}(R) \times \frac{D_{ls}}{D_s}

        where:

        - :math:`M(<R)` is the enclosed mass at projected radius :math:`R`
        - :math:`D_l`, :math:`D_s`, :math:`D_{ls}` are angular diameter distances to lens,
          source, and between lens and source, respectively.

        Parameters
        ----------
        R : :class:`~unyt.unyt_quantity` or :class:`~unyt.unyt_array`
            Projected radius in the lens plane with length units.
        mode : {"physical", "angular"}, optional
            Type of deflection to compute. Default is ``angular``.
        units : str or :class:`~unyt.Unit`, optional
            Output units. For angular deflection, defaults to ``arcsec``.
        G : :class:`~unyt.unyt_quantity`, optional
            Gravitational constant to use. Defaults to :data:`unyt.physical_constants.gravitational_constant`.
        z_lens : float, optional
            Redshift of the lens (required for angular mode if distances not provided).
        z_source : float, optional
            Redshift of the source (required for angular mode if distances not provided).
        D_l : :class:`~unyt.unyt_quantity`, optional
            Angular diameter distance to the lens.
        D_s : :class:`~unyt.unyt_quantity`, optional
            Angular diameter distance to the source.
        D_ls : :class:`~unyt.unyt_quantity`, optional
            Angular diameter distance between lens and source.
        cosmology : :class:`astropy.cosmology.Cosmology`, optional
            Cosmology used to compute distances if not provided. Defaults to
            ``pisces_config['physics.default_cosmology']``.
        **kwargs
            Additional arguments passed to :meth:`compute_enclosed_mass`.

        Returns
        -------
        alpha : :class:`~unyt.unyt_quantity` or :class:`~unyt.unyt_array`
            Deflection angle at each radius, in specified units.

        Raises
        ------
        ValueError
            If necessary distances are missing or redshifts are inconsistent.

        Examples
        --------

        .. code-block:: python

            import unyt
            from astropy.cosmology import Planck18
            from pisces.profiles.density import (
                NFWDensityProfile,
            )

            profile = NFWDensityProfile(rho_0=0.03, r_s=15)
            R = unyt.unyt_array([1, 5, 10], "kpc")

            alpha_arcsec = profile.compute_deflection_angle(
                R,
                mode="angular",
                units="arcsec",
                z_lens=0.3,
                z_source=2.0,
                cosmology=Planck18,
            )

            print(alpha_arcsec)

        See Also
        --------
        :meth:`compute_enclosed_mass`, :class:`astropy.cosmology.Cosmology`

        """
        import astropy.cosmology as cosmo

        mode = mode.lower()
        if mode not in {"physical", "angular"}:
            raise ValueError(f"Invalid mode '{mode}'. Choose 'physical' or 'angular'.")

        G = G if G is not None else unyt.physical_constants.gravitational_constant
        c = unyt.physical_constants.speed_of_light

        R_array = unyt.array.unyt_array(R)
        m_enc = self.compute_enclosed_mass(R_array, **kwargs)

        # Physical deflection term: length units
        alpha_phys = (4 * G * m_enc) / (c**2 * R_array)

        if mode == "physical":
            return alpha_phys.to(units) if units else alpha_phys

        # Angular deflection requires distances
        if D_l is None or D_s is None or D_ls is None:
            if z_lens is None or z_source is None:
                raise ValueError(
                    "Must provide either (D_l, D_s, D_ls) or (z_lens and z_source) for angular deflection."
                )

            if cosmology is None:
                _default = pisces_config["physics.default_cosmology"]
                try:
                    cosmology = getattr(cosmo, _default)
                except Exception as exp:
                    raise ValueError(f"Default cosmology `{_default}` is not available in astropy.cosmology.") from exp

            D_s = cosmology.angular_diameter_distance(z_source).to("kpc")
            D_ls = cosmology.angular_diameter_distance_z1z2(z_lens, z_source).to("kpc")

        # Apply distance ratio for angular deflection
        alpha_ang = alpha_phys * (D_ls / D_s)
        return alpha_ang.to(units) if units else alpha_ang

    def compute_einstein_radius(
        self,
        z_lens: float,
        z_source: float,
        units: _UnitType | None = "arcsec",
        G: unyt.unyt_quantity | None = None,
        cosmology: Optional["Cosmology"] = None,
        **kwargs,
    ) -> _UnitValue:
        r"""Compute the Einstein ring angular radius :math:`\\theta_E` for a perfectly aligned source-lens system.

        The Einstein radius satisfies:

        .. math::

            \alpha( \theta_E D_l ) = \theta_E

        where:

        - :math:`\alpha` is the angular deflection angle,
        - :math:`D_l` is the angular diameter distance to the lens.

        Parameters
        ----------
        z_lens : float
            Redshift of the lens.
        z_source : float
            Redshift of the source (must satisfy :math:`z_{\mathrm{source}} > z_{\mathrm{lens}}`).
        units : str or :class:`~unyt.Unit`, optional
            Desired output units for :math:`\\theta_E`. Default is ``arcsec``.
        G : :class:`~unyt.unyt_quantity`, optional
            Gravitational constant to use. Defaults to :data:`unyt.physical_constants.gravitational_constant`.
        cosmology : :class:`astropy.cosmology.Cosmology`, optional
            Cosmology used to compute distances. Defaults to ``pisces_config['physics.default_cosmology']``.
        **kwargs
            Additional arguments passed to :meth:`compute_enclosed_mass`.

        Returns
        -------
        theta_E : :class:`~unyt.unyt_quantity`
            Einstein radius in specified angular units.

        Raises
        ------
        ValueError
            If redshifts are invalid or distances cannot be computed.

        Examples
        --------

        .. code-block:: python

            from astropy.cosmology import Planck18
            from pisces.profiles.density import (
                HernquistDensityProfile,
            )

            profile = HernquistDensityProfile(
                rho_0=0.02, r_s=5
            )
            theta_E = profile.compute_einstein_radius(
                z_lens=0.3, z_source=2.0, cosmology=Planck18
            )

            print(f"Einstein radius: {theta_E:.3f}")

        See Also
        --------
        :meth:`compute_deflection_angle`, :class:`astropy.cosmology.Cosmology`

        """
        import astropy.cosmology as cosmo
        from scipy.optimize import brentq

        if z_source <= z_lens:
            raise ValueError("Source redshift must be greater than lens redshift.")

        if cosmology is None:
            _default = pisces_config["physics.default_cosmology"]
            try:
                cosmology = getattr(cosmo, _default)
            except Exception as exp:
                raise ValueError(f"Default cosmology `{_default}` is not available in astropy.cosmology.") from exp

        D_l = cosmology.angular_diameter_distance(z_lens).to("kpc")

        # Residual function: deflection angle minus angular radius
        def _residual(theta):
            R_proj = theta * D_l
            alpha = self.compute_deflection_angle(
                R_proj,
                mode="angular",
                units="rad",
                G=G,
                z_lens=z_lens,
                z_source=z_source,
                cosmology=cosmology,
                **kwargs,
            ).to_value("rad")
            return alpha - theta

        # Reasonable bracketing in radians: microarcsecond to 1 rad
        theta_min = (1e-6 * unyt.Unit("arcsec")).to_value("rad")
        theta_max = (1.0 * unyt.Unit("rad")).to_value("rad")

        # Root finding
        theta_E_rad = brentq(_residual, theta_min, theta_max)

        return (theta_E_rad * unyt.Unit("rad")).to(units)

    def compute_lensing_convergence(self, R, z_lens, z_source, cosmology=None, units=None, **kwargs):
        r"""Compute the lensing convergence :math:`\kappa(R)` at projected radius :math:`R`.

        The convergence is defined as:

        .. math::

            \kappa(R) = \frac{\Sigma(R)}{\Sigma_{\mathrm{crit}}}

        where:

        - :math:`\Sigma(R)` is the projected surface mass density,
        - :math:`\Sigma_{\mathrm{crit}}` is the critical surface density:

          .. math::

              \Sigma_{\mathrm{crit}} = \frac{c^2}{4 \pi G} \\frac{D_s}{D_l D_{ls}}

        Parameters
        ----------
        R : :class:`~unyt.unyt_quantity` or :class:`~unyt.unyt_array`
            Projected radius with length units.
        z_lens : float
            Redshift of the lens.
        z_source : float
            Redshift of the source (must satisfy :math:`z_{\mathrm{source}} > z_{\mathrm{lens}}`).
        cosmology : :class:`astropy.cosmology.Cosmology`, optional
            Cosmology to compute distances. Defaults to ``pisces_config['physics.default_cosmology']``.
        units : str or :class:`~unyt.Unit`, optional
            Output units for :math:`\\kappa`. Default is dimensionless.
        **kwargs
            Additional arguments passed to :meth:`compute_surface_density`.

        Returns
        -------
        kappa : :class:`~unyt.unyt_quantity` or :class:`~unyt.unyt_array`
            Lensing convergence at each radius.

        Examples
        --------

        .. code-block:: python

            import unyt
            from astropy.cosmology import Planck18
            from pisces.profiles.density import (
                NFWDensityProfile,
            )

            profile = NFWDensityProfile(rho_0=0.04, r_s=20)
            R = unyt.unyt_array([0.5, 1, 2, 5], "kpc")

            kappa = profile.compute_lensing_convergence(
                R,
                z_lens=0.3,
                z_source=2.0,
                cosmology=Planck18,
            )

            print(kappa)

        See Also
        --------
        :meth:`compute_surface_density`, :meth:`compute_deflection_angle`, :class:`astropy.cosmology.Cosmology`

        """
        import astropy.cosmology as cosmo

        Sigma = self.compute_surface_density(R, units="g/cm**2", **kwargs)

        if cosmology is None:
            _default = pisces_config["physics.default_cosmology"]
            cosmology = getattr(cosmo, _default)

        D_l = cosmology.angular_diameter_distance(z_lens).to("cm")
        D_s = cosmology.angular_diameter_distance(z_source).to("cm")
        D_ls = cosmology.angular_diameter_distance_z1z2(z_lens, z_source).to("cm")

        c = unyt.physical_constants.speed_of_light.to("cm/s")
        G = unyt.physical_constants.gravitational_constant.to("cm**3/g/s**2")

        Sigma_crit = (c**2) / (4 * np.pi * G) * (D_s / (D_l * D_ls))

        kappa = Sigma / Sigma_crit

        return kappa.to(units) if units else kappa


class NFWDensityProfile(BaseSphericalDensityProfile):
    r"""Navarro-Frenk-White (NFW) Density Profile.

    This profile is commonly used in astrophysics to describe the dark matter halo density
    in a spherical, isotropic system :footcite:p:`NFWProfile`. It is derived from simulations of structure formation.

    .. math::

        \rho(r) = \frac{\rho_0}{\frac{r}{r_s} \left(1 + \frac{r}{r_s}\right)^2}

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_s` is the scale radius.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`NFWDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_s``
             - :math:`r_s`
             - Scale radius

    References
    ----------
    .. footbibliography::

    Example
    -------
    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import (
        ...     NFWDensityProfile,
        ... )

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = NFWDensityProfile(rho_0=1.0, r_s=1.0)
        >>> rho = profile(r)

        >>> _ = plt.loglog(r, rho, "k-", label="NFW Profile")
        >>> _ = plt.xlabel("Radius (r)")
        >>> _ = plt.ylabel("Density (rho)")
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    HernquistDensityProfile, CoredNFWDensityProfile, SingularIsothermalDensityProfile

    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/pc**3"),
        "r_s": unyt.unyt_quantity(1.0, "pc"),
    }

    # ------------------------------ #
    # Derived Profile Implementation #
    # ------------------------------ #
    @derived_profile("enclosed_mass")
    @classmethod
    def _enclosed_mass(cls):
        def _func(r, **params):
            rho0, r_s = params["rho_0"], params["r_s"]
            return 4 * sp.pi * rho0 * r_s**3 * (sp.log((r + r_s) / r_s) - r / (r + r_s))

        def _unit_func(_, **params):
            return params["rho_0"].units * params["r_s"].units ** 3

        return _func, _unit_func, ["r"], cls.__PARAMETERS__.copy()

    @derived_profile("gravitational_potential")
    @classmethod
    def _gravitational_potential(cls):
        def _func(r, **params):
            rho0, r_s = params["rho_0"], params["r_s"]
            G = params["G"]
            return -4 * sp.pi * G * rho0 * r_s**2 * sp.log(1 + r / r_s) / r

        def _unit_func(r, **params):
            return params["G"].units * params["rho_0"].units * params["r_s"].units ** 2 / r

        # Add G to the available parameters.
        parameters = cls.__PARAMETERS__.copy()
        # noinspection PyUnresolvedReferences
        parameters["G"] = unyt.physical_constants.gravitational_constant

        return _func, _unit_func, ["r"], parameters

    @derived_profile("gravitational_field")
    @classmethod
    def _gravitational_field(cls):
        def _func(r, **params):
            rho0, r_s = params["rho_0"], params["r_s"]
            G = params["G"]
            term1 = sp.log(1 + r / r_s) / r**2
            term2 = 1 / (r * (r + r_s))
            return -4 * sp.pi * G * rho0 * r_s**2 * (term1 - term2)

        def _unit_func(r, **params):
            return params["G"].units * params["rho_0"].units * params["r_s"].units ** 2 / r**2

        parameters = cls.__PARAMETERS__.copy()
        parameters["G"] = unyt.physical_constants.gravitational_constant

        return _func, _unit_func, ["r"], parameters

    @classmethod
    def __function__(cls, r, rho_0=1.0, r_s=1.0):
        return rho_0 / (r / r_s * (1 + r / r_s) ** 2)

    @classmethod
    def __function_units__(
        cls,
        r,
        rho_0=unyt.unyt_quantity(1, "Msun/pc**3"),
        r_s=unyt.unyt_quantity(1, "pc"),
    ):
        r, rho_0, r_s = unyt.Unit(r), rho_0.units, r_s.units
        return rho_0 / (r / r_s)


class HernquistDensityProfile(BaseSphericalDensityProfile):
    r"""Hernquist Density Profile.

    This profile is often used to model the density distribution of elliptical galaxies and
    bulges :footcite:p:`HernquistProfile`.

    .. math::
        \rho(r) = \frac{\rho_0}{\frac{r}{r_s} \left(1 + \frac{r}{r_s}\right)^3}

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_s` is the scale radius.

    .. dropdown:: Parameters

        .. list-table:: Hernquist Profile Parameters
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_s``
             - :math:`r_s`
             - Scale radius


    References
    ----------
    .. footbibliography::

    Example
    -------
    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import (
        ...     HernquistDensityProfile,
        ... )

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = HernquistDensityProfile(
        ...     rho_0=1.0, r_s=1.0
        ... )
        >>> rho = profile(r)

        >>> _ = plt.loglog(
        ...     r, rho, "k-", label="Hernquist Profile"
        ... )
        >>> _ = plt.xlabel("Radius (r)")
        >>> _ = plt.ylabel("Density (rho)")
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    NFWDensityProfile, CoredNFWDensityProfile, SingularIsothermalDensityProfile

    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/pc**3"),
        "r_s": unyt.unyt_quantity(1.0, "pc"),
    }

    # ------------------------------ #
    # Derived Profile Implementation #
    # ------------------------------ #
    @derived_profile("enclosed_mass")
    @classmethod
    def _enclosed_mass(cls):
        def _func(r, **params):
            rho0, r_s = params["rho_0"], params["r_s"]
            M_tot = 2 * sp.pi * rho0 * r_s**3
            return M_tot * r**2 / (r + r_s) ** 2

        def _unit_func(_, **params):
            return params["rho_0"].units * params["r_s"].units ** 3

        return _func, _unit_func, ["r"], cls.__PARAMETERS__.copy()

    @derived_profile("gravitational_potential")
    @classmethod
    def _gravitational_potential(cls):
        def _func(r, **params):
            rho0, r_s, G = params["rho_0"], params["r_s"], params["G"]
            M_tot = 2 * sp.pi * rho0 * r_s**3
            return -G * M_tot / (r + r_s)

        def _unit_func(r, **params):
            return params["G"].units * params["rho_0"].units * params["r_s"].units ** 3 / r

        parameters = cls.__PARAMETERS__.copy()
        parameters["G"] = unyt.physical_constants.gravitational_constant

        return _func, _unit_func, ["r"], parameters

    @derived_profile("gravitational_field")
    @classmethod
    def _gravitational_field(cls):
        def _func(r, **params):
            rho0, r_s, G = params["rho_0"], params["r_s"], params["G"]
            M_tot = 2 * sp.pi * rho0 * r_s**3
            return -G * M_tot / (r + r_s) ** 2

        def _unit_func(r, **params):
            return params["G"].units * params["rho_0"].units * params["r_s"].units ** 3 / r**2

        parameters = cls.__PARAMETERS__.copy()
        parameters["G"] = unyt.physical_constants.gravitational_constant

        return _func, _unit_func, ["r"], parameters

    @classmethod
    def __function__(cls, r, rho_0=1.0, r_s=1.0):
        return rho_0 / ((r / r_s) * (1 + r / r_s) ** 3)

    @classmethod
    def __function_units__(
        cls,
        r,
        rho_0=unyt.unyt_quantity(1, "Msun/pc**3"),
        r_s=unyt.unyt_quantity(1, "pc"),
    ):
        r, rho_0, r_s = unyt.Unit(r), rho_0.units, r_s.units
        return rho_0 / (r / r_s)


class EinastoDensityProfile(BaseSphericalDensityProfile):
    r"""Einasto Density Profile.

    This profile provides a flexible model for dark matter halos with a gradual density decline
    :footcite:p:`EinastoProfile`.

    .. math::
        \rho(r) = \rho_0 \exp\left(-2 \alpha \left[\left(\frac{r}{r_s}\right)^\alpha - 1\right]\right)

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_s` is the scale radius.
    - :math:`\alpha` is a shape parameter that controls the profile steepness.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`EinastoDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_s``
             - :math:`r_s`
             - Scale radius
           * - ``alpha``
             - :math:`\alpha`
             - Shape parameter controlling profile steepness



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`EinastoDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base


    References
    ----------
    .. footbibliography::

    Example
    -------
    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import (
        ...     EinastoDensityProfile,
        ... )

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = EinastoDensityProfile(
        ...     rho_0=1.0, r_s=1.0, alpha=0.18
        ... )
        >>> rho = profile(r)

        >>> _ = plt.loglog(
        ...     r, rho, "k-", label="Einasto Profile"
        ... )
        >>> _ = plt.xlabel("Radius (r)")
        >>> _ = plt.ylabel("Density (rho)")
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    NFWDensityProfile, HernquistDensityProfile, CoredNFWDensityProfile

    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/pc**3"),
        "r_s": unyt.unyt_quantity(1.0, "pc"),
        "alpha": 0.18,
    }

    @classmethod
    def __function__(cls, r, rho_0=1.0, r_s=1.0, alpha=0.18):
        return rho_0 * sp.exp(-2 * alpha * ((r / r_s) ** alpha - 1))

    @classmethod
    def __function_units__(
        cls,
        r,
        rho_0=unyt.unyt_quantity(1, "Msun/pc**3"),
        r_s=unyt.unyt_quantity(1, "pc"),
        alpha=0.18,
    ):
        return rho_0.units


class SingularIsothermalDensityProfile(BaseSphericalDensityProfile):
    r"""Singular Isothermal Sphere (SIS) Density Profile.

    The SIS :footcite:p:`BinneyTremaine` profile is a simple model commonly used to describe the density distribution
    of dark matter in galaxies and galaxy clusters under the assumption of an isothermal system.

    .. math::
        \rho(r) = \frac{\rho_0}{r^2}

    where:

    - :math:`\rho_0` is the central density.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`SingularIsothermalDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`SingularIsothermalDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``spherical_mass``
             - :math:`M(r) = 4\pi\,\rho_0\,r`
             - Diverges as :math:`r\to\infty`.
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base



    References
    ----------
    .. footbibliography::

    Example
    -------
    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import (
        ...     SingularIsothermalDensityProfile,
        ... )

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = SingularIsothermalDensityProfile(
        ...     rho_0=1.0
        ... )
        >>> rho = profile(r)

        >>> _ = plt.loglog(r, rho, "k-", label="SIS Profile")
        >>> _ = plt.xlabel("Radius (r)")
        >>> _ = plt.ylabel("Density (rho)")
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    NFWDensityProfile, HernquistDensityProfile, CoredIsothermalDensityProfile

    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {"rho_0": unyt.unyt_quantity(1.0, "Msun/kpc")}

    @derived_profile("enclosed_mass")
    @classmethod
    def _enclosed_mass(cls):
        def _func(r, **params):
            rho0 = params["rho_0"]
            return 4 * sp.pi * rho0 * r

        def _unit_func(r, **params):
            return params["rho_0"].units * r.units

        return _func, _unit_func, ["r"], cls.__PARAMETERS__.copy()

    @derived_profile("gravitational_potential")
    @classmethod
    def _gravitational_potential(cls):
        def _func(r, **params):
            rho0, G = params["rho_0"], params["G"]
            return -4 * sp.pi * G * rho0 * sp.log(r)

        def _unit_func(r, **params):
            return params["G"].units * params["rho_0"].units * r.units**2 / r

        parameters = cls.__PARAMETERS__.copy()
        parameters["G"] = unyt.physical_constants.gravitational_constant
        return _func, _unit_func, ["r"], parameters

    @derived_profile("gravitational_field")
    @classmethod
    def _gravitational_field(cls):
        def _func(r, **params):
            rho0, G = params["rho_0"], params["G"]
            return -4 * sp.pi * G * rho0 / r

        def _unit_func(r, **params):
            return params["G"].units * params["rho_0"].units * r.units**2 / r**2

        parameters = cls.__PARAMETERS__.copy()
        parameters["G"] = unyt.physical_constants.gravitational_constant
        return _func, _unit_func, ["r"], parameters

    @classmethod
    def __function__(cls, r, rho_0=1.0):
        return rho_0 / r**2

    @classmethod
    def __function_units__(cls, r, rho_0=unyt.unyt_quantity(1, "Msun/kpc")):
        r, rho_0 = unyt.Unit(r), rho_0.units
        return rho_0 / r**2


class CoredIsothermalDensityProfile(BaseSphericalDensityProfile):
    r"""Cored Isothermal Sphere Density Profile.

    This profile modifies the Singular Isothermal Sphere (SIS) by introducing a core radius
    to account for the central flattening of the density distribution :footcite:p:`BinneyTremaine`.

    .. math::
        \rho(r) = \frac{\rho_0}{1 + \left(\frac{r}{r_c}\right)^2}

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_c` is the core radius.

    .. dropdown:: Parameters

        .. list-table:: Cored Isothermal Profile Parameters
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_c``
             - :math:`r_c`
             - Core radius



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`CoredIsothermalDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base



    References
    ----------
    .. footbibliography::

    Example
    -------
    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import (
        ...     CoredIsothermalDensityProfile,
        ... )

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = CoredIsothermalDensityProfile(
        ...     rho_0=1.0, r_c=1.0
        ... )
        >>> rho = profile(r)

        >>> _ = plt.loglog(
        ...     r, rho, "k-", label="Cored Isothermal Profile"
        ... )
        >>> _ = plt.xlabel("Radius (r)")
        >>> _ = plt.ylabel("Density (rho)")
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    SingularIsothermalDensityProfile, NFWDensityProfile, BurkertDensityProfile

    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3"),
        "r_c": unyt.unyt_quantity(1.0, "kpc"),
    }

    @classmethod
    def __function__(cls, r, rho_0=1.0, r_c=1.0):
        return rho_0 / (1 + (r / r_c) ** 2)

    @classmethod
    def __function_units__(cls, r, rho_0=unyt.unyt_quantity(1, "Msun/kpc**3"), **kwargs):
        return rho_0.units


class PlummerDensityProfile(BaseSphericalDensityProfile):
    r"""Plummer Density Profile.

    The Plummer profile is commonly used to model the density distribution of star clusters
    or spherical galaxies. It features a central core and a steep falloff at larger radii :footcite:p:`PlummerProfile`.

    .. math::
        \rho(r) = \frac{3M}{4\pi r_s^3} \left(1 + \left(\frac{r}{r_s}\right)^2\right)^{-5/2}

    where:

    - :math:`M` is the total mass.
    - :math:`r_s` is the scale radius.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`PlummerDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``M``
             - :math:`M`
             - Total mass
           * - ``r_s``
             - :math:`r_s`
             - Scale radius



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`PlummerDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``spherical_mass``
             - :math:`M(r) = M\,\bigl(\tfrac{r}{\sqrt{r^2 + r_s^2}}\bigr)^{3}`
             - None
           * - ``spherical_potential``
             - :math:`\Phi(r) = -\,\frac{M}{\sqrt{r^2 + r_s^2}}`
             - Multiplicative constants (e.g. `G`) can be included externally
           * - ``surface_density``
             - :math:`\Sigma(R) = \frac{M\,r_s^2}{\pi\,\bigl(r_s^2 + R^2\bigr)^{2}}`
             - On-demand expression for projected (2D) density
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base



    References
    ----------
    .. footbibliography::

    Example
    -------
    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import (
        ...     PlummerDensityProfile,
        ... )

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = PlummerDensityProfile(M=1.0, r_s=1.0)
        >>> rho = profile(r)

        >>> _ = plt.loglog(
        ...     r, rho, "k-", label="Plummer Profile"
        ... )
        >>> _ = plt.xlabel("Radius (r)")
        >>> _ = plt.ylabel("Density (rho)")
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    HernquistDensityProfile, NFWDensityProfile, JaffeDensityProfile

    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "M": unyt.unyt_quantity(1.0, "Msun"),
        "r_s": unyt.unyt_quantity(1.0, "pc"),
    }

    @derived_profile("enclosed_mass")
    @classmethod
    def _enclosed_mass(cls):
        def _func(r, **params):
            M_tot, r_s = params["M"], params["r_s"]
            return M_tot * r**3 / (r**2 + r_s**2) ** (3 / 2)

        def _unit_func(_, **params):
            return params["M"].units

        return _func, _unit_func, ["r"], cls.__PARAMETERS__.copy()

    @derived_profile("gravitational_potential")
    @classmethod
    def _gravitational_potential(cls):
        def _func(r, **params):
            M_tot, r_s, G = params["M"], params["r_s"], params["G"]
            return -G * M_tot / sp.sqrt(r**2 + r_s**2)

        def _unit_func(r, **params):
            return params["G"].units * params["M"].units / r.units

        parameters = cls.__PARAMETERS__.copy()
        parameters["G"] = unyt.physical_constants.gravitational_constant
        return _func, _unit_func, ["r"], parameters

    @derived_profile("gravitational_field")
    @classmethod
    def _gravitational_field(cls):
        def _func(r, **params):
            M_tot, r_s, G = params["M"], params["r_s"], params["G"]
            return -G * M_tot * r / (r**2 + r_s**2) ** (3 / 2)

        def _unit_func(r, **params):
            return params["G"].units * params["M"].units / r.units**2

        parameters = cls.__PARAMETERS__.copy()
        parameters["G"] = unyt.physical_constants.gravitational_constant
        return _func, _unit_func, ["r"], parameters

    @classmethod
    def __function__(cls, r, M=1.0, r_s=1.0):
        return (3 * M) / (4 * sp.pi * r_s**3) * (1 + (r / r_s) ** 2) ** (-5 / 2)

    @classmethod
    def __function_units__(cls, r, M=unyt.unyt_quantity(1, "Msun"), r_s=unyt.unyt_quantity(1, "pc")):
        M, r_s = M.units, r_s.units
        return M / r_s**3


class DehnenDensityProfile(BaseSphericalDensityProfile):
    r"""Dehnen Density Profile.

    This profile is widely used in modeling galactic bulges and elliptical galaxies.
    It generalizes other profiles like Hernquist and Jaffe with an adjustable inner slope :footcite:p:`DehnenProfile`.

    .. math::
        \rho(r) = \frac{(3 - \gamma)M}{4\pi r_s^3}
        \left(\frac{r}{r_s}\right)^{-\gamma} \left(1 + \frac{r}{r_s}\right)^{\gamma - 4}

    where:

    - :math:`M` is the total mass.
    - :math:`r_s` is the scale radius.
    - :math:`\gamma` controls the inner density slope.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`DehnenDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``M``
             - :math:`M`
             - Total mass
           * - ``r_s``
             - :math:`r_s`
             - Scale radius
           * - ``gamma``
             - :math:`\gamma`
             - Controls the inner density slope



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`DehnenDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``spherical_mass``
             - :math:`M(r) = M\,\Bigl(\frac{r}{r + r_s}\Bigr)^{3 - \gamma}`
             - Contains Hernquist (gamma=1) and Jaffe (gamma=2) as special cases
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base



    References
    ----------
    .. footbibliography::

    Example
    -------
    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import (
        ...     DehnenDensityProfile,
        ... )

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = DehnenDensityProfile(
        ...     M=1.0, r_s=1.0, gamma=1.0
        ... )
        >>> rho = profile(r)

        >>> _ = plt.loglog(
        ...     r, rho, "k-", label="Dehnen Profile"
        ... )
        >>> _ = plt.xlabel("Radius (r)")
        >>> _ = plt.ylabel("Density (rho)")
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    HernquistDensityProfile, JaffeDensityProfile, NFWDensityProfile

    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "M": unyt.unyt_quantity(1.0, "Msun"),
        "r_s": unyt.unyt_quantity(1.0, "pc"),
        "gamma": 1.0,
    }

    @classmethod
    def __function__(cls, r, M=1.0, r_s=1.0, gamma=1.0):
        return ((3 - gamma) * M) / (4 * sp.pi * r_s**3) * (r / r_s) ** (-gamma) * (1 + r / r_s) ** (gamma - 4)

    @classmethod
    def __function_units__(
        cls,
        r,
        M=unyt.unyt_quantity(1, "Msun"),
        r_s=unyt.unyt_quantity(1, "pc"),
        gamma=1.0,
    ):
        M, r_s = M.units, r_s.units
        return M / r_s**3


class JaffeDensityProfile(BaseSphericalDensityProfile):
    r"""Jaffe Density Profile.

    This profile is commonly used to describe the density distribution of elliptical galaxies
    :footcite:p:`JaffeProfile`.

    .. math::
        \rho(r) = \frac{\rho_0}{\frac{r}{r_s} \left(1 + \frac{r}{r_s}\right)^2}

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_s` is the scale radius.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`JaffeDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_s``
             - :math:`r_s`
             - Scale radius



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`JaffeDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``spherical_mass``
             - :math:`M(r) = 4\pi\,\rho_0\,r_s^3\,\Bigl[\ln(r + r_s) + \ln(r_s) \;-\; 1 \;+\; \frac{r_s}{r + r_s}\Bigr]`
             - (As coded; can simplify further)
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base



    References
    ----------
    .. footbibliography::

    Example
    -------
    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import (
        ...     JaffeDensityProfile,
        ... )

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = JaffeDensityProfile(rho_0=1.0, r_s=1.0)
        >>> rho = profile(r)

        >>> _ = plt.loglog(
        ...     r, rho, "k-", label="Jaffe Profile"
        ... )
        >>> _ = plt.xlabel("Radius (r)")
        >>> _ = plt.ylabel("Density (rho)")
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    HernquistDensityProfile, DehnenDensityProfile, NFWDensityProfile

    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3"),
        "r_s": unyt.unyt_quantity(1.0, "pc"),
    }

    @derived_profile("enclosed_mass")
    @classmethod
    def _enclosed_mass(cls):
        def _func(r, **params):
            M_tot, a = params["M_tot"], params["a"]
            return M_tot * r / (r + a)

        def _unit_func(_, **params):
            return params["M_tot"].units

        return _func, _unit_func, ["r"], cls.__PARAMETERS__.copy()

    @classmethod
    def __function__(cls, r, rho_0=1.0, r_s=1.0):
        return rho_0 / ((r / r_s) * (1 + r / r_s) ** 2)

    @classmethod
    def __function_units__(
        cls,
        r,
        rho_0=unyt.unyt_quantity(1, "Msun/kpc**3"),
        r_s=unyt.unyt_quantity(1, "pc"),
    ):
        r, rho_0, r_s = unyt.Unit(r), rho_0.units, r_s.units
        return rho_0 / (r / r_s)


class BurkertDensityProfile(BaseSphericalDensityProfile):
    r"""Burkert Density Profile.

    This profile describes dark matter halos with a flat density core, often used to
    fit rotation curves of dwarf galaxies :footcite:p:`BurkertProfile`.

    .. math::
        \rho(r) = \frac{\rho_0}{\left(1 + \frac{r}{r_s}\right) \left(1 + \left(\frac{r}{r_s}\right)^2\right)}

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_s` is the scale radius.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`BurkertDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_s``
             - :math:`r_s`
             - Scale radius



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`BurkertDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base


    References
    ----------
    .. footbibliography::

    Example
    -------
    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import (
        ...     BurkertDensityProfile,
        ... )

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = BurkertDensityProfile(
        ...     rho_0=1.0, r_s=1.0
        ... )
        >>> rho = profile(r)

        >>> _ = plt.loglog(
        ...     r, rho, "k-", label="Burkert Profile"
        ... )
        >>> _ = plt.xlabel("Radius (r)")
        >>> _ = plt.ylabel("Density (rho)")
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    NFWDensityProfile, CoredNFWDensityProfile, KingDensityProfile

    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3"),
        "r_s": unyt.unyt_quantity(1.0, "pc"),
    }

    @derived_profile("enclosed_mass")
    @classmethod
    def _enclosed_mass(cls):
        def _func(r, **params):
            rho0, r_c = params["rho_0"], params["r_c"]
            return (
                2 * sp.pi * rho0 * r_c**3 * (sp.log(1 + r / r_c) - sp.atan(r / r_c) + 0.5 * sp.log(1 + (r / r_c) ** 2))
            )

        def _unit_func(_, **params):
            return params["rho_0"].units * params["r_c"].units ** 3

        return _func, _unit_func, ["r"], cls.__PARAMETERS__.copy()

    @classmethod
    def __function__(cls, r, rho_0=1.0, r_s=1.0):
        return rho_0 / ((1 + r / r_s) * (1 + (r / r_s) ** 2))

    @classmethod
    def __function_units__(cls, r, rho_0=unyt.unyt_quantity(1, "Msun/kpc**3"), **kwargs):
        return rho_0.units


class MooreDensityProfile(BaseSphericalDensityProfile):
    r"""Moore Density Profile.

    This profile describes the density of dark matter halos with a steeper central slope compared to NFW
    :footcite:p:`MooreProfile`.

    .. math::
        \rho(r) = \frac{\rho_0}{\left(\frac{r}{r_s}\right)^{3/2} \left(1 + \frac{r}{r_s}\right)^{3/2}}

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_s` is the scale radius.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`MooreDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_s``
             - :math:`r_s`
             - Scale radius



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`MooreDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base



    References
    ----------
    .. footbibliography::

    Example
    -------
    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import (
        ...     MooreDensityProfile,
        ... )

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = MooreDensityProfile(rho_0=1.0, r_s=1.0)
        >>> rho = profile(r)

        >>> _ = plt.loglog(
        ...     r, rho, "k-", label="Moore Profile"
        ... )
        >>> _ = plt.xlabel("Radius (r)")
        >>> _ = plt.ylabel("Density (rho)")
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    NFWDensityProfile, CoredNFWDensityProfile, HernquistDensityProfile

    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3"),
        "r_s": unyt.unyt_quantity(1.0, "pc"),
    }

    @classmethod
    def __function__(cls, r, rho_0=1.0, r_s=1.0):
        return rho_0 / ((r / r_s) ** (3 / 2) * (1 + r / r_s) ** (3 / 2))

    @classmethod
    def __function_units__(
        cls,
        r,
        rho_0=unyt.unyt_quantity(1, "Msun/kpc**3"),
        r_s=unyt.unyt_quantity(1, "pc"),
    ):
        r, rho_0, r_s = unyt.Unit(r), rho_0.units, r_s.units
        return rho_0


class CoredNFWDensityProfile(BaseSphericalDensityProfile):
    r"""Cored Navarro-Frenk-White (NFW) Density Profile.

    This profile modifies the standard NFW profile by introducing a core, leading to
    a shallower density slope near the center :footcite:p:`CNFWProfile`.

    .. math::
        \rho(r) = \frac{\rho_0}{\left(1 + \left(\frac{r}{r_s}\right)^2\right) \left(1 + \frac{r}{r_s}\right)^2}

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_s` is the scale radius.

    .. dropdown:: Parameters

        .. list-table:: Cored NFW Profile Parameters
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_s``
             - :math:`r_s`
             - Scale radius



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`CoredNFWDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base



    References
    ----------
    .. footbibliography::

    Example
    -------
    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import (
        ...     CoredNFWDensityProfile,
        ... )

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = CoredNFWDensityProfile(
        ...     rho_0=1.0, r_s=1.0
        ... )
        >>> rho = profile(r)

        >>> _ = plt.loglog(
        ...     r, rho, "k-", label="Cored NFW Profile"
        ... )
        >>> _ = plt.xlabel("Radius (r)")
        >>> _ = plt.ylabel("Density (rho)")
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    NFWDensityProfile, HernquistDensityProfile, BurkertDensityProfile

    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3"),
        "r_s": unyt.unyt_quantity(1.0, "pc"),
    }

    @classmethod
    def __function__(cls, r, rho_0=1.0, r_s=1.0):
        return rho_0 / ((1 + (r / r_s) ** 2) * (1 + r / r_s) ** 2)

    @classmethod
    def __function_units__(
        cls,
        r,
        rho_0=unyt.unyt_quantity(1, "Msun/kpc**3"),
        r_s=unyt.unyt_quantity(1, "pc"),
    ):
        r, rho_0, r_s = unyt.Unit(r), rho_0.units, r_s.units
        return rho_0


class KingDensityProfile(BaseSphericalDensityProfile):
    r"""King Density Profile.

    This profile describes the density distribution in globular clusters and galaxy clusters,
    accounting for truncation at larger radii :footcite:p:`KingProfile`.

    .. math::
        \rho(r) = \rho_0 \left[\left(1 + \left(\frac{r}{r_c}\right)^2\right)^{-3/2}
        - \left(1 + \left(\frac{r_t}{r_c}\right)^2\right)^{-3/2}\right]

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_c` is the core radius.
    - :math:`r_t` is the truncation radius.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`KingDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_c``
             - :math:`r_c`
             - Core radius
           * - ``r_t``
             - :math:`r_t`
             - Truncation radius



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`KingDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base



    References
    ----------
    .. footbibliography::

    Example
    -------
    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import (
        ...     KingDensityProfile,
        ... )

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = KingDensityProfile(
        ...     rho_0=1.0, r_c=1.0, r_t=5.0
        ... )
        >>> rho = profile(r)

        >>> _ = plt.loglog(r, rho, "k-", label="King Profile")
        >>> _ = plt.xlabel("Radius (r)")
        >>> _ = plt.ylabel("Density (rho)")
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    NFWDensityProfile, BurkertDensityProfile, PlummerDensityProfile

    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3"),
        "r_c": unyt.unyt_quantity(1.0, "kpc"),
        "r_t": unyt.unyt_quantity(1.0, "kpc"),
    }

    @classmethod
    def __function__(cls, r, rho_0=1.0, r_c=1.0, r_t=1.0):
        return rho_0 * ((1 + (r / r_c) ** 2) ** (-1.5) - (1 + (r_t / r_c) ** 2) ** (-1.5))

    @classmethod
    def __function_units__(cls, r, rho_0=unyt.unyt_quantity(1, "Msun/kpc**3"), **kwargs):
        return rho_0.units


class VikhlininDensityProfile(BaseSphericalDensityProfile):
    r"""Vikhlinin Density Profile.

    This profile is used to model the density of galaxy clusters, incorporating
    a truncation at large radii and additional flexibility for inner slopes :footcite:p:`VikhlininProfile`.

    .. math::
        \rho(r) = \rho_0 \left(\frac{r}{r_c}\right)^{-0.5 \alpha}
        \left(1 + \left(\frac{r}{r_c}\right)^2\right)^{-1.5 \beta + 0.25 \alpha}
        \left(1 + \left(\frac{r}{r_s}\right)^{\gamma}\right)^{-0.5 \epsilon / \gamma}

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_c` is the core radius.
    - :math:`r_s` is the truncation radius.
    - :math:`\alpha, \beta, \gamma, \epsilon` control the slope and truncation behavior.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`VikhlininDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_c``
             - :math:`r_c`
             - Core radius
           * - ``r_s``
             - :math:`r_s`
             - Truncation radius
           * - ``alpha``
             - :math:`\alpha`
             - Controls the innermost slope
           * - ``beta``
             - :math:`\beta`
             - Governs the outer slope
           * - ``epsilon``
             - :math:`\epsilon`
             - Steepens the outer decline
           * - ``gamma``
             - :math:`\gamma`
             - Exponent in the truncation factor



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`VikhlininDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base



    References
    ----------
    .. footbibliography::

    Example
    -------
    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import (
        ...     VikhlininDensityProfile,
        ... )

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = VikhlininDensityProfile(
        ...     rho_0=1.0,
        ...     r_c=1.0,
        ...     r_s=5.0,
        ...     alpha=1.0,
        ...     beta=1.0,
        ...     epsilon=1.0,
        ...     gamma=3.0,
        ... )
        >>> rho = profile(r)

        >>> _ = plt.loglog(
        ...     r, rho, "k-", label="Vikhlinin Profile"
        ... )
        >>> _ = plt.xlabel("Radius (r)")
        >>> _ = plt.ylabel("Density (rho)")
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    KingDensityProfile, NFWDensityProfile, BurkertDensityProfile

    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3"),
        "r_c": unyt.unyt_quantity(1.0, "kpc"),
        "r_s": unyt.unyt_quantity(1.0, "pc"),
        "alpha": 1.0,
        "beta": 1.0,
        "epsilon": 1.0,
        "gamma": 3.0,
    }

    @classmethod
    def __function__(cls, r, rho_0=1.0, r_c=1.0, r_s=1.0, alpha=1.0, beta=1.0, epsilon=1.0, gamma=3.0):
        return (
            rho_0
            * (r / r_c) ** (-0.5 * alpha)
            * (1 + (r / r_c) ** 2) ** (-1.5 * beta + 0.25 * alpha)
            * (1 + (r / r_s) ** gamma) ** (-0.5 * epsilon / gamma)
        )

    @classmethod
    def __function_units__(cls, r, rho_0=unyt.unyt_quantity(1, "Msun/kpc**3"), **kwargs):
        return rho_0.units


class AM06DensityProfile(BaseSphericalDensityProfile):
    r"""Ascasibar and Markevitch (2006) Density Profile.

    This density profile is a generalized model that allows flexibility in fitting
    the density distributions of dark matter halos. It includes additional parameters
    for controlling inner and outer slopes, truncation, and other scaling properties :footcite:p:`AM06Profile`.

    .. math::
        \rho(r) = \rho_0 \left(1 + \frac{r}{a_c}\right) \left(1 + \frac{r}{a_c c}\right)^{\alpha}
        \left(1 + \frac{r}{a}\right)^{\beta}

    where:

    - :math:`\rho_0` is the central density.
    - :math:`a_c` is the core radius.
    - :math:`c` is the concentration parameter.
    - :math:`a` is the scale radius.
    - :math:`\alpha` controls the slope of the transition near the core.
    - :math:`\beta` controls the outer slope.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`AM06DensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``a_c``
             - :math:`a_c`
             - Core radius
           * - ``c``
             - :math:`c`
             - Concentration parameter
           * - ``a``
             - :math:`a`
             - Scale radius
           * - ``alpha``
             - :math:`\alpha`
             - Controls slope near the core
           * - ``beta``
             - :math:`\beta`
             - Controls outer slope



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`AM06DensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base


    Use Case
    --------
    This profile is well-suited for modeling dark matter halos with detailed inner and outer slope behaviors.


    References
    ----------
    .. footbibliography::

    Example
    -------
    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import (
        ...     AM06DensityProfile,
        ... )

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = AM06DensityProfile(
        ...     rho_0=1.0,
        ...     a_c=1.0,
        ...     c=2.0,
        ...     a=3.0,
        ...     alpha=1.0,
        ...     beta=3.0,
        ... )
        >>> rho = profile(r)

        >>> _ = plt.loglog(r, rho, "k-", label="AM06 Profile")
        >>> _ = plt.xlabel("Radius (r)")
        >>> _ = plt.ylabel("Density (rho)")
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    NFWDensityProfile, VikhlininDensityProfile, HernquistDensityProfile

    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3"),
        "a": unyt.unyt_quantity(1.0, "kpc"),
        "a_c": unyt.unyt_quantity(1.0, "kpc"),
        "c": 1.0,
        "alpha": 1.0,
        "beta": 1.0,
    }

    @classmethod
    def __function__(cls, r, rho_0=1.0, a=1.0, a_c=1.0, c=1.0, alpha=1.0, beta=1.0):
        return rho_0 * (1 + r / a_c) * (1 + r / (a_c * c)) ** alpha * (1 + r / a) ** beta

    @classmethod
    def __function_units__(cls, r, rho_0=unyt.unyt_quantity(1, "Msun/kpc**3"), **kwargs):
        return rho_0.units


class SNFWDensityProfile(BaseSphericalDensityProfile):
    r"""Simplified Navarro-Frenk-White (SNFW) Density Profile.

    This profile is a simplified version of the NFW profile, widely used for modeling
    dark matter halos with specific scaling :footcite:p:`SNFWProfile`.

    .. math::
        \rho(r) = \frac{3M}{16\pi a^3} \frac{1}{\frac{r}{a} \left(1 + \frac{r}{a}\right)^{2.5}}

    where:

    - :math:`M` is the total mass.
    - :math:`a` is the scale radius.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`SNFWDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``M``
             - :math:`M`
             - Total mass
           * - ``a``
             - :math:`a`
             - Scale radius



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`SNFWDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base (no direct mass/potential expression coded here)



    References
    ----------
    .. footbibliography::

    Example
    -------
    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import (
        ...     SNFWDensityProfile,
        ... )

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = SNFWDensityProfile(M=1.0, a=1.0)
        >>> rho = profile(r)

        >>> _ = plt.loglog(r, rho, "k-", label="SNFW Profile")
        >>> _ = plt.xlabel("Radius (r)")
        >>> _ = plt.ylabel("Density (rho)")
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    NFWDensityProfile, TNFWDensityProfile, HernquistDensityProfile

    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "M": unyt.unyt_quantity(1.0, "Msun"),
        "a": unyt.unyt_quantity(1.0, "kpc"),
    }

    @classmethod
    def __function__(cls, r, M=1.0, a=1.0):
        return 3 * M / (16 * sp.pi * a**3) / ((r / a) * (1 + r / a) ** 2.5)

    @classmethod
    def __function_units__(cls, r, M=unyt.unyt_quantity(1, "Msun"), a=unyt.unyt_quantity(1, "kpc")):
        M, a = M.units, a.units
        return M / a**3


class TNFWDensityProfile(BaseSphericalDensityProfile):
    r"""Truncated Navarro-Frenk-White (TNFW) Density Profile.

    This profile is a modification of the NFW profile with an additional truncation
    term to account for finite halo sizes :footcite:p:`TNFWProfile`.

    .. math::
        \rho(r) = \frac{\rho_0}{\frac{r}{r_s} \left(1 + \frac{r}{r_s}\right)^2}
        \frac{1}{1 + \left(\frac{r}{r_t}\right)^2}

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_s` is the scale radius.
    - :math:`r_t` is the truncation radius.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`TNFWDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_s``
             - :math:`r_s`
             - Scale radius
           * - ``r_t``
             - :math:`r_t`
             - Truncation radius



    .. dropdown:: Expressions

        .. list-table:: Expressions for :py:class:`TNFWDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Notes**
           * - ``ellipsoidal_psi``
             - :math:`\psi(r) = 2\int_{0}^{r}\!\xi\,\rho(\xi)\,d\xi`
             - Inherited from base



    References
    ----------
    .. footbibliography::

    Example
    -------
    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import (
        ...     TNFWDensityProfile,
        ... )

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = TNFWDensityProfile(
        ...     rho_0=1.0, r_s=1.0, r_t=10.0
        ... )
        >>> rho = profile(r)

        >>> _ = plt.loglog(r, rho, "k-", label="TNFW Profile")
        >>> _ = plt.xlabel("Radius (r)")
        >>> _ = plt.ylabel("Density (rho)")
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    NFWDensityProfile, CoredNFWDensityProfile, SNFWDensityProfile

    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3"),
        "r_s": unyt.unyt_quantity(1.0, "pc"),
        "r_t": unyt.unyt_quantity(1.0, "kpc"),
    }

    @classmethod
    def __function__(cls, r, rho_0=1.0, r_s=1.0, r_t=1.0):
        return rho_0 / ((r / r_s) * (1 + r / r_s) ** 2) / (1 + (r / r_t) ** 2)

    @classmethod
    def __function_units__(cls, r, rho_0=unyt.unyt_quantity(1, "Msun/kpc**3"), **kwargs):
        return rho_0.units


class PseudoIsothermalDensityProfile(BaseSphericalDensityProfile):
    r"""Pseudo-Isothermal Sphere (PISO) Density Profile.

    Commonly used to model dark matter halos with a constant density core, avoiding the central singularity of singular
    isothermal models.

    .. math::

        \rho(r) = \frac{\rho_0}{1 + \left(\frac{r}{r_c}\right)^2}

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_c` is the core radius.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`PseudoIsothermalDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_c``
             - :math:`r_c`
             - Core radius

    Example
    -------
    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import (
        ...     PseudoIsothermalDensityProfile,
        ... )

        >>> r = np.logspace(-1, 1, 100)
        >>> profile = PseudoIsothermalDensityProfile(
        ...     rho_0=1.0, r_c=1.0
        ... )
        >>> rho = profile(r)

        >>> plt.loglog(r, rho, label="PISO Profile")
        >>> plt.xlabel("Radius (r)")
        >>> plt.ylabel("Density (rho)")
        >>> plt.legend()
        >>> plt.show()

    See Also
    --------
    CoredIsothermalDensityProfile, SingularIsothermalDensityProfile

    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3"),
        "r_c": unyt.unyt_quantity(1.0, "kpc"),
    }

    @classmethod
    def __function__(cls, r, rho_0=1.0, r_c=1.0):
        return rho_0 / (1 + (r / r_c) ** 2)

    @classmethod
    def __function_units__(
        cls,
        r,
        rho_0=unyt.unyt_quantity(1, "Msun/kpc**3"),
        r_c=unyt.unyt_quantity(1, "kpc"),
    ):
        r, rho_0, r_c = unyt.Unit(r), rho_0.units, r_c.units
        return rho_0


class DoublePowerLawDensityProfile(BaseSphericalDensityProfile):
    r"""Double Power-Law Density Profile.

    Flexible, general profile with independent control over inner and outer slopes,
    frequently used to describe dark matter halos and galaxies.

    .. math::

        \rho(r) = \frac{\rho_0}{\left(\frac{r}{r_s}\right)^{\gamma}
         \left(1 + \left(\frac{r}{r_s}\right)^{\alpha}\right)^{(\beta - \gamma)/\alpha}}

    where:

    - :math:`\rho_0` is the normalization density.
    - :math:`r_s` is the scale radius.
    - :math:`\alpha` governs the sharpness of the transition.
    - :math:`\beta` is the outer slope.
    - :math:`\gamma` is the inner slope.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`DoublePowerLawDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Normalization density
           * - ``r_s``
             - :math:`r_s`
             - Scale radius
           * - ``alpha``
             - :math:`\alpha`
             - Sharpness of slope transition
           * - ``beta``
             - :math:`\beta`
             - Outer slope
           * - ``gamma``
             - :math:`\gamma`
             - Inner slope

    Example
    -------
    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import (
        ...     DoublePowerLawDensityProfile,
        ... )

        >>> r = np.logspace(-1, 1, 100)
        >>> profile = DoublePowerLawDensityProfile(
        ...     rho_0=1.0,
        ...     r_s=1.0,
        ...     alpha=1.0,
        ...     beta=3.0,
        ...     gamma=1.0,
        ... )
        >>> rho = profile(r)

        >>> plt.loglog(r, rho, label="Double Power-Law")
        >>> plt.xlabel("Radius (r)")
        >>> plt.ylabel("Density (rho)")
        >>> plt.legend()
        >>> plt.show()

    See Also
    --------
    NFWDensityProfile, DehnenDensityProfile, HernquistDensityProfile

    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3"),
        "r_s": unyt.unyt_quantity(1.0, "pc"),
        "alpha": 1.0,
        "beta": 3.0,
        "gamma": 1.0,
    }

    @classmethod
    def __function__(cls, r, rho_0=1.0, r_s=1.0, alpha=1.0, beta=3.0, gamma=1.0):
        return rho_0 / (r / r_s) ** gamma / (1 + (r / r_s) ** alpha) ** ((beta - gamma) / alpha)

    @classmethod
    def __function_units__(cls, r, rho_0=unyt.unyt_quantity(1, "Msun/kpc**3"), **kwargs):
        return rho_0.units


class CoredPowerLawDensityProfile(BaseSphericalDensityProfile):
    r"""Cored Power-Law Density Profile.

    This profile represents a flattened core transitioning to a power-law decline at large radii,
    often used for galaxies or halos with a central density core.

    .. math::

        \rho(r) = \rho_0 \left(1 + \left(\frac{r}{r_c}\right)^2\right)^{-\gamma/2}

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_c` is the core radius.
    - :math:`\gamma` controls the outer slope.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`CoredPowerLawDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_c``
             - :math:`r_c`
             - Core radius
           * - ``gamma``
             - :math:`\gamma`
             - Outer slope parameter

    Example
    -------
    .. plot::
        :include-source:


        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import (
        ...     CoredPowerLawDensityProfile,
        ... )

        >>> r = np.logspace(-1, 1, 100)
        >>> profile = CoredPowerLawDensityProfile(
        ...     rho_0=1.0, r_c=1.0, gamma=2.0
        ... )
        >>> rho = profile(r)

        >>> plt.loglog(r, rho, label="Cored Power-Law")
        >>> plt.xlabel("Radius (r)")
        >>> plt.ylabel("Density (rho)")
        >>> plt.legend()
        >>> plt.show()

    See Also
    --------
    PseudoIsothermalDensityProfile, CoredIsothermalDensityProfile, DoublePowerLawDensityProfile

    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3"),
        "r_c": unyt.unyt_quantity(1.0, "kpc"),
        "gamma": 1.0,
    }

    @classmethod
    def __function__(cls, r, rho_0=1.0, r_c=1.0, gamma=1.0):
        return rho_0 * (1 + (r / r_c) ** 2) ** (-gamma / 2)

    @classmethod
    def __function_units__(cls, r, rho_0=unyt.unyt_quantity(1, "Msun/kpc**3"), **kwargs):
        return rho_0.units


class GeneralizedNFWDensityProfile(BaseSphericalDensityProfile):
    r"""Generalized NFW (GNFW) Density Profile.

    Extends the Navarro-Frenk-White (NFW) profile with flexible inner and outer slopes, commonly used to model
    dark matter halos and galaxy clusters :footcite:p:`GeneralizedNFWProfile`.

    .. math::

        \rho(r) = \frac{\rho_0}{\left(\frac{r}{r_s}\right)^{\gamma}
        \left(1 + \left(\frac{r}{r_s}\right)^{\alpha}\right)^{(\beta - \gamma)/\alpha}}

    where:

    - :math:`\rho_0` is the normalization density.
    - :math:`r_s` is the scale radius.
    - :math:`\alpha` controls the sharpness of the transition.
    - :math:`\beta` is the outer slope.
    - :math:`\gamma` is the inner slope.

    The standard NFW profile corresponds to :math:`\alpha = 1`, :math:`\beta = 3`, :math:`\gamma = 1`.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`GeneralizedNFWDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Normalization density
           * - ``r_s``
             - :math:`r_s`
             - Scale radius
           * - ``alpha``
             - :math:`\alpha`
             - Sharpness of slope transition
           * - ``beta``
             - :math:`\beta`
             - Outer slope
           * - ``gamma``
             - :math:`\gamma`
             - Inner slope

    Example
    -------
    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import (
        ...     GeneralizedNFWDensityProfile,
        ... )

        >>> r = np.logspace(-1, 1, 100)
        >>> profile = GeneralizedNFWDensityProfile(
        ...     rho_0=1.0,
        ...     r_s=1.0,
        ...     alpha=1.0,
        ...     beta=3.0,
        ...     gamma=1.0,
        ... )
        >>> rho = profile(r)

        >>> plt.loglog(r, rho, label="GNFW Profile")
        >>> plt.xlabel("Radius (r)")
        >>> plt.ylabel("Density (rho)")
        >>> plt.legend()
        >>> plt.show()

    See Also
    --------
    NFWDensityProfile, DoublePowerLawDensityProfile, EinastoDensityProfile

    References
    ----------
    .. footbibliography::

    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3"),
        "r_s": unyt.unyt_quantity(1.0, "pc"),
        "alpha": 1.0,
        "beta": 3.0,
        "gamma": 1.0,
    }

    @classmethod
    def __function__(cls, r, rho_0=1.0, r_s=1.0, alpha=1.0, beta=3.0, gamma=1.0):
        return rho_0 / (r / r_s) ** gamma / (1 + (r / r_s) ** alpha) ** ((beta - gamma) / alpha)

    @classmethod
    def __function_units__(cls, r, rho_0=unyt.unyt_quantity(1, "Msun/kpc**3"), **kwargs):
        return rho_0.units


class BetaModelDensityProfile(BaseSphericalDensityProfile):
    r"""Beta Model Density Profile.

    Frequently used to describe gas density in galaxy clusters, X-ray halos, and intracluster media,
     based on an isothermal, spherically symmetric gas distribution :footcite:p:`BetaProfile`.

    .. math::

        \rho(r) = \rho_0 \left(1 + \left(\frac{r}{r_c}\right)^2 \right)^{-3\beta/2}

    where:

    - :math:`\rho_0` is the central density.
    - :math:`r_c` is the core radius.
    - :math:`\beta` controls the outer slope.

    This form corresponds to the classic "single-beta" model. More complex systems
    sometimes use multiple beta components.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`BetaModelDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_c``
             - :math:`r_c`
             - Core radius
           * - ``beta``
             - :math:`\beta`
             - Outer slope parameter

    Example
    -------
    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import (
        ...     BetaModelDensityProfile,
        ... )

        >>> r = np.logspace(-1, 1, 100)
        >>> profile = BetaModelDensityProfile(
        ...     rho_0=1.0, r_c=1.0, beta=2 / 3
        ... )
        >>> rho = profile(r)

        >>> _ = plt.loglog(r, rho, label="Beta Model")
        >>> _ = plt.xlabel("Radius (r)")
        >>> _ = plt.ylabel("Density (rho)")
        >>> _ = plt.legend()
        >>> plt.show()

    See Also
    --------
    CoredPowerLawDensityProfile, PseudoIsothermalDensityProfile

    References
    ----------
    .. footbibliography::

    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3"),
        "r_c": unyt.unyt_quantity(1.0, "kpc"),
        "beta": 1.0,
    }

    @classmethod
    def __function__(cls, r, rho_0=1.0, r_c=1.0, beta=1.0):
        return rho_0 * (1 + (r / r_c) ** 2) ** (-1.5 * beta)

    @classmethod
    def __function_units__(cls, r, rho_0=unyt.unyt_quantity(1, "Msun/kpc**3"), **kwargs):
        return rho_0.units


# ======================================= #
# Disk Density Profiles                   #
# ======================================= #
class BaseDiskDensityProfile(BaseCylindricalDiskProfile, ABC):
    r"""Base class for disk density profiles.

    This class provides a foundation for cylindrical disk profiles, defining the basic structure and methods
    that all disk profiles should implement. It is not intended to be instantiated directly.
    """

    __IS_ABSTRACT__ = True
    __VARIABLES__ = ["r", "theta"]
    __VARIABLES_LATEX__ = [r"r", r"\theta"]


class ExponentialDiskDensityProfile(BaseDiskDensityProfile):
    r"""Exponential Disk Density Profile.

    Commonly used for stellar and gas disks in spiral galaxies, this profile declines
    exponentially with radius and height.

    .. math::

        \rho(r, z) = \rho_0 \, \exp\left( -\frac{r}{r_s} \right) \, \exp\left( -\frac{|z|}{z_s} \right)

    where:

    - :math:`\\rho_0` is the central midplane density.
    - :math:`r_s` is the radial scale length.
    - :math:`z_s` is the vertical scale height.

    .. dropdown:: Parameters

        .. list-table:: Parameters for :py:class:`ExponentialDiskDensityProfile`
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\\rho_0`
             - Central midplane density
           * - ``r_s``
             - :math:`r_s`
             - Radial scale length
           * - ``z_s``
             - :math:`z_s`
             - Vertical scale height

    Example
    -------
    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> from pisces.profiles.density import (
        ...     ExponentialDiskDensityProfile,
        ... )

        >>> r = np.linspace(0, 15, 200)
        >>> z = 0.0  # Midplane

        >>> profile = ExponentialDiskDensityProfile(
        ...     rho_0=1.0, r_s=3.0, z_s=0.3
        ... )
        >>> rho = profile(r, z)

        >>> plt.semilogy(
        ...     r, rho, label="Exponential Disk (z=0)"
        ... )
        >>> plt.xlabel("Radius (r)")
        >>> plt.ylabel("Density (rho)")
        >>> plt.legend()
        >>> plt.show()

    See Also
    --------
    BaseDiskDensityProfile, DoubleExponentialDiskDensityProfile

    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3"),
        "r_s": unyt.unyt_quantity(3.0, "kpc"),
        "z_s": unyt.unyt_quantity(0.3, "kpc"),
    }

    @classmethod
    def __function__(cls, r, z, rho_0=1.0, r_s=1.0, z_s=1.0):
        return rho_0 * sp.exp(-r / r_s) * sp.exp(-sp.Abs(z) / z_s)

    @classmethod
    def __function_units__(cls, r, z, rho_0=unyt.unyt_quantity(1, "Msun/kpc**3"), **kwargs):
        return rho_0.units


class SechSquaredDiskDensityProfile(BaseDiskDensityProfile):
    r"""Sech² Vertical Disk Density Profile.

    Models a disk with exponential radial falloff and hyperbolic secant squared vertical structure:

    .. math::

        \rho(r, z) = \rho_0 \, \exp\left( -\frac{r}{r_s} \right) \, \mathrm{sech}^2 \left( \frac{z}{z_s} \right)

    Widely used for isothermal, self-gravitating stellar disks.

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_s``
             - :math:`r_s`
             - Radial scale length
           * - ``z_s``
             - :math:`z_s`
             - Vertical scale height

    Example
    -------
    .. plot::
        :include-source:

        >>> from pisces.profiles.density import (
        ...     SechSquaredDiskDensityProfile,
        ... )
        >>> r, z = np.meshgrid(
        ...     np.linspace(0, 10, 100),
        ...     np.linspace(-2, 2, 100),
        ... )
        >>> profile = SechSquaredDiskDensityProfile(
        ...     rho_0=1.0, r_s=3.0, z_s=0.3
        ... )
        >>> rho = profile(r, z)

    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3"),
        "r_s": unyt.unyt_quantity(1.0, "kpc"),
        "z_s": unyt.unyt_quantity(0.3, "kpc"),
    }

    @classmethod
    def __function__(cls, r, z, rho_0=1.0, r_s=1.0, z_s=0.3):
        return rho_0 * sp.exp(-r / r_s) * (1 / sp.cosh(z / z_s)) ** 2

    @classmethod
    def __function_units__(cls, r, z, rho_0=unyt.unyt_quantity(1, "Msun/kpc**3"), **kwargs):
        return rho_0.units


class SechDiskDensityProfile(BaseDiskDensityProfile):
    r"""Sech Vertical Disk Density Profile.

    Models a disk with exponential radial decline and hyperbolic secant vertical structure:

    .. math::

        \rho(r, z) = \rho_0 \, \exp\left( -\frac{r}{r_s} \right) \, \mathrm{sech} \left( \frac{z}{z_s} \right)

    Compared to the :py:class:`SechSquaredDiskDensityProfile`, this version provides a
    thicker vertical profile and is often used for diffuse components or gas layers.

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_s``
             - :math:`r_s`
             - Radial scale length
           * - ``z_s``
             - :math:`z_s`
             - Vertical scale height

    Example
    -------
    .. plot::
        :include-source:

        >>> from pisces.profiles.density import (
        ...     SechDiskDensityProfile,
        ... )
        >>> r, z = np.meshgrid(
        ...     np.linspace(0, 10, 100),
        ...     np.linspace(-2, 2, 100),
        ... )
        >>> profile = SechDiskDensityProfile(
        ...     rho_0=1.0, r_s=3.0, z_s=0.3
        ... )
        >>> rho = profile(r, z)

    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3"),
        "r_s": unyt.unyt_quantity(1.0, "kpc"),
        "z_s": unyt.unyt_quantity(0.3, "kpc"),
    }

    @classmethod
    def __function__(cls, r, z, rho_0=1.0, r_s=1.0, z_s=0.3):
        return rho_0 * sp.exp(-r / r_s) * (1 / sp.cosh(z / z_s))

    @classmethod
    def __function_units__(cls, r, z, rho_0=unyt.unyt_quantity(1, "Msun/kpc**3"), **kwargs):
        return rho_0.units


class FlaredDiskDensityProfile(BaseDiskDensityProfile):
    r"""Flared Exponential Disk Density Profile.

    Extends the exponential disk by allowing the vertical scale height to increase with radius:

    .. math::

        \rho(r, z) = \rho_0 \, \exp\left( -\frac{r}{r_s} \right) \, \exp\left( -\frac{|z|}{z_s(r)} \right)

    where the scale height grows with radius as:

    .. math::

        z_s(r) = z_0 \left( 1 + \frac{r}{r_f} \right)^{\delta}

    This profile captures disk thickening ("flaring") at large radii, as seen in stellar and gas disks.

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_0``
             - :math:`\rho_0`
             - Central density
           * - ``r_s``
             - :math:`r_s`
             - Radial scale length
           * - ``z_0``
             - :math:`z_0`
             - Base vertical scale height
           * - ``r_f``
             - :math:`r_f`
             - Flare radius
           * - ``delta``
             - :math:`\delta`
             - Flare steepness

    Example
    -------
    .. plot::
        :include-source:

        >>> from pisces.profiles.density import (
        ...     FlaredDiskDensityProfile,
        ... )
        >>> r, z = np.meshgrid(
        ...     np.linspace(0, 15, 200),
        ...     np.linspace(-3, 3, 200),
        ... )
        >>> profile = FlaredDiskDensityProfile(
        ...     rho_0=1.0,
        ...     r_s=3.0,
        ...     z_0=0.3,
        ...     r_f=5.0,
        ...     delta=0.5,
        ... )
        >>> rho = profile(r, z)

    """

    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3"),
        "r_s": unyt.unyt_quantity(3.0, "kpc"),
        "z_0": unyt.unyt_quantity(0.3, "kpc"),
        "r_f": unyt.unyt_quantity(5.0, "kpc"),
        "delta": 0.5,
    }

    @classmethod
    def __function__(cls, r, z, rho_0=1.0, r_s=1.0, z_0=0.3, r_f=5.0, delta=0.5):
        z_s_r = z_0 * (1 + r / r_f) ** delta
        return rho_0 * sp.exp(-r / r_s) * sp.exp(-sp.Abs(z) / z_s_r)

    @classmethod
    def __function_units__(cls, r, z, rho_0=unyt.unyt_quantity(1, "Msun/kpc**3"), **kwargs):
        return rho_0.units


# ======================================= #
# Surface Density Profiles                #
# ======================================= #
class SersicProfile(BaseProfile):
    r"""Sérsic Surface Density Profile.

    Describes the surface brightness or surface mass density distribution of galaxies, particularly elliptical
    galaxies and bulges. The profile is defined as:

    .. math::

        \Sigma(R) = \Sigma_0 \, \exp\left( -b_n \left[ \left( \frac{R}{R_e} \right)^{1/n} - 1 \right] \right)

    where:

    - :math:`\Sigma_0` is the central surface density.
    - :math:`R_e` is the effective radius enclosing half the total projected mass.
    - :math:`n` is the Sérsic index controlling the shape (with :math:`n=1` being exponential,
      and :math:`n=4` the de Vaucouleurs profile).
    - :math:`b_n` is a normalization constant dependent on :math:`n`.

    .. note::

        The constant :math:`b_n` is approximately given by:

        .. math::

            b_n \approx 2n - \frac{1}{3} + \frac{4}{405n} + \frac{46}{25515n^2}

        which ensures that :math:`R_e` encloses half the total projected mass.

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``Sigma_0``
             - :math:`\Sigma_0`
             - Central surface density
           * - ``R_e``
             - :math:`R_e`
             - Effective (half-light) radius
           * - ``n``
             - :math:`n`
             - Sérsic index (controls profile shape)

    Example
    -------
    .. plot::
        :include-source:

        >>> from pisces.profiles.density import SersicProfile
        >>> import matplotlib.pyplot as plt
        >>> R = np.linspace(0.01, 10, 200)
        >>> profile = SersicProfile(
        ...     Sigma_0=1.0, R_e=2.0, n=4.0
        ... )
        >>> Sigma = profile(R)
        >>> plt.semilogy(R, Sigma)
        >>> plt.xlabel("Radius (R)")
        >>> plt.ylabel("Surface Density (Sigma)")
        >>> plt.title("Sérsic Surface Profile (n=4)")
        >>> plt.grid(True)
        >>> plt.show()

    See Also
    --------
    BaseProfile, ExponentialDiskDensityProfile

    """

    __IS_ABSTRACT__ = False
    __VARIABLES__ = ["R"]
    __VARIABLES_LATEX__ = [r"R"]
    __PARAMETERS__ = {
        "Sigma_0": unyt.unyt_quantity(1.0, "Msun/kpc**2"),
        "R_e": unyt.unyt_quantity(2.0, "kpc"),
        "n": 4.0,
    }

    @staticmethod
    def _b_n(n):
        """Approximate b_n for given Sérsic index n."""
        return 2 * n - 1 / 3 + 4 / (405 * n) + 46 / (25515 * n**2)

    @classmethod
    def __function__(cls, R, Sigma_0=1.0, R_e=1.0, n=1.0):
        b_n = cls._b_n(n)
        return Sigma_0 * sp.exp(-b_n * ((R / R_e) ** (1 / n) - 1))

    @classmethod
    def __function_units__(cls, R, Sigma_0=unyt.unyt_quantity(1, "Msun/kpc**2"), **kwargs):
        return Sigma_0.units


class GaussianRingProfile(BaseProfile):
    r"""Gaussian Ring Surface Density Profile.

    Models an annular structure with a peak surface density at a characteristic radius,
    declining in both inward and outward directions as a Gaussian:

    .. math::

        \Sigma(R) = \Sigma_0 \, \exp\left( -\frac{(R - R_0)^2}{2 \sigma^2} \right)

    This profile is useful for representing ring galaxies, star-forming rings, or resonance features.

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``Sigma_0``
             - :math:`\Sigma_0`
             - Peak surface density at :math:`R = R_0`
           * - ``R_0``
             - :math:`R_0`
             - Radius of the peak surface density
           * - ``sigma``
             - :math:`\sigma`
             - Width of the ring

    Example
    -------
    .. plot::
        :include-source:

        >>> from pisces.profiles.density import (
        ...     GaussianRingProfile,
        ... )
        >>> import matplotlib.pyplot as plt
        >>> R = np.linspace(0, 10, 200)
        >>> profile = GaussianRingProfile(
        ...     Sigma_0=1.0, R_0=5.0, sigma=0.5
        ... )
        >>> Sigma = profile(R)
        >>> plt.plot(R, Sigma)
        >>> plt.xlabel("Radius (R)")
        >>> plt.ylabel("Surface Density (Sigma)")
        >>> plt.title("Gaussian Ring Profile")
        >>> plt.grid(True)
        >>> plt.show()

    """

    __IS_ABSTRACT__ = False
    __VARIABLES__ = ["R"]
    __VARIABLES_LATEX__ = [r"R"]
    __PARAMETERS__ = {
        "Sigma_0": unyt.unyt_quantity(1.0, "Msun/kpc**2"),
        "R_0": unyt.unyt_quantity(5.0, "kpc"),
        "sigma": unyt.unyt_quantity(0.5, "kpc"),
    }

    @classmethod
    def __function__(cls, R, Sigma_0=1.0, R_0=1.0, sigma=0.5):
        return Sigma_0 * sp.exp(-((R - R_0) ** 2) / (2 * sigma**2))

    @classmethod
    def __function_units__(cls, R, Sigma_0=unyt.unyt_quantity(1, "Msun/kpc**2"), **kwargs):
        return Sigma_0.units


class CoreSersicProfile(BaseProfile):
    r"""Core-Sérsic Surface Density Profile.

    Models galaxies with depleted cores and outer Sérsic-like falloff:

    .. math::

        \Sigma(R) = \Sigma_b \left[1 + \left( \frac{R_b}{R} \right)^\alpha \right]^{\gamma/\alpha}
        \exp\left( -b_n \left( \left( \frac{R^\alpha + R_b^\alpha}{R_e^\alpha} \right)^{1/(\alpha n)}
         - \left( \frac{R_b}{R_e} \right)^{1/n} \right) \right)

    where:

    - :math:`\Sigma_b` is the surface density at the break radius.
    - :math:`R_b` is the break radius between core and outer profile.
    - :math:`\gamma` controls the inner power-law slope.
    - :math:`\alpha` controls sharpness of transition.
    - :math:`R_e` is the effective radius of the outer Sérsic portion.
    - :math:`n` is the Sérsic index.
    - :math:`b_n` is the usual Sérsic constant.

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``Sigma_b``
             - :math:`\Sigma_b`
             - Surface density at break radius
           * - ``R_b``
             - :math:`R_b`
             - Break radius
           * - ``R_e``
             - :math:`R_e`
             - Sérsic effective radius
           * - ``n``
             - :math:`n`
             - Sérsic index
           * - ``gamma``
             - :math:`\gamma`
             - Inner slope
           * - ``alpha``
             - :math:`\alpha`
             - Transition sharpness

    Example
    -------
    .. plot::
        :include-source:

        >>> from pisces.profiles.density import (
        ...     CoreSersicProfile,
        ... )
        >>> import matplotlib.pyplot as plt
        >>> R = np.linspace(0.1, 20, 300)
        >>> profile = CoreSersicProfile(
        ...     Sigma_b=1.0,
        ...     R_b=2.0,
        ...     R_e=5.0,
        ...     n=4.0,
        ...     gamma=0.5,
        ...     alpha=5.0,
        ... )
        >>> Sigma = profile(R)
        >>> plt.semilogy(R, Sigma)
        >>> plt.xlabel("Radius (R)")
        >>> plt.ylabel("Surface Density (Sigma)")
        >>> plt.title("Core-Sérsic Surface Profile")
        >>> plt.grid(True)
        >>> plt.show()

    """

    __IS_ABSTRACT__ = False
    __VARIABLES__ = ["R"]
    __VARIABLES_LATEX__ = [r"R"]
    __PARAMETERS__ = {
        "Sigma_b": unyt.unyt_quantity(1.0, "Msun/kpc**2"),
        "R_b": unyt.unyt_quantity(2.0, "kpc"),
        "R_e": unyt.unyt_quantity(5.0, "kpc"),
        "n": 4.0,
        "gamma": 0.5,
        "alpha": 5.0,
    }

    @staticmethod
    def _b_n(n):
        return 2 * n - 1 / 3 + 4 / (405 * n) + 46 / (25515 * n**2)

    @classmethod
    def __function__(cls, R, Sigma_b=1.0, R_b=2.0, R_e=5.0, n=4.0, gamma=0.5, alpha=5.0):
        b_n = cls._b_n(n)
        x = (R**alpha + R_b**alpha) / R_e**alpha
        term1 = (1 + (R_b / R) ** alpha) ** (gamma / alpha)
        term2 = sp.exp(-b_n * (x ** (1 / (alpha * n)) - (R_b / R_e) ** (1 / n)))
        return Sigma_b * term1 * term2

    @classmethod
    def __function_units__(cls, R, Sigma_b=unyt.unyt_quantity(1, "Msun/kpc**2"), **kwargs):
        return Sigma_b.units
