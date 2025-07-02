"""
Density profiles for use in Pisces models.

The :mod:`density` module provides a number of built-in profiles to model density and surface density
profiles in various contexts, including for galaxies, clusters of galaxies, stars, etc.
"""
from abc import ABC
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import sympy as sp
import unyt
from scipy.integrate import quad, quad_vec

from pisces.profiles.base import BaseSphericalRadialProfile, derived_profile
from pisces.utilities.config import pisces_config
from pisces.utilities.math_ops import integrate_mass

# Type Hints
_UnitType = Union[str, unyt.Unit]
_UnitValue = Union[unyt.unyt_array, unyt.unyt_quantity]

if TYPE_CHECKING:
    from astropy.cosmology import Cosmology


# ======================================= #
# Radial Profiles (Spherical)             #
# ======================================= #
class BaseSphericalDensityProfile(BaseSphericalRadialProfile, ABC):
    """
    Abstract base class for spherically symmetric density profiles with
    standard gravitationally derived attached profiles.

    **Derived Profiles:**

    - Enclosed Mass : ``enclosed_mass``
    - Gravitational Field (acceleration) : ``gravitational_field``
    - Gravitational Potential : ``gravitational_potential``

    Notes
    -----
    - Gravitational constant ``G`` is included as a parameter with default units.
    - All derived profiles propagate units based on the density expression.
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

        def _unit_func(r, **param_units):
            G = param_units.pop("G")
            rho_unit = cls.__function_units__(r, **param_units)
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
            G = param_units.pop("G")
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
    def compute_surface_density(
        self, R: _UnitValue, units: Optional[_UnitType] = None, **kwargs
    ) -> _UnitValue:
        r"""
        Numerically compute the projected surface density at radius R.

        .. math::

            \Sigma(R) = 2 \int_0^\infty \rho \left( \sqrt{R^2 + z^2} \right) dz

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
            Surface density at each R, with correct units.
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
        result = (
            2
            * quad_vec(
                _integrand_, 0, np.inf, args=(r_arr,), full_output=False, **kwargs
            )[0]
        )

        # Determine the units. We have standard propagation
        # to get rho, and then we integrate through by z, which
        # should ARBITRARILY have the same units as _rr, so we
        # just multiply by a factor of the length unit.
        sd_unit = self.__function_units__(r_units, **self.__parameter_units__) * r_units
        sd = result * sd_unit

        if units is not None:
            sd = sd.to(units)

        if np.isscalar(R):
            return sd[0]
        return sd

    def compute_enclosed_mass(
        self, r: _UnitValue, units: Optional[_UnitType] = None, **kwargs
    ) -> _UnitValue:
        r"""
        Numerically compute the enclosed mass profile.

        .. math::

            M(r) = 4 \pi \int_0^r \rho(r') \, r'^2 \, dr'

        Parameters
        ----------
        r : ~unyt.array.unyt_quantity or ~unyt.array.unyt_array
            Radius or radii at which to compute the enclosed mass (must carry length units).
        units: ~unyt.unit_object.Unit or str, optional
            The output units to use.
        kwargs : dict
            Additional keyword arguments passed to :func:`~scipy.integrate.quad`.

        Returns
        -------
        mass : ~unyt.array.unyt_quantity or ~unyt.array.unyt_array
            Enclosed mass at each radius, with correct units.
        """
        r_array = unyt.array.unyt_array(r)
        r_values, r_unit = r_array.value, r_array.units

        # Units for density and mass
        rho_unit = self.__function_units__(r_unit, **self.__parameter_units__)
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

    def compute_total_mass(
        self, units: Optional[_UnitType] = None, **kwargs
    ) -> unyt.unyt_quantity:
        r"""
        Numerically compute the total mass of the profile.

        .. math::

            M_{\mathrm{tot}} = 4 \pi \int_0^\infty \rho(r) \, r^2 \, dr

        Parameters
        ----------
        units: str or ~unyt.unit_object.Unit, optional
            The units in which to return the mass. By default, this is :math:`M_{\rm sun}`.
        kwargs : dict
            Additional keyword arguments passed to :func:`~scipy.integrate.quad`.

        Returns
        -------
        mass : unyt_quantity
            Total mass with appropriate units.

        Raises
        ------
        RuntimeError
            If the integral is divergent or fails to converge to finite precision.
        """
        import warnings

        from scipy.integrate import IntegrationWarning

        # Use arbitrary length scale (units cancel internally)
        r_unit = unyt.Unit("kpc")
        rho_unit = self.__function_units__(r_unit, **self.__parameter_units__)
        mass_unit = rho_unit * r_unit**3

        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always", IntegrationWarning)

            def _density_fn(r_):
                return self(r_).value

            result, err = quad(
                lambda r_: _density_fn(r_) * 4 * np.pi * r_**2, 0, np.inf, **kwargs
            )[:2]

            # Check for divergence (nan/inf result)
            if not np.isfinite(result):
                raise RuntimeError(
                    "Total mass integral did not converge: result is infinite or NaN."
                )

            # Check for integration warnings (e.g., max subdivisions exceeded)
            for warn in warning_list:
                if issubclass(warn.category, IntegrationWarning):
                    raise RuntimeError(
                        f"Total mass integral raised an IntegrationWarning: {warn.message}"
                    )

        # Handle the units.
        value = result * mass_unit

        if units is not None:
            return value.to(units)
        else:
            return value

    def compute_fractional_mass_radius(
        self, rmin, rmax, fraction=0.5, units: Optional[_UnitType] = "kpc", **kwargs
    ):
        r"""
        Compute the radius enclosing a given fraction of the total mass.

        This method requires the profile to have finite total mass. It uses
        :func:`~scipy.optimize.brentq` to numerically solve for the radius enclosing
        the specified fraction of the total mass.

        Parameters
        ----------
        rmin : float or ~unyt.unyt_quantity
            Minimum search radius, interpreted as having units specified by ``units``.
        rmax : float or ~unyt.unyt_quantity
            Maximum search radius, interpreted as having units specified by ``units``.
        fraction : float, optional
            Fraction of total mass to enclose (must be between 0 and 1). Default is 0.5 (half-mass radius).
        units : str or ~unyt.Unit, optional
            Length units for ``rmin`` and ``rmax``. Default is "kpc".
        kwargs : dict
            Additional arguments passed to :meth:`compute_total_mass` and :meth:`compute_enclosed_mass`.

        Returns
        -------
        radius : ~unyt.unyt_quantity
            Radius enclosing the desired mass fraction, with specified units.

        Raises
        ------
        ValueError
            If fraction is invalid or total mass does not converge.
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
        r"""
        Compute the spherical overdensity profile relative to the critical density.

        .. math::

            \Delta(R) = \frac{3 M(R)}{4 \pi R^3 \rho_{\mathrm{crit}}(z)}

        Parameters
        ----------
        z : float
            Redshift.
        R : array-like or scalar
            Radii at which to compute overdensity.
        cosmology : ~astropy.cosmology.Cosmology, optional
            The cosmology to use for the computation. This is used to compute
            the critical density :math:`\rho_{\rm crit}(z)` which is then used
            in the computation.

            `cosmology` must be provided as a valid AstroPy cosmology. By
            default, the default cosmology from pisces configuration is used.

        kwargs : dict
            Additional arguments for mass integration.

        Returns
        -------
        overdensity : array-like or scalar
            Overdensity at each radius (dimensionless).
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
                raise ValueError(
                    f"Default cosmology `{_default}` is not available in astropy cosmology."
                ) from exp

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
            raise ValueError(
                f"Provided cosmology object failed to compute critical density: {exp}"
            ) from exp

        # Compute the mass and the relevant overdensities.
        rho_crit = rho_crit * density_array.units
        mass = self.compute_enclosed_mass(R_array, **kwargs)
        delta = (3 * mass) / (4 * np.pi * R_array**3 * rho_crit)

        return delta.d

    def compute_cosmological_overdensity_radius(
        self,
        z: float,
        delta_target: float,
        rmin: float,
        rmax: float,
        units: Optional[_UnitType] = "kpc",
        cosmology: Optional["Cosmology"] = None,
        **kwargs,
    ):
        """
        Compute the radius enclosing a target overdensity relative to the critical density.

        Uses :func:`~scipy.optimize.brentq` to solve for the radius enclosing a
        specified overdensity relative to :math:`\rho_{\rm crit}(z)`.

        Parameters
        ----------
        z : float
            Redshift.
        delta_target : float
            Desired overdensity relative to critical density.
        rmin : float
            Minimum search radius, interpreted as having units specified by ``units``.
        rmax : float
            Maximum search radius, interpreted as having units specified by ``units``.
        units : str or ~unyt.Unit, optional
            Length units for ``rmin`` and ``rmax``. Default is "kpc".
        cosmology : ~astropy.cosmology.Cosmology, optional
            The cosmology to use. Defaults to the value from ``pisces_config`` if None.
        kwargs : dict
            Additional arguments passed to :meth:`compute_enclosed_mass`.

        Returns
        -------
        radius : ~unyt.unyt_quantity
            Radius enclosing the desired overdensity, with correct units.

        Raises
        ------
        ValueError
            If the search bounds are invalid or cosmology is not provided.
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
                raise ValueError(
                    f"Default cosmology `{_default}` is not available in astropy.cosmology."
                ) from exp

        # Get critical density with units consistent to profile density
        rho_unit = self.__function_units__(unit, **self.__parameter_units__)
        try:
            rho_crit = cosmology.critical_density(z).to_value(str(rho_unit))
        except Exception as exp:
            raise ValueError(
                f"Failed to compute critical density with provided cosmology: {exp}"
            ) from exp

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
        units: Optional[_UnitType] = None,
        G: Optional[unyt.unyt_quantity] = None,
        **kwargs,
    ) -> _UnitValue:
        r"""
        Compute the circular velocity at radius ``r``.

        .. math::

            v_c(r) = \sqrt{ \frac{G M(r)}{r} }

        Parameters
        ----------
        r : ~unyt.unyt_quantity or array-like
            Radius or radii at which to compute the circular velocity.
        units : str or ~unyt.Unit, optional
            Desired output units. By default, this will use the built-in
            units of the various parameters.
        G: ~unyt.array.unyt_quantity
            The value of the gravitational constant. By default, this
            is set to the standard value as implemented in unyt.
        kwargs : dict
            Additional arguments passed to :meth:`compute_enclosed_mass`.

        Returns
        -------
        v_c : ~unyt.unyt_quantity or ~unyt.unyt_array
            Circular velocity at each radius, with specified units.
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
        units: Optional[_UnitType] = None,
        G: Optional[unyt.unyt_quantity] = None,
        **kwargs,
    ) -> _UnitValue:
        r"""
        Compute the escape velocity at radius ``r``.

        .. math::

            v_{\mathrm{esc}}(r) = \sqrt{ \frac{2 G M(r)}{r} }

        Parameters
        ----------
        r : ~unyt.unyt_quantity or array-like
            Radius or radii at which to compute the escape velocity.
        units : str or ~unyt.Unit, optional
            Desired output units. By default, this will use the internal
            units of the various parameters.
        G : ~unyt.unyt_quantity, optional
            Gravitational constant to use. Defaults to the standard value from unyt.
        kwargs : dict
            Additional arguments passed to :meth:`compute_enclosed_mass`.

        Returns
        -------
        v_esc : ~unyt.unyt_quantity or ~unyt.unyt_array
            Escape velocity at each radius, with specified units.
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
        units: Optional[_UnitType] = "arcsec",
        G: Optional[unyt.unyt_quantity] = None,
        z_lens: Optional[float] = None,
        z_source: Optional[float] = None,
        D_l: Optional[_UnitValue] = None,
        D_s: Optional[_UnitValue] = None,
        D_ls: Optional[_UnitValue] = None,
        cosmology: Optional["Cosmology"] = None,
        **kwargs,
    ) -> _UnitValue:
        r"""
        Compute the gravitational lensing deflection angle at projected radius ``R``.

        This method can return:

        - **Physical deflection angle** in length units (default: kpc), computed as:

          .. math::

              \alpha_{\mathrm{phys}}(R) = \frac{4 G M(<R)}{c^2 R}

        - **Angular deflection angle** in angular units (default: arcsec), computed as:

          .. math::

              \alpha_{\mathrm{ang}}(R) = \alpha_{\mathrm{phys}}(R) \times \frac{D_{ls}}{D_s}

          where :math:`D_l`, :math:`D_s`, and :math:`D_{ls}` are the angular diameter distances to lens,
          source, and between lens and source, respectively.

        Parameters
        ----------
        R : ~unyt.unyt_quantity or array-like
            Projected radius in the lens plane, with length units.
        mode : {"physical", "angular"}, optional
            Whether to return physical or angular deflection. Default is "angular".
        units : str or ~unyt.Unit, optional
            Output units. Default is "arcsec" for angular mode, or profile-consistent length units for physical mode.
        G : ~unyt.unyt_quantity, optional
            Gravitational constant to use. Defaults to `unyt.physical_constants.gravitational_constant`.
        z_lens : float, optional
            Redshift of the lens. Required if mode is "angular" and distances not provided.
        z_source : float, optional
            Redshift of the source. Required if mode is "angular" and distances not provided.
        D_l : ~unyt.unyt_quantity, optional
            Angular diameter distance to the lens.
        D_s : ~unyt.unyt_quantity, optional
            Angular diameter distance to the source.
        D_ls : ~unyt.unyt_quantity, optional
            Angular diameter distance between lens and source.
        cosmology : ~astropy.cosmology.Cosmology, optional
            Cosmology to use for distance calculations. Defaults to ``pisces_config['physics.default_cosmology']``.
        kwargs : dict
            Additional arguments passed to :meth:`compute_enclosed_mass`.

        Returns
        -------
        alpha : ~unyt.unyt_quantity or ~unyt.unyt_array
            Deflection angle at each radius, in specified units.

        Raises
        ------
        ValueError
            If required cosmological distances are missing for angular deflection.
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
                    raise ValueError(
                        f"Default cosmology `{_default}` is not available in astropy.cosmology."
                    ) from exp

            D_s = cosmology.angular_diameter_distance(z_source).to("kpc")
            D_ls = cosmology.angular_diameter_distance_z1z2(z_lens, z_source).to("kpc")

        # Apply distance ratio for angular deflection
        alpha_ang = alpha_phys * (D_ls / D_s)
        return alpha_ang.to(units) if units else alpha_ang

    def compute_einstein_radius(
        self,
        z_lens: float,
        z_source: float,
        units: Optional[_UnitType] = "arcsec",
        G: Optional[unyt.unyt_quantity] = None,
        cosmology: Optional["Cosmology"] = None,
        **kwargs,
    ) -> _UnitValue:
        r"""
        Compute the Einstein ring angular radius for a perfectly aligned source-lens system.

        Solves for the angular radius :math:`\theta_E` satisfying:

        .. math::

            \alpha( \theta_E D_l ) = \theta_E

        where:

        - :math:`\alpha` is the angular deflection angle at projected radius :math:`R`
        - :math:`D_l` is the angular diameter distance to the lens

        Parameters
        ----------
        z_lens : float
            Redshift of the lens.
        z_source : float
            Redshift of the source (must be > z_lens).
        units : str or ~unyt.Unit, optional
            Desired output units for the Einstein radius (default "arcsec").
        G : ~unyt.unyt_quantity, optional
            Gravitational constant to use. Defaults to `unyt.physical_constants.gravitational_constant`.
        cosmology : ~astropy.cosmology.Cosmology, optional
            Cosmology to use. Defaults to ``pisces_config['physics.default_cosmology']``.
        kwargs : dict
            Additional arguments passed to :meth:`compute_enclosed_mass`.

        Returns
        -------
        theta_E : ~unyt.unyt_quantity
            Einstein ring angular radius, in specified units.

        Raises
        ------
        ValueError
            If redshift ordering is invalid or cosmology is not provided.
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
                raise ValueError(
                    f"Default cosmology `{_default}` is not available in astropy.cosmology."
                ) from exp

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

    def compute_lensing_convergence(
        self, R, z_lens, z_source, cosmology=None, units=None, **kwargs
    ):
        r"""
        Compute lensing convergence :math:`\kappa(R)` at projected radius ``R``.

        Parameters
        ----------
        R : ~unyt.unyt_quantity
            Projected radius with length units.
        z_lens : float
            Redshift of the lens.
        z_source : float
            Redshift of the source.
        cosmology : ~astropy.cosmology.Cosmology, optional
            Cosmology to use. Defaults to ``pisces_config``.
        units : str or ~unyt.Unit, optional
            Desired output units for :math:`\kappa`. Default is dimensionless.

        Returns
        -------
        kappa : ~unyt.unyt_quantity or array
            Lensing convergence at each radius.
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
    r"""
    Navarro-Frenk-White :footcite:p:`NFWProfile` (NFW) Density Profile.

    This profile is commonly used in astrophysics to describe the dark matter halo density
    in a spherical, isotropic system. It is derived from simulations of structure formation.

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
        >>> from pisces.profiles.density import NFWDensityProfile

        >>> r = np.linspace(0.1, 10, 100)
        >>> profile = NFWDensityProfile(rho_0=1.0, r_s=1.0)
        >>> rho = profile(r)

        >>> _ = plt.loglog(r, rho,'k-', label='NFW Profile')
        >>> _ = plt.xlabel('Radius (r)')
        >>> _ = plt.ylabel('Density (rho)')
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

    @classmethod
    def __function__(cls, r, rho_0=1.0, r_s=1.0):
        return rho_0 / (r / r_s * (1 + r / r_s) ** 2)

    @classmethod
    def __function_units__(cls, r, rho_0="", r_s="") -> unyt.Unit:
        r, rho_0, r_s = unyt.Unit(r), unyt.Unit(rho_0), unyt.Unit(r_s)
        return rho_0 / (r / r_s) ** 3


class HernquistDensityProfile(BaseSphericalDensityProfile):
    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/pc**3"),
        "r_s": unyt.unyt_quantity(1.0, "pc"),
    }

    @classmethod
    def __function__(cls, r, rho_0=1.0, r_s=1.0):
        return rho_0 / ((r / r_s) * (1 + r / r_s) ** 3)

    @classmethod
    def __function_units__(cls, r, rho_0="", r_s=""):
        r, rho_0, r_s = unyt.Unit(r), unyt.Unit(rho_0), unyt.Unit(r_s)
        return rho_0 / (r / r_s) ** 3


class EinastoDensityProfile(BaseSphericalDensityProfile):
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
    def __function_units__(cls, r, rho_0="", r_s="", alpha=""):
        return unyt.Unit(rho_0)


class SingularIsothermalDensityProfile(BaseSphericalDensityProfile):
    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {"rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3")}

    @classmethod
    def __function__(cls, r, rho_0=1.0):
        return rho_0 / r**2

    @classmethod
    def __function_units__(cls, r, rho_0=""):
        r, rho_0 = unyt.Unit(r), unyt.Unit(rho_0)
        return rho_0 / r**2


class CoredIsothermalDensityProfile(BaseSphericalDensityProfile):
    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3"),
        "r_c": unyt.unyt_quantity(1.0, "kpc"),
    }

    @classmethod
    def __function__(cls, r, rho_0=1.0, r_c=1.0):
        return rho_0 / (1 + (r / r_c) ** 2)

    @classmethod
    def __function_units__(cls, r, rho_0="", r_c=""):
        r, rho_0, r_c = unyt.Unit(r), unyt.Unit(rho_0), unyt.Unit(r_c)
        return rho_0


class PlummerDensityProfile(BaseSphericalDensityProfile):
    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "M": unyt.unyt_quantity(1.0, "Msun"),
        "r_s": unyt.unyt_quantity(1.0, "pc"),
    }

    @classmethod
    def __function__(cls, r, M=1.0, r_s=1.0):
        return (3 * M) / (4 * sp.pi * r_s**3) * (1 + (r / r_s) ** 2) ** (-5 / 2)

    @classmethod
    def __function_units__(cls, r, M="", r_s=""):
        r, M, r_s = unyt.Unit(r), unyt.Unit(M), unyt.Unit(r_s)
        return M / r_s**3


class DehnenDensityProfile(BaseSphericalDensityProfile):
    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "M": unyt.unyt_quantity(1.0, "Msun"),
        "r_s": unyt.unyt_quantity(1.0, "pc"),
        "gamma": 1.0,
    }

    @classmethod
    def __function__(cls, r, M=1.0, r_s=1.0, gamma=1.0):
        return (
            ((3 - gamma) * M)
            / (4 * sp.pi * r_s**3)
            * (r / r_s) ** (-gamma)
            * (1 + r / r_s) ** (gamma - 4)
        )

    @classmethod
    def __function_units__(cls, r, M="", r_s="", gamma=""):
        r, M, r_s = unyt.Unit(r), unyt.Unit(M), unyt.Unit(r_s)
        return M / r_s**3


class JaffeDensityProfile(BaseSphericalDensityProfile):
    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3"),
        "r_s": unyt.unyt_quantity(1.0, "pc"),
    }

    @classmethod
    def __function__(cls, r, rho_0=1.0, r_s=1.0):
        return rho_0 / ((r / r_s) * (1 + r / r_s) ** 2)

    @classmethod
    def __function_units__(cls, r, rho_0="", r_s=""):
        r, rho_0, r_s = unyt.Unit(r), unyt.Unit(rho_0), unyt.Unit(r_s)
        return rho_0 / (r / r_s)


class BurkertDensityProfile(BaseSphericalDensityProfile):
    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3"),
        "r_s": unyt.unyt_quantity(1.0, "pc"),
    }

    @classmethod
    def __function__(cls, r, rho_0=1.0, r_s=1.0):
        return rho_0 / ((1 + r / r_s) * (1 + (r / r_s) ** 2))

    @classmethod
    def __function_units__(cls, r, rho_0="", r_s=""):
        r, rho_0, r_s = unyt.Unit(r), unyt.Unit(rho_0), unyt.Unit(r_s)
        return rho_0


class MooreDensityProfile(BaseSphericalDensityProfile):
    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3"),
        "r_s": unyt.unyt_quantity(1.0, "pc"),
    }

    @classmethod
    def __function__(cls, r, rho_0=1.0, r_s=1.0):
        return rho_0 / ((r / r_s) ** (3 / 2) * (1 + r / r_s) ** (3 / 2))

    @classmethod
    def __function_units__(cls, r, rho_0="", r_s=""):
        r, rho_0, r_s = unyt.Unit(r), unyt.Unit(rho_0), unyt.Unit(r_s)
        return rho_0 / ((r / r_s) ** (3 / 2))


class CoredNFWDensityProfile(BaseSphericalDensityProfile):
    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3"),
        "r_s": unyt.unyt_quantity(1.0, "pc"),
    }

    @classmethod
    def __function__(cls, r, rho_0=1.0, r_s=1.0):
        return rho_0 / ((1 + (r / r_s) ** 2) * (1 + r / r_s) ** 2)

    @classmethod
    def __function_units__(cls, r, rho_0="", r_s=""):
        r, rho_0, r_s = unyt.Unit(r), unyt.Unit(rho_0), unyt.Unit(r_s)
        return rho_0


class KingDensityProfile(BaseSphericalDensityProfile):
    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3"),
        "r_c": unyt.unyt_quantity(1.0, "kpc"),
        "r_t": unyt.unyt_quantity(1.0, "kpc"),
    }

    @classmethod
    def __function__(cls, r, rho_0=1.0, r_c=1.0, r_t=1.0):
        return rho_0 * (
            (1 + (r / r_c) ** 2) ** (-1.5) - (1 + (r_t / r_c) ** 2) ** (-1.5)
        )

    @classmethod
    def __function_units__(cls, r, rho_0="", r_c="", r_t=""):
        r, rho_0, r_c, r_t = (
            unyt.Unit(r),
            unyt.Unit(rho_0),
            unyt.Unit(r_c),
            unyt.Unit(r_t),
        )
        return rho_0


class VikhlininDensityProfile(BaseSphericalDensityProfile):
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
    def __function__(
        cls, r, rho_0=1.0, r_c=1.0, r_s=1.0, alpha=1.0, beta=1.0, epsilon=1.0, gamma=3.0
    ):
        return (
            rho_0
            * (r / r_c) ** (-0.5 * alpha)
            * (1 + (r / r_c) ** 2) ** (-1.5 * beta + 0.25 * alpha)
            * (1 + (r / r_s) ** gamma) ** (-0.5 * epsilon / gamma)
        )

    @classmethod
    def __function_units__(
        cls, r, rho_0="", r_c="", r_s="", alpha="", beta="", epsilon="", gamma=""
    ):
        r, rho_0, r_c, r_s = (
            unyt.Unit(r),
            unyt.Unit(rho_0),
            unyt.Unit(r_c),
            unyt.Unit(r_s),
        )
        return rho_0


class AM06DensityProfile(BaseSphericalDensityProfile):
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
        return (
            rho_0 * (1 + r / a_c) * (1 + r / (a_c * c)) ** alpha * (1 + r / a) ** beta
        )

    @classmethod
    def __function_units__(cls, r, rho_0="", a="", a_c="", c="", alpha="", beta=""):
        r, rho_0, a, a_c = unyt.Unit(r), unyt.Unit(rho_0), unyt.Unit(a), unyt.Unit(a_c)
        return rho_0


class SNFWDensityProfile(BaseSphericalDensityProfile):
    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "M": unyt.unyt_quantity(1.0, "Msun"),
        "a": unyt.unyt_quantity(1.0, "kpc"),
    }

    @classmethod
    def __function__(cls, r, M=1.0, a=1.0):
        return 3 * M / (16 * sp.pi * a**3) / ((r / a) * (1 + r / a) ** 2.5)

    @classmethod
    def __function_units__(cls, r, M="", a=""):
        r, M, a = unyt.Unit(r), unyt.Unit(M), unyt.Unit(a)
        return M / (a**3 * (r / a))


class TNFWDensityProfile(BaseSphericalDensityProfile):
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
    def __function_units__(cls, r, rho_0="", r_s="", r_t=""):
        r, rho_0, r_s, r_t = (
            unyt.Unit(r),
            unyt.Unit(rho_0),
            unyt.Unit(r_s),
            unyt.Unit(r_t),
        )
        return rho_0 / (r / r_s)


class PseudoIsothermalDensityProfile(BaseSphericalDensityProfile):
    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3"),
        "r_c": unyt.unyt_quantity(1.0, "kpc"),
    }

    @classmethod
    def __function__(cls, r, rho_0=1.0, r_c=1.0):
        return rho_0 / (1 + (r / r_c) ** 2)

    @classmethod
    def __function_units__(cls, r, rho_0="", r_c=""):
        r, rho_0, r_c = unyt.Unit(r), unyt.Unit(rho_0), unyt.Unit(r_c)
        return rho_0


class DoublePowerLawDensityProfile(BaseSphericalDensityProfile):
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
        return (
            rho_0
            / (r / r_s) ** gamma
            / (1 + (r / r_s) ** alpha) ** ((beta - gamma) / alpha)
        )

    @classmethod
    def __function_units__(cls, r, rho_0="", r_s="", alpha="", beta="", gamma=""):
        r, rho_0, r_s = unyt.Unit(r), unyt.Unit(rho_0), unyt.Unit(r_s)
        return rho_0 / (r / r_s) ** gamma


class CoredPowerLawDensityProfile(BaseSphericalDensityProfile):
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
    def __function_units__(cls, r, rho_0="", r_c="", gamma=""):
        r, rho_0, r_c = unyt.Unit(r), unyt.Unit(rho_0), unyt.Unit(r_c)
        return rho_0


class GeneralizedNFWDensityProfile(BaseSphericalDensityProfile):
    __IS_ABSTRACT__ = False
    __PARAMETERS__ = {
        "rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3"),
        "r_s": unyt.unyt_quantity(1.0, "pc"),
        "gamma": 1.0,
    }

    @classmethod
    def __function__(cls, r, rho_0=1.0, r_s=1.0, gamma=1.0):
        return rho_0 / (r / r_s) ** gamma / (1 + r / r_s) ** (3 - gamma)

    @classmethod
    def __function_units__(cls, r, rho_0="", r_s="", gamma=""):
        r, rho_0, r_s = unyt.Unit(r), unyt.Unit(rho_0), unyt.Unit(r_s)
        return rho_0 / (r / r_s) ** gamma


class BetaModelDensityProfile(BaseSphericalDensityProfile):
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
    def __function_units__(cls, r, rho_0="", r_c="", beta=""):
        r, rho_0, r_c = unyt.Unit(r), unyt.Unit(rho_0), unyt.Unit(r_c)
        return rho_0


# ======================================= #
# Disk Profiles (Cylindrical)             #
# ======================================= #
#
# class MiyamotoNagaiDensityProfile(BaseSymmetricCylindricalProfile):
#     __IS_ABSTRACT__ = False
#     __PARAMETERS__ = {"M": unyt.unyt_quantity(1.0, "Msun"), "a": unyt.unyt_quantity(1.0, "kpc"), "b": 1.0}
#
#     @classmethod
#   def __function__(cls, R, z, M=1.0, a=1.0, b=1.0):
#         B = sp.sqrt(z**2 + b**2)
#         denominator = (R**2 + (a + B)**2) ** 2.5
#         return (b**2 * M) / (4 * sp.pi) * (a * R**2 + (a + 3 * B) * (a + B)**2) / (denominator * B**3)
#
#     @classmethod
#    def __function_units__(cls, R, z, M="", a="", b=""):
#         R, z, M, a, b = unyt.Unit(R), unyt.Unit(z), unyt.Unit(M), unyt.Unit(a), unyt.Unit(b)
#         return M / a ** 3
#
# class ExponentialDiskDensityProfile(BaseSymmetricCylindricalProfile):
#     __IS_ABSTRACT__ = False
#     __PARAMETERS__ = {"rho_0": unyt.unyt_quantity(1.0, "Msun/kpc**3"), "R_d": 1.0, "z_d": 1.0}
#
#     @classmethod
#    def __function__(cls, R, z, rho_0=1.0, R_d=1.0, z_d=1.0):
#         return rho_0 * sp.exp(-R / R_d) * sp.sech(z / z_d) ** 2
#
#     @classmethod
#    def __function_units__(cls, R, z, rho_0="", R_d="", z_d=""):
#         R, z, rho_0, R_d, z_d = unyt.Unit(R), unyt.Unit(z), unyt.Unit(rho_0), unyt.Unit(R_d), unyt.Unit(z_d)
#         return rho_0
