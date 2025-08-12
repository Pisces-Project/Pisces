"""Polytropic models of stellar structure.

This module contains a variety of models which all rely on an assumed polytropic equation of state and
the resulting Lane-Emden equation to compute the structure of a star. The models are designed to be used
in a variety of contexts, from simple stellar models to more complex systems that require polytropic
equations of state.
"""

from pathlib import Path

import numpy as np
import unyt
from scipy.interpolate import InterpolatedUnivariateSpline
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from unyt import Unit, unyt_array, unyt_quantity
from unyt.physical_constants import G, kb, mp

from pisces.geometry.coordinates import SphericalCoordinateSystem
from pisces.geometry.grids.core import GenericGrid
from pisces.math_utils.integration import (
    compute_lame_emden_solution,
    integrate,
    integrate_mass,
)
from pisces.models.core.base import BaseModel
from pisces.utilities import pisces_config

from ._hooks import PolytropicParticleGenerationHook


class PolytropicStarModel(BaseModel, PolytropicParticleGenerationHook):
    r"""Spherical polytropic stellar structure model.

    This model represents a self-gravitating, spherically symmetric star in hydrostatic equilibrium,
    governed by a polytropic equation of state:

    .. math::

        P = K \rho^{1 + 1/n}

    where :math:`n` is the polytropic index and :math:`K` is a constant determined by core conditions.

    The stellar structure is computed by solving the Lane-Emden equation:

    .. math::

        \frac{1}{\xi^2} \frac{d}{d\xi} \left( \xi^2 \frac{d\theta}{d\xi} \right) = -\theta^n

    where :math:`\theta(\xi)` is the dimensionless density and :math:`\xi` is the dimensionless radius.

    Fields
    ======

    The following physical fields are computed and stored in the model:

    .. csv-table:: Polytropic Star Model Fields
       :file: ../../../docs/source/tables/model_fields/PolytropicStarModel_fields.csv
       :header-rows: 1
       :widths: 15, 40, 25, 25

    Metadata
    ========

    The following pieces of metadata are computed and stored in the model:

    .. csv-table:: Polytropic Star Model Metadata
       :file: ../../../docs/source/tables/model_fields/PolytropicStarModel_meta.csv
       :header-rows: 1
       :widths: 15, 40, 25, 25

    Methodology
    ===========

    The model can be constructed in one of two ways:

    .. tab-set::

       .. tab-item:: Core Density + Core Temperature

          .. hint::

             This mode is available through :meth:`from_density_and_temperature`.

          1. Specify the central density :math:`\rho_c` and central temperature :math:`T_c`.
          2. Use these to compute the polytropic constant :math:`K` and scale length :math:`\alpha`.
          3. Solve the Lane-Emden equation for the given index :math:`n`.
          4. Use :math:`\theta(\xi)` to compute physical profiles:

             - Density: :math:`\rho(r) = \rho_c \theta^n`
             - Pressure: :math:`P(r) = K \rho^{1 + 1/n}`
             - Temperature: from ideal gas law
             - Enclosed mass via: :math:`M(<r) = \int_0^r 4\pi r^2 \rho(r) \, dr`
             - Potential: using total mass and gravitational integration

       .. tab-item:: Total Mass + Radius

          .. hint::

             This mode is available through :meth:`from_mass_and_radius`.

          1. Specify the total stellar mass :math:`M_{\rm tot}` and radius :math:`R`.
          2. Solve Lane-Emden equation to obtain :math:`\xi_{\rm max}` and compute:

             .. math::

                \alpha = \frac{R}{\xi_{\rm max}}

          3. Use Lane-Emden integral to solve for core density:

             .. math::

                \rho_c = \frac{M_{\rm tot}}{4\pi \alpha^3 \int_0^{\xi_{\rm max}} \theta^n \xi^2 d\xi}

          4. Compute central temperature via hydrostatic equilibrium and ideal gas law:

             .. math::

                T_c = \rho_c \cdot \frac{4\pi \mu m_p G \alpha^2}{(n+1) k_B}

          5. Pass results to density+temperature constructor to build the model.

    Assumptions
    ===========

    - **Spherical symmetry**: all fields are radial functions.
    - **Polytropic equation of state**: gas follows :math:`P = K \rho^{1 + 1/n}`.
    - **Hydrostatic equilibrium**: neglects rotation, magnetic fields, or turbulence.
    - **Ideal gas law**: used to relate temperature, pressure, and density.
    - **Fixed mean molecular weight**: assumed constant throughout the star.

    """

    __IS_ABSTRACT__ = False

    # ========================================= #
    # Model Generator Methods                   #
    # ========================================= #
    # These methods are the generator methods for the model. They
    # should take some set of inputs (whatever is needed for the model)
    # and yield a resulting model.
    @classmethod
    def _construct_radial_grid(
        cls,
        min_radius: unyt_quantity,
        max_radius: unyt_quantity,
        num_points: int,
        spacing: str = "log",
    ) -> "GenericGrid":
        """Construct a radial grid between `min_radius` and `max_radius`.

        Parameters
        ----------
        min_radius : ~unyt.array.unyt_quantity
            The minimum radius (e.g., 1 * kpc).
        max_radius : ~unyt.array.unyt_quantity
            The maximum radius (e.g., 1 * Mpc).
        num_points : int
            Number of radial samples to generate.
        spacing : {"log", "linear"}, default: "log"
            The type of spacing to use for the radial grid.

        Returns
        -------
        GenericGrid
            Array of radial values with units attached.

        Raises
        ------
        ValueError
            If spacing type is unrecognized or min/max radius are invalid.

        """
        if min_radius >= max_radius:
            raise ValueError("min_radius must be less than max_radius.")

        if spacing == "log":
            grid = np.geomspace(min_radius.to_value("km"), max_radius.to_value("km"), num=num_points + 1) * Unit("km")
        elif spacing == "linear":
            grid = np.linspace(min_radius.to_value("km"), max_radius.to_value("km"), num=num_points + 1) * Unit("km")
        else:
            raise ValueError(f"Unsupported spacing type '{spacing}'. Use 'log' or 'linear'.")

        # Build the actual grid object in spherical coordinates using a
        # single (radial) axis.
        cs = SphericalCoordinateSystem()
        grid = GenericGrid(
            cs, grid, axes=["r"], units={"r": "km", "theta": "", "phi": ""}, fill_values={"theta": 0.0, "phi": 0.0}
        )

        cls.logger.debug(
            "Constructed %s-spaced radial grid from %.2e to %.2e (%d points)",
            spacing,
            min_radius.to_value("km"),
            max_radius.to_value("km"),
            num_points,
        )

        return grid

    @classmethod
    def _compute_potential(
        cls,
        radii: unyt_array,
        total_density_field: unyt_array,
        total_mass_field: unyt_array,
    ) -> unyt_array:
        r"""Compute the gravitational potential profile using the total density.

        This solves the spherical gravitational potential using the formula:

        .. math::

            \Phi(r) = -G \left[ \frac{M(<r)}{r} + 4\pi \int_0^r \rho(r') \, r' \, dr' \right]

        Parameters
        ----------
        radii : unyt_array
            Radial grid (:math:`r`). Must have units of length (e.g., :math:`\mathrm{kpc}`).
        total_density_field : unyt_array
            Total density profile :math:`\rho(r)` in units of mass/volume (e.g., :math:`M_\odot / \mathrm{kpc}^3`).
        total_mass_field : unyt_array
            Enclosed mass profile :math:`M(<r)` in units of mass (e.g., :math:`M_\odot`).

        Returns
        -------
        unyt_array
            Gravitational potential :math:`\\Phi(r)` in units of :math:`\mathrm{kpc}^2 / \mathrm{Myr}^2`.

        """
        # Extract the radius and the density.
        rr = radii.d
        rho = total_density_field.d
        rho_interp = InterpolatedUnivariateSpline(rr, rho)

        # Perform the first integral term.
        integrand = lambda r: rho_interp(r) * r
        integral_term = integrate(integrand, rr)
        integral_term = unyt_array(
            4.0 * np.pi * integral_term,
            units=total_density_field.units * radii.units**2,
        )

        # Complete the potential calculation.
        potential = -G * (total_mass_field / radii + integral_term)

        return potential.to("kpc**2/Myr**2")

    @classmethod
    def _compute_polytropic_constants(cls, rho_c, T_c, n):
        """Use the core density, core temperature, and polytropic index, to compute the critical constants."""
        mu = cls.config["mean_molecular_weight"]

        K = rho_c ** (-1 / n) * (kb * T_c) / (mu * mp)
        alpha_squared = ((n + 1) * K * rho_c ** ((1 / n) - 1)) / (4 * np.pi * G)
        alpha = np.sqrt(alpha_squared).to("km")

        return K, alpha

    @classmethod
    def _compute_fields_from_theta(cls, theta, rho_c, n, K):
        """Use the theta profile (`theta`) to compute the density, pressure, and temperature profiles."""
        # Start with the density. This is just rho = rho_c * theta**n.
        density = (rho_c * theta**n).to("kg/m**3")

        # Now compute the pressure using the polytropic relation P = K * rho**(1 + 1/n).
        pressure = (K * density ** (1 + 1 / n)).to("Pa")

        # Finally, compute the temperature using the ideal gas law T = P / (rho * mu * mp). Thus
        # needs to be done a little bit carefully because we have div/0 issues that can crop up
        # due to the cutoff on the radius.
        temperature = unyt_array(np.zeros_like(theta), units="K")
        valid_mask = theta > 0

        temperature[valid_mask] = (
            (pressure[valid_mask] * cls.config["mean_molecular_weight"] * mp) / (density[valid_mask] * kb)
        ).to("K")

        return {
            "density": density,
            "pressure": pressure,
            "temperature": temperature,
        }

    @classmethod
    def from_density_and_temperature(
        cls,
        filename: Path | str,
        core_density: unyt_quantity,
        core_temperature: unyt_quantity,
        polytropic_index: float = 1.0,
        rmin: unyt_quantity = unyt_quantity(1, "km"),
        rmax: unyt_quantity = unyt_quantity(100_000, "km"),
        num_points: int = 1000,
        overwrite: bool = False,
        **kwargs,
    ):
        """
        Construct a polytropic star model from core density and core temperature.

        This method generates a complete stellar model assuming a polytropic equation of state
        defined by the given core density, core temperature, and polytropic index. It solves
        the Lane-Emden equation, builds physical profiles, and writes out all relevant fields.

        Parameters
        ----------
        filename : Path or str
            The path to the file where the resulting model will be stored. If the file already exists,
            it will not be overwritten unless `overwrite` is set to True.
        core_density : unyt_quantity
            The central density of the star (e.g., in kg/m**3).
        core_temperature : unyt_quantity
            The central temperature of the star (e.g., in K).
        polytropic_index : float
            The polytropic index `n` governing the stellar structure equation. By
            default, this is set to 1.0.
        rmin : unyt_quantity
            Minimum radial value of the stellar model grid. Default is 1 km.
        rmax : unyt_quantity
            Maximum radial value of the stellar model grid. Default is 100,000 km.
        num_points : int
            Number of radial points in the model grid. Default is 1000.
        overwrite : bool, default: False
            Whether to overwrite the existing file if it already exists.
        **kwargs
            Additional keyword arguments:

            - spacing : {"log", "linear"}, default: "log"
                Whether to use logarithmic or linear spacing for the radial grid.

        Returns
        -------
        PolytropicStarModel
            An instance of the constructed polytropic star model.

        Raises
        ------
        ValueError
            If the Lane-Emden equation does not converge before hitting the maximum value of `xi`.

        """
        # Print out the leading logger information.
        cls.logger.info("Generating PolytropicStarModel...")

        # Enter the model generation context with logging redirection
        # and the step-by-step progress bar.
        with logging_redirect_tqdm(loggers=[cls.logger]):
            progress_bar = tqdm(
                total=7,
                desc="Preparing metadata...",
                disable=pisces_config["system.appearance.disable_progress_bars"],
                leave=False,
            )

            # --- Prepare Metadata, Profiles, and Fields --- #
            # (STEP 1)
            # This section prepares the metadata, profiles, and fields
            # for the model. It initializes the fields dictionary,
            # metadata dictionary, and profiles dictionary.
            fields: dict[str, unyt_array] = {}
            metadata: dict[str, str] = {}
            profiles: dict[str, str] = {}

            # --- Create the radial grid --- #
            # (STEP 2)
            # The next step is to build the radial grid.
            progress_bar.update(1)
            progress_bar.set_description("Generating grid...")

            grid = cls._construct_radial_grid(rmin, rmax, num_points, spacing=kwargs.pop("spacing", "log"))
            radii: unyt.unyt_array = grid["r"]
            # --- Compute Constants --- #
            # (STEP 3)
            # Now we compute the polytropic constants K and alpha. These are
            # needed to define the polytropic star model.
            progress_bar.update(1)
            progress_bar.set_description("Computing model constants...")

            _K, _alpha = cls._compute_polytropic_constants(core_density, core_temperature, polytropic_index)
            metadata["K_constant"] = _K
            metadata["alpha_constant"] = _alpha.to("km")

            # --- Solving Lame-Emden Equation --- #
            # (STEP 4)
            # In this step, we proceed with solving the Lane-Emden equation. To do so,
            # we construct a xi_array based on the radii and the alpha constant. We then
            # solve, compute the theta values and get the cutoff radius. From there, we're
            # ready to compute the density, pressure, and temperature profiles.
            progress_bar.update(1)
            progress_bar.set_description("Solving Lame-Emden Eq...")

            # Setup the xi array for the evaluation. We also need to
            # minimum xi flag because of the numerical issues when solving
            # the equation at very small radii.
            _xi = (radii / _alpha).to_value("dimensionless")
            _xi_min = 1e-6

            # Compute the solution to the Lane-Emden equation.
            le_solution = compute_lame_emden_solution(
                xi_max=_xi.max(),
                n=polytropic_index,
                xi_min=_xi_min,
                end_at_first_zero=True,
                t_eval=_xi[_xi >= _xi_min],  # Evaluate only at valid xi values
                max_step=kwargs.pop("max_step", 0.1),
            )
            xi_sol, theta_sol = le_solution.t, le_solution.y[0]
            # Extract the cutoff xi and the radius of that cutoff if
            # it exists.
            _cutoff_xi = xi_sol[-1]
            _cutoff_radius = _cutoff_xi * _alpha.to("km")

            metadata["stellar_radius"] = _cutoff_radius.to("km")

            # Create the field for theta by interpolating the solution
            # using an InterpolatedUnivariateSpline and then enforcing the
            # cutoff radius.
            _theta_interpolator = InterpolatedUnivariateSpline(xi_sol, theta_sol)
            fields["Theta"] = _theta_interpolator(_xi)
            fields["Theta"][_xi > _cutoff_xi] = 0.0  # Enforce cutoff
            fields["Theta"] = unyt_array(fields["Theta"], units="dimensionless")

            # --- Derive Density, Pressure, and Temperature Profiles --- #
            # (STEP 5)
            # Now we derive the density, pressure, and temperature profiles from
            # the theta values. The density is computed using the polytropic relation,
            # and the pressure and temperature are derived from the density.
            progress_bar.update(1)
            progress_bar.set_description("Computing fields...")

            fields.update(cls._compute_fields_from_theta(fields["Theta"], core_density, polytropic_index, _K))

            # --- Compute the gravitational field --- #
            # (STEP 6)
            # Next, we compute the enclosed mass, gravitational field, and potential.
            progress_bar.update(1)
            progress_bar.set_description("Computing gravitational potential...")

            # We start with the total mass. This requires interpolating the density
            # and integrating as far as the cutoff radius. Past that, we'll require constant values.
            _x, _y = radii, fields["density"].to_value("kg/km**3")
            _xmax = _cutoff_radius.to_value("km")

            _density_interpolator = InterpolatedUnivariateSpline(_x[_x <= _xmax], _y[_x <= _xmax])

            # Now integrate the mass out to the cutoff radius.
            _mass_array = np.zeros_like(_x)
            _mass_array[_x <= _xmax] = integrate_mass(
                _density_interpolator,
                _x[_x <= _xmax],
            )
            _mass_array[_x > _xmax] = _mass_array.max()
            fields["mass"] = unyt_array(_mass_array, units="kg").to("Msun")
            metadata["total_mass"] = fields["mass"].max().to("Msun")

            # With the mass and the density available, we can now compute the gravitational potential.
            fields["potential"] = unyt_array(np.zeros_like(radii), "km**2/s**2")
            fields["potential"][_x <= _xmax] = cls._compute_potential(
                radii=radii[_x <= _xmax],
                total_density_field=fields["density"][_x <= _xmax],
                total_mass_field=fields["mass"][_x <= _xmax],
            )
            fields["potential"][_x > _xmax] = -G * fields["mass"].max() / radii[_x > _xmax]

            # We also compute the gravitational field, which is the negative gradient of the potential.
            # we use a spline interpolation.
            _pot_interp = InterpolatedUnivariateSpline(
                radii.to_value("km"),
                fields["potential"].to_value("km**2/s**2"),
            )
            fields["gravitational_field"] = _pot_interp(radii.to_value("km"), 1) * Unit("km/s**2")

            # End the progress bar.
            progress_bar.update(1)
            progress_bar.set_description("Finalizing model...")
            progress_bar.close()

        # Return the model as a new instance of the class.
        return cls.from_components(filename, grid, fields, profiles, metadata, overwrite=overwrite)

    @classmethod
    def from_mass_and_radius(
        cls,
        filename: Path | str,
        mass: unyt_quantity,
        radius: unyt_quantity,
        polytropic_index: float = 1.0,
        rmin: unyt_quantity = unyt_quantity(1, "km"),
        rmax: unyt_quantity = None,
        num_points: int = 1000,
        overwrite: bool = False,
        **kwargs,
    ):
        """
        Construct a polytropic star model from its total mass and radius.

        This method determines the core density and temperature required to match the
        given total mass and radius for a specified polytropic index. These parameters
        are then passed to the :meth:`from_density_and_temperature` constructor to build the model.

        Parameters
        ----------
        filename : Path or str
            Output filename to write the model to. If the file already exists,
            then it will not be overwritten unless `overwrite` is set to True.
        mass : unyt_quantity
            Total stellar mass (e.g., in Msun).
        radius : unyt_quantity
            Total stellar radius (e.g., in Rsun).
        polytropic_index : float, default: 1.0
            The polytropic index `n` for the equation of state.
        rmin : unyt_quantity, default: 1 km
            Minimum radial value for the model grid.
        rmax : unyt_quantity, optional
            Maximum radial value for the model grid. If None, uses `radius`.
        num_points : int, default: 1000
            Number of radial grid points.
        overwrite : bool, default: False
            Whether to overwrite an existing file.
        **kwargs
            Additional keyword arguments forwarded to :meth:`from_density_and_temperature`.

        Returns
        -------
        PolytropicStarModel
            An instance of the constructed model.

        Raises
        ------
        ValueError
            If the Lane-Emden equation fails to converge.
        """
        # Extract xi_max from kwargs if provided (default is 10)
        xi_max = kwargs.pop("xi_max", 10.0)

        # Compute the core density and temperature from mass and radius
        rho_c, T_c = cls.compute_parameters_from_mass_and_radius(
            mass=mass,
            radius=radius,
            polytropic_index=polytropic_index,
            xi_max=xi_max,
        )

        # Set default rmax if not explicitly provided
        if rmax is None:
            rmax = radius.to("km")

        # Pass to the density+temperature model constructor
        return cls.from_density_and_temperature(
            filename=filename,
            core_density=rho_c,
            core_temperature=T_c,
            polytropic_index=polytropic_index,
            rmin=rmin,
            rmax=rmax,
            num_points=num_points,
            overwrite=overwrite,
            **kwargs,
        )

    # ========================================= #
    # Scaling Methods and Relations             #
    # ========================================= #
    @classmethod
    def compute_parameters_from_mass_and_radius(
        cls,
        mass: unyt_quantity,
        radius: unyt_quantity,
        polytropic_index: float = 1.0,
        mu: float = None,
        xi_max: float = 10.0,
    ) -> tuple[unyt_quantity, unyt_quantity]:
        r"""
        Compute the central density and temperature of a polytropic star from its mass and radius.

        This method derives the core properties of a polytropic stellar model, specifically the central
        density (:math:`\rho_c`) and central temperature (:math:`T_c`), based on the total stellar mass,
        total stellar radius, and the chosen polytropic index. It numerically solves the Lane-Emden
        equation for the given index to determine the structure of the polytrope, and uses scaling
        relations to connect the dimensionless solution to physical units.

        The computed central density is determined by matching the Lane-Emden mass integral to the
        desired total mass. The central temperature is then derived from the ideal gas law assuming
        hydrostatic equilibrium and a known mean molecular weight.

        Parameters
        ----------
        mass : ~unyt.unyt_quantity
            Total stellar mass (e.g., in solar masses). This must be consistent with units of mass.
            If no unit is provided, it defaults to solar masses (Msun).
        radius : ~unyt.unyt_quantity
            Total stellar radius (e.g., in solar radii). This must have dimensions of length.
            If no unit is provided, it defaults to solar radii (Rsun).
        polytropic_index : float, default: 1.0
            The polytropic index :math:`n` controlling the equation of state.
        mu : float, optional
            Mean molecular weight. If None, pulled from the class configuration.
        xi_max : float, default: 10
            Maximum dimensionless radius to use when integrating the Lane-Emden equation. If this is
            too small, the Lane-Emden equation will not hit its first zero before reaching the truncation
            point and an error will be raised.

            If this occurs, it is generally safe to simply increase this value to ensure the solution is valid.

        Returns
        -------
        rho_c : ~unyt.unyt_quantity
            Central mass density of the star.
        T_c : ~unyt.unyt_quantity
            Central temperature of the star.

        Raises
        ------
        ValueError
            If the Lane-Emden solver does not encounter a zero crossing before `xi_max`.

        Notes
        -----
        For a given polytropic index :math:`n`, the Lane-Emden equation is solved numerically via a call to
        the low-level :func:`~math_units.integration.compute_lame_emden_solution`. Given the truncation point
        :math:`\xi_{\rm max}`, the scale parameter :math:`\alpha` is computed as the ratio of the total radius to the
        truncation point, i.e., :math:`\alpha = R / \xi_{\rm max}`.

        The total mass of the star is

        .. math::

            M_{\rm total} = 4\pi \alpha^3 \rho_c \int_0^{\xi_{\rm max}} \theta^n \xi^2 d\xi.

        As such, the integral is computed and used to determine the core density :math:`\rho_c` as.

        Finally, the central temperature is computed using the ideal gas law.

        """
        # Load the relevant configuration / sort of the
        # units and ensure everything is in the right place.
        mass, radius = unyt.unyt_quantity(mass, "Msun"), unyt.unyt_quantity(radius, "Rsun")
        mu = mu if mu is not None else cls.config["mean_molecular_weight"]

        # Use the polytropic index to compute the solution to the Lane-Emden equation.
        # This will then give us the maximum radius and we can use that to scale things.
        le_solution = compute_lame_emden_solution(
            xi_max=xi_max,
            n=polytropic_index,
            end_at_first_zero=True,
            max_step=0.01,  # Ensure we have a reasonable step size for convergence)
        )

        # Check that the solution reached a termination event.
        if le_solution.status != 1:
            raise ValueError(
                "The Lane-Emden equation did not encounter a zero within xi_max. Please"
                " increase the threshold value for xi_max to ensure the solution is valid."
            )

        xi, theta = le_solution.t, le_solution.y[0]

        # Using the solution of the LEE, we can compute the special integral
        # int_0^xi_max theta^n xi^2 dxi, which is used to compute the central density.
        _interpolator = InterpolatedUnivariateSpline(xi, theta**polytropic_index)
        _mass_integral = integrate_mass(_interpolator, xi)[-1]

        # Using the xi bound, we can compute the alpha parameter on the scale.
        alpha = radius / xi[-1]  # Scale radius by the truncation point.

        # Now combine these to compute the central density value.
        rho_c = mass / (alpha**3 * _mass_integral)

        # Finally, we compute the central temperature using the ideal gas law.
        T_c = rho_c * (4 * np.pi * mu * mp * G * alpha**2) / ((polytropic_index + 1) * kb)

        # Return the values
        return rho_c, T_c


if __name__ == "__main__":
    # Example usage of the PolytropicStarModel class
    model = PolytropicStarModel.from_density_and_temperature(
        filename="polytropic_star_model.h5",
        core_density=unyt.unyt_quantity(1e3, "kg/m**3"),
        core_temperature=unyt.unyt_quantity(1e6, "K"),
        polytropic_index=1.0,
        rmin=unyt.unyt_quantity(1, "km"),
        rmax=unyt.unyt_quantity(100_000, "km"),
        num_points=1000,
        overwrite=True,
    )
    print(model)
