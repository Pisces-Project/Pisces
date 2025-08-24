"""Spherical galaxy cluster models.

This module provides classes and methods for constructing and analyzing spherically symmetric galaxy cluster
models, including both standard and magnetized variants. It supports the generation of cluster models
from analytic or tabulated profiles of gas density, temperature, entropy, and total density,
and can include optional stellar and magnetic field components.

This module is intended for use in astrophysical modeling and analysis of galaxy clusters,
particularly in the context of hydrostatic equilibrium and multi-component mass modeling.
"""

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import h5py
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from unyt import Unit, unyt_array, unyt_quantity
from unyt.physical_constants import G, mp

from pisces.geometry.coordinates import SphericalCoordinateSystem
from pisces.geometry.grids.core import GenericGrid
from pisces.math_utils import integrate
from pisces.models.core.base import BaseModel
from pisces.models.galaxy_clusters._hooks import SGCParticleGenerationHook
from pisces.physics import (
    compute_mean_molecular_weight,
    compute_mean_molecular_weight_per_electron,
)
from pisces.physics.virialization.eddington import compute_eddington_distribution
from pisces.profiles.base import BaseProfile
from pisces.utilities import pisces_config

if TYPE_CHECKING:
    from pisces.profiles.density import BaseSphericalDensityProfile
    from pisces.profiles.entropy import BaseSphericalEntropyProfile
    from pisces.profiles.temperature import BaseSphericalTemperatureProfile


class SphericalGalaxyClusterModel(BaseModel, SGCParticleGenerationHook):
    r"""Spherical, hydrostatic galaxy cluster model.

    This model represents a galaxy cluster composed of 3 components:

    1. **Gas**: the hot, diffuse plasma filling the cluster.
    2. **Dark Matter**: the dominant mass component, providing gravitational binding.
    3. **Stellar**: the stellar component, if provided.

    The model is fully spherical and assumes hydrostatic equilibrium, meaning the gas pressure balances the
    gravitational pull of the total mass (gas + dark matter + stellar). This model
    ignores the presence of non-thermal pressure components (e.g., cosmic rays, magnetic fields).

    Fields
    ======

    The following physical fields are computed and stored in the model:

    .. csv-table:: Spherical Galaxy Cluster Model Fields
       :file: ../../../docs/source/tables/model_fields/SphericalGalaxyClusterModel.csv
       :header-rows: 1
       :widths: 15, 40, 25, 25

    Methodology
    ===========

    This model makes a number of assumptions across all of its construction methods:

    - **Spherical symmetry**: all profiles depend only on radius :math:`r`.
    - **Fully thermal hydrostatic equilibrium (HE)**: non-thermal pressure components are ignored.
    - **Single-phase gas**: temperature and density are well-defined scalars at each point.
    - **Ionic equilibrium**: used for electron density and entropy computation.
    - **Ideal gas law**: pressure and temperature are related via a constant mean molecular weight.

    The process to generate a model follows one of three modes depending on the inputs.

    .. tab-set::

       .. tab-item:: Density & Total Density

          .. hint::

            This mode is available through :meth:`from_density_and_total_density`.

          Given gas density :math:`\rho_{\mathrm{gas}}(r)`, total density :math:`\rho_{\mathrm{tot}}(r)`,
          and an optional stellar density :math:`\rho_{\star}(r)`, we perform the following steps:

          1. **Compute Dark Matter Density**:

             Requiring that the total density is the sum of gas, stellar, and dark matter densities:

             .. math::

                \rho_{\mathrm{DM}}(r) = \rho_{\mathrm{tot}}(r) - \rho_{\mathrm{gas}}(r) - \rho_{\star}(r)

          1. **Compute mass profiles** via:

             .. math::

                M_i(<r) = 4\pi \int_0^r \rho_i(r') \, r'^2 \, dr'

             for each component :math:`i \in \{\mathrm{gas}, \mathrm{tot}, {\rm dm},  \mathrm{stellar}\}`.

          2. **Compute the gravitational field**:

             .. math::
                \nabla \Phi(r) = \frac{G M_{\mathrm{tot}}(<r)}{r^2}

          3. Use hydrostatic equilibrium to **integrate for pressure**:

             .. math::

                \nabla P = - \rho_{\mathrm{gas}}(r) \nabla \Phi(r)

          4. **Solve for temperature** using the ideal gas law:

             .. math::
                T(r) = \frac{P(r)}{n(r)} = \frac{P(r)}{\rho_{\mathrm{gas}}(r)} \cdot \mu m_p

       .. tab-item:: Temperature + Density

          .. hint::

            This mode is available through :meth:`from_temperature_and_density`.

          Given :math:`T(r)` and :math:`\rho_{\mathrm{gas}}(r)`, we compute:

          1. **Pressure** from the ideal gas law:

             .. math::
                P(r) = \rho_{\mathrm{gas}}(r) \cdot T(r) / (\mu m_p)

          2. **Gravitational field** via hydrostatic equilibrium:

             .. math::
                g(r) = -\frac{1}{\rho_{\mathrm{gas}}(r)} \cdot \frac{dP}{dr}

          3. **Total mass profile**:

             .. math::
                M_{\mathrm{tot}}(<r) = \frac{r^2 \cdot g(r)}{G}

          4. Derive :math:`\rho_{\mathrm{tot}}(r)` via:

             .. math::
                \rho_{\mathrm{tot}}(r) = \frac{1}{4\pi r^2} \frac{dM_{\mathrm{tot}}}{dr}

       .. tab-item:: Entropy + Density

          .. hint::

            This mode is available through :meth:`from_entropy_and_density`.

          Given entropy :math:`K(r)` and gas density :math:`\rho_{\mathrm{gas}}(r)`, we use:

          1. **Entropy definition**:

             .. math::

                K(r) = \frac{T(r)}{n_e(r)^{2/3}}

             with

             .. math::
                n_e(r) = \frac{\rho_{\mathrm{gas}}(r)}{\mu_e m_p}

          2. **Invert to get temperature**:

             .. math::
                T(r) = K(r) \cdot \left(\frac{\rho_{\mathrm{gas}}(r)}{\mu_e m_p}\right)^{2/3}

          3. Compute pressure from ideal gas law and proceed as in the temperature + density pathway.

    .. note::

       In all modes, the gravitational potential is computed via:

       .. math::
          \Phi(r) = -G \left[ \frac{M(<r)}{r} + 4\pi \int_0^r \rho_{\mathrm{tot}}(r') \cdot r' \, dr' \right]

    """

    __IS_ABSTRACT__ = False

    # ========================================= #
    # Model Generator Methods                   #
    # ========================================= #
    # These methods are the generator methods for the model. They
    # should take some set of inputs (whatever is needed for the model)
    # and yield a resulting model.
    @classmethod
    def _construct_model_file_skeleton(cls, filepath: str | Path, overwrite: bool = False) -> Path:
        """Create an empty HDF5 file with the necessary structure for a model.

        This method should be called before any data is written to the file.
        It sets up the basic structure of the HDF5 file, including groups for
        fields and profiles.

        Parameters
        ----------
        filepath : str or Path
            The path to the HDF5 file to create.
        overwrite: bool, default: False
            If True, allows overwriting an existing file at the specified path.

        """
        # Ensure that the filepath is a valid path object and
        # then handle issues with overwriting existing files.
        filepath = Path(filepath)
        if filepath.exists() and not overwrite:
            raise FileExistsError(f"File {filepath} already exists. Use 'overwrite=True' to overwrite it.")

        # Open the HDF5 file in write mode and start managing the
        # relevant structure. This can be overwritten in subclasses as
        # needed to add additional structure. By default, we simply
        # create the FIELDS and PROFILES groups, which are the
        # two main components of the model. We then add the class
        # name as an attribute on the file to ensure that we can
        # identify the model type later on.
        with h5py.File(filepath, mode="w") as handle:
            handle.create_group("FIELDS")
            handle.create_group("PROFILES")
            handle.create_group("DISTRIBUTION_FUNCTIONS")
            handle.create_group("GRID")

            # Add the class name as an attribute on the file.
            handle.attrs["__model_class__"] = cls.__name__

            # Add any other necessary groups or attributes here.

        return filepath

    @classmethod
    def _construct_radial_grid(
        cls,
        min_radius: unyt_quantity,
        max_radius: unyt_quantity,
        num_points: int,
        spacing: str = "log",
    ) -> GenericGrid:
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
        unyt_array
            Array of radial values with units attached.

        Raises
        ------
        ValueError
            If spacing type is unrecognized or min/max radius are invalid.

        """
        if min_radius >= max_radius:
            raise ValueError("min_radius must be less than max_radius.")

        if spacing == "log":
            grid = np.geomspace(min_radius.to_value("kpc"), max_radius.to_value("kpc"), num=num_points) * Unit("kpc")
        elif spacing == "linear":
            grid = np.linspace(min_radius.to_value("kpc"), max_radius.to_value("kpc"), num=num_points) * Unit("kpc")
        else:
            raise ValueError(f"Unsupported spacing type '{spacing}'. Use 'log' or 'linear'.")

        cs = SphericalCoordinateSystem()
        grid = GenericGrid(
            cs,
            grid,
            axes=["r"],
            units={"r": "kpc", "theta": "", "phi": ""},
            fill_values={"theta": 0.0, "phi": 0.0},
        )

        cls.logger.debug(
            "Constructed %s-spaced radial grid from %.2e to %.2e (%d points)",
            spacing,
            min_radius.to_value("kpc"),
            max_radius.to_value("kpc"),
            num_points,
        )

        return grid

    @classmethod
    def _solve_hydrostatic_equilibrium(
        cls,
        radii: unyt_array,
        density_profile: "BaseSphericalDensityProfile",
        gravitational_field: unyt_array = None,
        pressure_field: unyt_array = None,
    ) -> unyt_array:
        r"""Solve the hydrostatic equilibrium equation in one direction.

        - If ``gravitational_field`` is given, integrate inward to solve for the pressure profile.
        - If ``pressure_field`` is given, differentiate to solve for the gravitational field profile.

        The equation of hydrostatic equilibrium is:

        .. math::

            \frac{dP}{dr} = -\rho(r) \, g(r)

        which can also be rearranged as:

        .. math::

            g(r) = -\frac{1}{\\rho(r)} \frac{dP}{dr}

        Parameters
        ----------
        radii : unyt_array
            Radial grid (:math:`r`), in units of length.
        density_profile : BaseSphericalDensityProfile
            Gas density profile :math:`\\rho(r)`.
        gravitational_field : unyt.array.unyt_array, optional
            Gravitational field :math:`g(r)`, used to compute pressure.
        pressure_field : unyt.array.unyt_array, optional
            Pressure profile :math:`P(r)`, used to compute gravitational field.

        Returns
        -------
        unyt_array
            Either:
            - Pressure profile (:math:`P(r)`), if ``gravitational_field`` is provided.
            - Gravitational field (:math:`g(r)`), if ``pressure_field`` is provided.

        Raises
        ------
        ValueError
            If neither or both of ``gravitational_field`` and ``pressure_field`` are provided.

        """
        if (gravitational_field is None) == (pressure_field is None):
            raise ValueError("Provide exactly one of `gravitational_field` or `pressure_field`.")

        # Determine the direction in which we are solving the equation. If we are given
        # the gravitational field, then we solve for the pressure. If we are given the
        # pressure, then we need to solve for the potential.
        _solving_for = "pressure" if gravitational_field is not None else "potential"

        if _solving_for == "pressure":
            # This case is dPhi = -1/rho * dP. We integrate outward given
            # the gravitational field. For the boundary we assume a 1/r^2 drop off.
            g_spline = InterpolatedUnivariateSpline(radii.d, gravitational_field.d)
            r_edge, g_edge = radii[-1].d, gravitational_field[-1].d

            # Construct the integrands.
            integrand_internal = lambda r: -1 * density_profile(r, no_units=True) * g_spline(r)
            integrand_external = lambda r: -1 * density_profile(r, no_units=True) * g_edge * (r_edge / r) ** 2

            # Compute the value, determine the units and then
            # return.
            P = -integrate(integrand_internal, radii.d)
            P -= quad(integrand_external, r_edge, np.inf, limit=100)[0]

            units = density_profile.get_output_units(radii.units) * gravitational_field.units * radii.units
            return unyt_array(P, units).to("Msun/kpc/Myr**2")

        else:
            # Case: Solve for gravitational potential from dΦ/dr = -1/ρ * dP/dr
            P_spine = InterpolatedUnivariateSpline(radii.d, pressure_field.d)

            gravitational_field = -P_spine(radii.d, 1) / density_profile(radii.d, no_units=True)

            units = pressure_field.units / density_profile.get_output_units(radii.units) / radii.units
            return unyt_array(gravitational_field, units).to("kpc/Myr**2")

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

            \Phi(r) = -G \left[ \frac{M(<r)}{r} + 4\pi \int_0^\infty \rho(r') \, r' \, dr' \right]

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
    def _fields_from_temperature_and_density(
        cls,
        fields: dict[str, unyt_array],
        density_profile: "BaseSphericalDensityProfile",
        radii: unyt_array,
        progress_bar,
        stellar_density_profile: Optional["BaseSphericalDensityProfile"] = None,
    ):
        """Compute core fields from gas density and temperature fields.

        Parameters
        ----------
        fields : dict
            Dictionary of field arrays including at least 'gas_density' and 'temperature'.
        density_profile : BaseSphericalDensityProfile
            Analytic gas density profile used for mass calculations.
        stellar_density_profile : ~profiles.density.BaseSphericalDensityProfile, optional
            Optional stellar density profile. If provided, the stellar density will
            be included in the model. Otherwise, the stellar density is assumed to
            be zero and no stellar component is included.

        radii : unyt_array
            Radial grid points.
        progress_bar : tqdm.tqdm
            TQDM progress bar to update during processing.

        """
        # --- Compute Pressure --- #
        # (Step 4 [temp], Step _ [entr])
        # Compute the pressure of the field based on the idea gas
        # law.
        XH = cls.config["hydrogen_abundance"]
        mu = compute_mean_molecular_weight(XH)
        fields["pressure"] = fields["gas_density"] * fields["temperature"] / (mp * mu)
        fields["pressure"].convert_to_units("Msun/(kpc*Myr**2)")

        # --- Solve HSE Equation --- #
        # (Step 5 [temp], Step 6 [entr])
        # Solve the HSE equation to get the gravitational field.
        progress_bar.set_description("Computing gravitational field...")
        progress_bar.update(1)

        fields["gravitational_field"] = cls._solve_hydrostatic_equilibrium(
            radii, density_profile, pressure_field=fields["pressure"]
        )

        # --- Compute Masses --- #
        # (Step 6 [temp], Step 7 [entr])
        # Solve the HSE equation to get the gravitational field.
        progress_bar.set_description("Computing Mass Profiles...")
        progress_bar.update(1)

        fields["gas_mass"] = density_profile.compute_enclosed_mass(radii, units="Msun")
        if stellar_density_profile is not None:
            fields["stellar_mass"] = stellar_density_profile.compute_enclosed_mass(radii, units="Msun")
        else:
            fields["stellar_mass"] = unyt_array(np.zeros_like(radii), units="Msun")

        fields["total_mass"] = radii**2 * fields["gravitational_field"] / G
        fields["dark_matter_mass"] = fields["total_mass"] - fields["gas_mass"] - fields["stellar_mass"]

        # --- Densities & Potential --- #
        # (Step 7 [temp], Step 8 [entr])
        # Finally take derivatives to compute the density profiles
        # and calculate the potentials.
        progress_bar.set_description("Computing potential...")
        progress_bar.update(1)

        for key in ("total", "dark_matter"):
            spline = InterpolatedUnivariateSpline(radii.d, fields[f"{key}_mass"].d)
            dMdr = unyt_array(spline(radii.d, 1), units="Msun/kpc")
            fields[f"{key}_density"] = dMdr / (4.0 * np.pi * radii**2)

        fields["gravitational_potential"] = cls._compute_potential(radii, fields["total_density"], fields["total_mass"])

        return fields

    @classmethod
    def from_density_and_total_density(
        cls,
        density_profile: "BaseSphericalDensityProfile",
        total_density_profile: "BaseSphericalDensityProfile",
        filename: str | Path,
        min_radius: unyt_quantity | str = unyt_quantity(1, "kpc"),
        max_radius: unyt_quantity | str = unyt_quantity(1, "Mpc"),
        num_points: int = 1000,
        overwrite: bool = False,
        stellar_density_profile: "BaseSphericalDensityProfile" = None,
        **kwargs,
    ):
        r"""Generate a spherical cluster model from analytic density profiles.

        Parameters
        ----------
        density_profile : ~profiles.density.BaseSphericalDensityProfile
            Profile object representing the radial **gas** density.
        total_density_profile : ~profiles.density.BaseSphericalDensityProfile
            Profile object representing the **total** density.
        filename : str or Path
            The file at which to save the model. If the file already
            exists, then it will raise an error unless ``overwrite=True``.

            This should be an HDF5 file with a ``.h5`` or ``.hdf5`` extension.

        min_radius : ~unyt.array.unyt_quantity or str, optional
            Minimum radius for sampling (default: 1 kpc).
        max_radius : ~unyt.array.unyt_quantity or str, optional
            Maximum radius for sampling (default: 1 Mpc).
        num_points : int, optional
            Number of radial samples (default: 1000).
        overwrite : bool, optional
            Whether to overwrite existing file (default: False).
        stellar_density_profile : ~profiles.density.BaseSphericalDensityProfile, optional
            Optional stellar density profile. If provided, the stellar density will
            be included in the model. Otherwise, the stellar density is assumed to
            be zero and no stellar component is included.
        **kwargs:
            Additional keyword arguments passed to the density profile constructors.

        Notes
        -----
        Given gas density :math:`\rho_{\mathrm{gas}}(r)`, total density :math:`\rho_{\mathrm{tot}}(r)`,
        and an optional stellar density :math:`\rho_{\star}(r)`, we perform the following steps:

        1. **Compute Dark Matter Density**:

           Requiring that the total density is the sum of gas, stellar, and dark matter densities:

           .. math::

             \rho_{\mathrm{DM}}(r) = \rho_{\mathrm{tot}}(r) - \rho_{\mathrm{gas}}(r) - \rho_{\star}(r)

        1. **Compute mass profiles** via:

           .. math::

             M_i(<r) = 4\pi \int_0^r \rho_i(r') \, r'^2 \, dr'

        for each component :math:`i \in \{\mathrm{gas}, \mathrm{tot}, {\rm dm},  \mathrm{stellar}\}`.

        2. **Compute the gravitational field**:

           .. math::

             \nabla \Phi(r) = \frac{G M_{\mathrm{tot}}(<r)}{r^2}

        3. Use hydrostatic equilibrium to **integrate for pressure**:

           .. math::

             \nabla P = - \rho_{\mathrm{gas}}(r) \nabla \Phi(r)

        4. **Solve for temperature** using the ideal gas law:

           .. math::

             T(r) = \frac{P(r)}{n(r)} = \frac{P(r)}{\rho_{\mathrm{gas}}(r)} \cdot \mu m_p

        """
        # --- Preambular Setup --- #
        # This section of the pathway just configures the
        # progress bar, the logger, and some other setup
        # data.
        cls.logger.info("Generating SphericalGalaxyClusterModel from density and total density profiles.")

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
            profiles: dict[str, BaseProfile] = {
                "gas_density": density_profile,
                "total_density": total_density_profile,
            }
            if stellar_density_profile is not None:
                profiles["stellar_density"] = stellar_density_profile

            # --- Create Radial Grid (Logarithmic Spacing) --- #
            # (STEP 2)
            # This section creates a radial grid for the model.
            progress_bar.update(1)
            progress_bar.set_description("Generating grid...")

            grid = cls._construct_radial_grid(min_radius, max_radius, num_points, spacing=kwargs.pop("spacing", "log"))
            radii = grid["r"]

            # --- Create Density Fields --- #
            # (STEP 3)
            # This section computes the gas and total density profiles
            # at the specified radii, and also computes the stellar density
            # if provided.
            progress_bar.update(1)
            progress_bar.set_description("Building density...")

            fields["gas_density"] = density_profile(radii)
            fields["total_density"] = total_density_profile(radii)
            if stellar_density_profile is not None:
                fields["stellar_density"] = stellar_density_profile(radii)
            else:
                fields["stellar_density"] = unyt_array(np.zeros_like(radii), units="Msun/kpc**3")

            fields["dark_matter_density"] = fields["total_density"] - fields["gas_density"] - fields["stellar_density"]

            # --- Computing Mass Fields --- #
            # (STEP 4)
            # This section computes the enclosed mass profiles for gas,
            # total, and stellar components. It also computes the dark matter
            # mass as the difference between total and gas + stellar masses.
            progress_bar.update(1)
            progress_bar.set_description("Computing masses...")

            fields["gas_mass"] = density_profile.compute_enclosed_mass(radii, units="Msun")
            fields["total_mass"] = total_density_profile.compute_enclosed_mass(radii, units="Msun")
            if stellar_density_profile is not None:
                fields["stellar_mass"] = stellar_density_profile.compute_enclosed_mass(radii, units="Msun")
            else:
                fields["stellar_mass"] = unyt_array(np.zeros_like(radii), units="Msun")

            fields["dark_matter_mass"] = fields["total_mass"] - fields["gas_mass"] - fields["stellar_mass"]

            # --- Compute Gravitational Field & Potential --- #
            # (STEP 5)
            # This section computes the gravitational field and potential
            # using the total mass profile. It uses Newton's law of gravitation
            # to compute the gravitational field and potential at each radius.
            progress_bar.update(1)
            progress_bar.set_description("Calculating potential...")

            fields["gravitational_field"] = G * fields["total_mass"] / (grid["r"] ** 2)
            fields["gravitational_potential"] = cls._compute_potential(
                radii, fields["total_density"], fields["total_mass"]
            )

            # --- Compute Temperature and Pressure --- #
            # (STEP 6)
            # To compute the pressure, we have - dP = rho dPhi, but
            # we then need to integrate to infinity to get the full pressure profile.
            progress_bar.update(1)
            progress_bar.set_description("Calculating temperature...")

            fields["pressure"] = cls._solve_hydrostatic_equilibrium(
                radii,
                density_profile,
                gravitational_field=fields["gravitational_field"],
            )

            XH = cls.config["hydrogen_abundance"]
            mu = compute_mean_molecular_weight(XH)
            fields["temperature"] = fields["pressure"] * mp * mu / fields["gas_density"]
            fields["temperature"].convert_to_units("keV")

            # --- Compute Extras --- #
            # (STEP 7)
            # This section computes additional fields such as sound speed, entropy, and baryon fraction.
            # Compute sound speed.
            progress_bar.update(1)
            progress_bar.set_description("Performing additional computations...")

            gamma = 5 / 3
            fields["sound_speed"] = np.sqrt(gamma * fields["pressure"] / fields["gas_density"])
            fields["sound_speed"].convert_to_units("km/s")

            # Compute the baryon fraction.
            fields["baryon_fraction"] = (fields["gas_mass"] + fields["stellar_mass"]) / fields["total_mass"]

            # Compute the electron number density.
            mue = compute_mean_molecular_weight_per_electron(XH)
            fields["electron_density"] = fields["gas_density"] / (mue * mp)
            fields["electron_density"].convert_to_units("1/cm**3")

            # Compute the entropy.
            fields["entropy"] = fields["temperature"] / fields["electron_density"] ** (2 / 3)
            fields["entropy"].convert_to_units("keV*cm**2")

            # Compute the internal energy / unit mass
            fields["internal_energy_per_unit_mass"] = fields["temperature"] / (mp * mu * (2 / 3))
            fields["internal_energy_per_unit_mass"].convert_to_units("km**2/s**2")

            progress_bar.set_description("COMPLETE")
            progress_bar.update(1)
            progress_bar.close()

        return cls.from_components(filename, grid, fields, profiles, metadata, overwrite=overwrite)

    @classmethod
    def from_temperature_and_density(
        cls,
        density_profile: "BaseSphericalDensityProfile",
        temperature_profile: "BaseSphericalTemperatureProfile",
        filename: str | Path,
        min_radius: unyt_quantity | str = unyt_quantity(1, "kpc"),
        max_radius: unyt_quantity | str = unyt_quantity(1, "Mpc"),
        num_points: int = 1000,
        overwrite: bool = False,
        stellar_density_profile: "BaseSphericalDensityProfile" = None,
        **kwargs,
    ):
        r"""Generate a spherical galaxy cluster model from gas density and temperature profiles.

        Parameters
        ----------
        density_profile : ~profiles.density.BaseSphericalDensityProfile
            Profile object representing the radial **gas** density.
        temperature_profile : ~profiles.temperature.BaseSphericalTemperatureProfile
            Profile object representing the radial **gas** temperature.
        filename : str or Path
            Output HDF5 file path.
        min_radius : ~unyt.array.unyt_quantity or str, optional
            Minimum radius for sampling (default: 1 kpc).
        max_radius : ~unyt.array.unyt_quantity or str, optional
            Maximum radius for sampling (default: 1 Mpc).
        num_points : int, optional
            Number of radial samples (default: 1000).
        overwrite : bool, optional
            Whether to overwrite existing file (default: False).
        stellar_density_profile : ~profiles.density.BaseSphericalDensityProfile, optional
            Optional stellar density profile. If provided, the stellar density will
            be included in the model. Otherwise, the stellar density is assumed to
            be zero and no stellar component is included.
        **kwargs:
            Additional keyword arguments for radial grid construction (e.g., spacing).

        Notes
        -----
        Given :math:`T(r)` and :math:`\rho_{\mathrm{gas}}(r)`, we compute:

        1. **Pressure** from the ideal gas law:

        .. math::

            P(r) = \rho_{\mathrm{gas}}(r) \cdot T(r) / (\mu m_p)

        2. **Gravitational field** via hydrostatic equilibrium:

        .. math::

            g(r) = -\frac{1}{\rho_{\mathrm{gas}}(r)} \cdot \frac{dP}{dr}

        3. **Total mass profile**:

        .. math::

            M_{\mathrm{tot}}(<r) = \frac{r^2 \cdot g(r)}{G}

        4. Derive :math:`\rho_{\mathrm{tot}}(r)` via:

        .. math::

            \rho_{\mathrm{tot}}(r) = \frac{1}{4\pi r^2} \frac{dM_{\mathrm{tot}}}{dr}

        """
        # --- Preambular Setup --- #
        # This section of the pathway just configures the
        # progress bar, the logger, and some other setup
        # data.
        cls.logger.info("Generating SphericalGalaxyClusterModel from temperature and density profiles.")

        with logging_redirect_tqdm(loggers=[cls.logger]):
            progress_bar = tqdm(
                total=8,
                desc="Prepare metadata, profiles, and fields...",
                disable=pisces_config["system.appearance.disable_progress_bars"],
                leave=False,
            )

            # --- Step 1: Initialize data --- #
            # (STEP 1)
            # This section initializes the fields, metadata, and profiles
            # for the model. It prepares the fields dictionary, metadata
            # dictionary, and profiles dictionary.
            fields: dict[str, unyt_array] = {}
            metadata: dict[str, str] = {}
            profiles: dict[str, BaseSphericalDensityProfile] = {
                "gas_density": density_profile,
                "temperature": temperature_profile,
            }
            if stellar_density_profile is not None:
                profiles["stellar_density"] = stellar_density_profile

            # --- Step 2: Construct radial grid --- #
            # (STEP 2)
            # This section constructs a radial grid for the model.
            progress_bar.update(1)
            progress_bar.set_description("Creating grid...")

            grid = cls._construct_radial_grid(min_radius, max_radius, num_points, spacing=kwargs.pop("spacing", "log"))
            radii = grid["r"]

            # --- Step 3: Evaluate profiles --- #
            # This section evaluates the gas density and temperature profiles
            # at the specified radii. It also computes the pressure profile
            # using the ideal gas law: P = rho * kT / (mu * m_p), where mu is the
            # mean molecular weight and m_p is the proton mass.
            progress_bar.update(1)
            progress_bar.set_description("Building profile fields...")

            fields["gas_density"] = density_profile(radii)
            fields["temperature"] = temperature_profile(radii)

            # --- Step 4: Compute Fields --- #
            # These computations are shared with the entropy path
            # and are therefore abstracted out. This completes
            # the hydrostatic calculation and then computes the
            # masses and gravitational quantities.
            progress_bar.update(1)
            fields = cls._fields_from_temperature_and_density(
                fields,
                density_profile,
                radii,
                progress_bar,
                stellar_density_profile=stellar_density_profile,
            )

            # --- Compute Extras --- #
            # (Step 8)
            # This section computes additional fields such as sound speed, entropy, and baryon fraction.
            # Compute sound speed.
            progress_bar.update(1)
            progress_bar.set_description("Performing additional computations...")

            gamma = 5 / 3
            mu = compute_mean_molecular_weight(cls.config["hydrogen_abundance"])
            fields["sound_speed"] = np.sqrt(gamma * fields["pressure"] / fields["gas_density"])
            fields["sound_speed"].convert_to_units("km/s")

            # Compute the baryon fraction.
            fields["baryon_fraction"] = (fields["gas_mass"] + fields["stellar_mass"]) / fields["total_mass"]

            # Compute the electron number density.
            mue = compute_mean_molecular_weight_per_electron(cls.config["hydrogen_abundance"])
            fields["electron_density"] = fields["gas_density"] / (mue * mp)
            fields["electron_density"].convert_to_units("1/cm**3")

            # Compute the entropy.
            fields["entropy"] = fields["temperature"] / fields["electron_density"] ** (2 / 3)
            fields["entropy"].convert_to_units("keV*cm**2")

            # Compute the internal energy / unit mass
            fields["internal_energy_per_unit_mass"] = fields["temperature"] / (mp * mu * (2 / 3))
            fields["internal_energy_per_unit_mass"].convert_to_units("km**2/s**2")

            progress_bar.set_description("COMPLETE")
            progress_bar.update(1)
            progress_bar.close()

        return cls.from_components(
            filename,
            grid,
            fields,
            profiles,
            metadata=metadata,
            overwrite=overwrite,
        )

    @classmethod
    def from_entropy_and_density(
        cls,
        density_profile: "BaseSphericalDensityProfile",
        entropy_profile: "BaseSphericalEntropyProfile",
        filename: str | Path,
        min_radius: unyt_quantity | str = unyt_quantity(1, "kpc"),
        max_radius: unyt_quantity | str = unyt_quantity(1, "Mpc"),
        num_points: int = 1000,
        overwrite: bool = False,
        stellar_density_profile: "BaseSphericalDensityProfile" = None,
        **kwargs,
    ):
        r"""Generate a spherical cluster model from gas density and entropy profiles.

        Parameters
        ----------
        density_profile : ~profiles.density.BaseSphericalDensityProfile
            Profile object representing the radial **gas** density.
        entropy_profile : ~profiles.entropy.BaseSphericalEntropyProfile
            Profile object representing the entropy of the ICM gas.
        filename : str or Path
            Output HDF5 file path.
        min_radius : ~unyt.array.unyt_quantity or str, optional
            Minimum radius for sampling (default: 1 kpc).
        max_radius : ~unyt.array.unyt_quantity or str, optional
            Maximum radius for sampling (default: 1 Mpc).
        num_points : int, optional
            Number of radial samples (default: 1000).
        overwrite : bool, optional
            Whether to overwrite existing file (default: False).
        stellar_density_profile : ~profiles.density.BaseSphericalDensityProfile, optional
            Optional stellar density profile. If provided, the stellar density will
            be included in the model. Otherwise, the stellar density is assumed to
            be zero and no stellar component is included.
        **kwargs:
            Additional keyword arguments for radial grid construction.

        Notes
        -----
        Given entropy :math:`K(r)` and gas density :math:`\rho_{\mathrm{gas}}(r)`, we use:

        1. **Entropy definition**:

        .. math::

            K(r) = \frac{T(r)}{n_e(r)^{2/3}}

         with

        .. math::

            n_e(r) = \frac{\rho_{\mathrm{gas}}(r)}{\mu_e m_p}

        2. **Invert to get temperature**:

        .. math::

            T(r) = K(r) \cdot \left(\frac{\rho_{\mathrm{gas}}(r)}{\mu_e m_p}\right)^{2/3}

        3. Compute pressure from ideal gas law and proceed as in the temperature + density pathway.

        """
        # --- Preambular Setup --- #
        # This section of the pathway just configures the
        # progress bar, the logger, and some other setup
        # data.
        cls.logger.info("Generating SphericalGalaxyClusterModel from entropy and density profiles.")
        with logging_redirect_tqdm(loggers=[cls.logger]):
            progress_bar = tqdm(
                total=9,
                desc="Prepare metadata, profiles, and fields...",
                disable=pisces_config["system.appearance.disable_progress_bars"],
                leave=False,
            )

            # --- Step 1: Initialize data --- #
            # This section initializes the fields, metadata, and profiles
            # for the model. It prepares the fields dictionary, metadata
            # dictionary, and profiles dictionary.
            fields: dict[str, unyt_array] = {}
            metadata: dict[str, str] = {}
            profiles: dict[str, BaseSphericalDensityProfile] = {
                "gas_density": density_profile,
                "entropy": entropy_profile,
            }
            if stellar_density_profile is not None:
                profiles["stellar_density"] = stellar_density_profile

            # --- Step 2: Construct radial grid --- #
            # This section constructs a radial grid for the model.
            progress_bar.update(1)
            progress_bar.set_description("Creating grid...")

            grid = cls._construct_radial_grid(min_radius, max_radius, num_points, spacing=kwargs.pop("spacing", "log"))
            radii = grid["r"]

            # --- Construct Radial Fields --- #
            # (STEP 3)
            # This section evaluates the gas density and entropy profiles
            # at the specified radii.
            progress_bar.update(1)
            progress_bar.set_description("Building profile fields...")

            fields["gas_density"] = density_profile(radii)
            fields["entropy"] = entropy_profile(radii)

            mue = compute_mean_molecular_weight_per_electron(cls.config["hydrogen_abundance"])
            mu = compute_mean_molecular_weight(cls.config["hydrogen_abundance"])
            fields["electron_density"] = fields["gas_density"] / (mue * mp)
            fields["electron_density"].convert_to_units("1/cm**3")

            # --- Compute Temperature --- #
            # (STEP 4)
            # This allows us to then complete the execution by
            # calling out to the same pipeline that was used for
            # the temperature and density approach.
            progress_bar.update(1)
            progress_bar.set_description("Computing Temperature...")

            fields["temperature"] = fields["entropy"] * fields["electron_density"] ** (2 / 3)
            fields["temperature"].convert_to_units("keV")

            # --- Step 5: Compute Fields --- #
            # These computations are shared with the entropy path
            # and are therefore abstracted out. This completes
            # the hydrostatic calculation and then computes the
            # masses and gravitational quantities.
            fields = cls._fields_from_temperature_and_density(
                fields,
                density_profile,
                radii,
                progress_bar,
                stellar_density_profile=stellar_density_profile,
            )

            # --- Compute Extras --- #
            # (STEP 9)
            # This section computes additional fields such as sound speed, entropy, and baryon fraction.
            # Compute sound speed.
            progress_bar.update(1)
            progress_bar.set_description("Performing additional computations...")

            gamma = 5 / 3
            fields["sound_speed"] = np.sqrt(gamma * fields["pressure"] / fields["gas_density"])
            fields["sound_speed"].convert_to_units("km/s")

            # Compute the baryon fraction.
            fields["baryon_fraction"] = (fields["gas_mass"] + fields["stellar_mass"]) / fields["total_mass"]

            # Compute the internal energy / unit mass
            fields["internal_energy_per_unit_mass"] = fields["temperature"] / (mp * mu * (2 / 3))
            fields["internal_energy_per_unit_mass"].convert_to_units("km**2/s**2")

            progress_bar.set_description("COMPLETE")
            progress_bar.update(1)
            progress_bar.close()

        return cls.from_components(
            filename,
            grid,
            fields,
            profiles,
            metadata=metadata,
            overwrite=overwrite,
        )

    # ========================================= #
    # Virialization Hooks                       #
    # ========================================= #
    # These are hooks for accessing the distribution function
    # for the model.

    def get_df(self, species: str) -> tuple[unyt_array, unyt_array]:
        r"""Retrieve the distribution function and energy grid for a given species.

        The distribution functions are expressed as function of the effective energy
        :math:`\mathcal{E}`. Thus,

        .. math::

            P(a<\mathcal{E}<b) = \int_a^b {\rm DF}(\mathcal{E}) d\mathcal{E}.

        Inside the underlying HDF5 file, distribution functions are stored in the
        ``"DISTRIBUTION_FUNCTIONS"``.

        Parameters
        ----------
        species : str
            The species name (e.g., "stellar", "gas", "dm"). This will look for
            keys named ``"E_<species>"`` and ``"DF_<species>"`` in the ``"DISTRIBUTION_FUNCTIONS"``
            group of the model HDF5 file.

        Returns
        -------
        Tuple of unyt_array
            A tuple ``(E, DF)`` containing the energy grid and corresponding distribution function
            as unyt arrays with units.

        Raises
        ------
        KeyError
            If the species' distribution function or energy grid is missing.

        """
        # Construct the keys and grab out the DF group.
        group = self.handle["DISTRIBUTION_FUNCTIONS"]
        E_key = f"E_{species}"
        DF_key = f"DF_{species}"

        # Check for existence.
        if E_key not in group or DF_key not in group:
            raise KeyError(
                f"Distribution function for species '{species}' not found. "
                f"Missing keys: "
                f"{E_key if E_key not in group else ''} "
                f"{DF_key if DF_key not in group else ''}"
            )

        # Extract data and report.
        E_data = unyt_array(group[E_key][...], units=group[E_key].attrs["units"])
        DF_data = unyt_array(group[DF_key][...], units=group[DF_key].attrs["units"])

        return E_data, DF_data

    def compute_df(
        self,
        species: Literal["stellar", "dark_matter"],
        num_points: int = 1000,
        scale: Literal["linear", "log"] = "linear",
        boundary_value: unyt_quantity = unyt_quantity(0, "km**2/s**2"),
        save: bool = True,
    ) -> tuple[unyt_array, unyt_array]:
        r"""Compute the isotropic Eddington distribution function for a given species.

        This method applies the Eddington inversion formula to recover the distribution
        function :math:`f(\mathcal{E})` from the density profile :math:`\rho(r)` and
        gravitational potential :math:`\Phi(r)` of the model, assuming spherical symmetry
        and velocity isotropy.

        Parameters
        ----------
        species : {"stellar", "dark_matter"}
            The species to compute the distribution function for. Must correspond to a field
            in the model named ``"<species>_density"``.

        num_points : int, optional
            Number of energy samples to evaluate (default is 1000).

        scale : {"linear", "log"}, optional
            Sampling strategy for the relative energy grid :math:`\mathcal{E}`.

            - "linear": Uniform spacing from 0 to :math:`\Psi_0`.
            - "log": Logarithmic spacing over a fixed dynamic range (default: "linear").

        boundary_value : ~unyt.array.unyt_quantity, optional
            The potential at the outer boundary :math:`\Phi_0`, used to define the
            relative potential :math:`\Psi = - (\Phi - \Phi_0)`. Default is 0.

        save : bool, optional
            If True (default), stores the computed distribution function and energy grid
            to the model HDF5 file in the ``"DISTRIBUTION_FUNCTIONS"`` group using keys
            ``"E_<species>"`` and ``"DF_<species>"``.

        Returns
        -------
        energy : unyt_array
            The relative energy grid :math:`\mathcal{E}`, with units of velocity squared.

        distribution : unyt_array
            The corresponding distribution function :math:`f(\mathcal{E})`, with units of
            :math:`[\rho][\mathcal{E}]^{-3/2}`.

        Raises
        ------
        ValueError
            If the model does not contain the required fields for the given species.

        """
        density_key = f"{species}_density"
        potential_key = "gravitational_potential"

        if density_key not in self.fields or potential_key not in self.fields:
            raise ValueError(
                f"Model must contain '{density_key}' and '{potential_key}' fields "
                f"to compute the DF for species '{species}'."
            )

        density = self.fields[density_key]
        potential = self.fields[potential_key]

        E, DF = compute_eddington_distribution(
            density=density,
            potential=potential,
            num_points=num_points,
            boundary_value=boundary_value,
            scale=scale,
        )

        if save:
            group = self.handle.require_group("DISTRIBUTION_FUNCTIONS")

            E_dset = group.create_dataset(f"E_{species}", data=E.value)
            DF_dset = group.create_dataset(f"DF_{species}", data=DF.value)

            E_dset.attrs["units"] = str(E.units)
            DF_dset.attrs["units"] = str(DF.units)

        return E, DF

    # ========================================= #
    # Properties                                #
    # ========================================= #
    @property
    def stellar_df(self) -> tuple[unyt_array, unyt_array]:
        r"""Return (E, DF) for the stellar component."""
        return self.get_df("stellar")

    @property
    def dark_matter_df(self) -> tuple[unyt_array, unyt_array]:
        r"""Return (E, DF) for the dark matter component."""
        return self.get_df("dark_matter")

    @property
    def has_stellar_df(self) -> bool:
        r"""Check whether the stellar DF is stored in the model file."""
        group = self.handle.get("DISTRIBUTION_FUNCTIONS", {})
        return "E_stellar" in group and "DF_stellar" in group

    @property
    def has_dark_matter_df(self) -> bool:
        r"""Check whether the dark matter DF is stored in the model file."""
        group = self.handle.get("DISTRIBUTION_FUNCTIONS", {})
        return "E_dark_matter" in group and "DF_dark_matter" in group


class MagnetizedSphericalGalaxyClusterModel(SphericalGalaxyClusterModel):
    r"""A spherically symmetric galaxy cluster model featuring a magnetic field.

    This model extends the basic spherical galaxy cluster model to include
    magnetic field profiles and their effects on the cluster dynamics. In particular,
    the magnetic field strength is expressed via the :math:`\beta` parameter defined
    such that

    .. math::

        \beta = \frac{P_{\rm thermal}}{P_{\rm magnetic}}.


    Fields
    ======

    The following physical fields are computed and stored in the model:

    .. csv-table:: Magnetized Spherical Galaxy Cluster Model Fields
       :file: ../../../docs/source/tables/model_fields/MagnetizedSphericalGalaxyClusterModel.csv
       :header-rows: 1
       :widths: 15, 40, 25, 25

    Methodology
    ===========

    .. note::

        In most respects, the methodology here mirrors that of :class:`SphericalGalaxyClusterModel`
        with the addition of magnetic field profiles. For details on the various available pathways,
        see the documentation for that class.

    For a profile with :math:`\beta(r)`, the effect is simply to offset the necessary thermal pressure. Thus,
    the hydrostatic equilibrium equation is modified to account for the magnetic pressure:

    .. math::

        - \rho \nabla \Phi = \nabla P = \nabla P_{\rm thermal} + \nabla P_{\rm magnetic} =
        \nabla \left[\left(1+\frac{1}{\beta}\right) P_{\rm thermal}\right]

    where :math:`P_{\rm magnetic}(r) = P_{\rm thermal}(r) / \beta(r)`. As such, the inclusion of the
    :math:`\beta` parameter allows for a full treatment of the non-thermal pressure. The corresponding
    magnetic field strength is given by

    .. math::

        B(r) = \sqrt{2 \beta(r) P_{\rm magnetic}(r)}.

    """

    # ========================================= #
    # Model Generator Methods                   #
    # ========================================= #
    # These methods are the generator methods for the model. They
    # should take some set of inputs (whatever is needed for the model)
    # and yield a resulting model.
    @classmethod
    def _fields_from_temperature_and_density(
        cls,
        fields: dict[str, unyt_array],
        density_profile: "BaseSphericalDensityProfile",
        radii: unyt_array,
        progress_bar,
        stellar_density_profile: Optional["BaseSphericalDensityProfile"] = None,
    ):
        """Compute core fields rom gas density and temperature fields.

        Parameters
        ----------
        fields : dict
            Dictionary of field arrays including at least 'gas_density' and 'temperature'.
        density_profile : BaseSphericalDensityProfile
            Analytic gas density profile used for mass calculations.
        stellar_density_profile : ~profiles.density.BaseSphericalDensityProfile, optional
            Optional stellar density profile. If provided, the stellar density will
            be included in the model. Otherwise, the stellar density is assumed to
            be zero and no stellar component is included.

        radii : unyt_array
            Radial grid points.
        progress_bar : tqdm.tqdm
            TQDM progress bar to update during processing.

        """
        # --- Compute Pressure --- #
        # (Step 4 [temp], Step 5 [entr])
        # Compute the pressure of the field based on the idea gas
        # law.
        XH = cls.config["hydrogen_abundance"]
        mu = compute_mean_molecular_weight(XH)
        fields["pressure"] = fields["gas_density"] * fields["temperature"] * mp * mu
        fields["pressure"].convert_to_units("Msun/(kpc*Myr**2)")

        # --- Compute Magnetic and Thermal Components --- #
        # (STEP 5 [temp], Step 6 [entr])
        # Compute the magnetic and total pressure from the
        # thermal pressure profile.
        progress_bar.set_description("Computing Pressure Components...")
        progress_bar.update(1)

        fields["pressure_magnetic"] = fields["pressure_thermal"] / fields["beta_parameter"]
        fields["pressure"] = fields["pressure_thermal"] + fields["pressure_magnetic"]

        # --- Solve HSE Equation --- #
        # (Step 6 [temp], Step 7 [entr])
        # Solve the HSE equation to get the gravitational field.
        progress_bar.set_description("Computing gravitational field...")
        progress_bar.update(1)

        fields["gravitational_field"] = cls._solve_hydrostatic_equilibrium(
            radii, density_profile, pressure_field=fields["pressure"]
        )

        # --- Compute Masses --- #
        # (Step 7 [temp], Step 8 [entr])
        # Solve the HSE equation to get the gravitational field.
        progress_bar.set_description("Computing Mass Profiles...")
        progress_bar.update(1)

        fields["gas_mass"] = density_profile.compute_enclosed_mass(radii, units="Msun")
        if stellar_density_profile is not None:
            fields["stellar_mass"] = stellar_density_profile.compute_enclosed_mass(radii, units="Msun")
        else:
            fields["stellar_mass"] = unyt_array(np.zeros_like(radii), units="Msun")

        fields["total_mass"] = radii**2 * fields["gravitational_field"] / G
        fields["dark_matter_mass"] = fields["total_mass"] - fields["gas_mass"] - fields["stellar_mass"]

        # --- Densities & Potential --- #
        # (Step 8 [temp], Step 9 [entr])
        # Finally take derivatives to compute the density profiles
        # and calculate the potentials.
        progress_bar.set_description("Computing potential...")
        progress_bar.update(1)

        for key in ("total", "dark_matter"):
            spline = InterpolatedUnivariateSpline(radii.d, fields[f"{key}_mass"].d)
            dMdr = unyt_array(spline(radii.d, 1), units="Msun/kpc")
            fields[f"{key}_density"] = dMdr / (4.0 * np.pi * radii**2)

        fields["gravitational_potential"] = cls._compute_potential(radii, fields["total_density"], fields["total_mass"])

        return fields

    @classmethod
    def from_density_and_total_density(
        cls,
        density_profile: "BaseSphericalDensityProfile",
        total_density_profile: "BaseSphericalDensityProfile",
        filename: str | Path,
        min_radius: unyt_quantity | str = unyt_quantity(1, "kpc"),
        max_radius: unyt_quantity | str = unyt_quantity(1, "Mpc"),
        num_points: int = 1000,
        overwrite: bool = False,
        stellar_density_profile: "BaseSphericalDensityProfile" = None,
        beta_profile: BaseProfile | Callable | float = None,
        **kwargs,
    ):
        r"""Generate a spherical cluster model from analytic density profiles.

        Parameters
        ----------
        density_profile : ~profiles.density.BaseSphericalDensityProfile
            Profile object representing the radial **gas** density.
        total_density_profile : ~profiles.density.BaseSphericalDensityProfile
            Profile object representing the **total** density.
        filename : str or Path
            The file at which to save the model. If the file already
            exists, then it will raise an error unless ``overwrite=True``.

            This should be an HDF5 file with a ``.h5`` or ``.hdf5`` extension.

        min_radius : ~unyt.array.unyt_quantity or str, optional
            Minimum radius for sampling (default: 1 kpc).
        max_radius : ~unyt.array.unyt_quantity or str, optional
            Maximum radius for sampling (default: 1 Mpc).
        num_points : int, optional
            Number of radial samples (default: 1000).
        overwrite : bool, optional
            Whether to overwrite existing file (default: False).
        stellar_density_profile : ~profiles.density.BaseSphericalDensityProfile, optional
            Optional stellar density profile. If provided, the stellar density will
            be included in the model. Otherwise, the stellar density is assumed to
            be zero and no stellar component is included.
        beta_profile : ~profiles.density.BaseSphericalDensityProfile or float or callable, optional
            The magnetic pressure fraction :math:`\beta`. This is the ratio between the
            thermal pressure and the magnetic pressure. By default, it is set to :math:`\infty`,
            corresponding to an unmagnetized cluster. For larger values of :math:`\beta`, the cluster
            will become less magnetized.
        **kwargs :
            Additional keyword arguments passed to the radial grid construction.

        Notes
        -----
        Given gas density :math:`\rho_{\mathrm{gas}}(r)`, total density :math:`\rho_{\mathrm{tot}}(r)`,
        and an optional stellar density :math:`\rho_{\star}(r)`, we perform the following steps:

        1. **Compute Dark Matter Density**:

           Requiring that the total density is the sum of gas, stellar, and dark matter densities:

           .. math::

             \rho_{\mathrm{DM}}(r) = \rho_{\mathrm{tot}}(r) - \rho_{\mathrm{gas}}(r) - \rho_{\star}(r)

        1. **Compute mass profiles** via:

           .. math::

             M_i(<r) = 4\pi \int_0^r \rho_i(r') \, r'^2 \, dr'

        for each component :math:`i \in \{\mathrm{gas}, \mathrm{tot}, {\rm dm},  \mathrm{stellar}\}`.

        2. **Compute the gravitational field**:

           .. math::

             \nabla \Phi(r) = \frac{G M_{\mathrm{tot}}(<r)}{r^2}

        3. Use hydrostatic equilibrium to **integrate for pressure**:

           .. math::

             \nabla P = - \rho_{\mathrm{gas}}(r) \nabla \Phi(r)

        4. **Solve for temperature** using the ideal gas law:

           .. math::

             T(r) = \frac{P(r)}{n(r)} = \frac{P(r)}{\rho_{\mathrm{gas}}(r)} \cdot \mu m_p

        """
        from unyt.physical_constants import mu_0

        # --- Preambular Setup --- #
        # This section of the pathway just configures the
        # progress bar, the logger, and some other setup
        # data.
        cls.logger.info("Generating MagnetizedSphericalGalaxyClusterModel from density and total density profiles.")

        with logging_redirect_tqdm(loggers=[cls.logger]):
            progress_bar = tqdm(
                total=8,
                desc="Preparing metadata...",
                disable=pisces_config["system.appearance.disable_progress_bars"],
                leave=False,
            )

            # --- Prepare Metadata, Profiles, and Fields --- #
            # (STEP 1)
            # This section prepares the metadata, profiles, and fields
            # for the model. It initializes the fields dictionary,
            # metadata dictionary, and profiles dictionary.
            # NOTE: we don't keep beta because its not a standard profile and
            # users might want to specify it as a generic function.
            fields: dict[str, unyt_array] = {}
            metadata: dict[str, str] = {}
            profiles: dict[str, BaseProfile] = {
                "gas_density": density_profile,
                "total_density": total_density_profile,
            }
            if stellar_density_profile is not None:
                profiles["stellar_density"] = stellar_density_profile

            # --- Create Radial Grid (Logarithmic Spacing) --- #
            # (STEP 2)
            # This section creates a radial grid for the model.
            progress_bar.update(1)
            progress_bar.set_description("Generating grid...")

            grid = cls._construct_radial_grid(min_radius, max_radius, num_points, spacing=kwargs.pop("spacing", "log"))
            radii = grid["r"]

            # --- Create Density Fields --- #
            # (STEP 3)
            # This section computes the gas and total density profiles
            # at the specified radii, and also computes the stellar density
            # if provided.
            progress_bar.update(1)
            progress_bar.set_description("Building density...")

            fields["gas_density"] = density_profile(radii)
            fields["total_density"] = total_density_profile(radii)
            if stellar_density_profile is not None:
                fields["stellar_density"] = stellar_density_profile(radii)
            else:
                fields["stellar_density"] = unyt_array(np.zeros_like(radii), units="Msun/kpc**3")

            # --- Add the Beta Profile --- #
            # (STEP 4)
            # Add the beta profile as a field based on what we got
            # from the user. By default, we use inf to have no magnetic
            # field. We only add it as a profile if we get a true profile.
            progress_bar.update(1)
            progress_bar.set_description("Adding Beta Profile...")

            if beta_profile is None:
                fields["beta_parameter"] = np.full_like(radii.d, np.inf)
            elif isinstance(beta_profile, (float, int)):
                fields["beta_parameter"] = np.full_like(radii.d, beta_profile)
            elif isinstance(beta_profile, BaseProfile):
                fields["beta_parameter"] = beta_profile(radii)

                # The beta profile was actually a proper profile, so we'll add
                # it to the profiles dictionary.
                profiles["beta_profile"] = beta_profile

            elif callable(beta_profile):
                fields["beta_parameter"] = beta_profile(radii.to_value("kpc"))
            else:
                raise ValueError("`beta_parameter` must be a callable, profile, None, or a number.")

            # --- Computing Mass Fields --- #
            # (STEP 5)
            # This section computes the enclosed mass profiles for gas,
            # total, and stellar components. It also computes the dark matter
            # mass as the difference between total and gas + stellar masses.
            progress_bar.update(1)
            progress_bar.set_description("Computing masses...")

            fields["gas_mass"] = density_profile.compute_enclosed_mass(radii, units="Msun")
            fields["total_mass"] = total_density_profile.compute_enclosed_mass(radii, units="Msun")
            if stellar_density_profile is not None:
                fields["stellar_mass"] = stellar_density_profile.compute_enclosed_mass(radii, units="Msun")
            else:
                fields["stellar_mass"] = unyt_array(np.zeros_like(radii), units="Msun")

            fields["dark_matter_mass"] = fields["total_mass"] - fields["gas_mass"] - fields["stellar_mass"]

            # --- Compute Gravitational Field & Potential --- #
            # (STEP 6)
            # This section computes the gravitational field and potential
            # using the total mass profile. It uses Newton's law of gravitation
            # to compute the gravitational field and potential at each radius.
            progress_bar.update(1)
            progress_bar.set_description("Calculating potential...")

            fields["gravitational_field"] = G * fields["total_mass"] / (grid["r"] ** 2)
            fields["gravitational_potential"] = cls._compute_potential(
                radii, fields["total_density"], fields["total_mass"]
            )

            # --- Compute Temperature and Pressure --- #
            # (STEP 7)
            # To compute the pressure, we have - dP = rho dPhi, but
            # we then need to integrate to infinity to get the full pressure profile.
            # This time, because we have a magnetic field, we need to reduce the full pressure
            # into its components.
            progress_bar.update(1)
            progress_bar.set_description("Calculating temperature...")

            fields["pressure"] = cls._solve_hydrostatic_equilibrium(
                radii,
                density_profile,
                gravitational_field=fields["gravitational_field"],
            )
            fields["pressure_thermal"] = fields["pressure"] / (1 + 1 / fields["beta_parameter"])
            fields["pressure_magnetic"] = fields["pressure"] - fields["pressure_thermal"]

            XH = cls.config["hydrogen_abundance"]
            mu = compute_mean_molecular_weight(XH)
            fields["temperature"] = fields["pressure_thermal"] * mp * mu / fields["gas_density"]
            fields["temperature"].convert_to_units("keV")

            # --- Compute Extras --- #
            # (STEP 8)
            # This section computes additional fields such as sound speed, entropy, and baryon fraction.
            # Compute sound speed.
            progress_bar.update(1)
            progress_bar.set_description("Performing additional computations...")

            # Compute the sound speed.
            gamma = 5 / 3
            fields["sound_speed"] = np.sqrt(gamma * fields["pressure_thermal"] / fields["gas_density"])
            fields["sound_speed"].convert_to_units("km/s")

            # Compute the magnetic field strength.
            fields["magnetic_field"] = np.sqrt(2 * mu_0 * fields["pressure_magnetic"]).to("tesla").to("gauss")

            # Compute the Alfven Speed
            fields["alfven_velocity"] = np.sqrt(2 * fields["pressure_magnetic"] / fields["gas_density"]).to("km/s")

            # Compute the baryon fraction.
            fields["baryon_fraction"] = (fields["gas_mass"] + fields["stellar_mass"]) / fields["total_mass"]

            # Compute the electron number density.
            mue = compute_mean_molecular_weight_per_electron(XH)
            fields["electron_density"] = fields["gas_density"] / (mue * mp)
            fields["electron_density"].convert_to_units("1/cm**3")

            # Compute the entropy.
            fields["entropy"] = fields["temperature"] / fields["electron_density"] ** (2 / 3)
            fields["entropy"].convert_to_units("keV*cm**2")

            # Complete and close the bar.
            progress_bar.set_description("COMPLETE")
            progress_bar.update(1)
            progress_bar.close()

        # Return.
        return cls.from_components(filename, grid, fields, profiles, metadata, overwrite=overwrite)

    @classmethod
    def from_temperature_and_density(
        cls,
        density_profile: "BaseSphericalDensityProfile",
        temperature_profile: "BaseSphericalTemperatureProfile",
        filename: str | Path,
        min_radius: unyt_quantity | str = unyt_quantity(1, "kpc"),
        max_radius: unyt_quantity | str = unyt_quantity(1, "Mpc"),
        num_points: int = 1000,
        overwrite: bool = False,
        stellar_density_profile: "BaseSphericalDensityProfile" = None,
        beta_profile: BaseProfile | Callable | float = None,
        **kwargs,
    ):
        r"""Generate a spherical galaxy cluster model from gas density and temperature profiles.

        Parameters
        ----------
        density_profile : ~profiles.density.BaseSphericalDensityProfile
            Profile object representing the radial **gas** density.
        temperature_profile : ~profiles.temperature.BaseSphericalTemperatureProfile
            Profile object representing the radial **gas** temperature.
        filename : str or Path
            Output HDF5 file path.
        min_radius : ~unyt.array.unyt_quantity or str, optional
            Minimum radius for sampling (default: 1 kpc).
        max_radius : ~unyt.array.unyt_quantity or str, optional
            Maximum radius for sampling (default: 1 Mpc).
        num_points : int, optional
            Number of radial samples (default: 1000).
        overwrite : bool, optional
            Whether to overwrite existing file (default: False).
        stellar_density_profile : ~profiles.density.BaseSphericalDensityProfile, optional
            Optional stellar density profile. If provided, the stellar density will
            be included in the model. Otherwise, the stellar density is assumed to
            be zero and no stellar component is included.
        beta_profile : ~profiles.density.BaseSphericalDensityProfile or float or callable, optional
            The magnetic pressure fraction :math:`\beta`. This is the ratio between the
            thermal pressure and the magnetic pressure. By default, it is set to :math:`\infty`,
            corresponding to an unmagnetized cluster. For larger values of :math:`\beta`, the cluster
            will become less magnetized.
        **kwargs:
            Additional keyword arguments passed to the radial grid construction.

        Notes
        -----
        Given :math:`T(r)` and :math:`\rho_{\mathrm{gas}}(r)`, we compute:

        1. **Pressure** from the ideal gas law:

        .. math::

            P(r) = \rho_{\mathrm{gas}}(r) \cdot T(r) / (\mu m_p)

        2. **Gravitational field** via hydrostatic equilibrium:

        .. math::

            g(r) = -\frac{1}{\rho_{\mathrm{gas}}(r)} \cdot \frac{dP}{dr}

        3. **Total mass profile**:

        .. math::

            M_{\mathrm{tot}}(<r) = \frac{r^2 \cdot g(r)}{G}

        4. Derive :math:`\rho_{\mathrm{tot}}(r)` via:

        .. math::

            \rho_{\mathrm{tot}}(r) = \frac{1}{4\pi r^2} \frac{dM_{\mathrm{tot}}}{dr}

        """
        from unyt.physical_constants import mu_0

        # --- Preambular Setup --- #
        # This section of the pathway just configures the
        # progress bar, the logger, and some other setup
        # data.
        cls.logger.info("Generating SphericalGalaxyClusterModel from temperature and density profiles.")

        with logging_redirect_tqdm(loggers=[cls.logger]):
            progress_bar = tqdm(
                total=8,
                desc="Prepare metadata, profiles, and fields...",
                disable=pisces_config["system.appearance.disable_progress_bars"],
                leave=False,
            )

            # --- Step 1: Initialize data --- #
            # This section initializes the fields, metadata, and profiles
            # for the model. It prepares the fields dictionary, metadata
            # dictionary, and profiles dictionary.
            fields: dict[str, unyt_array] = {}
            metadata: dict[str, str] = {}
            profiles: dict[str, BaseSphericalDensityProfile] = {
                "gas_density": density_profile,
                "temperature": temperature_profile,
            }
            if stellar_density_profile is not None:
                profiles["stellar_density"] = stellar_density_profile

            # --- Step 2: Construct radial grid --- #
            # This section constructs a radial grid for the model.
            progress_bar.update(1)
            progress_bar.set_description("Creating grid...")

            grid = cls._construct_radial_grid(min_radius, max_radius, num_points, spacing=kwargs.pop("spacing", "log"))
            radii = grid["r"]

            # --- Step 3: Evaluate profiles --- #
            # This section evaluates the gas density and temperature profiles
            # at the specified radii. It also computes the pressure profile
            # using the ideal gas law: P = rho * kT / (mu * m_p), where mu is the
            # mean molecular weight and m_p is the proton mass.
            progress_bar.update(1)
            progress_bar.set_description("Building profile fields...")

            fields["gas_density"] = density_profile(radii)
            fields["temperature"] = temperature_profile(radii)

            # Add the beta profile as a field based on what we got
            # from the user.
            if beta_profile is None:
                fields["beta_parameter"] = np.full_like(radii.d, np.inf)
            elif isinstance(beta_profile, (float, int)):
                fields["beta_parameter"] = np.full_like(radii.d, beta_profile)
            elif isinstance(beta_profile, BaseProfile):
                fields["beta_parameter"] = beta_profile(radii)

                # The beta profile was actually a proper profile, so we'll add
                # it to the profiles dictionary.
                profiles["beta_profile"] = beta_profile

            elif callable(beta_profile):
                fields["beta_parameter"] = beta_profile(radii.to_value("kpc"))
            else:
                raise ValueError("`beta_parameter` must be a callable, profile, None, or a number.")

            fields = cls._fields_from_temperature_and_density(
                fields,
                density_profile,
                radii,
                progress_bar,
                stellar_density_profile=stellar_density_profile,
            )

            # --- Compute Extras --- #
            # This section computes additional fields such as sound speed, entropy, and baryon fraction.
            # Compute sound speed.
            progress_bar.set_description("Performing additional computations...")
            gamma = 5 / 3
            fields["sound_speed"] = np.sqrt(gamma * fields["pressure_thermal"] / fields["gas_density"])
            fields["sound_speed"].convert_to_units("km/s")

            # Compute the magnetic field strength.
            fields["magnetic_field"] = np.sqrt(2 * mu_0 * fields["pressure_magnetic"]).to("tesla").to("gauss")

            # Compute the Alfven Speed
            fields["alfven_velocity"] = np.sqrt(2 * fields["pressure_magnetic"] / fields["gas_density"]).to("km/s")

            # Compute the baryon fraction.
            fields["baryon_fraction"] = (fields["gas_mass"] + fields["stellar_mass"]) / fields["total_mass"]

            # Compute the electron number density.
            mue = compute_mean_molecular_weight_per_electron(cls.config["hydrogen_abundance"])
            fields["electron_density"] = fields["gas_density"] / (mue * mp)
            fields["electron_density"].convert_to_units("1/cm**3")

            # Compute the entropy.
            fields["entropy"] = fields["temperature"] / fields["electron_density"] ** (2 / 3)
            fields["entropy"].convert_to_units("keV*cm**2")

            progress_bar.set_description("COMPLETE")
            progress_bar.update(1)
            progress_bar.close()

        return cls.from_components(
            filename,
            grid,
            fields,
            profiles,
            metadata=metadata,
            overwrite=overwrite,
        )

    @classmethod
    def from_entropy_and_density(
        cls,
        density_profile: "BaseSphericalDensityProfile",
        entropy_profile: "BaseSphericalEntropyProfile",
        filename: str | Path,
        min_radius: unyt_quantity | str = unyt_quantity(1, "kpc"),
        max_radius: unyt_quantity | str = unyt_quantity(1, "Mpc"),
        num_points: int = 1000,
        overwrite: bool = False,
        stellar_density_profile: "BaseSphericalDensityProfile" = None,
        beta_profile: BaseProfile | Callable | float = None,
        **kwargs,
    ):
        r"""Generate a spherical cluster model from gas density and entropy profiles.

        Parameters
        ----------
        density_profile : ~profiles.density.BaseSphericalDensityProfile
            Profile object representing the radial **gas** density.
        entropy_profile : ~profiles.entropy.BaseSphericalEntropyProfile
            Profile object representing the entropy of the ICM gas.
        filename : str or Path
            Output HDF5 file path.
        min_radius : ~unyt.array.unyt_quantity or str, optional
            Minimum radius for sampling (default: 1 kpc).
        max_radius : ~unyt.array.unyt_quantity or str, optional
            Maximum radius for sampling (default: 1 Mpc).
        num_points : int, optional
            Number of radial samples (default: 1000).
        overwrite : bool, optional
            Whether to overwrite existing file (default: False).
        stellar_density_profile : ~profiles.density.BaseSphericalDensityProfile, optional
            Optional stellar density profile. If provided, the stellar density will
            be included in the model. Otherwise, the stellar density is assumed to
            be zero and no stellar component is included.
        beta_profile : ~profiles.density.BaseSphericalDensityProfile or float or callable, optional
            The magnetic pressure fraction :math:`\beta`. This is the ratio between the
            thermal pressure and the magnetic pressure. By default, it is set to :math:`\infty`,
            corresponding to an unmagnetized cluster. For larger values of :math:`\beta`, the cluster
            will become less magnetized.
        **kwargs : dict, optional
            Additional keyword arguments for radial grid construction, such as `spacing`.

        Notes
        -----
        Given entropy :math:`K(r)` and gas density :math:`\rho_{\mathrm{gas}}(r)`, we use:

        1. **Entropy definition**:

        .. math::

            K(r) = \frac{T(r)}{n_e(r)^{2/3}}

         with

        .. math::

            n_e(r) = \frac{\rho_{\mathrm{gas}}(r)}{\mu_e m_p}

        2. **Invert to get temperature**:

        .. math::

            T(r) = K(r) \cdot \left(\frac{\rho_{\mathrm{gas}}(r)}{\mu_e m_p}\right)^{2/3}

        3. Compute pressure from ideal gas law and proceed as in the temperature + density pathway.

        """
        from unyt.physical_constants import mu_0

        # --- Preambular Setup --- #
        # This section of the pathway just configures the
        # progress bar, the logger, and some other setup
        # data.
        cls.logger.info("Generating SphericalGalaxyClusterModel from entropy and density profiles.")
        with logging_redirect_tqdm(loggers=[cls.logger]):
            progress_bar = tqdm(
                total=9,
                desc="Prepare metadata, profiles, and fields...",
                disable=pisces_config["system.appearance.disable_progress_bars"],
                leave=False,
            )

            # --- Step 1: Initialize data --- #
            # This section initializes the fields, metadata, and profiles
            # for the model. It prepares the fields dictionary, metadata
            # dictionary, and profiles dictionary.
            fields: dict[str, unyt_array] = {}
            metadata: dict[str, str] = {}
            profiles: dict[str, BaseSphericalDensityProfile] = {
                "gas_density": density_profile,
                "entropy": entropy_profile,
            }
            if stellar_density_profile is not None:
                profiles["stellar_density"] = stellar_density_profile
            progress_bar.update(1)
            progress_bar.set_description("Creating grid...")

            # --- Step 2: Construct radial grid --- #
            # This section constructs a radial grid for the model.
            grid = cls._construct_radial_grid(min_radius, max_radius, num_points, spacing=kwargs.pop("spacing", "log"))
            radii = grid["r"]
            progress_bar.update(1)
            progress_bar.set_description("Building profile fields...")

            # --- Construct Radial Fields --- #
            # This section evaluates the gas density and entropy profiles
            # at the specified radii.
            fields["gas_density"] = density_profile(radii)
            fields["entropy"] = entropy_profile(radii)

            progress_bar.update(1)
            progress_bar.set_description("Computing temperature...")

            mue = compute_mean_molecular_weight_per_electron(cls.config["hydrogen_abundance"])
            fields["electron_density"] = fields["gas_density"] / (mue * mp)
            fields["electron_density"].convert_to_units("1/cm**3")

            # Compute the entropy.
            fields["temperature"] = fields["entropy"] * fields["electron_density"] ** (2 / 3)
            fields["temperature"].convert_to_units("keV")

            # Add the beta profile as a field based on what we got
            # from the user.
            progress_bar.update(1)
            progress_bar.set_description("Adding beta parameter...")

            if beta_profile is None:
                fields["beta_parameter"] = np.full_like(radii.d, np.inf)
            elif isinstance(beta_profile, (float, int)):
                fields["beta_parameter"] = np.full_like(radii.d, beta_profile)
            elif isinstance(beta_profile, BaseProfile):
                fields["beta_parameter"] = beta_profile(radii)

                # The beta profile was actually a proper profile, so we'll add
                # it to the profiles dictionary.
                profiles["beta_profile"] = beta_profile

            elif callable(beta_profile):
                fields["beta_parameter"] = beta_profile(radii.to_value("kpc"))
            else:
                raise ValueError("`beta_parameter` must be a callable, profile, None, or a number.")

            # --- Step 4: Compute Fields --- #
            # These computations are shared with the entropy path
            # and are therefore abstracted out. This completes
            # the hydrostatic calculation and then computes the
            # masses and gravitational quantities.
            progress_bar.update(1)
            progress_bar.set_description("Computing Pressure...")
            fields = cls._fields_from_temperature_and_density(
                fields,
                density_profile,
                radii,
                progress_bar,
                stellar_density_profile=stellar_density_profile,
            )

            # --- Compute Extras --- #
            # This section computes additional fields such as sound speed, entropy, and baryon fraction.
            # Compute sound speed.
            progress_bar.set_description("Performing additional computations...")
            gamma = 5 / 3
            fields["sound_speed"] = np.sqrt(gamma * fields["pressure_thermal"] / fields["gas_density"])
            fields["sound_speed"].convert_to_units("km/s")

            # Compute the magnetic field strength.
            fields["magnetic_field"] = np.sqrt(2 * mu_0 * fields["pressure_magnetic"]).to("tesla").to("gauss")

            # Compute the Alfven Speed
            fields["alfven_velocity"] = np.sqrt(2 * fields["pressure_magnetic"] / fields["gas_density"]).to("km/s")

            # Compute the baryon fraction.
            fields["baryon_fraction"] = (fields["gas_mass"] + fields["stellar_mass"]) / fields["total_mass"]

            progress_bar.set_description("COMPLETE")
            progress_bar.update(1)
            progress_bar.close()

        return cls.from_components(
            filename,
            grid,
            fields,
            profiles,
            metadata=metadata,
            overwrite=overwrite,
        )
