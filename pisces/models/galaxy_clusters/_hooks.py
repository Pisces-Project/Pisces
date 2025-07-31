"""Hook implementations for Galaxy Cluster models."""

from pathlib import Path
from typing import Self

import numpy as np
import unyt
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from pisces.models.core.hooks import SphericalParticleGenerationHook
from pisces.particles.base import ParticleDataset
from pisces.physics.virialization.eddington import sample_eddington_velocities
from pisces.utilities import pisces_config

# ========================================= #
# Particle Generation Hooks                 #
# ========================================= #
# These hooks are written to support particle generation in
# spherically symmetric galaxy cluster models. They implement the logic
# required to convert a spherical galaxy cluster model into a particle-based
# dataset suitable for use in simulation or analysis. The generated particles
# are distributed according to the radial mass profiles defined by the model
# (gas, dark matter, and stellar components). Key physical quantities from the
# model are linearly interpolated onto the particles, and velocity sampling is
# performed for collisionless species using Eddington inversion.


class SGCParticleGenerationHook(SphericalParticleGenerationHook):
    """Particle generation hook for spherical galaxy cluster models.

    .. important::

        This hook class is specific to the :class:`~models.galaxy_clusters.SphericalGalaxyClusterModel` class
        and should not be associated with other model classes except in cases where the logic is mirrored identically.

    This hook implements the logic required to convert a spherical galaxy cluster
    model into a particle-based dataset suitable for use in simulation or analysis.

    The generated particles are distributed according to the radial mass profiles
    defined by the model (gas, dark matter, and stellar components). Key physical
    quantities from the model are linearly interpolated onto the particles, and
    velocity sampling is performed for collisionless species using Eddington inversion.

    Supported particle types include:

        - 'gas' (thermal particles with interpolated thermodynamic fields)
        - 'dark_matter' (collisionless particles with velocity sampling)
        - 'stars' (collisionless particles with velocity sampling)

    The particle generation process includes the following steps for each species:

        1. **Sampling Positions**:
           Radial coordinates are sampled from the cumulative mass profile via
           inverse transform sampling. Particles are distributed isotropically
           over the sphere to obtain 3D positions.

        2. **Interpolating Fields**:
           Model fields such as density, temperature, pressure, and gravitational
           potential are interpolated onto the particle positions using 1D linear
           interpolation along the radius.

        3. **Sampling Velocities** (for collisionless components):
           If the species is 'dark_matter' or 'stars', particle velocities are drawn
           from the distribution function obtained via Eddington inversion.

    Users can control the number of particles generated for each species via
    the `num_particles` dictionary passed to :meth:`generate_particles`. The output is
    a :class:`~particles.base.ParticleDataset` object saved to disk at the specified path.
    """

    __SGCParticleGenerationHook_HOOK_ENABLED__ = True
    __SGCParticleGenerationHook_IS_TEMPLATE__ = False

    # ----------------------------------- #
    # Generator Settings                  #
    # ----------------------------------- #
    _SGCParticleGenerationHook_PTYPES = ("dark_matter", "gas", "stars")
    """tuple of str: Allowed particle species this hook can generate.

    These names correspond to particle groups in the generated dataset.
    A species is only included if specified in the `num_particles` argument
    to `generate_particles()`.
    """

    _SGCParticleGenerationHook_CDF_FIELDS = {
        "gas": "gas_mass",
        "dark_matter": "dark_matter_mass",
        "stars": "stellar_mass",
    }
    """dict: Mapping of particle types to cumulative mass fields used for sampling.

    Each entry maps a particle species to the name of the field in the model that
    defines its cumulative radial mass profile. This profile serves as a CDF
    for inverse transform sampling of particle radii.
    """

    _SGCParticleGenerationHook_INTERPOLATED_FIELDS = {
        "gas": {
            "density": "gas_density",
            "gravitational_potential": "gravitational_potential",
            "gravitational_field": "gravitational_field",
            "pressure": "pressure",
            "kT": "temperature",
            "entropy": "entropy",
            "sound_speed": "sound_speed",
        },
        "dark_matter": {
            "density": "dark_matter_density",
            "gravitational_potential": "gravitational_potential",
            "gravitational_field": "gravitational_field",
        },
        "stars": {
            "density": "stellar_density",
            "gravitational_potential": "gravitational_potential",
            "gravitational_field": "gravitational_field",
        },
    }
    """dict: Fields to interpolate from the model onto each particle.

    For each particle type, this maps target field names (in the particle dataset)
    to source field names (in the model). These fields are linearly interpolated
    onto particle radii after sampling.
    """

    # ----------------------------------- #
    # Generator Methods                   #
    # ----------------------------------- #
    # This section of the hook should be used to
    # encapsulate the logic for generating the particle dataset.
    def _SGCParticleGenerationHook_generate_velocities(
        self: Self,
        particle_dataset: "ParticleDataset",
        particle_type: str,
        df_kwargs: dict = None,
    ):
        """Generate velocities for particles of a given type using Eddington inversion.

        Parameters
        ----------
        particle_dataset : ParticleDataset
            The dataset into which the velocity field will be written.
        particle_type : str
            The species of particle (must be 'dark_matter' or 'stars').
        df_kwargs : dict, optional
            Extra keyword arguments for DF generation if needed.

        """
        if df_kwargs is None:
            df_kwargs = {}

        if particle_type not in {"dark_matter", "stars"}:
            return

        df_type = "dark_matter" if particle_type == "dark_matter" else "stellar"

        # Try loading the distribution function; fallback to computing it
        try:
            rel_energy, dist_function = self.get_df(df_type)
        except Exception:
            self.logger.warning("DF for '%s' not found, attempting to compute it...", df_type)
            try:
                rel_energy, dist_function = self.compute_df(df_type, **df_kwargs)
            except Exception as err:
                raise RuntimeError(f"Failed to compute DF for '{df_type}': {err}") from err

        # Compute relative potential = -ϕ
        rel_potential = -particle_dataset.get_particle_field(particle_type, "gravitational_potential")

        # Sample velocities using the Eddington DF
        velocities = sample_eddington_velocities(rel_energy, dist_function, rel_potential).T

        particle_dataset.add_particle_field(particle_type, "particle_velocity", velocities)

    def _SGCParticleGenerationHook_generate_particle_species(
        self: Self,
        particle_dataset: "ParticleDataset",
        particle_type: str,
        num_particles: int,
        df_kwargs: dict = None,
    ):
        """Generate particles for a single species and populate the particle dataset.

        This method performs the full particle generation workflow for a given species:
        sampling radial positions, computing 3D coordinates, interpolating model fields,
        and (for collisionless components) generating velocities via Eddington inversion.

        Parameters
        ----------
        particle_dataset : ParticleDataset
            The dataset to which the particle data will be written.

        particle_type : str
            The name of the species to generate particles for. Must be one of the
            allowed types defined in `_SGCParticleGenerationHook_PTYPES` (e.g., "gas", "dark_matter", "stars").

        num_particles : int
            Number of particles to generate for the given species. If set to zero or
            a negative value, no particles are created.

        df_kwargs : dict, optional
            Optional keyword arguments to pass to `compute_df()` if a distribution
            function must be generated for velocity sampling (only used for
            "dark_matter" and "stars").

        Raises
        ------
        ValueError
            If `particle_type` is invalid (i.e., not listed in `_SGCParticleGenerationHook_PTYPES`).

        """
        # Introductory logging statement.
        self.logger.info("Generating %d %s particles...", num_particles, particle_type)

        # Validate the particle type and count. Then generate
        # an empty group in the particle dataset for this particle type.
        if particle_type not in self._SGCParticleGenerationHook_PTYPES:
            raise ValueError(
                f"Invalid particle type: {particle_type}. Must be "
                f"one of the _SGCParticleGenerationHook_PTYPES: {self._SGCParticleGenerationHook_PTYPES}."
            )
        if num_particles <= 0:
            return

        particle_dataset.add_particle_type(particle_type, num_particles)

        # --- Sample Particles --- #
        # The first step in the particle generation process is to generate
        # the particle positions and the radii. This is done via inverse
        # transform sampling and is encapsulated in the `_SGCParticleGenerationHook_sample_particle_radii` method.
        radii, positions = self._SphericalParticleGenerationHook_sample_particle_radii(
            "radii",
            self._SGCParticleGenerationHook_CDF_FIELDS[particle_type],
            num_particles,
        )
        particle_dataset.add_particle_field(particle_type, "radius", radii)
        particle_dataset.add_particle_field(particle_type, "particle_position", positions)

        # --- Set the Particle Masses --- #
        # The next step is to set the particle masses. We do this by
        # assigning equal mass to each particle of a given type.
        particle_mass = self.fields[self._SGCParticleGenerationHook_CDF_FIELDS[particle_type]].d[-1] / num_particles
        particle_mass = unyt.unyt_array(
            np.full(num_particles, particle_mass),
            self.fields[self._SGCParticleGenerationHook_CDF_FIELDS[particle_type]].units,
        )
        particle_dataset.add_particle_field(particle_type, "particle_mass", particle_mass)

        # --- Interpolate Fields --- #
        # The next step is to interpolate any relevant fields onto the particle dataset.
        # To do this, we use a linear interpolator.
        interpolated_fields = self._SGCParticleGenerationHook_INTERPOLATED_FIELDS[particle_type]
        for interpolated_particle_field, interpolated_model_field in tqdm(
            interpolated_fields.items(),
            desc=f"Interpolating {particle_type} fields",
            disable=pisces_config["system.appearance.disable_progress_bars"],
            unit="fields",
            leave=False,
        ):
            self._SphericalParticleGenerationHook_interpolate_particle_field(
                particle_dataset, "radii", particle_type, interpolated_particle_field, interpolated_model_field
            )

        # --- Generate Velocities --- #
        self._SGCParticleGenerationHook_generate_velocities(particle_dataset, particle_type, df_kwargs=df_kwargs)

    def generate_particles(
        self: Self,
        filename: str | Path,
        num_particles: dict[str, int],
        overwrite: bool = False,
    ) -> "ParticleDataset":
        """Convert this galaxy cluster model into a particle dataset.

        This method creates a particle representation of the model by sampling positions,
        interpolating model fields onto particles, and (if appropriate) assigning velocities
        via Eddington inversion. The output is written to disk in the form
        of a :class:`particles.base.ParticleDataset`.

        This is a common step in preparing a model for simulations as most simulation codes
        require that the collisionless components of the model be represented as particles.

        Parameters
        ----------
        filename : str or ~pathlib.Path
            Filesystem path where the output particle dataset should be saved.
            If the path already exists, it will be overwritten if `overwrite=True`.
        num_particles : dict of str, int
            The number of each type of particle to generate. The dictionary may contain any
            of the following keys: ``['gas','dark_matter','stars']`` and values should correspond
            to the number of particles to be generated of each species.

            If a species is provided but is not one of the allowed particle types, then
            an error will be raised.
        overwrite : bool, optional
            Whether to overwrite an existing file at `path`. Default is False.

        Returns
        -------
        ~particles.base.ParticleDataset
            The newly created and populated particle dataset. For each particle type,
            the particle positions, velocities, and masses will be included along with any
            other relevant fields that were interpolated from the model.

        Notes
        -----
        The particle generation process follows three main steps:

        1. **Sampling**: Radial positions are sampled from the species-specific mass profile
           (interpreted as a CDF), and converted into 3D Cartesian coordinates. This ensures that
           particles are distributed according to the model's density. Masses are assigned to each
           particle species so that each species has equal masses across all particles.

        2. **Interpolation**: Once the particle positions are established, the values of various model
           fields can be assigned to each particle via interpolation.

        3. **Velocity Assignment**: Velocity sampling is the most complex element of this procedure. For
           the collisionless species (dark matter and stars), the velocities are sampled from the the
           distribution function (the solution of the collisionless Boltzmann equation) using the Eddington
           inversion method.

           This ensures that the generated particles are correctly virialized.

           For the gas particles, velocities are set uniformly to zero indicating that they have (by default)
           no bulk motion and that their entire dispersive motion is thermal in nature.

        Examples
        --------
        To generate particles from a model, simple call this method:

        .. code-block:: python

            from pisces.models.galaxy_clusters import (
                SphericalGalaxyClusterModel,
            )

            # Load an existing model from disk.
            model = SphericalGalaxyClusterModel(
                "path/to/model.hdf5"
            )

            # Generate the particles for a specific number of each type.
            num_particles = {
                "gas": 100000,
                "dark_matter": 50000,
                "stars": 20000,
            }
            particle_dataset = model.generate_particles(
                path="path/to/particles.hdf5",
                num_particles=num_particles,
                overwrite=True,
            )


        """
        # Begin by generating the blank particle dataset into
        # which we will be writing the particle data. This will get passed
        # through to the various sub-methods to allow for logically clear
        # separation of concerns.
        particle_dataset = ParticleDataset.build_particle_dataset(filename, overwrite=overwrite)

        # For each of the particle types being generated, we'll follow
        # the sample basic process: create the particle group, sample positions, then
        # interpolate.
        # We delegate particle velocities until the end of the process to ensure that
        # we have good logical separation (since that's the hard part).
        with logging_redirect_tqdm(loggers=[self.logger]):
            for particle_type, particle_count in tqdm(
                num_particles.items(),
                desc="Generating particles",
                disable=pisces_config["system.appearance.disable_progress_bars"],
                unit="species",
            ):
                # Generate the particle species.
                self._SGCParticleGenerationHook_generate_particle_species(
                    particle_dataset,
                    particle_type,
                    particle_count,
                )

        return particle_dataset
