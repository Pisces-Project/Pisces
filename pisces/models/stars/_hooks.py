"""Hook implementations for stellar models."""

from pathlib import Path
from typing import Self

import numpy as np
import unyt
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from pisces.models.core.hooks import SphericalParticleGenerationHook
from pisces.particles.base import ParticleDataset
from pisces.utilities import pisces_config

# ========================================= #
# Particle Generation Hooks                 #
# ========================================= #
# These hooks are used to convert models into particle datasets.


class PolytropicParticleGenerationHook(SphericalParticleGenerationHook):
    """Hook for generating particles from a polytropic stellar model.

    This mixin class provides the necessary methods to convert a polytropic stellar model
    into a particle dataset. It handles the sampling of particle positions, and interpolation
    of model fields onto the particles.

    This hook is specific to :class:`~pisces.models.stars.polytropes.PolytropicStarModel`.
    """

    __PolytropicParticleGenerationHook_HOOK_ENABLED__ = True
    __PolytropicParticleGenerationHook_IS_TEMPLATE__ = False

    # ----------------------------------- #
    # Generator Settings                  #
    # ----------------------------------- #
    _PolytropicParticleGenerationHook_PTYPES = ("gas",)
    _PolytropicParticleGenerationHook_CDF_FIELDS = {"gas": "mass"}
    _PolytropicParticleGenerationHook_INTERPOLATED_FIELDS = {
        "gas": {
            "density": "density",
            "gravitational_potential": "potential",
            "gravitational_field": "gravitational_field",
            "pressure": "pressure",
            "temperature": "temperature",
        },
    }

    # ----------------------------------- #
    # Generator Methods                   #
    # ----------------------------------- #
    # This section of the hook should be used to
    # encapsulate the logic for generating the particle dataset.
    def _PolytropicParticleGenerationHook_generate_particle_species(
        self: Self,
        particle_dataset: "ParticleDataset",
        particle_type: str,
        num_particles: int,
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

        Raises
        ------
        ValueError
            If `particle_type` is invalid (i.e., not listed in `_SGCParticleGenerationHook_PTYPES`).

        """
        # Introductory logging statement.
        self.logger.info("Generating %d %s particles...", num_particles, particle_type)

        # Validate the particle type and count. Then generate
        # an empty group in the particle dataset for this particle type.
        if particle_type not in self._PolytropicParticleGenerationHook_PTYPES:
            raise ValueError(
                f"Invalid particle type: {particle_type}. Must be "
                f"one of the _PolytropicParticleGenerationHook_PTYPES: {self._PolytropicParticleGenerationHook_PTYPES}."
            )
        if num_particles <= 0:
            return

        particle_dataset.add_particle_type(particle_type, num_particles)

        # --- Sample Particles --- #
        # The first step in the particle generation process is to generate
        # the particle positions and the radii. This is done via inverse
        # transform sampling and is encapsulated in the `_SGCParticleGenerationHook_sample_particle_radii` method.
        radii, positions = self._SphericalParticleGenerationHook_sample_particle_radii(
            self._PolytropicParticleGenerationHook_CDF_FIELDS[particle_type],
            num_particles,
        )
        particle_dataset.add_particle_field(particle_type, "radius", radii)
        particle_dataset.add_particle_field(particle_type, "particle_position", positions)

        # --- Set the Particle Masses --- #
        # The next step is to set the particle masses. We do this by
        # assigning equal mass to each particle of a given type.
        particle_mass = (
            self.fields[self._PolytropicParticleGenerationHook_CDF_FIELDS[particle_type]].d[-1] / num_particles
        )
        particle_mass = unyt.unyt_array(
            np.full(num_particles, particle_mass),
            self.fields[self._PolytropicParticleGenerationHook_CDF_FIELDS[particle_type]].units,
        )
        particle_dataset.add_particle_field(particle_type, "particle_mass", particle_mass)

        # --- Interpolate Fields --- #
        # The next step is to interpolate any relevant fields onto the particle dataset.
        # To do this, we use a linear interpolator.
        interpolated_fields = self._PolytropicParticleGenerationHook_INTERPOLATED_FIELDS[particle_type]
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

    def generate_particles(
        self: Self,
        filename: str | Path,
        num_particles: dict[str, int],
        overwrite: bool = False,
    ) -> "ParticleDataset":
        """Convert this galaxy cluster model into a particle dataset.

        This method creates a particle representation of the model by sampling positions, and
        interpolating model fields onto particles. The output is written to disk in the form
        of a :class:`particles.base.ParticleDataset`.

        Parameters
        ----------
        filename : str or ~pathlib.Path
            Filesystem path where the output particle dataset should be saved.
            If the path already exists, it will be overwritten if `overwrite=True`.
        num_particles : dict of str, int
            The number of each type of particle to generate. The dictionary may contain any
            of the following keys: ``['gas',]`` and values should correspond
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
        The particle generation process follows two main steps:

        1. **Sampling**: Radial positions are sampled from the species-specific mass profile
           (interpreted as a CDF), and converted into 3D Cartesian coordinates. This ensures that
           particles are distributed according to the model's density. Masses are assigned to each
           particle species so that each species has equal masses across all particles.

        2. **Interpolation**: Once the particle positions are established, the values of various model
           fields can be assigned to each particle via interpolation.

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
                self._PolytropicParticleGenerationHook_generate_particle_species(
                    particle_dataset,
                    particle_type,
                    particle_count,
                )

        return particle_dataset
