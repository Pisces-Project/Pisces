"""
Frontends for Gadget simulations codes.

This module provides frontend classes for working with Gadget-4 simulation codes.
These frontends facilitate the generation of initial conditions and particle datasets
compatible with Gadget-4, leveraging the underlying infrastructure provided by
the Pisces framework.
"""

from pathlib import Path

import numpy as np

from pisces.particles.gadget import Gadget4ParticleDataset

from ..core import GadgetLikeFrontend, InitialConditions3DCartesian

# Construct the local path to this file for looking up
# the default configuration file(s).
file_path = Path(__file__).absolute().parent


# --------------------------------------- #
# FRONTEND CLASSES                        #
# --------------------------------------- #
class Gadget4Frontend(GadgetLikeFrontend):
    """
    Frontend for generating initial conditions compatible with Gadget-4.

    This class provides frontend access for generating initial conditions which can
    be fed directly to Gadget-4 simulations. Detailed notes on the usage
    of this frontend and the relevant settings / configuration options found
    in Gadget-4 can be found on the associated documentation page: :ref:`simulations_gadget`.
    """

    # --------------------------------------- #
    # Class Variables and Constants           #
    # --------------------------------------- #
    # These overwrite the baseclass implementations to customize
    # for Gadget-4.
    __default_configuration_path__: Path = file_path / "gadget_4_default_config.yaml"
    __particle_dataset_type__ = Gadget4ParticleDataset
    __frontend_name__ = "Gadget4Frontend"
    _simulation_types_map: dict[type, str] = {InitialConditions3DCartesian: "3D"}
    _allowed_ic_types = (InitialConditions3DCartesian,)

    # --------------------------------------- #
    # IC Generation Methods                   #
    # --------------------------------------- #
    # Most of the detail here is handled by the superclass GadgetLikeFrontend.
    # We only need to implement the method that generates the particle dataset as this
    # will determine some details about the file header and other relevant structures.
    def _generate_particle_dataset(self, *args, **kwargs):
        """
        Generate the particle dataset for the entire simulation.

        Notes
        -----
        For Gadget-like frontends, this method constructs the complete
        particle dataset by iterating over each model in the initial conditions,
        mapping their native particle types and fields to the Gadget-4
        equivalents, and writing the data into the appropriate locations
        in the overall particle dataset.

        It will need to be reimplemented in subclasses if the particle dataset
        construction requires different parameters or additional steps.
        """
        # Parse out the arguments that are relevant to the generation of
        # the particle dataset.
        path, *args = args
        # Begin by creating the IC particles dataset skeleton so that we can
        # populate it with data from each of the models.
        #
        # THIS MAY NEED MODIFICATION in subclasses to handle new / different
        # inputs to the particle dataset builder.
        # noinspection PyArgumentList
        ic_particles = self.__class__.__particle_dataset_type__.build_particle_dataset(
            path,  # The path we generate the particles at.
            self._count_particles(),  # The number of particles.
            self.config["parameters.boxsize"],  # The box size.
            ntypes=self.config["makefile.number_of_particle_types"],  # Number of particle types.
            unit_system=self.unit_system,  # The unit system to use.
            double_precision=True,  # Use double precision for positions / velocities.
            id_precision=self.config["makefile.nbits_id"],  # The particle ID precision.
        )

        # With the skeleton written, we now cycle through each of the models,
        # extract their particle datasets, cast the names to the right things, and
        # then proceed to write the data into the file.
        _particle_count_offsets = np.zeros(self.config["makefile.number_of_particle_types"], dtype=np.uint64)
        for model_name in self.initial_conditions.list_models():
            # Extract the model's configuration data from the frontend
            # configuration file and load the particle dataset that we
            # are going to be using.
            model_config = self.config[f"models.{model_name}"]
            particle_dataset = self.initial_conditions.load_particles(model_name)

            # We iterate through each of the initial conditions' particle
            # types and map them to Gadget-4 types.
            for gptype in range(self.config["makefile.number_of_particle_types"]):
                # Extract relevant metadata.
                gpkey = f"ParticleType{gptype}"
                ipkey = model_config[gpkey]["name"]

                # Check if the model even includes this particle type.
                if ipkey not in particle_dataset.particle_groups:
                    self.logger.debug(
                        f"Model `{model_name}` does not have particles of type `{ipkey}`!"
                        f" Skipping {self.__frontend_name__} type `{gpkey}`."
                    )
                    continue

                for gfield, ifield in model_config[gpkey]["fields"].items():
                    # Skip the ID column because we are going to handle
                    # that ourselves at the very end once all the
                    # particles have been written.
                    if gfield == "ParticleIDs":
                        continue

                    # Extract the particle array from the particle file
                    # for this model.
                    if f"{ipkey}.{ifield}" not in particle_dataset:
                        raise RuntimeError(
                            f"Model `{model_name}` is missing required field"
                            f" `{ifield}` for particle type `{ipkey}`!\n"
                            f"HINT: If it exists, is it named something different? Modify the"
                            f" configuration file to match the field name in the particle dataset.\n"
                            f"HINT: If it doesn't exist, you may need to derive it manually."
                        )

                    # Access the dataset in the Gadget-4 particle dataset and
                    # dump our data into it.
                    field_handle = ic_particles.get_particle_field_handle(gpkey, gfield)
                    field_unit = ic_particles.get_field_units(gpkey, gfield)

                    # Now we write the array data into the dataset by virtue of
                    # the tracked slicing.
                    _slc = slice(
                        int(_particle_count_offsets[gptype]),
                        int(_particle_count_offsets[gptype] + particle_dataset.num_particles[ipkey]),
                    )
                    field_handle[_slc, ...] = particle_dataset.get_particle_field(ipkey, ifield).to_value(field_unit)

                # After writing all the fields, we need to increment the particle offsets
                _particle_count_offsets[gptype] += particle_dataset.num_particles[ipkey]
                self.logger.info(
                    f"[{self.__class__.__name__}]: Model `{model_name}` wrote"
                    f" {particle_dataset.num_particles[ipkey]} particles of type"
                    f" `{ipkey}` to {self.__frontend_name__} type `{gpkey}`."
                )

        return ic_particles

    def generate_initial_conditions(self, filename: str, *args, overwrite: bool = False, **kwargs):
        """
        Generate Gadget-4–compatible initial condition files for this frontend.

        This method wraps the base
        :meth:`~pisces.extensions.simulation.core.frontends.GadgetLikeFrontend.generate_initial_conditions`
        lifecycle, providing the standard Gadget-4 interface for writing fully formatted
        HDF5 initial condition files.

        It performs a complete validation and generation sequence:

        1. **Runtime validation** — Ensures that all models have generated particles
           and that the current configuration is consistent with the Gadget-4
           compile-time and runtime settings.
        2. **Initial condition generation** — Constructs and writes the Gadget-4
           particle dataset, including all particle types, field mappings, and metadata.

        Parameters
        ----------
        filename : str
            Name of the output HDF5 file to generate within the initial conditions
            directory (e.g., ``"ClusterICs.hdf5"``). The path is resolved automatically
            relative to :attr:`ic_directory`.
        overwrite : bool, default=False
            If ``True``, any existing file with the same name will be replaced.
            If ``False``, a :class:`FileExistsError` is raised when the file exists.

        Notes
        -----
        - This is the preferred entry point for creating Gadget-4 initial conditions.
          It provides automatic logging, file overwrite handling, and lifecycle consistency.
        - Subclasses should not override this method unless a different output mechanism
          is required.
        - All generated files conform to the standard Gadget-4 ``ICFormat=3`` HDF5 layout.
        """
        super().generate_initial_conditions(filename, *args, overwrite=overwrite, **kwargs)
