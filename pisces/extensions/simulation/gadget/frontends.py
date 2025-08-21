"""
Frontends for Gadget-2 simulation code.

This module provides the :class`Gadget2Frontend` class, which is responsible for
managing initial conditions and generating input files for the Gadget-2 simulation code.

"""

from pathlib import Path

import h5py
import numpy as np
import unyt
from extensions.simulation import InitialConditions

from pisces.extensions.simulation.core.frontends import SimulationFrontend

# Fetch a reference to this directory so that we can
# setup configuration paths.
__local_path__ = Path(__file__).parents[0]


class Gadget2Frontend(SimulationFrontend):
    """
    Simulation frontend for the Gadget-2 N-body/SPH code.

    This class translates Pisces :class:`~pisces.extensions.simulation.core.initial_conditions.InitialConditions`
    objects into Gadget-2–compatible HDF5 initial condition (IC) files. It manages
    configuration, validation, preprocessing, and file generation to ensure that
    particle data and metadata are correctly written in Gadget-2’s expected format.

    """

    # --------------------------------------- #
    # Class Variables and Constants           #
    # --------------------------------------- #
    # These are class-level variables which define connections to
    # the configuration along with some other aspects of the
    # frontend's behavior.
    __default_configuration_path__: Path = __local_path__ / "gadget_2_default_config.yaml"
    """The path to the default configuration file.
    """

    # --------------------------------------- #
    # Initialization and Configuration        #
    # --------------------------------------- #
    def _validate_input_ic(self, initial_conditions: InitialConditions) -> bool:
        # We don't need to do any special validation for the Gadget-2 ICs since
        # there shouldn't be any special requirements beyond the base class.
        #
        # We will need to validate before runtime to ensure that the user
        # has generated particle files for each model.
        self.logger.debug(f"{initial_conditions} passed validation on {self.__class__.__name__} frontend...")
        return super()._validate_input_ic(initial_conditions)

    def _setup_config(self):
        """
        Configure the simulation configuration for the Gadget-2 frontend.

        For Gadget-2, we need to take the ``model_DEFAULT`` entry from the configuration
        and copy it through for each model in the ``model`` section so that the users
        can modify the field and particle type names present in the particle data.
        """
        # Fetch the default model configuration from the yaml.
        default_model_config = dict(self.config["model_DEFAULT"])

        # For each model, we need to copy this into the configuration at
        # the right position.
        self.config["models"] = {}
        for model_name in self.initial_conditions.list_models():
            self.logger.debug(f"Added {model_name} to the Gadget-2 configuration models...")
            self.config["models"][model_name] = dict(default_model_config.copy())

    def __post_init__(self):
        """
        Finalize the Gadget-2 frontend after initialization.

        Aside from the ``_setup_config`` step, we don't need to do anything
        special during post-initialization for the Gadget-2 frontend.
        """
        pass

    # --------------------------------------- #
    # IC GENERATION METHODS                   #
    # --------------------------------------- #
    def _validate_runtime_configuration(self, *args, **kwargs):
        """
        Validate the runtime configuration for the Gadget-2 frontend.

        For the Gadget-2 frontend, we need to ensure a couple of things. The first
        is that each model appears in the configuration file correctly (providing
        us with mappings to fields and particle types). The second is that each
        model has an existing particle file available in the initial conditions.

        If either of these checks fails, we alert the user and raise an error.
        In general, these should occur because the user just hasn't correctly followed
        the structure of the configuration file or hasn't generated the
        initial conditions correctly.

        Parameters
        ----------
        *args, **kwargs
            Any arguments passed through from `generate_initial_conditions`.

        Returns
        -------
        None
        """
        # Start by ensuring that we have particles for each model in the
        # initial conditions.
        for model in self.initial_conditions.list_models():
            if not self.initial_conditions.has_particles(model):
                raise RuntimeError(
                    f"Model `{model}` is missing particles in the ICs!\n"
                    "HINT: Ensure that you have generated particles for each model before calling the"
                    " Gadget-2 frontend.\n"
                    "HINT: You can use `add_particles_to_model` on the IC object to generate particles."
                )

        # All of the models have particle files. We now just ensure that each
        # model appears in the configuration list before allowing things
        # to proceed. We DON'T check for the full structure (too much logical complexity),
        # and instead let errors arise naturally if the user misconfigures things.
        for model in self.initial_conditions.list_models():
            if model not in self.config["models"]:
                raise RuntimeError(
                    f"Model `{model}` is missing from the Gadget-2 frontend configuration!\n"
                    "HINT: Ensure that you have added an entry for each model in the `models` "
                    "section of the"
                    " configuration file.\n"
                    "HINT: You can use the `model_DEFAULT` section to set default"
                    " values for each model."
                )

        # Everything looks good! We'll log this and return.
        return None

    def _preprocess_particle_dataset(self):
        # Begin by fetching the simulation boxsize and then convert that
        # into a bounding box we can use to cut the particles as needed.
        try:
            simulation_boxsize = self.config["parameters.boxsize"] * self.config["parameters.units.length"]
        except Exception as exp:
            raise RuntimeError(
                "Failed to load the boxsize (parameters.boxsize) from the Gadget-2 configuration!"
            ) from exp

        simulation_bbox = np.stack([[0, simulation_boxsize] for _ in range(3)], axis=1)
        simulation_bbox = unyt.unyt_array(simulation_bbox, simulation_boxsize.units)

        # We now cycle through the models and fetch their particle datasets
        # so that we can begin the reductions.
        for model_name in self.initial_conditions.list_models():
            # Extract the particle dataset and the relevant orientation, spin, offset,
            # etc. for this model.
            particle_dataset = self.initial_conditions.load_particles(model_name)
            model_position, model_velocity, model_orientation, model_spin = (
                self.initial_conditions.config[f"models.{model_name}.position"],
                self.initial_conditions.config[f"models.{model_name}.velocity"],
                self.initial_conditions.config[f"models.{model_name}.orientation"],
                self.initial_conditions.config[f"models.{model_name}.spin"],
            )

            self.logger.debug(
                f"Model `{model_name}` parameters:\n"
                f"\tPosition: {model_position}\n"
                f"\tVelocity: {model_velocity}\n"
                f"\tOrientation: {model_orientation}\n"
                f"\tSpin: {model_spin}\n"
            )

            # --- Reorient model --- #
            # The objective here is to realign the particle dataset to the
            # correct orientation. We do this by computing the relevant
            # rotation matrices. All of this is handled at the particle dataset
            # level.
            #
            # We ONLY REORIENT the position and velocity since there is no support
            # for magnetic fields or other vector fields in the Gadget-2 frontend.
            particle_dataset.reorient_particles(
                model_orientation,
                spin=model_spin,
            )

            # We now perform the offsets of the COM of the system
            # by the relevant velocity and position.
            particle_dataset.offset_particle_positions(model_position)
            particle_dataset.offset_particle_velocities(model_velocity)

            # --- Cut to bounding box --- #
            # With the particles reoriented and offset, we can now
            # cut them down to the bounding box of the simulation.
            particle_dataset.cut_particles_to_bbox(simulation_bbox)

        # We now quickly log to the user that we have completed
        # the model preprocessing step. This includes catching the
        # new particle counts for each model and logging them.
        _pcount_log_statements = "\n".join(
            [
                f"\tModel `{mname}` has {self.initial_conditions.get_particle_count(mname)} particles"
                f" after preprocessing."
                for mname in self.initial_conditions.list_models()
            ]
        )
        self.logger.info(f"{self.__class__.__name__}:\t Preprocessing... [DONE]\n" + _pcount_log_statements)

    def _count_particles(self):
        """Count particles of each Gadget-2 type across all models.

        Returns
        -------
        np.ndarray
            1D array of length = number of Gadget-2 particle types,
            dtype is uint64.
        """
        # Choose correct integer dtype for Gadget-2
        dtype = np.uint64

        # Number of Gadget-2 particle types (typically 6)
        n_types = 6
        final_count = np.zeros(n_types, dtype=dtype)

        # Iterate over models defined in initial conditions
        for model_name in self.initial_conditions.list_models():
            model_config = self.config[f"models.{model_name}"]
            particle_counts = self.initial_conditions.get_particle_count(model_name)

            # Map model's native particle types into Gadget-2 ordering
            for i, (_gadget_type, cfg) in enumerate(model_config.items()):
                native_type = cfg["name"]
                count = particle_counts.get(native_type, 0)
                final_count[i] += count

        return final_count

    def _construct_header_particle_specifiers(self):
        """Construct the three Gadget-2 particle count arrays for the header.

        This generates:
          - ``Npart``  (per-file counts, 6-element array)
          - ``Nall``   (global totals, 6-element array, low 32 bits)
          - ``NallHW`` (high word of 64-bit totals, 6-element array)

        Returns
        -------
        tuple of np.ndarray
            (Npart, Nall, NallHW), each a 1D array of length 6,
            dtype = uint32.
        """
        # Global particle totals (64-bit safe in Python)
        global_counts = self._count_particles()

        # Initialize arrays
        Npart = np.zeros(6, dtype=np.uint32)
        Nall = np.zeros(6, dtype=np.uint32)
        NallHW = np.zeros(6, dtype=np.uint32)

        # Split into low/high 32-bit words
        for i, total in enumerate(global_counts):
            low = int(int(total) & 0xFFFFFFFF)  # lower 32 bits
            high = int(int(total) >> 32)  # upper 32 bits

            Npart[i] = low  # this file contains the same number as global (single-file case)
            Nall[i] = low  # store lower 32 bits of global total
            NallHW[i] = high  # store upper 32 bits (zero if <2^32)

        return Npart, Nall, NallHW

    def _write_header_attributes(self, header_group: h5py.Group):
        # Start with the trivial attributes which are constant and / or
        # zero in the IC file setup. These are the majority of the header
        # fields.
        header_group.attrs["Time"] = np.array(0.0, dtype=np.double)
        header_group.attrs["Redshift"] = np.array(0.0, dtype=np.double)
        header_group.attrs["Flag_Sfr"] = np.array(0, dtype=np.int32)
        header_group.attrs["Flag_Feedback"] = np.array(0, dtype=np.int32)
        header_group.attrs["Flag_Cooling"] = np.array(0, dtype=np.int32)
        header_group.attrs["Flag_Metals"] = np.array(0, dtype=np.int32)
        header_group.attrs["Flag_StellarAge"] = np.array(0, dtype=np.int32)
        header_group.attrs["Omega0"] = np.array(0.0, dtype=np.double)
        header_group.attrs["OmegaLambda"] = np.array(0.0, dtype=np.double)
        header_group.attrs["HubbleParam"] = np.array(0.0, dtype=np.double)
        header_group.attrs["NumFileserSnapshot"] = np.array(1, dtype=np.int32)
        header_group.attrs["Flag_Entropy_ICs"] = np.array(0, dtype=np.int32)
        header_group.attrs["MassTable"] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.double)
        # Now we need to handle particle counts and we need to handle
        # the boxsize. The easier of the two is the boxsize.
        boxsize = self.config["parameters.boxsize"]
        header_group.attrs["BoxSize"] = np.array(boxsize, dtype=np.double)

        # We now need to handle the particle counts. This is a bit more
        # involved since we need to map from the native particle types
        # in the initial conditions to the Gadget-2 particle types.
        Npart, Nall, NallHW = self._construct_header_particle_specifiers()
        header_group.attrs["NumPart_ThisFile"] = np.asarray(Npart, dtype=np.uint32)
        header_group.attrs["NumPart_Total"] = np.asarray(Nall, dtype=np.uint32)
        header_group.attrs["NumPart_Total_HW"] = np.asarray(NallHW, dtype=np.uint32)

    def _generate_initial_conditions(self, filename: str, overwrite: bool = False, *args, **kwargs):
        """
        Generate the initial condition files for the target simulation code.

        This is the core implementation method that *must* be overridden
        by subclasses. It should read from `self.config` and
        `self.initial_conditions` and write any necessary files in the format
        expected by the simulation code.

        Parameters
        ----------
        filename: str
            The name of the output file to write the Gadget-2 initial conditions to.
        overwrite: bool
            Whether to overwrite existing files. If `False` and the file
            already exists, an error should be raised.

        """
        # Begin by handling the file checking and overwrite language
        # before proceeding to the actual generation element of the method.
        path = self.ic_directory / filename
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output file {path} already exists and overwrite is set to False!")
        elif path.exists() and overwrite:
            self.logger.debug(f"Overwriting existing Gadget-2 IC file at {path}...")
            path.unlink()
        else:
            pass

        # We can now pre-process our particle datasets so that
        # they are contained within the specified bounding box and
        # so that they have the correct offsets, rotations, and velocities.
        self._preprocess_particle_dataset()

        # Now, with pre-processing complete, we can start managing
        # the structure of the file system for the file.

        # Generate the the file that we're going to be using.
        with h5py.File(path, "w") as ic_file:
            self.logger.debug(f"Generating Gadget-2 IC file at {path}...")

            # Generate the various top level elements (groups) that are going
            # to be necessary for the structure.
            _header_group = ic_file.create_group("Header")
            _part_groups = {
                "Type0": ic_file.create_group("Type0"),  # Gas
                "Type1": ic_file.create_group("Type1"),  # Dark Matter
                "Type2": ic_file.create_group("Type2"),  # Disk Stars
                "Type3": ic_file.create_group("Type3"),  # Bulge Stars
                "Type4": ic_file.create_group("Type4"),  # Stars
                "Type5": ic_file.create_group("Type5"),  # Boundary Particles
            }

            # Now we're going to write the header group's attributes
            # in self._write_header_attributes.
            self.logger.debug("Writing Gadget-2 header attributes...")
            self._write_header_attributes(_header_group)
            self.logger.debug("Finished writing Gadget-2 header attributes.")

            # Setup up a unit system for the Gadget-2 ICs so that we can
            # arbitrarily convert from the units in the initial conditions
            _time_unit = (1 * self.config["parameters.units.length"]) / (1 * self.config["parameters.units.velocity"])
            _time_unit = _time_unit.to("Myr")

            _gadget_unit_system = unyt.unit_systems.UnitSystem(
                "gadget_simulation_system",
                length_unit=self.config["parameters.units.length"],
                mass_unit=self.config["parameters.units.mass"],
                time_unit=unyt.Unit(_time_unit).simplify(),
            )
            self.logger.debug(
                f"Using Gadget-2 unit system:\n"
                f"\tLength Unit: {_gadget_unit_system['length']}\n"
                f"\tMass Unit: {_gadget_unit_system['mass']}\n"
                f"\tTime Unit: {_gadget_unit_system['time']}\n"
                f"\tVelocity Unit: {_gadget_unit_system['velocity']}\n"
            )

            # ========================================== #
            # Write particle data for each Gadget-2 type #
            # ========================================== #
            # This is the most logically complex part of the IC generation
            # process. We proceed as follows:
            #
            # - Cycle through each model in the initial conditions.
            #   - Cycle through each gadget-2 particle type (0-5).
            #     - Check that the model has particles of this type. If not,
            #       skip to the next type.
            #     - If it does, fetch the relevant fields and
            #     - Iterate through each field, converting to the correct units
            #       and writing to the correct dataset in the file.

            # We need to keep track of the particle IDs that we are attaching
            # to the particles. As such, we create the _pidx array of particle
            # indices for each particle type and the total_particles counter.
            particle_offsets = np.zeros(6, dtype=np.uint64)
            total_particles = self._count_particles()

            # --- Cycle through each model -- #

            for model_name in self.initial_conditions.list_models():
                # Extract relevant metadata.
                model_config = self.config[f"models.{model_name}"]

                # Load the particle dataset for this model.
                particle_dataset = self.initial_conditions.load_particles(model_name)

                for gptype in range(6):
                    # Extract relevant metadata.
                    gpkey = f"Type{gptype}"
                    ipkey = model_config[gpkey]["name"]

                    # Check if the model even includes this particle type.
                    if ipkey not in particle_dataset.particle_groups:
                        self.logger.debug(
                            f"Model `{model_name}` does not have particles of type `{ipkey}`!"
                            f" Skipping Gadget-2 type `{gpkey}`."
                        )
                        continue

                    for gfield, ifield in model_config[gpkey]["fields"].items():
                        # Skip the ID column because we are going to handle
                        # that ourselves at the very end once all the
                        # particles have been written.
                        if gfield == "ID":
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

                        # Ensure that the dataset we're writing into exists. This
                        # will require us keeping track of the shape.
                        if gfield in ("Coordinates", "Velocities"):
                            dset_shape = (total_particles[gptype], 3)
                            dset_dtype = np.float32
                        else:
                            dset_shape = (total_particles[gptype],)
                            dset_dtype = np.float32

                        hdf5_dataset = _part_groups[gpkey].require_dataset(gfield, shape=dset_shape, dtype=dset_dtype)

                        # Now, with the dataset handle available and the particles
                        # loaded, we need to transfer. We'll load the particle data into
                        # memory and coerce the units to the Gadget-2 unit system.
                        _i_data_array = particle_dataset.get_particle_field(ipkey, ifield)
                        _i_data_array = _i_data_array.in_base(_gadget_unit_system).d

                        # Now we write the array data into the dataset by virtue of
                        # the tracked slicing.
                        _slc = slice(
                            int(particle_offsets[gptype]), int(particle_offsets[gptype] + _i_data_array.shape[0])
                        )
                        hdf5_dataset[_slc] = _i_data_array

                    # After writing all the fields, we need to increment the particle offsets
                    particle_offsets[gptype] += particle_dataset.num_particles[ipkey]
                    self.logger.info(
                        f"[{self.__class__.__name__}]: Model `{model_name}` wrote"
                        f" {particle_dataset.num_particles[ipkey]} particles of type"
                        f" `{ipkey}` to Gadget-2 type `{gpkey}`."
                    )
