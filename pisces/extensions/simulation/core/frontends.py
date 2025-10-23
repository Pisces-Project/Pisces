"""
Abstract base classes for simulation frontends in Pisces.

A simulation frontend is the bridge between Pisces' in-house
:class:`~pisces.extensions.simulation.core.initial_conditions.InitialConditions`
objects and an external simulation code's native input files and configuration
requirements.

"""

import shutil
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import unyt

from pisces.particles import ParticleDataset
from pisces.utilities.config import ConfigManager

from .initial_conditions import InitialConditions, InitialConditions3DCartesian


class SimulationFrontend(ABC):
    """
    Abstract base class defining the behavior of simulation frontends.

    This class forms the core logic for all simulation frontends in Pisces,
    and includes a number of abstract methods and modifiable hooks to alter
    various aspects of the simulation code's initial conditions generation.

    The :class:`SimulationFrontend` class takes a
    :class:`~pisces.extensions.simulation.core.initial_conditions.InitialConditions`
    object as input, which it will then convert into simulation code-specific
    files and configurations for running the simulation.

    To facilitate this, the frontend class manages a configuration file
    in which the user may specify the details of the conversion process.
    Each frontend has a default configuration file in ``../frontend_configs``,
    which is copied into the initial conditions directory when the frontend
    is initialized. This allows the user to modify the configuration
    for each simulation run without affecting the default template.

    For more detailed information on how to implement a custom frontend,
    see :ref:`frontend_dev`. For the user guide documentation, see :ref:`simulation_frontends`.
    """

    # --------------------------------------- #
    # Class Variables and Constants           #
    # --------------------------------------- #
    # These are class-level variables which define connections to
    # the configuration along with some other aspects of the
    # frontend's behavior.
    __default_configuration_path__: Path = None
    """Path to the default configuration file template for this frontend.

    This should be set in subclasses to point to the default YAML
    configuration file that defines the expected parameters for
    this frontend. The file should be located in the
    `pisces/extensions/simulation/core/frontend_configs` directory.
    """
    __default_configuration_manager_class__: type = ConfigManager
    """The class used to manage the configuration file for this frontend.

    This should be set in subclasses to specify the type of
    :class:`~pisces.utilities.config.ConfigManager` or a subclass that
    will be used to handle the configuration file. By default,
    it is set to :class:`~pisces.utilities.config.ConfigManager`, but
    subclasses may override this to use a custom configuration manager
    that provides additional functionality or validation specific to
    the simulation code being interfaced with.
    """

    # --------------------------------------- #
    # Initialization and Configuration        #
    # --------------------------------------- #
    # At the core, the frontend initialization is broken down
    # into just a few steps:
    #
    # 1. Validate that the input initial conditions are
    #    valid / suitable for this frontend. This is done
    #    by the `_validate_input_ic` method, which can be
    #    overridden by subclasses to implement specific
    #    validation rules.
    # 2. Copy / ensure that the default configuration
    #    file for this frontend exists in the initial
    #    conditions directory. This is done by the
    #    `_ensure_config` method, which will copy the
    #    default configuration file from the class-level
    #    `__default_configuration_path__` to the IC's
    #    working directory, or reset it if `reset` is True.
    #    [This generally doesn't need to be overridden!]
    # 3. Load the configuration file into a ConfigManager
    #    instance, which provides a convenient interface
    #    for accessing and modifying the configuration.
    #
    # Modify any relevant parts of this process to customize
    # the behavior of the frontend. The most common
    # modification here is to override _validate_input_ic.
    #
    @abstractmethod
    def _validate_input_ic(self, initial_conditions: InitialConditions) -> bool:
        """
        Validate that the provided initial conditions are suitable for this frontend.

        This method is called during initialization to ensure that
        the initial conditions object meets the requirements of the
        specific simulation code this frontend is designed for.

        Parameters
        ----------
        initial_conditions : InitialConditions
            The initial conditions object to validate.

        Returns
        -------
        bool
            True if the initial conditions are valid, False otherwise.
        """
        if not isinstance(initial_conditions, InitialConditions):
            raise TypeError(f"Expected an InitialConditions object, got {type(initial_conditions)}.")
        return True

    def _ensure_config(self, ic_directory: Path, reset: bool = False) -> tuple[Path, str]:
        """
        Ensure that a configuration file for this frontend exists.

        This method checks if a configuration file already exists in the
        specified initial conditions directory. If it does not exist,
        it copies the default configuration template from the class-level
        `__default_configuration_path__` to the IC directory. If a file
        already exists and `reset` is True, it overwrites the existing
        configuration file with the default template. If `reset` is False,
        it leaves the existing configuration file untouched.

        Parameters
        ----------
        ic_directory : Path
            The directory where the initial conditions are stored.
        reset : bool, optional
            If True, overwrite any existing configuration file with the
            default configuration template.

        Returns
        -------
        Path
            The full path to the configuration file in the IC directory.
        str
            The status of the configuration file. This will be either ``"created"``
            or ``"existing"`` depending on whether a new file was created or an existing
            file was found. This is used further down in the initialization process to
            ensure that the configuration is loaded correctly.
        """
        # --- Paths for configuration management --- #
        # Destination is always inside the IC's working directory, and
        # its filename is derived from the frontend class name to avoid
        # collisions across different frontends.
        config_destination = ic_directory / f"{self.__class__.__name__}_config.yaml"

        # The source path should be a class-level constant defined in subclasses
        # pointing to the frontend's default configuration file template.
        config_source = self.__class__.__default_configuration_path__

        # --- Verify that the default configuration exists --- #
        if config_source is None or not config_source.exists():
            raise FileNotFoundError(f"Default configuration file not found at {config_source}.")

        # --- Handle existing vs. new configuration file --- #
        if config_destination.exists():
            if reset:
                # Overwrite the existing configuration with the default.
                self.logger.debug(f"Resetting configuration file at {config_destination}.")
                shutil.copy(config_source, config_destination)
                status = "created"
            else:
                # Keep the existing configuration file untouched.
                self.logger.debug(f"Using existing configuration file at {config_destination}.")
                status = "existing"
        else:
            # No configuration exists — copy in the default template.
            self.logger.debug(f"Copying default configuration file from {config_source} to {config_destination}.")
            shutil.copy(config_source, config_destination)
            status = "created"

        return config_destination, status

    def _setup_config(self):
        """
        Perform additional setup after the configuration file is ensured.

        This method is called after the configuration file has been
        ensured and loaded into the ConfigManager. Subclasses may
        override this method to perform any additional setup steps
        that depend on the configuration being present.

        By default, this method does nothing.
        """
        return

    def __init__(self, initial_conditions: InitialConditions, reset_configuration: bool = False, **kwargs):
        """
        Construct a simulation frontend and attach it to an IC object.

        On initialization, the frontend will:

          1. Validate the provided initial conditions object (via
             :meth:`_validate_input_ic`).
          2. Ensure that a per-frontend configuration file exists in the
             initial conditions directory. If no file is present, the default
             configuration template is copied in. If a file already exists,
             it is either preserved or reset depending on ``reset_configuration``.
          3. Load the configuration into a :class:`~pisces.utilities.config.ConfigManager`
             instance and apply any keyword overrides passed through ``kwargs``.
          4. Call :meth:`__post_init__` to allow subclasses to perform
             additional setup.

        Parameters
        ----------
        initial_conditions : InitialConditions
            The Pisces initial conditions object to bind this frontend to.
            Provides access to the working directory, logging, and data
            structures that will be translated into native simulation input.
        reset_configuration : bool, optional
            If ``True``, overwrite any existing configuration file in the
            initial conditions directory with the default template for this
            frontend. If ``False`` (default), an existing configuration file
            will be preserved.
        **kwargs
            Arbitrary keyword arguments that will be merged into the loaded
            configuration after it is created. This allows programmatic
            updates without requiring manual edits to the YAML file.

        Raises
        ------
        FileNotFoundError
            If the default configuration template for this frontend cannot
            be found.
        """
        # --- Setup IC connection --- #
        # At this stage, we just take the IC that we got passed
        # and ensure it gets connected to the class.
        self.__initial_conditions__ = initial_conditions
        initial_conditions.logger.info(f"[{self.__class__.__name__}] Linking IC: {initial_conditions}...")

        # --- Input validation hook --- #
        # Perform any frontend-specific validation on the
        # provided initial conditions object.
        if not self._validate_input_ic(initial_conditions):
            raise ValueError(
                f"Initial conditions {initial_conditions} are not valid for {self.__class__.__name__} frontend."
            )
        self.logger.debug(f"{initial_conditions} passed validation on {self.__class__.__name__} frontend...")

        # --- Setup Configuration file --- #
        # We first pass off to _ensure_config to make sure that
        # that the configuration file exists in the IC directory. That
        # also provides a status indicating whether the file was created
        # or already existed. If it was created, then we need to setup the
        # configuration from defaults (_setup_config). We also need to update
        # with the kwargs.
        self.__config_path__, _load_status = self._ensure_config(
            ic_directory=initial_conditions.__directory__, reset=reset_configuration
        )
        self.__config__ = self.__class__.__default_configuration_manager_class__(self.__config_path__, autosave=True)

        if _load_status == "created":
            # We need to perform further setup procedures.
            self._setup_config()
        else:
            pass

        self.__config__.update(kwargs)

        # --- Subclass post-initialization hook --- #
        self.__post_init__()
        initial_conditions.logger.info(f"[{self.__class__.__name__}] Linking IC: {initial_conditions}... [DONE]")

    def __post_init__(self):  # noqa: B027
        """
        Perform any additional setup after initialization.

        This is called after the configuration has been loaded and updated
        in __init__. Override this in a subclass to perform any setup steps
        that depend on the configuration or IC object.
        """
        pass

    # --------------------------------------- #
    # Properties                              #
    # --------------------------------------- #
    @property
    def initial_conditions(self) -> InitialConditions:
        """
        The IC object that this frontend is bound to.

        Provides access to:
          - The working directory for file generation
          - Data structures describing the initial state of the system
          - A logger instance for status messages

        This is the central object that will be translated into the
        simulation code's native input format.
        """
        return self.__initial_conditions__

    @property
    def config(self) -> ConfigManager:
        """
        The :class:`~pisces.utilities.config.ConfigManager` instance associated with this frontend.

        This object provides a programmatic interface for reading and
        updating the YAML configuration file used to guide initial
        condition generation. It always corresponds to the file at
        :attr:`config_path`.

        Notes
        -----
        Updates made through this object may or may not automatically
        persist to disk, depending on the ConfigManager implementation.
        Call a dedicated ``save()`` method if persistence is required.
        """
        return self.__config__

    @property
    def config_path(self) -> Path:
        """
        Absolute path to the configuration file for this frontend.

        The filename is derived from the frontend's class name to
        avoid collisions across multiple frontends, e.g.::

            <ic_directory>/<FrontendClassName>_config.yaml

        Returns
        -------
        pathlib.Path
            Filesystem path to the YAML configuration file.
        """
        return self.__config_path__

    @property
    def default_configuration_path(self) -> Path:
        """
        Absolute path to the default configuration file template for this frontend.

        This is a class-level constant set by each frontend subclass,
        usually pointing to a file in the ``frontend_configs`` resource
        directory. It is copied into :attr:`ic_directory` if no existing
        configuration file is found.

        Returns
        -------
        pathlib.Path
            Filesystem path to the frontend's default YAML template.
        """
        return self.__class__.__default_configuration_path__

    @property
    def ic_directory(self) -> Path:
        """
        Filesystem directory where the initial conditions object is stored.

        This directory serves as the root for:
          - The per-frontend configuration file (:attr:`config_path`)
          - Any generated native input files
          - Metadata or provenance logs

        Returns
        -------
        pathlib.Path
            Directory path backing the connected
            :class:`~pisces.extensions.simulation.core.initial_conditions.InitialConditions`.
        """
        return self.initial_conditions.__directory__

    @property
    def logger(self):
        """
        Logger instance associated with this frontend.

        Delegated from :attr:`initial_conditions`. Provides a consistent
        way to emit information, warnings, or debug messages during
        configuration handling or IC generation.
        """
        return self.initial_conditions.logger

    # --------------------------------------- #
    # DUNDER METHODS                          #
    # --------------------------------------- #
    def __str__(self) -> str:
        """
        Return a human-readable string representation of the frontend.

        Includes the frontend class name and the path to its configuration file.
        Useful for logging and user-facing output.
        """
        return f"<{self.__class__.__name__} | IC={self.initial_conditions.directory.name}>"

    def __repr__(self) -> str:
        """
        Return an unambiguous string representation of the frontend.

        Includes the class name, associated InitialConditions object,
        and absolute configuration path for debugging.
        """
        return (
            f"{self.__class__.__name__}("
            f"initial_conditions={repr(self.initial_conditions)}, "
            f"config_path='{self.config_path}')"
        )

    def __getitem__(self, key):
        """
        Access configuration values by key.

        This is a convenience method that delegates directly to the
        underlying :class:`~pisces.utilities.config.ConfigManager` object.


        Parameters
        ----------
        key : str
            The configuration key to access.

        Returns
        -------
        Any
            The value associated with the key in the configuration.
        """
        return self.config[key]

    def __setitem__(self, key, value):
        """
        Set a configuration value by key.

        This is a convenience method that delegates directly to the
        underlying :class:`~pisces.utilities.config.ConfigManager`.

        Parameters
        ----------
        key : str
            Configuration key.
        value : Any
            New value to assign.
        """
        self.config[key] = value

    # --------------------------------------- #
    # IC GENERATION UTILITIES                 #
    # --------------------------------------- #
    # These are utility methods that can be used in generation procedures
    # when they are useful. They are not used in the default lifecycle,
    # but may be called by subclasses as needed.
    def _validate_models_have_particles(self):
        """
        Ensure that all models in the IC have associated particles.

        This method checks that each model defined in the
        :class:`~pisces.extensions.simulation.core.initial_conditions.InitialConditions`
        object has particles generated. If any model is found to be missing
        particles, a RuntimeError is raised with a descriptive message.

        Raises
        ------
        RuntimeError
            If any model in the initial conditions is missing particles.
        """
        for model in self.initial_conditions.list_models():
            if not self.initial_conditions.has_particles(model):
                raise RuntimeError(
                    f"Model `{model}` is missing particles in the ICs!\n"
                    "HINT: Ensure that you have generated particles for each model before calling the"
                    f" {self.__class__.__name__} frontend.\n"
                    "HINT: You can use `add_particles_to_model` on the IC object to generate particles."
                )

    # --------------------------------------- #
    # IC GENERATION METHODS                   #
    # --------------------------------------- #
    # These methods define the lifecycle for generating initial conditions
    # through a frontend. Subclasses are expected to override
    # `_generate_initial_conditions` with their own logic, and may optionally
    # override `_validate_runtime_configuration` for code-specific checks.
    #
    @abstractmethod
    def _validate_runtime_configuration(self, *args, **kwargs):
        """
        Validate the runtime configuration before generating initial conditions.

        This method is called immediately before initial condition
        generation to ensure that the loaded configuration is valid
        for the target simulation code. By default, it performs no checks.

        Subclasses may override this method to:
          - Enforce presence of required configuration keys.
          - Check unit consistency or ranges.
          - Perform any simulation-specific sanity checks
            before writing files.

        Parameters
        ----------
        *args, **kwargs
            Any arguments passed through from `generate_initial_conditions`.

        Returns
        -------
        None
        """
        return None

    @abstractmethod
    def _generate_initial_conditions(self, *args, **kwargs):
        """
        Generate the initial condition files for the target simulation code.

        This is the core implementation method that *must* be overridden
        by subclasses. It should read from `self.config` and
        `self.initial_conditions` and write any necessary files in the format
        expected by the simulation code.

        Parameters
        ----------
        *args
            Positional arguments specific to the subclass implementation.
        **kwargs
            Keyword arguments specific to the subclass implementation.

        """
        pass

    def generate_initial_conditions(self, *args, **kwargs) -> Any:
        """
        Generate all simulation-ready initial condition files for this frontend.

        This method serves as the **main entry point** for converting a
        Pisces :class:`~pisces.extensions.simulation.core.initial_conditions.InitialConditions`
        object into the native input format required by an external simulation code.

        It performs a complete two-step lifecycle:

        1. **Validate runtime configuration** using
           :meth:`_validate_runtime_configuration`, ensuring that the loaded
           configuration and attached initial conditions are physically and
           structurally consistent with the simulation code's expectations.

        2. **Generate native input files** via
           :meth:`_generate_initial_conditions`, which performs the actual
           file-writing procedure defined by the subclass implementation.

        Depending on the specific frontend, this process may involve:

        - Writing HDF5 or binary particle data files.
        - Generating code-specific metadata or parameter files.
        - Performing pre-processing (e.g., orientation, bounding box cuts).
        - Registering provenance or logging details to the IC directory.

        Parameters
        ----------
        *args
            Positional arguments forwarded to both
            :meth:`_validate_runtime_configuration` and
            :meth:`_generate_initial_conditions`.
        **kwargs
            Keyword arguments forwarded to both methods. These may include
            code-specific runtime options (e.g., ``filename``, ``overwrite=True``).

        Returns
        -------
        Any
            The return value of :meth:`_generate_initial_conditions`, which varies
            depending on the frontend implementation. For example, Gadget-like
            frontends return a :class:`~pisces.particles.gadget.Gadget4ParticleDataset`
            instance representing the generated file.

        Notes
        -----
        - This method is **not** meant to be overridden in subclasses.
          Instead, subclasses should implement the two private lifecycle methods:
          :meth:`_validate_runtime_configuration` and
          :meth:`_generate_initial_conditions`.
        - Any log output or error handling performed here ensures a consistent
          user experience across all simulation frontends.

        """
        self.logger.info(f"[{self.__class__.__name__}] Generating ICs - {self.initial_conditions}...")

        # Start with the runtime validation procedure.
        self._validate_runtime_configuration(*args, **kwargs)
        self.logger.info(f"[{self.__class__.__name__}]\t Validating runtime configuration... [DONE]")

        # Then generate the initial conditions files.
        out = self._generate_initial_conditions(*args, **kwargs)
        self.logger.info(f"[{self.__class__.__name__}] Generating ICs - {self.initial_conditions}... [DONE]")

        return out


class GadgetLikeFrontend(SimulationFrontend, ABC):
    """Abstract skeleton for Gadget-like simulation frontends."""

    # --------------------------------------- #
    # Class Variables and Constants           #
    # --------------------------------------- #
    # These are class-level variables which define connections to
    # the configuration along with some other aspects of the
    # frontend's behavior.
    __default_configuration_path__: Path = None  # FILL IN SUBCLASSES.
    """The path to the default configuration file.

    This is the path by which the frontend will attempt to locate the
    default configuration file template for this frontend.
    """
    __frontend_name__: str = None  # FILL IN SUBCLASSES.
    """str: The name of this frontend.

    The name of the frontend is substituted into various logging
    calls throughout the lifecycle to provide context on which
    frontend is being used. This ensures that subclasses do not
    need to modify every logging call to include their own
    name, and instead can just set this class-level constant.
    """
    __particle_dataset_type__: "ParticleDataset" = None  # FILL IN SUBCLASSES.
    """type: The particle dataset type used by this frontend.
    """
    _allowed_ic_types: tuple[type] = (InitialConditions3DCartesian,)
    """tuple of type: The allowed initial conditions types for this frontend.

    This is used on initialization to ensure that the input initial conditions
    class is actually compatible with the frontend in question. In general, this can
    be used to support many (or few) IC types based on the degree of support the developer
    wishes to provide for the frontend.
    """
    _simulation_types_map: dict[type, str] = {
        InitialConditions3DCartesian: "3D",
    }
    """dict: Mapping of initial conditions types to simulation type strings.

    These are used throughout the frontend simply to identify the type of simulation
    being run. This mapping allows the frontend to convert from an IC class to a
    string representation of the simulation type.
    """

    # --------------------------------------- #
    # Properties                              #
    # --------------------------------------- #
    @property
    def unit_system(self) -> unyt.unit_systems.UnitSystem:
        """
        Fetch the unit system for this Gadget-like frontend.

        Returns
        -------
        unyt.unit_systems.UnitSystem
            The unit system used for this frontend.

        Notes
        -----
        This property constructs and returns a unit system based on the
        length, mass, and velocity units specified in the configuration. This
        allows for easy conversion between the internal units used by the
        simulation code and the physical units used in the initial conditions.

        By default, Gadget-Like codes utilize a length, mass, and velocity based unit system
        where the time unit is derived from the other three. This is the method used here.

        """
        # Fetch the unit values from the configuration file so that we have them
        # on deck.
        length_unit, mass_unit, velocity_unit = (
            self.config["parameters.units.length"],
            self.config["parameters.units.mass"],
            self.config["parameters.units.velocity"],
        )

        # Use the velocity and the length units to construct
        # the time unit.
        time_unit = (1 * length_unit) / (1 * velocity_unit)
        time_unit = time_unit.to("Myr")

        # Construct the unit system object from unyt.
        _gadget_unit_system = unyt.unit_systems.UnitSystem(
            "gadget_simulation_system",
            length_unit=length_unit,
            mass_unit=mass_unit,
            time_unit=unyt.Unit(time_unit).simplify(),
        )

        return _gadget_unit_system

    # --------------------------------------- #
    # Initialization and Configuration        #
    # --------------------------------------- #
    # When running a Gadget-like code, there are a few standard things that
    # should be checked during the validation step. The first is that the input initial
    # conditions are valid initial conditions at all, and then we check that they
    # are supported by this frontend.
    #
    # DEVELOPERS could add further refinements to subclasses to even more finely
    # control what ICs are supported, or to add additional validation steps.

    def _validate_input_ic(self, initial_conditions: InitialConditions) -> bool:
        """
        Validate that the provided initial conditions are suitable for this frontend.

        This method is called during initialization to ensure that
        the initial conditions object meets the requirements of the
        specific simulation code this frontend is designed for.

        Parameters
        ----------
        initial_conditions : InitialConditions
            The initial conditions object to validate. This object is
            checked to ensure that it is actually a
            :class:`~pisces.extensions.simulation.core.initial_conditions.InitialConditions`
            object and that it is one of the supported types for this frontend.

        Returns
        -------
        bool
            True if the initial conditions are valid, False otherwise.
        """
        # Check that this is even an InitialConditions object.
        if not isinstance(initial_conditions, InitialConditions):
            raise TypeError(f"Expected an InitialConditions object, got {type(initial_conditions)}.")

        # For Gadget-like frontends, there are some specific IC types that are
        # permitted and these correspond to very specific values of the simulation type.
        # Depending on the simulation code in question, these may or may not be the same.
        if not isinstance(initial_conditions, self._allowed_ic_types):
            raise TypeError(
                f"{self.__frontend_name__} frontend only supports the following initial conditions types:\n"
                f"  - " + "\n  - ".join([cls.__name__ for cls in self._allowed_ic_types]) + "\n"
                f"Got {initial_conditions.__class__.__name__}."
            )

        return True

    def _setup_config(self):
        """
        Set up the configuration for this frontend after it has been generated in the IC directory.

        This method is called after the initial configuration has been
        generated in the initial conditions directory. It modifies
        the configuration in-place to ensure that it is suitable for
        the specific initial conditions being used.

        Notes
        -----
        In Gadget-like codes, each particle is assigned to a particular particle type and each particle
        type has some set of fields. The names of these particle types and their fields may vary from code
        to code and the names of those fields / particles on the Pisces side of the API may also vary. As such,
        we include ``"model_template"`` in the configuration file which contains a "default" set of particle types
        and fields for a single model.

        Once the frontend is generated for a particular set of initial conditions, we copy this template out
        so that each model has a copy of it. The user may then edit the mapping as needed before generating the
        simulation input files.

        Gadget-like codes also specify a number of different "simulation types" based on the type of initial conditions
        being used. This is typically a string value that is set in the Makefile or parameters file. We automatically
        set this value based on the type of initial conditions being used by mapping the IC class to a simulation type
        string.
        """
        # Extract the model template configuration from the data. If it doesn't exist,
        # we need to raise an error.
        if "model_template" in self.config:
            default_model_config = dict(self.config["model_template"])
        else:
            raise RuntimeError(
                f"Failed to find the `model_template` section in the {self.__frontend_name__} configuration!"
                " This section is required to setup the model configurations."
            )

        # Now duplicate the model template for each of the models in the initial conditions.
        self.config["models"] = {}
        for model_name in self.initial_conditions.list_models():
            self.logger.debug(f"Added {model_name} to the {self.__frontend_name__} configuration models...")
            self.config["models"][model_name] = deepcopy(dict(default_model_config))

        # We also need to automatically set the simulation type based on the type of
        # initial condition being used.
        self.config["makefile.simulation_type"] = self._simulation_types_map.get(
            self.initial_conditions.__class__, None
        )
        if self.config["makefile.simulation_type"] is None:
            raise RuntimeError(
                f"Failed to map initial conditions type {self.initial_conditions.__class__.__name__} to a "
                f"simulation type for the {self.__frontend_name__} frontend!"
            )

    # --------------------------------------- #
    # IC GENERATION METHODS (STRUCTURES)      #
    # --------------------------------------- #
    # These methods are used when converting PISCES ICs into simulation-ready
    # input files. They handle the various steps needed to preprocess the
    # particle datasets prior to writing them out to disk.
    #
    # Subclasses may need to make any number of modifications to this section of the
    # code.
    def _construct_bounding_box(self):
        """
        Construct the simulation bounding box from the configuration.

        Notes
        -----
        In Gadget-like codes, the simulation box is typically defined by a single
        "BoxSize" parameter which defines the length of one side of the cubic
        simulation volume. We convert this into a bounding box format that can
        be used for particle cuts and other operations.
        """
        # Extract the simulation boxsize from the configuration and ensure that
        # it actually exists.
        simulation_boxsize = self.config.get("parameters.boxsize", None)
        if simulation_boxsize is None:
            raise RuntimeError(
                f"Failed to load the boxsize (parameters.boxsize) from the {self.__frontend_name__} configuration!"
            )
        if not isinstance(simulation_boxsize, unyt.unyt_quantity):
            raise TypeError(f"Boxsize must be a `unyt_quantity` with length units, got {type(simulation_boxsize)}!")

        # With the boxsize parameter available, we can start building the bounding box.
        simulation_boxsize = simulation_boxsize.to_value(self.config["parameters.units.length"])
        simulation_bbox = np.stack([[0, simulation_boxsize] for _ in range(3)], axis=1)
        simulation_bbox = unyt.unyt_array(simulation_bbox, self.config["parameters.units.length"])

        return simulation_bbox

    def _reorient_particles(self, model_name):
        """
        Reorient the particles for a given model based on its configuration.

        Notes
        -----
        In Gadget-like codes, particles can be reoriented and offset based on
        the model parameters specified in the initial conditions. This method
        handles the reorientation and offsetting of particles for a specific model.
        """
        # Extract the particles corresponding to this particular model
        # so that we can modify them. Then look up the simulation type
        # so that we can differentiate between different geometric
        # scenarios.
        particle_dataset = self.initial_conditions.load_particles(model_name)
        _simulation_type = self.config["makefile.simulation_type"]
        if _simulation_type not in self.__class__._simulation_types_map.values():
            raise RuntimeError(
                f"Failed to map initial conditions type {self.initial_conditions.__class__.__name__} to a "
                f"simulation type for the {self.__frontend_name__} frontend!"
            )

        # Determine the parameters for the model based on the simulation type and
        # the initial conditions. We track the position, velocity, orientation, and spin
        # of the model so that we can apply the relevant transformations.
        mpos, mvel, more, mspin = None, None, None, None  # <- minimum information case (spherical)

        # for particular simulation types, we need to fill these out.
        if _simulation_type in ["1D", "2D", "3D"]:
            mpos = self.initial_conditions.config[f"models.{model_name}.position"]
            mvel = self.initial_conditions.config[f"models.{model_name}.velocity"]

        if _simulation_type in ["2D", "3D"]:
            more = self.initial_conditions.config[f"models.{model_name}.orientation"]

        if _simulation_type == "3D":
            mspin = self.initial_conditions.config[f"models.{model_name}.spin"]

        self.logger.debug(
            f"Model `{model_name}` parameters:\n"
            f"\tPosition: {mpos}\n"
            f"\tVelocity: {mvel}\n"
            f"\tOrientation: {more}\n"
            f"\tSpin: {mspin}\n"
        )

        # --- Reorient model --- #
        # We are now ready to perform the reorientation of the particle dataset. We do
        # this, by default, just for the position and velocity fields since we don't natively
        # support B fields. If B fields need to be added, one can simply pass another call to
        # particle_dataset.reorient_particles for the B field (gas) only.
        particle_dataset.reorient_particles(
            more,
            spin=mspin,
        )

        # --- Offset Model --- #
        # We now move the model to the correct position and velocity in the simulation volume.
        particle_dataset.offset_particle_positions(mpos)
        particle_dataset.offset_particle_velocities(mvel)
        particle_dataset.handle.close()

    def _preprocess_particle_datasets(self):
        """
        Preprocess the particle datasets for all models prior to writing simulation input files.

        Notes
        -----
        This method performs the necessary preprocessing steps on the particle datasets
        for all models in the initial conditions. This includes reorienting the particles
        based on the model parameters and cutting the particles down to the simulation
        bounding box.
        """
        # Construct the bounding box for the simulation so that
        # we can use it for cutting the particle datasets.
        simulation_bbox = self._construct_bounding_box()

        # --- Cut Particle Files to Bounding Box --- #
        # We now cycle through the models and fetch their particle datasets
        # so that we can begin the reductions. For each of the ICs, we'll need
        # to check the dimension to determine how to handle the bounding box cut.
        for model_name in self.initial_conditions.list_models():
            # --- Reorient Particles --- #
            # First, we need to reorient the particles for this model
            # so that they are in the correct position and orientation
            # for the simulation.
            self._reorient_particles(model_name)

            # --- Cut to bounding box --- #
            # With the particles reoriented and offset, we can now
            # cut them down to the bounding box of the simulation.
            particle_dataset = self.initial_conditions.load_particles(model_name)
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
        """Count particles of each Gadget-4 type across all models.

        Returns
        -------
        np.ndarray
            1D array of length = number of Gadget-4 particle types,
            dtype is uint64.
        """
        # Number of Gadget-4 particle types (typically 6)
        n_types = self.config["makefile.number_of_particle_types"]
        final_count = np.zeros(n_types, dtype=int)

        # Iterate over models defined in initial conditions
        for model_name in self.initial_conditions.list_models():
            model_config = self.config[f"models.{model_name}"]
            particle_counts = self.initial_conditions.get_particle_count(model_name)

            # Map model's native particle types into Gadget-4 ordering
            for i, (_gadget_type, cfg) in enumerate(model_config.items()):
                native_type = cfg["name"]
                count = particle_counts.get(native_type, 0)
                final_count[i] += count

        return final_count

    # --------------------------------------- #
    # IC GENERATION METHODS                   #
    # --------------------------------------- #
    def _validate_runtime_configuration(self, *args, **kwargs):
        """Validate the runtime configuration for the Gadget-like frontend."""
        # Start by ensuring that we have particles for each model in the
        # initial conditions.
        self._validate_models_have_particles()

        # All of the models have particle files. We now just ensure that each
        # model appears in the configuration list before allowing things
        # to proceed. We DON'T check for the full structure (too much logical complexity),
        # and instead let errors arise naturally if the user misconfigures things.
        for model in self.initial_conditions.list_models():
            if model not in self.config["models"]:
                raise RuntimeError(
                    f"Model `{model}` is missing from the {self.__frontend_name__} frontend configuration!\n"
                    "HINT: Ensure that you have added an entry for each model in the `models` "
                    "section of the"
                    " configuration file.\n"
                    "HINT: You can use the `model_template` section to set default"
                    " values for each model."
                )

        # Everything looks good! We'll log this and return.
        return None

    @abstractmethod
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
        # Begin by creating the IC particles dataset skeleton so that we can
        # populate it with data from each of the models.
        #
        # THIS MAY NEED MODIFICATION in subclasses to handle new / different
        # inputs to the particle dataset builder.
        # noinspection PyArgumentList
        ic_particles = self.__class__.__particle_dataset_type__.build_particle_dataset(
            *args,
            **kwargs,
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

    def _generate_initial_conditions(self, filename: str, *args, overwrite: bool, **kwargs):
        # Begin by handling the file checking and overwrite language
        # before proceeding to the actual generation element of the method.
        path = self.ic_directory / filename
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output file {path} already exists and overwrite is set to False!")
        elif path.exists() and overwrite:
            self.logger.debug(f"Overwriting existing {self.__frontend_name__} IC file at {path}...")
            path.unlink()
        else:
            pass

        # --- Pre-Process Particle Data --- #
        # We can now pre-process our particle datasets so that
        # they are contained within the specified bounding box and
        # so that they have the correct offsets, rotations, and velocities.
        # This is a universal process that should need minimal (if any) manipulations
        # when passing down to subclasses.
        self._preprocess_particle_datasets()

        # Fetch the unit system so that we have it available
        # for the relevant manipulations.
        unit_system = self.unit_system
        self.logger.debug(
            f"Using {self.__frontend_name__} unit system:\n"
            f"\tLength Unit: {unit_system['length']}\n"
            f"\tMass Unit: {unit_system['mass']}\n"
            f"\tTime Unit: {unit_system['time']}\n"
            f"\tVelocity Unit: {unit_system['velocity']}\n"
        )

        # Load / generate the particle dataset for the entire simulation.
        ic_particles = self._generate_particle_dataset(path, overwrite=overwrite)

        # --- Add IDs --- #
        # With all of the other fields written, we now need to handle
        # the particle IDs. Gadget-4 expects these to be unique across
        # all particle types, so we need to be a bit careful about how
        # we generate them.
        if self.config["makefile.nbits_id"] == 64:
            dtype = np.uint64
        elif self.config["makefile.nbits_id"] == 32:
            dtype = np.uint32
        else:
            raise RuntimeError(
                f"Unsupported ID bit-width {self.config['makefile.nbits_id']} in {self.__frontend_name__} "
                f"configuration!"
            )

        ic_particles.add_particle_ids(policy="global", overwrite=True, start_id=1, dtype=dtype)

        return ic_particles

    def generate_initial_conditions(self, filename: str, *args, overwrite: bool = False, **kwargs):
        """
        Generate Gadget-compatible initial condition files for this frontend.

        This method provides a convenience wrapper around the base
        :meth:`~pisces.extensions.simulation.core.frontends.SimulationFrontend.generate_initial_conditions`
        method, adding a standard ``filename`` and ``overwrite`` interface for
        Gadget-like simulation codes.

        It performs the complete frontend lifecycle:

        1. **Validate runtime configuration** using
           :meth:`_validate_runtime_configuration`, ensuring that the current
           configuration and particle data are consistent and complete.
        2. **Write the simulation input files** by calling the subclass-specific
           :meth:`_generate_initial_conditions`, which performs file creation,
           unit system setup, particle preprocessing, and ID assignment.

        Parameters
        ----------
        filename : str
            Name of the output file to be written within the initial conditions
            directory (e.g., ``"ClusterICs.hdf5"``). The full path is automatically
            resolved from :attr:`ic_directory`.
        *args
            Additional positional arguments forwarded to both
            :meth:`_validate_runtime_configuration` and
            :meth:`_generate_initial_conditions`.
        overwrite : bool, default=False
            If ``True``, any existing file with the same name will be replaced.
            If ``False``, an existing file will raise a
            :class:`FileExistsError`.
        **kwargs
            Additional keyword arguments forwarded to both
            :meth:`_validate_runtime_configuration` and
            :meth:`_generate_initial_conditions`. These may include
            code-specific options (e.g., compression flags, chunking behavior, etc.).

        Returns
        -------
        Any
            The object returned by :meth:`_generate_initial_conditions`. For most
            Gadget-like frontends, this will be a
            :class:`~pisces.particles.gadget.Gadget4ParticleDataset` representing
            the generated simulation file.

        Notes
        -----
        - This method should not be overridden in subclasses unless the frontend
          requires a fundamentally different call signature.
        - The ``filename`` argument is always interpreted relative to
          :attr:`ic_directory`.
        - Logging messages will automatically indicate progress through
          validation and file generation stages.
        """
        return super().generate_initial_conditions(filename, *args, overwrite=overwrite, **kwargs)
