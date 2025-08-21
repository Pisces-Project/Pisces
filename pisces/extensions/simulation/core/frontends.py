"""
Abstract base classes for simulation frontends in Pisces.

A simulation frontend is the bridge between Pisces' in-house
:class:`~pisces.extensions.simulation.core.initial_conditions.InitialConditions`
objects and an external simulation code's native input files and configuration
requirements.

"""

import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from pisces.utilities.config import ConfigManager

from .initial_conditions import InitialConditions


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

    @abstractmethod
    def __post_init__(self):
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

    def generate_initial_conditions(self, *args, **kwargs):
        """
        Generate the necessary initial condition files from this frontend.

        This method serves as the main entry point for generating initial
        conditions. It first validates the runtime configuration using
        :meth:`_validate_runtime_configuration`, and then calls
        :meth:`_generate_initial_conditions` to perform the actual file
        generation.

        Depending on the frontend implementation, this may involve
        writing files in a specific format, creating metadata, or
        performing additional setup steps.

        Parameters
        ----------
        *args
            Positional arguments forwarded to both validation and generation.
        **kwargs
            Keyword arguments forwarded to both validation and generation.

        """
        self.logger.info(f"[{self.__class__.__name__}] Generating ICs - {self.initial_conditions}...")

        # Start with the runtime validation procedure.
        self._validate_runtime_configuration(*args, **kwargs)
        self.logger.info(f"[{self.__class__.__name__}]\t Validating runtime configuration... [DONE]")

        # Then generate the initial conditions files.
        self._generate_initial_conditions(*args, **kwargs)
        self.logger.info(f"[{self.__class__.__name__}] Generating ICs - {self.initial_conditions}... [DONE]")

        return
