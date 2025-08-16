"""
abstract base classes for **simulation frontends** in Pisces.

A simulation frontend is the bridge between Pisces' in-house
:class:`~pisces.extensions.simulation.core.initial_conditions.InitialConditions`
objects and an external simulation code's native input files and configuration
requirements.

Overview
--------
Simulation frontends serve three key purposes:

1. **Configuration management**
   Each frontend provides a default YAML configuration template
   (stored in ``frontend_configs``). When attached to an initial
   conditions object, this template is copied into the working directory,
   ensuring that the simulation run is reproducible and customizable
   without altering global defaults.

2. **Validation hooks**
   Frontends provide multiple hooks to check compatibility and correctness:
     - :meth:`SimulationFrontend._validate_input_ic` validates that an
       :class:`~pisces.extensions.simulation.core.initial_conditions.InitialConditions`
       object is suitable for the target simulation code.
     - :meth:`SimulationFrontend._validate_runtime_configuration` allows
       subclasses to enforce runtime constraints, such as required keys,
       units, or parameter ranges.

3. **Initial condition generation**
   The main task of a frontend is to translate Pisces IC objects into
   simulation-native files. This is done via
   :meth:`SimulationFrontend.generate_initial_conditions`, which first
   validates the runtime configuration, then calls the subclass-defined
   :meth:`SimulationFrontend._generate_initial_conditions`.

Extending this Module
---------------------
To implement a new simulation frontend:

1. Subclass :class:`SimulationFrontend`.
2. Define the class variable :attr:`__default_configuration_path__` to
   point to your YAML template in ``frontend_configs``.
3. Implement :meth:`_generate_initial_conditions` with logic for
   generating simulation-ready files.
4. (Optional) Override :meth:`_validate_input_ic` to ensure only compatible
   ICs are used.
5. (Optional) Override :meth:`_validate_runtime_configuration` to perform
   final checks before file generation.

Example
-------
.. code-block:: python

    from pisces.extensions.simulation.core.frontend import (
        SimulationFrontend,
    )


    class MySimFrontend(SimulationFrontend):
        __default_configuration_path__ = (
            __frontend_bin_path__
            / "mysim_config.yaml"
        )

        def _validate_input_ic(self, ic):
            if "required_field" not in ic.fields:
                raise ValueError(
                    "MySim requires 'required_field' in ICs."
                )
            return True

        def _generate_initial_conditions(
            self, *args, **kwargs
        ):
            # Custom logic for writing MySim input files
            ...


    # Usage:
    ic = InitialConditions(...)
    frontend = MySimFrontend(
        ic, reset_configuration=True
    )
    frontend.generate_initial_conditions()

Notes
-----
- Configuration files are **copied per IC object**, ensuring reproducibility
  and preventing accidental modification of shared defaults.
- The :class:`~pisces.utilities.config.ConfigManager` provides an interface
  for reading/updating YAML configurations programmatically.

"""

import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from pisces.utilities.config import ConfigManager

from .initial_conditions import InitialConditions

# Create a path reference to the frontend configuration
# bin directory so that we can seek / load the configuration
# files.
__frontend_bin_path__ = Path(__file__).parents[1] / "frontend_configs"


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

    def _ensure_config(self, ic_directory: Path, reset: bool = False) -> Path:
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
            else:
                # Keep the existing configuration file untouched.
                self.logger.debug(f"Using existing configuration file at {config_destination}.")
        else:
            # No configuration exists — copy in the default template.
            self.logger.debug(f"Copying default configuration file from {config_source} to {config_destination}.")
            shutil.copy(config_source, config_destination)

        return config_destination

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
        # --- Store and log IC association --- #
        self.__initial_conditions__ = initial_conditions
        initial_conditions.logger.info(f"Connecting {initial_conditions} to {self.__class__.__name__} frontend.")

        # --- Input validation hook --- #
        if not self._validate_input_ic(initial_conditions):
            raise ValueError(
                f"Initial conditions {initial_conditions} are not valid for {self.__class__.__name__} frontend."
            )

        # --- Ensure configuration file is present/up to date --- #
        self.__config_path__ = self._ensure_config(
            ic_directory=initial_conditions.__directory__, reset=reset_configuration
        )

        # --- Load configuration manager --- #
        self.__config__ = ConfigManager(self.__config_path__)

        # --- Apply configuration overrides --- #
        self.__config__.update(kwargs)

        # --- Subclass post-initialization hook --- #
        self.__post_init__()

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
        return f"<{self.__class__.__name__} | config={self.config_path.name}>"

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
        self.logger.info(f"Generating initial conditions files for {self.initial_conditions}.")

        self._validate_runtime_configuration(*args, **kwargs)

        return self._generate_initial_conditions(*args, **kwargs)
