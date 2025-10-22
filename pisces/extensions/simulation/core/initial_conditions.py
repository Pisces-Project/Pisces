"""
Core classes for facilitating simulation initial conditions.

This module defines the :class:`InitialConditions` class, which serves as a base for
all simulation initial conditions in Pisces. It allows users to insert models into
a 3D box, providing a framework for defining and manipulating initial conditions
in astrophysical simulations.

"""

import datetime
import shutil
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import unyt

from pisces.models.core.base import BaseModel
from pisces.models.core.utils import inspect_model_grid, load_model
from pisces.utilities.config import ConfigManager
from pisces.utilities.io_tools import unyt_yaml
from pisces.utilities.log import LogDescriptor

if TYPE_CHECKING:
    from logging import Logger

    from pisces.geometry.grids.base import Grid
    from pisces.particles.base import ParticleDataset


class InitialConditions(ABC):
    """
    Central base class for configuring simulation initial conditions.

    This class provides the backbone of simulation initial conditions (ICs) in Pisces.
    It effectively provides a wrapper by which to place multiple models into a cartesian
    space with various orientations, velocities, etc. The initial conditions can then
    be processed by the simulation frontends to generate input files for various
    astrophysical simulation codes.

    For more detailed information about using and interacting with initial conditions,
    read :ref:`initial_conditions_overview`.
    """

    # ============================== #
    # Class Attributes               #
    # ============================== #
    # The following attributes are defined at the class level and provide
    # shared, configurable resources for all instances of InitialConditions.
    #
    # Subclasses can override these attributes to customize behavior without
    # rewriting core logic. For example:
    #   - Replace `__YAML__` to change serialization format or settings
    #   - Replace `logger` to use a different logging configuration or target
    #
    # These are not intended to be mutated at the instance level.
    __YAML__ = unyt_yaml
    """
    The YAML (de)serialization utility used to read/write the
    ``IC_CONFIG.yaml`` file.

    By default, this is set to :data:`unyt_yaml`, which is a YAML
    representer/dumper configured to correctly handle `unyt` array types
    (units, quantities, etc.).

    Subclasses may override this attribute to:

    - Use a different serialization backend
    - Provide custom YAML formatting options
    - Support additional non-standard Python types
    """
    logger: "Logger" = LogDescriptor(mode="ics")
    """
    Logger instance for the InitialConditions class.

    This descriptor provides a class-scoped logger configured with the
    ``"ics"`` mode. It should be used for all diagnostic,
    info, warning, and error output within the class.

    Subclasses may override this attribute to:

    - Use a different logging category or name
    - Redirect output to a custom logging handler

    Settings for the logger may be adjusted in the pisces configuration
    file under the ``ics`` section of ``logging``.
    """
    _model_file_extensions: dict[str, dict[str, str]] = {"path": {"extension": ""}, "particles": {"extension": "_p"}}
    """
    Standard file extensions for model-related files.

    This dictionary maps keys used in the model configuration (e.g., ``"path"``,
    ``"particles"``) to their corresponding file extensions. The extensions are
    used when copying or moving model files into the initial conditions directory
    to avoid filename collisions.
    """

    # ============================== #
    # Class Flags                    #
    # ============================== #
    # These flags are easily modified settings that are used throughout
    # the base class and should be easily accessible for modification
    # by subclasses.
    _model_metadata_required_keys = []
    _model_metadata_allowed_keys = ["particles"]
    _ndim = 3

    # ============================== #
    # Initialization Methods         #
    # ============================== #
    def _validate_configuration(self, *_, **__):
        """
        Validate the loaded IC configuration for structural and metadata consistency.

        This method performs the following checks:

          1. Ensures that both the metadata section (``__metadata__``) and at least one
             model entry are present in the configuration.
          2. Verifies that the stored `class_name` in metadata matches the current
             class name. This helps prevent loading ICs with an incompatible class.
          3. Validates that all referenced model and particle files exist on disk.

        Subclasses
        ----------
        Subclasses can override this method to add more specialized validation rules,
        such as checking for specific metadata keys, required physical parameters, or
        additional file dependencies. If overriding, consider calling
        ``super()._validate_configuration(*args, **kwargs)`` to preserve base checks.

        Raises
        ------
        ValueError
            If required metadata keys are missing or inconsistent, or if models have
            mismatched dimensions.
        FileNotFoundError
            If any referenced model or particle file is missing.
        """
        self.logger.debug(f"Attempting configuration validation of {self}.")
        # Start by performing basic validation. Ensure that we have both
        # metadata and the models sections in the configuration.
        if "metadata" not in self.__config__:
            raise ValueError("Configuration file is missing 'metadata' section.")
        if "models" not in self.__config__:
            raise ValueError("Configuration file is missing 'models' section.")

        # Now ensure that the metadata validation is performed and passes.
        # This section of the mode might need to be altered in subclasses because
        # of newly added metadata which needs validation. In this default case,
        # we ensure that the ndim field exists and we check that the class name
        # matches the current class.
        metadata = dict(self.__config__["metadata"])

        # check the class name.
        class_name = metadata.get("class_name", None)
        if class_name != self.__class__.__name__:
            raise ValueError(
                f"Configuration class name '{class_name}' does not match the current class '{self.__class__.__name__}'."
            )

        # As long as these pass, we consider the configuration valid. We now need to
        # validate that the models in models each are valid. That is done
        # separately in the __init__ method.
        self.logger.debug(f"{self} passed configuration validation.")

    @classmethod
    @abstractmethod
    def _validate_model(cls, model_name: str, model_info: dict) -> None:
        # Ensure that the model info contains a path and that
        # the path exists.
        model_path = Path(model_info["path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file for '{model_name}' not found: {model_path}")

        # Ensure all REQUIRED keys are present
        _check_keys = set(cls._model_metadata_required_keys).union({"path"})
        for key in _check_keys:
            if key not in model_info:
                raise ValueError(f"Model '{model_name}' is missing required key '{key}'.")

    def __init__(self, directory: Union[str, Path]):
        """
        Load an existing set of initial conditions from a target directory.

        This base initializer is responsible for:

        1. Normalizing and storing the target directory path.
        2. Verifying that the directory exists and is a valid directory on disk.
        3. Locating and loading the ``IC_CONFIG.yaml`` file associated with the
         initial condition set.
        4. Initializing the internal :class:`~pisces.utilities.config.ConfigManager` for accessing and
         modifying configuration data.
        5. Running the default configuration validation via
         ``_validate_configuration``.

        Parameters
        ----------
        directory : str or pathlib.Path
            Filesystem path to an **existing** initial condition directory that
            contains a valid ``IC_CONFIG.yaml`` file.

        Raises
        ------
        FileNotFoundError
            If the provided directory does not exist or is not a directory.
        FileNotFoundError
            If no ``IC_CONFIG.yaml`` file is found in the specified directory.
        ValueError
            If the configuration file exists but is structurally invalid.

        Notes
        -----
        - Subclasses overriding ``__init__`` should call
          ``super().__init__(directory)`` to ensure the configuration is loaded
          and validated before performing subclass-specific setup.
        - The loaded configuration is stored in :attr:`config`, a
          :class:`~pisces.utilities.config.ConfigManager` instance, and the absolute directory path
          is available as :attr:`directory`.
        """
        # Normalize to a Path object for consistent behavior
        # this ensures that we can use all of the relevant Path methods.
        self.__directory__ = Path(directory)
        self.logger.info("Loading %s object from directory: %s", self.__class__.__name__, self.__directory__.absolute())

        # Verify that the target directory exists and is actually a directory.
        # If we fail to find the directory, we raise an error.
        if not self.__directory__.exists():
            raise FileNotFoundError(f"Directory '{self.__directory__}' does not exist.")

        # Locate the configuration file and ensure that it exists before proceeding.
        # We will additionally check the consistency of the configuration file contents
        # during the following step in the initialization process.
        config_path = self.__directory__ / "IC_CONFIG.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file '{config_path}' does not exist in the directory.")

        self.__config__ = ConfigManager(config_path, autosave=True)

        # With the configuration manager in place, we proceed to the validation
        # step. This passes off to ``._validate_configuration``, which could be altered
        # in subclasses, but essentially just checks that everything is in order.
        self._validate_configuration()

        # We now validate the models.
        for model_name, model_info in dict(self.__config__["models"]).items():
            self.logger.debug("Validating model '%s'...", model_name)
            self._validate_model(model_name, model_info)

    # ============================== #
    # Properties                     #
    # ============================== #
    @property
    def directory(self) -> Path:
        """
        The directory where the initial conditions are stored.

        Returns
        -------
        ~pathlib.Path
            The directory path as a `Path` object.
        """
        return self.__directory__

    @property
    def config(self) -> ConfigManager:
        """
        The configuration manager for the initial conditions.

        Returns
        -------
        ~pisces.utilities.config.ConfigManager
            The configuration manager instance containing the initial conditions data.
        """
        return self.__config__

    @property
    def models(self) -> dict[str, dict]:
        """
        The processed models used in the initial conditions.

        Returns
        -------
        dict
            A dictionary mapping model names to their properties.
        """
        return dict(self.__config__["models"])

    @property
    def particles(self) -> dict[str, Union[Path, None]]:
        """
        The particle files associated with the models in the initial conditions.

        Returns
        -------
        dict
            A dictionary mapping model names to their particle file paths as `Path`.
            If no particles are associated, the value is `None`.
        """
        return {name: Path(info.get("particles", None)) for name, info in self.models.items()}

    @property
    def metadata(self) -> dict:
        """
        The metadata associated with the initial conditions.

        Returns
        -------
        dict
            A dictionary containing metadata information.
        """
        return dict(self.__config__["metadata"])

    @property
    def ndim(self) -> int:
        """
        The number of spatial dimensions for the initial conditions.

        Returns
        -------
        int
            The number of spatial dimensions (e.g., 3 for 3D).
        """
        return self.__class__._ndim

    # ============================== #
    # MAGIC Methods                  #
    # ============================== #
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(dir='{self.__directory__}', models={len(self.models)})>"

    def __str__(self) -> str:
        """Human-readable summary of the InitialConditions object."""
        return f"{self.__class__.__name__}({self.__directory__.name})"

    def __len__(self) -> int:
        """Get the number of models in this initial condition set."""
        return len(self.models)

    def __getitem__(self, model_name: str) -> dict:
        """
        Retrieve a model’s configuration by name.

        Parameters
        ----------
        model_name : str
            The model identifier.

        Returns
        -------
        dict
            Dictionary of the model’s configuration.

        Raises
        ------
        KeyError
            If no model with the given name exists.
        """
        try:
            return self.models[model_name]
        except KeyError as exp:
            raise KeyError(f"Model '{model_name}' not found in initial conditions.") from exp

    def __contains__(self, model_name: str) -> bool:
        """Check if a model with the given name exists in this initial condition set."""
        return model_name in self.models

    def __iter__(self):
        """Iterate over the names of all models in the initial conditions."""
        return iter(self.models)

    # ============================== #
    # Methods - Model Interaction    #
    # ============================== #
    def load_model(self, name: str) -> BaseModel:
        """
        Load a specific model from the initial conditions.

        This method retrieves the file path for the given model name from the
        internal configuration, verifies that the file exists, and uses
        :func:`load_model` to instantiate and return the model.

        Parameters
        ----------
        name : str
            The name/identifier of the model to load. Must match one of the keys
            in :attr:`models`.

        Returns
        -------
        ~pisces.models.core.base.BaseModel
            An instance of the loaded model.

        Raises
        ------
        KeyError
            If no model with the given ``name`` exists in the configuration.
        FileNotFoundError
            If the referenced model file does not exist on disk.
        """
        models = self.__config__.get("models", {})
        if name not in models:
            raise KeyError(f"No model named '{name}' found in initial conditions.")

        model_path = Path(models[name]["path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found for '{name}': {model_path}")

        return load_model(model_path)

    def remove_model(self, name: str) -> None:
        """
        Remove a model and its associated files from the initial conditions.

        This method:
          1. Verifies that the specified model exists in the configuration.
          2. Deletes the model's main file and optional particle file from disk
             (if they exist).
          3. Removes the model entry from the configuration.

        Parameters
        ----------
        name : str
            The name/identifier of the model to remove. Must match one of the
            keys in :attr:`models`.

        Raises
        ------
        KeyError
            If no model with the given ``name`` exists in the configuration.
        """
        models = self.__config__.get("models", {})
        if name not in models:
            raise KeyError(f"No model named '{name}' found in initial conditions.")

        model_info = models[name]

        # Remove associated files if they exist
        for key in self._model_file_extensions:
            if key in model_info and model_info[key] is not None:
                try:
                    Path(model_info[key]).unlink()
                except FileNotFoundError:
                    pass  # File already gone; no issue

        # Remove from configuration
        del self.__config__[f"models.{name}"]

        self.logger.info(f"Removed model '{name}' from initial conditions.")

    def add_model(
        self,
        name: str,
        model: Union[str, Path, BaseModel],
        model_config: dict[str, Any],
        file_processing_mode: str = "copy",
        overwrite: bool = False,
        **kwargs,
    ):
        """
        Add a new model to the initial conditions set.

        This method can be used to incorporate a new model into the initial conditions.
        In doing so, the model is validated to ensure that it satisfies the conditions for
        inclusion in the ICs and then is added in the same way that the original dataset
        was generated. This includes copying or moving the model file into
        the IC directory and updating the configuration file.

        Parameters
        ----------
        name : str
            The name of the model being added.
        model : str, ~pathlib.Path or ~pisces.models.core.base.BaseModel
            The model to add, specified as either:

            - **Path-like (str or Path)**: Path to the model file on disk. The file
              will be copied or moved into the initial conditions directory according
              to ``file_processing_mode``.
            - **BaseModel instance**: An already loaded model object. Its source file
              path will be obtained from the model's metadata.

            In both cases, the model file must exist and be accessible before calling
            this method.

        model_config : tuple
            A tuple specifying the model configuration settings. This may vary from
            code to code and should be documented in the subclass. Commonly, this
            includes parameters such as:

            - ``position`` : unyt array or sequence of length ``ndim``
            - ``velocity`` : unyt array or sequence of length ``ndim``
            - ``orientation`` : sequence or array defining orientation vector
            - ``spin`` : scalar float


        file_processing_mode : {"copy", "move"}, default="copy"
            Determines how the provided model file is placed into the initial
            conditions directory:

            - ``"copy"``: The source file is copied into the IC directory,
              preserving the original file in its current location.
            - ``"move"``: The source file is moved into the IC directory,
              removing it from its original location.

        overwrite : bool, optional
            If ``True``, an existing model entry with the same ``name`` will be
            replaced, and any associated files will be deleted or overwritten as
            needed. If ``False`` (default), attempting to add a model with a duplicate
            name will raise a :class:`ValueError`.

        kwargs:
            Additional keyword arguments passed to the model processing function.
            This may include options specific to the subclass or model type.
        """
        # Ensure that the model name does not already exist.
        # If it does, we need tell the user to manually remove it first.
        if name in self.models:
            if not overwrite:
                raise ValueError(
                    f"Model '{name}' already exists. Set `overwrite=True` to replace it or remove it first."
                )
            else:
                self.logger.warning(f"Overwriting existing model '{name}'.")
                self.remove_model(name)

        # --- Process the new model --- #
        # This mirrors the process we go through for
        # the initialization of models when creating the class.
        model_config = dict(model_config)
        model_config["model_name"] = name
        model_config["model"] = model

        _validated_model = self.__class__._validate_input_model(model, self.models, **kwargs)
        _validated_model_name = _validated_model.pop("model_name")

        # Now we need to ensure that the model gets either copied or moved
        # into the directory as needed. This is done with the ``_process_model``
        # method.
        _validated_model = self.__class__._process_model(
            _validated_model_name, _validated_model, file_processing_mode=file_processing_mode, **kwargs
        )

        # Add the post-validation model to the dictionary of ready-to-go
        # models.
        self.__config__["models"][_validated_model_name] = _validated_model
        self.logger.info(f"Added model '{name}' to initial conditions.")

    def list_models(self):
        """
        List all models currently stored in these initial conditions.

        Returns
        -------
        list
            A list of model names (strings) present in the initial conditions.
        """
        return list(self.models.keys())

    def has_model(self, model_name: str):
        """
        Check if a model with the given name exists in these initial conditions.

        Parameters
        ----------
        model_name : str
            The name/identifier of the model to check.

        Returns
        -------
        bool
            ``True`` if the model exists, ``False`` otherwise.
        """
        return model_name in self.models

    def update_model(self, model_name: str, **parameters) -> None:
        """
        Update parameters of an existing model in the initial conditions.

        Parameters
        ----------
        model_name : str
            The name/identifier of the model to update.
        parameters : dict
            Key-value pairs of parameters to update.

        Raises
        ------
        KeyError
            If no model with the given ``model_name`` exists.
        ValueError
            If provided parameters are invalid or inconsistent with expected dimensions/units.
        """
        # Ensure that we have the model to update.
        if model_name not in self.models:
            raise ValueError(f"No model named '{model_name}' found in initial conditions.")

        # Cycle through the keys and values of the
        # parameters dictionary and update.
        _permitted_keys = set(self._model_metadata_required_keys).union(self._model_metadata_allowed_keys)
        for key, value in parameters.items():
            if key not in _permitted_keys:
                raise ValueError(f"Parameter '{key}' is not a valid model parameter. ")

            # Update the value
            self.__config__[f"models.{model_name}.{key}"] = value

    def get_model_info(self, model_name: str):
        """
        Retrieve the configuration information for a specific model.

        Parameters
        ----------
        model_name : str
            The name/identifier of the model to retrieve.

        Returns
        -------
        dict
            A dictionary containing the model's configuration information.

        Raises
        ------
        KeyError
            If no model with the given ``model_name`` exists.
        """
        if model_name not in self.models:
            raise KeyError(f"No model named '{model_name}' found in initial conditions.")
        return self.models[model_name]

    def get_model_fields(self, model_name: str):
        """
        Inspect the available fields in a stored model without fully loading it.

        This method uses :func:`~pisces.models.core.utils.inspect_model_fields`
        to open the model file, look inside its ``/FIELDS`` group, and return a
        list of field names and their shapes.

        Parameters
        ----------
        model_name : str
            The name/identifier of the model to inspect.

        Returns
        -------
        list of tuple
            A list of ``(field_name, shape)`` pairs.

        Raises
        ------
        KeyError
            If the model name is not found in the initial conditions.
        FileNotFoundError
            If the referenced model file does not exist.
        """
        from pisces.models.core.utils import inspect_model_fields

        if model_name not in self.models:
            raise KeyError(f"No model named '{model_name}' found in initial conditions.")

        model_path = Path(self.models[model_name]["path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found for '{model_name}': {model_path}")

        return inspect_model_fields(model_path)

    def get_model_metadata(self, model_name: str):
        """
        Inspect the metadata of a stored model without fully loading it.

        This method uses :func:`~pisces.models.core.utils.inspect_model_metadata`
        to open the model file, look inside its ``/METADATA`` group, and return
        a dictionary of metadata key-value pairs.

        Parameters
        ----------
        model_name : str
            The name/identifier of the model to inspect.

        Returns
        -------
        dict
            A dictionary containing the model's metadata.

        Raises
        ------
        KeyError
            If the model name is not found in the initial conditions.
        FileNotFoundError
            If the referenced model file does not exist.
        """
        from pisces.models.core.utils import inspect_model_metadata

        if model_name not in self.models:
            raise KeyError(f"No model named '{model_name}' found in initial conditions.")

        model_path = Path(self.models[model_name]["path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found for '{model_name}': {model_path}")

        return inspect_model_metadata(model_path)

    def get_model_coordinate_system(self, model_name: str):
        """
        Retrieve the coordinate system of a stored model without fully loading it.

        This method opens the model file, reads the metadata, and returns
        the coordinate system object.

        Parameters
        ----------
        model_name : str
            The name/identifier of the model to inspect.

        Returns
        -------
        CoordinateSystem
            The coordinate system object associated with the model.

        Raises
        ------
        KeyError
            If the model name is not found in the initial conditions.
        FileNotFoundError
            If the referenced model file does not exist.
        """
        from pisces.models.core.utils import inspect_model_coordinate_system

        if model_name not in self.models:
            raise KeyError(f"No model named '{model_name}' found in initial conditions.")

        model_path = Path(self.models[model_name]["path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found for '{model_name}': {model_path}")

        return inspect_model_coordinate_system(model_path)

    def get_model_grid(self, model_name: str) -> "Grid":
        """
        Retrieve the grid object of a stored model without fully loading it.

        This method opens the model file, reads the metadata, and returns
        the grid object.

        Parameters
        ----------
        model_name : str
            The name/identifier of the model to inspect.

        Returns
        -------
        Grid
            The grid object associated with the model.

        Raises
        ------
        KeyError
            If the model name is not found in the initial conditions.
        FileNotFoundError
            If the referenced model file does not exist.
        """
        from pisces.models.core.utils import inspect_model_grid

        if model_name not in self.models:
            raise KeyError(f"No model named '{model_name}' found in initial conditions.")

        model_path = Path(self.models[model_name]["path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found for '{model_name}': {model_path}")

        return inspect_model_grid(model_path)

    def get_model_summary(self, model_name: str, include_fields: bool = True):
        """
        Retrieve a combined summary of a stored model without fully loading it.

        Parameters
        ----------
        model_name : str
            The name/identifier of the model to inspect.
        include_fields : bool, optional
            If True (default), also include the model's fields and shapes.

        Returns
        -------
        dict
            A dictionary with keys:
                - ``metadata`` : dict
                - ``grid`` : Grid
                - ``coordinate_system`` : CoordinateSystem
                - ``fields`` : list[(name, shape)] (if ``include_fields`` is True)
        """
        summary = {
            "metadata": self.get_model_metadata(model_name),
            "grid": self.get_model_grid(model_name),
            "coordinate_system": self.get_model_coordinate_system(model_name),
        }
        if include_fields:
            summary["fields"] = self.get_model_fields(model_name)
        return summary

    # ================================ #
    #  Methods - Particle Interaction  #
    # ================================ #
    def has_particles(self, model_name: str) -> bool:
        """
        Check if a specific model has associated particles.

        Parameters
        ----------
        model_name : str
            The name/identifier of the model to check.

        Returns
        -------
        bool
            ``True`` if the model has associated particles, ``False`` otherwise.

        Raises
        ------
        KeyError
            If no model with the given ``model_name`` exists.
        """
        if model_name not in self.models:
            raise KeyError(f"No model named '{model_name}' found in initial conditions.")
        particle_path = self.models[model_name].get("particles", None)
        return particle_path is not None and Path(particle_path).exists()

    def list_models_with_particles(self) -> list[str]:
        """
        List all models that have associated particle datasets.

        Returns
        -------
        list
            A list of model names (strings) that have associated particles.
        """
        return [name for name, info in self.models.items() if info.get("particles", None) is not None]

    def get_particle_path(self, model_name: str) -> Union[Path, None]:
        """
        Retrieve the particle file path associated with a specific model.

        Parameters
        ----------
        model_name : str
            The name/identifier of the model to inspect.

        Returns
        -------
        ~pathlib.Path or None
            The path to the particle dataset file if it exists, otherwise `None`.

        Raises
        ------
        KeyError
            If no model with the given ``model_name`` exists.
        """
        if model_name not in self.models:
            raise KeyError(f"No model named '{model_name}' found in initial conditions.")
        particle_path = self.models[model_name].get("particles", None)
        return Path(particle_path) if particle_path is not None else None

    def get_particle_path_dict(self) -> dict[str, Union[Path, None]]:
        """
        Retrieve a dictionary mapping model names to their associated particle file paths.

        Returns
        -------
        dict
            A dictionary where keys are model names and values are paths to the
            particle dataset files (or `None` if no particles are associated).
        """
        return {
            name: Path(info["particles"]) if info.get("particles", None) is not None else None
            for name, info in self.models.items()
        }

    def load_particles(self, model_name: str, **kwargs) -> "ParticleDataset":
        """
        Load the particle dataset associated with a specific model.

        Parameters
        ----------
        model_name : str
            The name/identifier of the model whose particles to load.
        kwargs :
            Additional keyword arguments to pass to the
            :class:`~pisces.particles.base.ParticleDataset` constructor.

        Returns
        -------
        ~pisces.particles.base.ParticleDataset
            An instance of the loaded particle dataset.

        Raises
        ------
        KeyError
            If no model with the given ``model_name`` exists.
        FileNotFoundError
            If the referenced particle file does not exist.
        ValueError
            If no particles are associated with the specified model.
        """
        from pisces.particles import ParticleDataset

        if model_name not in self.models:
            raise KeyError(f"No model named '{model_name}' found in initial conditions.")

        particle_path = self.models[model_name].get("particles", None)
        if particle_path is None:
            raise ValueError(f"No particles associated with model '{model_name}'.")

        particle_path = Path(particle_path)
        if not particle_path.exists():
            raise FileNotFoundError(f"Particle file for '{model_name}' not found: {particle_path}")

        return ParticleDataset(particle_path, **kwargs)

    def add_particles_to_model(
        self,
        particle_path: Union[str, Path],
        model_name: str,
        file_processing_mode: str = "copy",
        overwrite: bool = False,
    ):
        """
        Attach a particle dataset file to an existing model in the initial conditions.

        This method:

          1. Validates that the target model exists in the configuration.
          2. Checks if the specified particle file exists on disk.
          3. Copies or moves the file into the initial conditions directory, naming it
             ``<model_name>_p.hdf5`` to avoid collisions.
          4. Updates the configuration to record the particle file path.
          5. Optionally overwrites any previously associated particle file if
             ``overwrite=True``.

        Parameters
        ----------
        particle_path : str or ~pathlib.Path
            Path to the particle dataset file to associate with the model.
        model_name : str
            The name/identifier of the model to which the particles will be added.
        file_processing_mode : {"copy", "move"}, default="copy"
            How to handle the provided file:
              * ``"copy"`` – Copy the file into the IC directory (original remains intact).
              * ``"move"`` – Move the file into the IC directory (original is removed).
        overwrite : bool, optional
            If ``True``, any existing particle file for this model will be overwritten.
            If ``False`` (default), an error will be raised if a particle file is already
            associated with the model.

        Raises
        ------
        KeyError
            If no model with the given ``model_name`` exists in the configuration.
        FileNotFoundError
            If the specified ``particle_path`` does not exist.
        ValueError
            If ``file_processing_mode`` is not one of ``"copy"`` or ``"move"``.
        FileExistsError
            If a particle file is already associated with the model and ``overwrite`` is False.
        """
        _particle_file_extension = self.__class__._model_file_extensions["particles"]["extension"]
        if model_name not in self.models:
            raise KeyError(f"No model named '{model_name}' found in initial conditions.")

        particle_path = Path(particle_path)
        if not particle_path.exists() or not particle_path.is_file():
            raise FileNotFoundError(f"Particle file '{particle_path}' does not exist.")

        dest_path = self.directory / f"{model_name}{_particle_file_extension}.hdf5"

        # Check for existing particle file
        existing_particle_path = self.models[model_name].get("particles")
        if existing_particle_path:
            existing_file = Path(existing_particle_path)
            if existing_file.exists():
                if not overwrite:
                    raise FileExistsError(
                        f"Model '{model_name}' already has a particle file at {existing_file}. "
                        "Use `overwrite=True` to replace it."
                    )
                else:
                    existing_file.unlink()
                    self.logger.warning(f"Overwriting existing particle file for model '{model_name}'.")

        # Copy or move the new particle file
        if file_processing_mode == "copy":
            shutil.copy2(particle_path, dest_path)
        elif file_processing_mode == "move":
            shutil.move(str(particle_path), str(dest_path))
        else:
            raise ValueError(f"Invalid file_processing_mode '{file_processing_mode}'; must be 'copy' or 'move'.")

        # Update configuration
        self.__config__[f"models.{model_name}.particles"] = str(dest_path)
        self.logger.info(f"Added particles from '{dest_path}' to model '{model_name}'.")

    def remove_particles_from_model(self, model_name: str, delete_particle_file: bool = True):
        """
        Detach and optionally delete the particle dataset associated with a specific model.

        This method:

          1. Validates that the target model exists in the configuration.
          2. Checks if a particle file is associated with the model.
          3. Optionally deletes the particle file from disk.
          4. Updates the configuration to remove the particle file reference.

        Parameters
        ----------
        model_name : str
            The name/identifier of the model from which to remove particles.
        delete_particle_file : bool, optional
            If ``True`` (default), the particle file will be deleted from disk.
            If ``False``, only the reference in the configuration will be removed.

        Raises
        ------
        KeyError
            If no model with the given ``model_name`` exists in the configuration.
        ValueError
            If no particles are associated with the specified model.
        """
        if model_name not in self.models:
            raise KeyError(f"No model named '{model_name}' found in initial conditions.")

        particle_path = self.models[model_name].get("particles", None)
        if particle_path is None:
            raise ValueError(f"No particles associated with model '{model_name}' to remove.")

        particle_path = Path(particle_path)
        if delete_particle_file and particle_path.exists():
            particle_path.unlink()
            self.logger.info(f"Deleted particle file '{particle_path}' for model '{model_name}'.")

        # Remove reference from configuration
        del self.__config__[f"models.{model_name}.particles"]
        self.logger.info(f"Removed particle association from model '{model_name}'.")

    def get_particle_count(self, model_name: str) -> dict[str, int]:
        """Inspect the number of particles in each species group for a stored model.

        This is a wrapper around
        :func:`~pisces.particles.utils.inspect_particle_count`.

        Parameters
        ----------
        model_name : str
            The name/identifier of the model whose particle counts should be
            inspected.

        Returns
        -------
        dict of str, int
            Mapping of particle species names to the number of particles in each.

        Raises
        ------
        KeyError
            If the model name is not found in the initial conditions.
        FileNotFoundError
            If the referenced particle file does not exist.
        ValueError
            If a particle group is missing the required ``NUMBER_OF_PARTICLES``
            attribute.
        """
        from pisces.particles.utils import inspect_particle_count

        if model_name not in self.models:
            raise KeyError(f"No model named '{model_name}' found in initial conditions.")

        particle_path = self.particles[model_name]
        if particle_path is None or not particle_path.is_file():
            raise FileNotFoundError(f"No particle file associated with model '{model_name}'.")

        return inspect_particle_count(particle_path)

    def get_particle_species(self, model_name: str) -> list[str]:
        """List the particle species present in the stored particle dataset for a given model.

        This is a wrapper around
        :func:`~pisces.particles.utils.inspect_species`.

        Parameters
        ----------
        model_name : str
            The name/identifier of the model whose particle species should be
            inspected.

        Returns
        -------
        list of str
            A list of particle species names (HDF5 group names) present in the
            particle file.

        Raises
        ------
        KeyError
            If the model name is not found in the initial conditions.
        FileNotFoundError
            If the referenced particle file does not exist.
        """
        from pisces.particles.utils import inspect_species

        if model_name not in self.models:
            raise KeyError(f"No model named '{model_name}' found in initial conditions.")

        particle_path = self.particles[model_name]
        if particle_path is None or not particle_path.is_file():
            raise FileNotFoundError(f"No particle file associated with model '{model_name}'.")

        return inspect_species(particle_path)

    def get_particle_fields(self, model_name: str) -> dict[str, list[tuple[str, tuple[int, ...]]]]:
        """
        Inspect the available fields for each particle species in a model.

        This is a wrapper around
        :func:`~pisces.particles.utils.inspect_fields`.

        Parameters
        ----------
        model_name : str
            The name/identifier of the model whose particle fields should be
            inspected.

        Returns
        -------
        dict
            Mapping of particle species names to lists of
            ``(field_name, element_shape)`` tuples.

        Raises
        ------
        KeyError
            If the model name is not found in the initial conditions.
        FileNotFoundError
            If the referenced particle file does not exist.
        """
        from pisces.particles.utils import inspect_fields

        if model_name not in self.models:
            raise KeyError(f"No model named '{model_name}' found in initial conditions.")

        particle_path = self.particles[model_name]
        if particle_path is None or not particle_path.is_file():
            raise FileNotFoundError(f"No particle file associated with model '{model_name}'.")

        return inspect_fields(particle_path)

    def generate_particles(
        self, model_name: str, num_particles: dict[str, int], overwrite: bool = False, **kwargs
    ) -> None:
        """
        Generate particles for a stored model using its internal particle generation method.

        This method will load the model, verify that it implements a
        ``generate_particles`` method, and invoke it with the provided particle
        counts.

        Parameters
        ----------
        model_name : str
            The name/identifier of the model for which to generate particles.
        num_particles : dict of str, int
            Mapping of particle species names to the number of particles to
            generate for each.
        overwrite : bool
            If ``True``, any existing particle file for this model will be
            overwritten. If ``False`` (default), an error will be raised if
            a particle file is already associated with the model.
        kwargs:
            Additional keyword arguments to pass to the model's
            ``generate_particles`` method.

        Raises
        ------
        KeyError
            If the specified model is not found in the initial conditions.
        NotImplementedError
            If the loaded model does not implement a ``generate_particles`` method.
        """
        # Ensure that we actually have access to this model
        # and then load the model into memory so that we can
        # have it generate the particles.
        if model_name not in self.models:
            raise KeyError(f"No model named '{model_name}' found in initial conditions.")

        if self.__config__.get(f"models.{model_name}.particles", None) is not None:
            if overwrite:
                self.remove_particles_from_model(model_name, delete_particle_file=True)
            else:
                raise FileExistsError(
                    f"Model '{model_name}' already has associated particles. "
                    "Set `overwrite=True` to regenerate and replace them."
                )

        model = self.load_model(model_name)

        # Ensure that the model actually supports particle generation before
        # going any further with the process.
        if not hasattr(model, "generate_particles"):
            raise NotImplementedError(f"The model '{model_name}' does not support particle generation.")

        # Delegate to the model's particle generation method
        _particle_file_extension = self.__class__._model_file_extensions["particles"]["extension"]

        # noinspection PyUnresolvedReferences
        _generated_particles = model.generate_particles(
            self.__directory__ / f"{model_name}{_particle_file_extension}.hdf5", num_particles, **kwargs
        )
        self.config[f"models.{model_name}.particles"] = (
            self.__directory__ / f"{model_name}{_particle_file_extension}.hdf5"
        )
        self.logger.info(f"Generated particles for model '{model_name}' with counts: {num_particles}")

    # ============================== #
    # Generator Methods              #
    # ============================== #
    # These methods are used to generate the skeleton of an initial conditions
    # object. The methods here should be overridden in subclasses to specialize
    # the behavior of the initial conditions for a specific simulation code.
    #
    # ``_process_models`` takes a set of models and associated metadata and
    #   proceeds to process them, returning a CONFIG-compatible dictionary of
    #   model names and their properties. It also moves or copies the model files
    #  into the initial conditions directory as needed.
    #
    # ``
    @classmethod
    def _process_metadata(cls, directory: Path, **kwargs) -> dict:
        """
        Generate core metadata for the initial conditions set.

        The base implementation records only invariant structural
        information required for all IC sets. Subclasses are
        expected to extend this to add simulation- or
        geometry-specific keys (e.g., ``ndim``, coordinate
        system, cosmology parameters).

        Parameters
        ----------
        directory : ~pathlib.Path
            The directory where ICs are being created.
        **kwargs :
            Ignored in the base implementation. Subclasses may
            interpret these values when extending this method.

        Returns
        -------
        dict
            Metadata dictionary to be stored alongside model data.
            Guaranteed keys:
              - ``created_at`` : ISO 8601 timestamp (UTC)
              - ``class_name`` : str, subclass name
              - ``directory`` : str, absolute resolved path
        """
        timestamp = datetime.datetime.now(datetime.UTC).isoformat()

        return {
            "metadata": {
                "created_at": timestamp,
                "class_name": cls.__name__,
                "directory": str(directory.resolve()),
            }
        }

    @classmethod
    @abstractmethod
    def _validate_input_model(cls, model: dict, existing_models: dict, **kwargs) -> dict:
        """
        Validate and normalize a single input model definition.

        This method enforces a consistent structure for model dictionaries
        before they are incorporated into an initial conditions set.
        It is the *primary extension point* for subclasses that need
        specialized rules (e.g., different dimensionalities, extra fields,
        spherical coordinates).

        .. rubric:: Validation guarantees

        By default, validation ensures that:

        - ``model_name`` is a unique string across all models in this IC set.
        - ``model`` points to a valid model file on disk
          (string/Path or :class:`~pisces.models.core.base.BaseModel`). If a model
          was provided directly originally, the path to its file is extracted and
          stored.
        - ``particles`` (if specified) points to a valid file on disk.
        - ``position`` is a :class:`~unyt.unyt_array` with length units.
        - ``velocity`` is a :class:`~unyt.unyt_array` with length/time units.

        This should be improved / extended for subclasses to ensure that
        the models conform to the expectations of the specific simulation code.

        .. rubric:: Subclass extension points

        - Add or modify required keys by overriding
          :attr:`_model_metadata_required_keys`.
        - Enforce dimensionality of vectors (e.g., 1D, 2D, spherical).
        - Validate additional fields (orientation, spin, metallicity, etc.).
        - Inject default values for optional parameters if not provided.

        Parameters
        ----------
        model : dict
            Raw model specification provided by the user. Must include at least
            the keys listed in :attr:`_model_metadata_required_keys`.
        existing_models : dict
            Dictionary of already-processed models. Used to ensure uniqueness
            of ``model_name`` and to cross-check consistency.
        **kwargs :
            Extra options passed down from higher-level calls. Subclasses may
            use these to specialize validation (e.g., enforcing ``ndim``).

        Returns
        -------
        dict
            A validated, standardized model dictionary ready for inclusion
            in the configuration. Keys are guaranteed to include:

              - ``model_name`` : str
              - ``path`` : Path
              - ``position`` : unyt_array
              - ``velocity`` : unyt_array
              - ``[particles_path]`` : Path (if provided)

            Subclasses may add additional validated keys.

        Raises
        ------
        ValueError
            If required keys are missing or invalid.
        TypeError
            If fields are of incorrect types or units.
        FileNotFoundError
            If the referenced model file does not exist.
        """
        # --- Check Required Keys [Invariant] --- #
        # Ensure that all of the required keys are present. This is invariant
        # across all subclasses and should not need to be overwritten.
        for _required_key in cls._model_metadata_required_keys:
            if _required_key not in model:
                raise ValueError(f"Model definition is missing required key: {_required_key}.")

        # --- Check Only Optional / Required Keys [Invariant] --- #
        # Ensure that no unexpected keys are present. This is invariant
        # across all subclasses and should not need to be overwritten.
        allowed_keys = set(cls._model_metadata_required_keys).union(cls._model_metadata_allowed_keys)
        allowed_keys = allowed_keys.union({"model_name", "model"})
        for key in model.keys():
            if (key not in allowed_keys) and (key != "model_name"):
                raise ValueError(f"Model definition contains unexpected key: {key}.")

        # --- Model Name Uniqueness [Invariant] --- #
        # Ensure that the model name is a string and is unique across
        # all models in this IC set. This is invariant across all
        # subclasses and should not need to be overwritten.
        model["model_name"] = str(model["model_name"])
        if model["model_name"] in existing_models:
            suffix = 1
            new_name = f"{model['model_name']}_{suffix}"
            while new_name in existing_models:
                suffix += 1
                new_name = f"{model['model_name']}_{suffix}"
            model["model_name"] = new_name

        # --- Model Processing [Invariant] --- #
        # At this stage, we validate the model itself and ensure
        # that the model exists, convert the model to a path if
        # necessary and then proceed.

        # Ensure that the model specification is actually a path.
        attached_model = model.pop("model")
        if isinstance(attached_model, (str, Path)):
            model_path = Path(attached_model)
            if not model_path.exists() or not model_path.is_file():
                raise FileNotFoundError(f"Model file '{model_path}' does not exist or is not a file.")
            model["path"] = model_path
        elif isinstance(attached_model, BaseModel):
            model_path = Path(attached_model.__path__)
            if not model_path.exists() or not model_path.is_file():
                raise FileNotFoundError(f"Model file '{model_path}' does not exist or is not a file.")
            model["path"] = model_path
        else:
            raise TypeError(
                f"Model must be a string, Path, or BaseModel instance, not {type(attached_model).__name__}."
            )

        # Once we complete the validation, we return the model.
        return model

    @classmethod
    def _process_model(
        cls, directory: Path, model_name: str, model_info: dict, file_processing_mode: str = "copy", **kwargs
    ) -> dict:
        """
        Finalize a validated model for inclusion in the IC directory.

        This method performs any required file operations (copy/move) and
        ensures that the model dictionary is in a config-ready format.
        Subclasses may extend this to perform additional tasks such as
        particle generation, orientation defaults, etc.

        Parameters
        ----------
        directory : ~pathlib.Path
            The target directory where the model file should be placed.
        model_name : str
            Unique identifier for the model. This is guaranteed to have been
            validated for uniqueness upstream in :meth:`_validate_input_model`.
        model_info : dict
            The validated model dictionary.
        file_processing_mode : {"copy", "move"}, default="copy"
            How to place the model file into the IC directory:

            * ``"copy"`` – Copy the source file, preserving the original.
            * ``"move"`` – Move the source file, removing the original.
        **kwargs :
            Extra options forwarded from higher-level calls. Subclasses may
            use these to specialize processing.

        Returns
        -------
        dict
            The finalized model dictionary.

        Raises
        ------
        FileNotFoundError
            If the source model file does not exist.
        ValueError
            If ``file_processing_mode`` is invalid.
        """
        # For each of the models, we need to go through all of the
        # filehook keys and move things.
        for file_key, info in cls._model_file_extensions.items():
            extension = info.get("extension", "")
            if file_key in model_info:
                src_path = Path(model_info[file_key])
                if not src_path.exists():
                    raise FileNotFoundError(f"File for key '{file_key}' not found for '{model_name}': {src_path}")
                dest_path = directory / f"{model_name}{extension}{src_path.suffix}"
                print(dest_path)
                if file_processing_mode == "copy":
                    shutil.copy2(src_path, dest_path)
                elif file_processing_mode == "move":
                    shutil.move(str(src_path), str(dest_path))
                else:
                    raise ValueError(
                        f"Invalid file_processing_mode '{file_processing_mode}'; must be 'copy' or 'move'."
                    )
                model_info[file_key] = dest_path

        cls.logger.debug(
            "Processed model '%s': stored at %s [mode=%s]",
            model_name,
            model_info["path"],
            file_processing_mode,
        )

        # Return the finalized dictionary for this model.
        return model_info

    @classmethod
    def create_ics(cls, directory: Union[str, Path], *models: dict, file_processing_mode: str = "copy", **kwargs):
        """
        Create a new initial conditions (IC) directory and return an :class:`InitialConditions` instance.

        This is the main entry point for building an IC set from scratch.
        It performs all required filesystem setup, model validation, and
        configuration writing.

        .. rubric:: Workflow

        1. **Directory setup**:

           - Create the target directory if it does not exist.
           - If it exists and is non-empty, ``overwrite=True`` is required
             in ``kwargs`` to clear it safely.

        2. **Model processing**:

           - Each model definition (dict) is validated via
             :meth:`_validate_input_model`.
           - Files are copied or moved into the IC directory via
             :meth:`_process_model`.

        3. **Configuration file**:

           - Metadata and processed model information are assembled.
           - A YAML file ``IC_CONFIG.yaml`` is written for later reload.

        Parameters
        ----------
        directory : str or ~pathlib.Path
            Target directory for the IC set. Must be empty unless
            ``overwrite=True`` is provided. All of the model files and
            any connected files will be copied / moved into this directory so
            that it becomes the centralized location for the IC set.
        *models : dict
            Model definitions. The expected keys in each model may vary from
            subclass to subclass, but at a minimum we expect:

            - ``"model_name"`` : str
              The unique name/identifier for this model.
            - ``"model"`` : str, Path, or BaseModel
               The model specification, either as a path to a model file
            - ``"position"`` : unyt_array with length units
              The position of the model in the simulation volume.
            - ``"velocity"`` : unyt_array with length/time units
              The bulk velocity of the model in the simulation volume.

        file_processing_mode: {"copy", "move"}, default="copy"
            How to handle the provided model files:

            * ``"copy"`` – Copy the files into the IC directory (originals remain intact).
            * ``"move"`` – Move the files into the IC directory (originals are removed).

        **kwargs :
            Additional keyword arguments which are subclass dependent.

        Returns
        -------
        InitialConditions
            A fully initialized instance pointing to the new directory.

        Raises
        ------
        FileExistsError
            If the directory exists and is non-empty, and ``overwrite`` is False.
        FileNotFoundError
            If a referenced model file does not exist.
        ValueError
            If a model definition is missing required keys or is otherwise invalid.

        Notes
        -----
        - This method is not intended to be overridden by subclasses.
          Instead, override the following hooks to customize behavior:

          * :meth:`_validate_input_model`
          * :meth:`_process_model`
          * :meth:`_process_metadata`

        - The returned instance is ready for immediate use in frontend
          converters (e.g., Gadget, AREPO).
        """
        # --- Directory Setup [INVARIANT] --- #
        # Process the provided directory. We check that it is a valid directory
        # and that it doesn't contain any existing files that need to be overwritten.
        #
        # This is a structural invariant of this class and should NOT be overwritten
        # by subclasses to ensure that the structure is contiguous.
        cls.logger.info("Creating initial conditions in directory: %s", directory)
        directory = Path(directory)

        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            cls.logger.debug("Created new directory: %s", directory)
        else:
            has_content = any(directory.iterdir())
            if has_content and kwargs.get("overwrite", False):
                shutil.rmtree(directory)
                directory.mkdir(parents=True, exist_ok=True)
                cls.logger.debug("Overwrote existing directory: %s", directory)
            elif has_content:
                raise FileExistsError(
                    f"Directory `{directory}` is not empty. Use `overwrite=True` to replace its contents."
                )

        # --- Model Processing --- #
        # At this stage, we move onto model processing. This is a two step process:
        # 1. We convert the metadata into a standardized dictionary that we can write
        #    directly into the IC file when we're ready.
        # 2. We copy or move the model files into the directory as needed.
        #
        # Subclasses may override parts of this process.
        _validated_models = {}
        for model in models:
            # Each model is a dictionary contained some set of keys. The first
            # step will be to ensure all of the expected model attributes are
            # present and to ensure that there are no unexpected keys. This is
            # do in the _validate_input_model method.
            _validated_model = cls._validate_input_model(model, _validated_models, **kwargs)
            _validated_model_name = _validated_model.pop("model_name")

            # Now we need to ensure that the model gets either copied or moved
            # into the directory as needed. This is done with the ``_process_model``
            # method.
            _validated_model = cls._process_model(
                directory, _validated_model_name, _validated_model, file_processing_mode=file_processing_mode, **kwargs
            )

            # Add the post-validation model to the dictionary of ready-to-go
            # models.
            _validated_models[_validated_model_name] = _validated_model

        # --- Create the IC_CONFIG File --- #
        # We create a configuration file that contains the processed models
        # and their properties. This is a simple YAML file that can be read later.
        with open(directory / "IC_CONFIG.yaml", "w") as f:
            metadata = cls._process_metadata(directory, **kwargs)
            metadata["models"] = _validated_models
            cls.__YAML__.dump(metadata, f)

        # Return the class initialized with the directory.
        cls.logger.info("Initial conditions created successfully in %s", directory)
        return cls(directory)


class InitialConditions1DSpherical(InitialConditions):
    """
    Initial conditions for 1D spherical simulations.

    This subclass enforces that all models are defined in a strictly spherical
    coordinate system with only a radial dependence (``r`` axis). Bulk positions
    and velocities are not used—models are implicitly located at the origin with
    zero translational motion.

    Use this class when constructing spherically symmetric ICs, e.g. radial
    gas/halo profiles. The class ensures that:

    - The model grid uses :class:`~pisces.geometry.coordinates.coordinate_systems.SphericalCoordinateSystem`.
    - The grid has exactly one active axis (``r``).
    - Only ``model_name`` and ``model`` are required keys; particles may be added.

    Subclasses may extend validation to enforce additional spherical-specific
    metadata (e.g., radial boundary conditions, outer cutoff radii).
    """

    # ============================== #
    # Class Flags                    #
    # ============================== #
    # These flags are easily modified settings that are used throughout
    # the base class and should be easily accessible for modification
    # by subclasses.
    _model_metadata_required_keys = []
    _model_metadata_allowed_keys = ["particles"]
    _ndim = 1

    # ============================== #
    # Initialization Methods         #
    # ============================== #
    @classmethod
    def _validate_model(cls, model_name: str, model_info: dict) -> None:
        super()._validate_model(model_name, model_info)

    # ============================== #
    # Generator Methods              #
    # ============================== #
    # These methods are used to generate the skeleton of an initial conditions
    # object. The methods here should be overridden in subclasses to specialize
    # the behavior of the initial conditions for a specific simulation code.
    #
    # ``_process_models`` takes a set of models and associated metadata and
    #   proceeds to process them, returning a CONFIG-compatible dictionary of
    #   model names and their properties. It also moves or copies the model files
    #  into the initial conditions directory as needed.
    #
    @classmethod
    def _validate_input_model(cls, model: dict, existing_models: dict, **kwargs) -> dict:
        # --- Check Required Keys [Invariant] --- #
        # Ensure that all of the required keys are present. This is invariant
        # across all subclasses and should not need to be overwritten.
        for _required_key in cls._model_metadata_required_keys:
            if _required_key not in model:
                raise ValueError(f"Model definition is missing required key: {_required_key}.")

        # --- Check Only Optional / Required Keys [Invariant] --- #
        # Ensure that no unexpected keys are present. This is invariant
        # across all subclasses and should not need to be overwritten.
        allowed_keys = set(cls._model_metadata_required_keys).union(cls._model_metadata_allowed_keys)
        for key in model.keys():
            if key not in allowed_keys:
                raise ValueError(f"Model definition contains unexpected key: {key}.")

        # --- Model Name Uniqueness [Invariant] --- #
        # Ensure that the model name is a string and is unique across
        # all models in this IC set. This is invariant across all
        # subclasses and should not need to be overwritten.
        model["model_name"] = str(model["model_name"])
        if model["model_name"] in existing_models:
            suffix = 1
            new_name = f"{model['model_name']}_{suffix}"
            while new_name in existing_models:
                suffix += 1
                new_name = f"{model['model_name']}_{suffix}"
            model["model_name"] = new_name

        # --- Model Processing [Invariant] --- #
        # At this stage, we validate the model itself and ensure
        # that the model exists, convert the model to a path if
        # necessary and then proceed.

        # Ensure that the model specification is actually a path.
        attached_model = model.pop("model")
        if isinstance(attached_model, (str, Path)):
            model_path = Path(attached_model)
            if not model_path.exists() or not model_path.is_file():
                raise FileNotFoundError(f"Model file '{model_path}' does not exist or is not a file.")
            model["path"] = model_path
        elif isinstance(attached_model, BaseModel):
            model_path = Path(attached_model.__path__)
            if not model_path.exists() or not model_path.is_file():
                raise FileNotFoundError(f"Model file '{model_path}' does not exist or is not a file.")
            model["path"] = model_path
        else:
            raise TypeError(
                f"Model must be a string, Path, or BaseModel instance, not {type(attached_model).__name__}."
            )

        # --- Ensure Spherical Coordinate System --- #
        # We access the model's grid and ensure that it is spherical and that
        # we only have 1 active axis (r).
        model_grid = inspect_model_grid(model["path"])

        if model_grid.coordinate_system.__class__.__name__ != "SphericalCoordinateSystem":
            raise ValueError(
                f"Model '{model['model_name']}' must use a spherical coordinate system, "
                f"not {model_grid.coordinate_system.__class__.__name__}."
            )

        if set(model_grid.active_axes) != {"r"}:
            raise ValueError(
                f"Model '{model['model_name']}' must have only a radial grid dependence, not {model_grid.active_axes}."
            )

        # Once we complete the validation, we return the model.
        return model


class InitialConditionsCartesian(InitialConditions, ABC):
    """
    Abstract base class for Cartesian initial conditions.

    Provides a general framework for simulations defined in Cartesian
    coordinates with ``ndim`` active spatial dimensions. This class
    enforces the presence and dimensionality of ``position`` and ``velocity``
    vectors and provides convenience accessors for them.

    Key features:
      - Ensures positions are length vectors of shape ``(ndim,)``.
      - Ensures velocities are length/time vectors of shape ``(ndim,)``.
      - Defines ``model_positions`` and ``model_velocities`` properties for
        retrieving validated unyt arrays with physical units.

    Subclasses specify the dimensionality by setting ``_ndim`` and may extend
    validation to include orientation, spin, or other Cartesian-specific keys.
    """

    # ============================== #
    # Class Flags                    #
    # ============================== #
    # These flags are easily modified settings that are used throughout
    # the base class and should be easily accessible for modification
    # by subclasses.
    _model_metadata_required_keys = []
    _model_metadata_allowed_keys = ["particles"]
    _ndim = 3

    # ============================== #
    # Initialization Methods         #
    # ============================== #
    @classmethod
    def _validate_model(cls, model_name: str, model_info: dict) -> None:
        super()._validate_model(model_name, model_info)

    # ============================== #
    # Properties                     #
    # ============================== #
    @property
    def model_positions(self) -> dict[str, unyt.unyt_array]:
        """
        The positions of the models in the initial conditions.

        Returns
        -------
        dict
            A dictionary mapping model names to their position vectors as `unyt.array.unyt_array`.
        """
        return {name: unyt.unyt_array(info["position"], units="m") for name, info in self.models.items()}

    @property
    def model_velocities(self) -> dict[str, unyt.unyt_array]:
        """
        The velocities of the models in the initial conditions.

        Returns
        -------
        dict
            A dictionary mapping model names to their velocity vectors as `unyt.array.unyt_array`.
        """
        return {name: unyt.unyt_array(info["velocity"], units="km/s") for name, info in self.models.items()}

    # ============================== #
    # Generator Methods              #
    # ============================== #
    # These methods are used to generate the skeleton of an initial conditions
    # object. The methods here should be overridden in subclasses to specialize
    # the behavior of the initial conditions for a specific simulation code.
    #
    # ``_process_models`` takes a set of models and associated metadata and
    #   proceeds to process them, returning a CONFIG-compatible dictionary of
    #   model names and their properties. It also moves or copies the model files
    #  into the initial conditions directory as needed.
    #
    @classmethod
    @abstractmethod
    def _validate_input_model(cls, model: dict, existing_models: dict, **kwargs) -> dict:
        # Perform the super-class initialization to ensure that we
        # have the basic structure in place.
        model = super()._validate_input_model(model, existing_models, **kwargs)

        # --- Parameter Processing [Extensible] --- #
        # At this stage, we validate and process the other parameters
        # in the model. This is the primary extension point for subclasses
        # that need to enforce different dimensionalities, coordinate systems,
        # or additional parameters.
        # Ensure that the model has its location and velocity specified correctly.
        model["position"] = unyt.unyt_array(model["position"])
        if model["position"].units.dimensions != unyt.dimensions.length:
            raise TypeError(f"Position must have length units, not {model['position'].units.dimensions}.")
        if model["position"].shape != (cls._ndim,):
            raise ValueError(f"Position must be a {cls._ndim}D vector, not shape {model['position'].shape}.")

        model["velocity"] = unyt.unyt_array(model["velocity"])
        if model["velocity"].units.dimensions != (unyt.dimensions.length / unyt.dimensions.time):
            raise TypeError(f"Velocity must have length/time units, not {model['velocity'].units.dimensions}.")
        if model["velocity"].shape != (cls._ndim,):
            raise ValueError(f"Velocity must be a {cls._ndim}D vector, not shape {model['velocity'].shape}.")

        # Once we complete the validation, we return the model.
        return model

    # ============================== #
    # Physics Methods                #
    # ============================== #
    def compute_center_of_mass(
        self,
        models: Union[str, list[str]] = "all",
        masses: Optional[dict[str, "unyt.unyt_quantity"]] = None,
    ):
        r"""Compute the mass-weighted center of mass (COM) position for one or more models in the initial conditions.

        The COM is calculated as:

        .. math::

            \mathbf{R}_{\mathrm{COM}} =
            \frac{\sum_i m_i \mathbf{r}_i}{\sum_i m_i}

        where ``m_i`` is the total mass of model ``i`` and ``r_i`` is its position.

        Mass values are obtained in the following order:

          1. If ``masses`` is provided, use those values (must map model name → scalar mass).
          2. Otherwise, attempt to read ``total_mass`` from the model's metadata via
             :meth:`get_model_metadata`.

        Parameters
        ----------
        models : str or list of str, optional
            Which models to include in the COM calculation. If ``"all"``, all models
            in :attr:`models` are used. Otherwise, provide an explicit list of model
            names. If a single model name is provided as a string, it will be treated
            as a list of one model.
        masses : dict of str, ~unyt.array.unyt_quantity, optional
            Optional mapping from model name to its total mass. If provided for a
            given model, this value overrides the ``total_mass`` from metadata.

        Returns
        -------
        unyt.array.unyt_array
            The COM position vector with shape ``(ndim,)`` and units of length.

        Raises
        ------
        KeyError
            If any specified model name is not found in the initial conditions.
        ValueError
            If a required mass is missing from both ``masses`` and the model metadata.
        """
        import unyt

        # Manage the model processing logic so that
        # we have a list of model names to work with.
        if models == "all":
            model_names = list(self.models.keys())
        elif isinstance(models, str):
            model_names = [models]
        elif isinstance(models, Sequence):
            model_names = list(models)
        else:
            raise TypeError("`models` must be 'all', a string, or a list of strings.")

        # Start the loop through all of the various
        # models to begin computing the center of mass
        # value and the weighted position value.
        total_mass_sum = 0.0 * unyt.Unit("Msun")
        weighted_position_sum = unyt.unyt_array([0.0] * self.ndim, units="Msun*pc")

        # Iterate through each of the models in our
        # initial conditions / those specified by the user.
        for name in model_names:
            if name not in self.models:
                raise KeyError(f"No model named '{name}' found in initial conditions.")

            # Determine mass
            if (masses is not None) and (name in masses):
                mass = masses[name]
                if not isinstance(mass, unyt.unyt_quantity):
                    mass = unyt.unyt_quantity(mass, "Msun")
            else:
                metadata = self.get_model_metadata(name)
                if "total_mass" not in metadata:
                    raise ValueError(
                        f"Model '{name}' is missing 'total_mass' in metadata and no override was provided."
                    )
                mass = metadata["total_mass"]
                if not isinstance(mass, unyt.unyt_quantity):
                    mass = unyt.unyt_quantity(mass, "Msun")

            # Position from stored config
            pos = self.model_positions[name]

            weighted_position_sum += mass * pos
            total_mass_sum += mass

        if total_mass_sum <= 0 * unyt.Unit("Msun"):
            raise ValueError("Total mass is zero; cannot compute center of mass.")

        return weighted_position_sum / total_mass_sum

    def compute_center_of_mass_frame_positions(
        self,
        models: Union[str, list[str]] = "all",
        masses: Optional[dict[str, "unyt.unyt_quantity"]] = None,
        com_position: Optional["unyt.unyt_array"] = None,
    ) -> dict[str, "unyt.unyt_array"]:
        """
        Compute the positions of models in the center-of-mass (COM) frame.

        This subtracts the COM position (mass-weighted) from each model's stored
        position.

        Parameters
        ----------
        models : str or list of str, optional
            Models to include in the computation. Defaults to ``"all"`` for all models.
        masses : dict of str, ~unyt.array.unyt_quantity, optional
            Optional mapping from model name to its total mass, used for COM
            calculation. Overrides ``total_mass`` in metadata if provided.
        com_position : ~unyt.array.unyt_array, optional
            Precomputed COM position vector. If not given, it will be computed using
            :meth:`compute_center_of_mass` with the same ``models`` and ``masses``.

        Returns
        -------
        dict of str, ~unyt.array.unyt_array
            Mapping of model names to their position vectors in the COM frame.

        Raises
        ------
        KeyError
            If any specified model name is not found.
        ValueError
            If mass data is missing for a model and COM computation is required.
        """
        # Normalize model list
        if models == "all":
            model_names = list(self.models.keys())
        elif isinstance(models, str):
            model_names = [models]
        elif isinstance(models, Sequence):
            model_names = list(models)
        else:
            raise TypeError("`models` must be 'all', a string, or a list of strings.")

        # Compute COM position if not provided
        if com_position is None:
            com_position = self.compute_center_of_mass(model_names, masses=masses)

        com_position = unyt.unyt_array(com_position, units="m")

        # Compute COM-frame positions
        com_frame_positions: dict[str, unyt.unyt_array] = {}
        for name in model_names:
            if name not in self.models:
                raise KeyError(f"No model named '{name}' found in initial conditions.")
            pos = self.model_positions[name]
            com_frame_positions[name] = pos - com_position

        return com_frame_positions

    def compute_center_of_mass_velocity(
        self,
        models: Union[str, list[str]] = "all",
        masses: Optional[dict[str, "unyt.unyt_quantity"]] = None,
    ):
        r"""
        Compute the mass-weighted center-of-mass (COM) velocity for one or more models in the initial conditions.

        The COM velocity is:

        .. math::

            \mathbf{V}_{\mathrm{COM}} =
            \frac{\sum_i m_i \mathbf{v}_i}{\sum_i m_i}

        where ``m_i`` is the total mass of model ``i`` and ``v_i`` is its velocity vector.

        Mass values are obtained in the following order:
          1. If ``masses`` is provided, use those values (must map model name → scalar mass).
          2. Otherwise, attempt to read ``total_mass`` from the model's metadata.

        Parameters
        ----------
        models : str or list of str, optional
            Which models to include in the COM velocity calculation. If ``"all"``,
            all models in :attr:`models` are used. If a single model name is given,
            it will be treated as a list of one.
        masses : dict of str, ~unyt.array.unyt_quantity, optional
            Optional mapping from model name to its total mass. Overrides
            ``total_mass`` in metadata if provided.

        Returns
        -------
        ~unyt.array.unyt_array
            The COM velocity vector with shape ``(ndim,)`` and units of velocity.

        Raises
        ------
        KeyError
            If any specified model name is not found in the initial conditions.
        ValueError
            If a required mass is missing from both ``masses`` and the model metadata.
        """
        # Normalize model list
        if models == "all":
            model_names = list(self.models.keys())
        elif isinstance(models, str):
            model_names = [models]
        elif isinstance(models, Sequence):
            model_names = list(models)
        else:
            raise TypeError("`models` must be 'all', a string, or a list of strings.")

        total_mass_sum = 0.0 * unyt.Unit("Msun")
        weighted_velocity_sum = unyt.unyt_array([0.0] * self.ndim, units="Msun*km/s")

        for name in model_names:
            if name not in self.models:
                raise KeyError(f"No model named '{name}' found in initial conditions.")

            # Determine mass
            if (masses is not None) and (name in masses):
                mass = masses[name]
                if not isinstance(mass, unyt.unyt_quantity):
                    mass = unyt.unyt_quantity(mass, "Msun")
            else:
                metadata = self.get_model_metadata(name)
                if "total_mass" not in metadata:
                    raise ValueError(
                        f"Model '{name}' is missing 'total_mass' in metadata and no override was provided."
                    )
                mass = metadata["total_mass"]
                if not isinstance(mass, unyt.unyt_quantity):
                    mass = unyt.unyt_quantity(mass, "Msun")

            vel = self.model_velocities[name]

            weighted_velocity_sum += mass * vel
            total_mass_sum += mass

        if total_mass_sum <= 0 * unyt.Unit("Msun"):
            raise ValueError("Total mass is zero; cannot compute center-of-mass velocity.")

        return weighted_velocity_sum / total_mass_sum

    def compute_center_of_mass_frame_velocities(
        self,
        models: Union[str, list[str]] = "all",
        masses: Optional[dict[str, "unyt.unyt_quantity"]] = None,
        com_velocity: Optional["unyt.unyt_array"] = None,
    ) -> dict[str, "unyt.unyt_array"]:
        """
        Compute the velocities of models in the center-of-mass (COM) frame.

        This subtracts the COM velocity (mass-weighted) from each model's stored
        velocity.

        Parameters
        ----------
        models : str or list of str, optional
            Models to include in the computation. Defaults to ``"all"`` for all models.
        masses : dict of str, ~unyt.array.unyt_quantity, optional
            Optional mapping from model name to its total mass, used for COM
            calculation. Overrides ``total_mass`` in metadata if provided.
        com_velocity : ~unyt.array.unyt_array, optional
            Precomputed COM velocity vector. If not given, it will be computed using
            :meth:`compute_center_of_mass_velocity` with the same ``models`` and ``masses``.

        Returns
        -------
        dict of str, ~unyt.array.unyt_array
            Mapping of model names to their velocity vectors in the COM frame.

        Raises
        ------
        KeyError
            If any specified model name is not found.
        ValueError
            If mass data is missing for a model and COM computation is required.
        """
        # Normalize model list
        if models == "all":
            model_names = list(self.models.keys())
        elif isinstance(models, str):
            model_names = [models]
        elif isinstance(models, Sequence):
            model_names = list(models)
        else:
            raise TypeError("`models` must be 'all', a string, or a list of strings.")

        # Compute COM velocity if not provided
        if com_velocity is None:
            com_velocity = self.compute_center_of_mass_velocity(model_names, masses=masses)

        com_velocity = unyt.unyt_array(com_velocity, units="km/s")

        com_frame_velocities: dict[str, unyt.unyt_array] = {}
        for name in model_names:
            if name not in self.models:
                raise KeyError(f"No model named '{name}' found in initial conditions.")
            vel = self.model_velocities[name]
            com_frame_velocities[name] = vel - com_velocity

        return com_frame_velocities

    def shift_to_COM_frame(
        self,
        models: Union[str, list[str]] = "all",
        masses: Optional[dict[str, "unyt.unyt_quantity"]] = None,
    ) -> None:
        """
        Shift the positions and velocities of models into the center-of-mass (COM) frame.

        This method:
          1. Computes the COM position and velocity for the specified models.
          2. Subtracts these values from each model's position and velocity.
          3. Updates the configuration in-place so the changes are persistent.

        Parameters
        ----------
        models : str or list of str, optional
            Models to include in the COM calculation and shifting. If ``"all"``
            (default), all models are included. If a single model name is given,
            it will be treated as a list of one.
        masses : dict of str, ~unyt.array.unyt_quantity, optional
            Optional mapping from model name to its total mass. Overrides
            ``total_mass`` in metadata if provided.

        Raises
        ------
        KeyError
            If any specified model name is not found in the initial conditions.
        ValueError
            If mass data is missing for a model and COM computation is required.
        """
        import unyt

        # Normalize model list
        if models == "all":
            model_names = list(self.models.keys())
        elif isinstance(models, str):
            model_names = [models]
        elif isinstance(models, Sequence):
            model_names = list(models)
        else:
            raise TypeError("`models` must be 'all', a string, or a list of strings.")

        # Compute COM position and velocity
        com_position = self.compute_center_of_mass(model_names, masses=masses)
        com_velocity = self.compute_center_of_mass_velocity(model_names, masses=masses)

        # Apply shifts and update config
        for name in model_names:
            if name not in self.models:
                raise KeyError(f"No model named '{name}' found in initial conditions.")

            pos_key = f"models.{name}.position"
            vel_key = f"models.{name}.velocity"

            pos = unyt.unyt_array(self.__config__[pos_key], units=self.model_positions[name].units)
            vel = unyt.unyt_array(self.__config__[vel_key], units=self.model_velocities[name].units)

            self.__config__[pos_key] = pos - com_position
            self.__config__[vel_key] = vel - com_velocity

        self.logger.info(
            f"Shifted {len(model_names)} models to COM frame: COM position={com_position}, COM velocity={com_velocity}"
        )

    def compute_total_mass(
        self,
        models: Union[str, list[str]] = "all",
        masses: Optional[dict[str, "unyt.unyt_quantity"]] = None,
    ) -> "unyt.unyt_quantity":
        """
        Compute the total mass of one or more models in the initial conditions.

        Mass values are determined in the same order as in
        :meth:`compute_center_of_mass`:

          1. If ``masses`` is provided, use those values (must map model name → scalar mass).
          2. Otherwise, attempt to read ``total_mass`` from the model's metadata via
             :meth:`get_model_metadata`.

        Parameters
        ----------
        models : str or list of str, optional
            Which models to include in the total mass calculation.
            If ``"all"`` (default), all models in :attr:`models` are used.
            If a single model name is provided as a string, it will be treated as a
            list containing that model.
        masses : dict of str, ~unyt.array.unyt_quantity, optional
            Optional mapping from model name to its total mass. If provided for a given
            model, this value overrides the ``total_mass`` from metadata.

        Returns
        -------
        ~unyt.array.unyt_quantity
            The total mass of the specified models, with units of mass.

        Raises
        ------
        KeyError
            If any specified model name is not found in the initial conditions.
        ValueError
            If a required mass is missing from both ``masses`` and the model metadata.
        """
        import unyt

        # Normalize model list
        if models == "all":
            model_names = list(self.models.keys())
        elif isinstance(models, str):
            model_names = [models]
        elif isinstance(models, Sequence):
            model_names = list(models)
        else:
            raise TypeError("`models` must be 'all', a string, or a list of strings.")

        total_mass_sum = 0.0 * unyt.Unit("Msun")

        for name in model_names:
            if name not in self.models:
                raise KeyError(f"No model named '{name}' found in initial conditions.")

            # Determine mass
            if masses is not None and name in masses:
                mass = masses[name]
                if not isinstance(mass, unyt.unyt_quantity):
                    mass = unyt.unyt_quantity(mass, "Msun")
            else:
                metadata = self.get_model_metadata(name)
                if "total_mass" not in metadata:
                    raise ValueError(
                        f"Model '{name}' is missing 'total_mass' in metadata and no override was provided."
                    )
                mass = metadata["total_mass"]
                if not isinstance(mass, unyt.unyt_quantity):
                    mass = unyt.unyt_quantity(mass, "Msun")

            total_mass_sum += mass

        return total_mass_sum


class InitialConditions1DCartesian(InitialConditionsCartesian):
    """
    Initial conditions for 1D Cartesian simulations.

    Specialized Cartesian subclass with ``ndim=1``. Models are placed along a
    one-dimensional line with scalar position and velocity. This is useful for
    toy models, 1D test problems, or simplified collapse/expansion scenarios.

    Extends :class:`InitialConditionsCartesian` with physics utilities for:
      - Mass-weighted center-of-mass (COM) position and velocity.
      - Shifting models into the COM frame.
      - Computing the total system mass.

    Models must include:
      - ``model_name`` and ``model`` (file reference).
      - ``position`` (1D vector with length units).
      - ``velocity`` (1D vector with length/time units).
    """

    # ============================== #
    # Class Flags                    #
    # ============================== #
    # These flags are easily modified settings that are used throughout
    # the base class and should be easily accessible for modification
    # by subclasses.
    _model_metadata_required_keys = ["position", "velocity"]
    _model_metadata_allowed_keys = ["particles"]
    _ndim = 1

    # ============================== #
    # Initialization Methods         #
    # ============================== #
    @classmethod
    def _validate_model(cls, model_name: str, model_info: dict) -> None:
        super()._validate_model(model_name, model_info)

    # ============================== #
    # Generator Methods              #
    # ============================== #
    # These methods are used to generate the skeleton of an initial conditions
    # object. The methods here should be overridden in subclasses to specialize
    # the behavior of the initial conditions for a specific simulation code.
    #
    # ``_process_models`` takes a set of models and associated metadata and
    #   proceeds to process them, returning a CONFIG-compatible dictionary of
    #   model names and their properties. It also moves or copies the model files
    #  into the initial conditions directory as needed.
    #
    @classmethod
    def _validate_input_model(cls, model: dict, existing_models: dict, **kwargs) -> dict:
        # Perform the super-class initialization to ensure that we
        # have the basic structure in place.
        model = super()._validate_input_model(model, existing_models, **kwargs)

        return model


class InitialConditions2DCartesian(InitialConditionsCartesian):
    """
    Initial conditions for 2D Cartesian simulations.

    Specialized Cartesian subclass with ``ndim=2``. Models are embedded in a
    planar (x, y) geometry and may include an orientation vector to define
    spin axes or angular alignment.

    Key features:
      - Validates that positions/velocities are 2D vectors with correct units.
      - Normalizes orientation vectors to unit length.
      - Provides ``model_orientations`` property for access.

    Models must include:
      - ``model_name``, ``model``, ``position``, ``velocity``.
      - Optional ``orientation`` (default = [0, 1]).
      - Optional ``particles``.
    """

    # ============================== #
    # Class Flags                    #
    # ============================== #
    # These flags are easily modified settings that are used throughout
    # the base class and should be easily accessible for modification
    # by subclasses.
    _model_metadata_required_keys = ["position", "velocity"]
    _model_metadata_allowed_keys = ["particles", "orientation"]
    _ndim = 2

    # ============================== #
    # Initialization Methods         #
    # ============================== #
    @classmethod
    def _validate_model(cls, model_name: str, model_info: dict) -> None:
        super()._validate_model(model_name, model_info)

    # ============================== #
    # Properties                     #
    # ============================== #
    @property
    def model_orientations(self) -> dict[str, np.ndarray]:
        """
        The orientations of the models in the initial conditions.

        Returns
        -------
        dict
            A dictionary mapping model names to their orientation vectors as `np.ndarray`.
        """
        return {name: np.asarray(info["orientation"], dtype=float) for name, info in self.models.items()}

    # ============================== #
    # Generator Methods              #
    # ============================== #
    # These methods are used to generate the skeleton of an initial conditions
    # object. The methods here should be overridden in subclasses to specialize
    # the behavior of the initial conditions for a specific simulation code.
    #
    # ``_process_models`` takes a set of models and associated metadata and
    #   proceeds to process them, returning a CONFIG-compatible dictionary of
    #   model names and their properties. It also moves or copies the model files
    #  into the initial conditions directory as needed.
    #
    @classmethod
    def _validate_input_model(cls, model: dict, existing_models: dict, **kwargs) -> dict:
        # Perform the super-class initialization to ensure that we
        # have the basic structure in place.
        model = super()._validate_input_model(model, existing_models, **kwargs)

        # Now we check the orientation to ensure that they are both valid and
        # correctly dimensioned. If they are not present, we set them to default values.
        model["orientation"] = np.asarray(model.get("orientation", np.asarray([0, 1], dtype="f8")))
        if model["orientation"].shape != (2,):
            raise ValueError(f"Orientation must be a 3D vector, not shape {model['orientation'].shape}.")
        if np.linalg.norm(model["orientation"]) <= 1e-8:
            raise ValueError("Orientation vector cannot be the zero vector.")
        model["orientation"] = model["orientation"] / np.linalg.norm(model["orientation"])

        return model


class InitialConditions3DCartesian(InitialConditionsCartesian):
    """
    Initial conditions for 3D Cartesian simulations.

    Specialized Cartesian subclass with ``ndim=3``. This is the most general
    Cartesian IC class, allowing full 3D positioning, velocities, orientations,
    and spins. It is designed for galaxy, cluster, and cosmological ICs where
    full spatial and kinematic degrees of freedom are needed.

    Key features:
      - Validates 3D position and velocity vectors.
      - Ensures orientation is a valid nonzero 3D unit vector.
      - Enforces ``spin`` to be a scalar float.
      - Provides convenience properties: ``model_spins``.

    Physics utilities include:
      - Integration of point-mass orbits with the `rebound` N-body package.
      - Future extensions may add angular momentum alignment, merging utilities,
        or orbit fitting.

    Models must include:
      - ``model_name``, ``model``, ``position``, ``velocity``.
      - Optional ``orientation`` (default = [0, 0, 1]).
      - Optional ``spin`` (default = 0.0).
      - Optional ``particles``.
    """

    # ============================== #
    # Class Flags                    #
    # ============================== #
    # These flags are easily modified settings that are used throughout
    # the base class and should be easily accessible for modification
    # by subclasses.
    _model_metadata_required_keys = ["position", "velocity"]
    _model_metadata_allowed_keys = ["particles", "orientation", "spin"]
    _ndim = 3

    # ============================== #
    # Initialization Methods         #
    # ============================== #
    @classmethod
    def _validate_model(cls, model_name: str, model_info: dict) -> None:
        super()._validate_model(model_name, model_info)

    # ============================== #
    # Properties                     #
    # ============================== #
    @property
    def model_spins(self) -> dict[str, float]:
        """
        The spins of the models in the initial conditions.

        Returns
        -------
        dict
            A dictionary mapping model names to their spin values as `float`.
        """
        return {name: float(info["spin"]) for name, info in self.models.items()}

    @property
    def model_orientations(self) -> dict[str, np.ndarray]:
        """
        The orientations of the models in the initial conditions.

        Returns
        -------
        dict
            A dictionary mapping model names to their orientation vectors as `np.ndarray`.
        """
        return {name: np.asarray(info["orientation"], dtype=float) for name, info in self.models.items()}

    # ============================== #
    # Generator Methods              #
    # ============================== #
    # These methods are used to generate the skeleton of an initial conditions
    # object. The methods here should be overridden in subclasses to specialize
    # the behavior of the initial conditions for a specific simulation code.
    #
    # ``_process_models`` takes a set of models and associated metadata and
    #   proceeds to process them, returning a CONFIG-compatible dictionary of
    #   model names and their properties. It also moves or copies the model files
    #  into the initial conditions directory as needed.
    #

    @classmethod
    def _validate_input_model(cls, model: dict, existing_models: dict, **kwargs) -> dict:
        # Perform the super-class initialization to ensure that we
        # have the basic structure in place.
        model = super()._validate_input_model(model, existing_models, **kwargs)

        # Now we check the orientation and the spin to ensure that they are both valid and
        # correctly dimensioned. If they are not present, we set them to default values.
        model["orientation"] = np.asarray(model.get("orientation", np.asarray([0, 0, 1], dtype="f8")))
        if model["orientation"].shape != (3,):
            raise ValueError(f"Orientation must be a 3D vector, not shape {model['orientation'].shape}.")
        if np.linalg.norm(model["orientation"]) <= 1e-8:
            raise ValueError("Orientation vector cannot be the zero vector.")
        model["orientation"] = model["orientation"] / np.linalg.norm(model["orientation"])

        # We now check the spin to ensure that it is a scalar value.
        model["spin"] = float(model.get("spin", 0.0))
        if not isinstance(model["spin"], float):
            raise TypeError(f"Spin must be a float, not {type(model['spin']).__name__}.")

        # Once we complete the validation, we return the model.
        return model

    # ============================== #
    # Physics Methods                #
    # ============================== #
    def integrate_point_mass_orbits(
        self,
        models: Union[str, list[str]] = "all",
        masses: Optional[dict[str, "unyt.unyt_quantity"]] = None,
        t_end: "unyt.unyt_quantity" = None,
        dt: Optional["unyt.unyt_quantity"] = None,
        integrator: str = "whfast",
    ):
        """
        Integrate the point-mass equivalent orbits for selected models.

        This method treats each selected model as a single massive particle with its
        center-of-mass position, velocity, and total mass. The integration is done
        in a self-consistent gravitational N-body simulation without hydrodynamics.

        Parameters
        ----------
        models : str or list of str, default="all"
            Models to include as point masses. If ``"all"``, uses all models.
            If a single string is given, it is treated as a list of one.
        masses : dict of str, ~unyt.array.unyt_quantity, optional
            Mapping from model name to its total mass. If provided for a given model,
            overrides the ``total_mass`` value from that model's metadata.
            All masses must be scalar `unyt_quantity` with units convertible to ``Msun``.
        t_end : ~unyt.array.unyt_quantity
            Total integration time **from the current epoch**. Must have time units.
        dt : ~unyt.array.unyt_quantity, optional
            Time step for the integrator. If not provided, defaults to ``t_end / 1000``.
            Must have time units.
        integrator : str, default="whfast"
            The REBOUND integrator to use. Options include:
              - ``"whfast"``: fast symplectic integrator (default)
              - ``"ias15"``: high-accuracy adaptive integrator
              - Other integrators supported by `rebound`

        Returns
        -------
        rebound.Simulation
            The `rebound` simulation object after integration to ``t_end``.
            Particle indices correspond to the order of models in the
            ``models`` parameter.

        Raises
        ------
        ImportError
            If the `rebound` package is not installed.
        KeyError
            If any requested model is not found in the configuration.
        ValueError
            If `t_end` is not provided, is non-positive, or if mass data is missing.
        """
        import unyt

        try:
            import rebound
        except ImportError as e:
            raise ImportError(
                "The `rebound` package is required for this method. Install it via: pip install rebound"
            ) from e

        # --- Normalize model selection
        if models == "all":
            model_names = list(self.models.keys())
        elif isinstance(models, str):
            model_names = [models]
        elif isinstance(models, Sequence):
            model_names = list(models)
        else:
            raise TypeError("`models` must be 'all', a string, or a list of strings.")

        if not model_names:
            raise ValueError("No models specified for integration.")

        # --- Validate time inputs
        if t_end is None:
            raise ValueError("`t_end` must be provided as a time quantity.")
        t_end = unyt.unyt_quantity(t_end)
        if t_end.units.dimensions != unyt.dimensions.time:
            raise ValueError("`t_end` must have time units.")
        if t_end <= 0 * t_end.units:
            raise ValueError("`t_end` must be positive.")

        if dt is None:
            dt = t_end / 1000
        dt = unyt.unyt_quantity(dt)
        if dt.units.dimensions != unyt.dimensions.time:
            raise ValueError("`dt` must have time units.")

        # --- Prepare model data
        positions = self.model_positions
        velocities = self.model_velocities
        total_masses: dict[str, unyt.unyt_quantity] = {}

        for name in model_names:
            if name not in self.models:
                raise KeyError(f"No model named '{name}' found in initial conditions.")

            # Determine mass (user-specified overrides metadata)
            if masses and name in masses:
                m_val = masses[name]
                if not isinstance(m_val, unyt.unyt_quantity):
                    m_val = unyt.unyt_quantity(m_val, "Msun")
            else:
                metadata = self.get_model_metadata(name)
                if "total_mass" not in metadata:
                    raise ValueError(f"Model '{name}' is missing 'total_mass' in metadata.")
                m_val = metadata["total_mass"]
                if not isinstance(m_val, unyt.unyt_quantity):
                    m_val = unyt.unyt_quantity(m_val, "Msun")

            total_masses[name] = m_val.to("Msun")

        # --- Build REBOUND simulation
        sim = rebound.Simulation()
        sim.units = ("pc", "Msun", "Myr")
        sim.integrator = integrator
        sim.dt = dt.to_value("Myr")

        for name in model_names:
            pos = positions[name].to("pc").value
            vel = velocities[name].to("pc/Myr").value
            m = total_masses[name].value
            sim.add(m=m, x=pos[0], y=pos[1], z=pos[2], vx=vel[0], vy=vel[1], vz=vel[2])

        # Shift to COM frame before integration
        sim.move_to_com()

        # --- Integrate
        sim.integrate(sim.t + t_end.to_value("Myr"))

        return sim


def load_ics(path: Union[str, Path]) -> InitialConditions:
    """
    Load an initial conditions directory from its configuration file.

    This function reads the ``IC_CONFIG.yaml`` file in the specified
    directory, inspects its metadata to determine the correct
    :class:`InitialConditions` subclass, and returns an instance
    of that class.

    Parameters
    ----------
    path : str or Path
        Path to the initial conditions directory containing an
        ``IC_CONFIG.yaml`` file.

    Returns
    -------
    InitialConditions
        An instance of the appropriate subclass populated from the file.

    Raises
    ------
    FileNotFoundError
        If the directory or its ``IC_CONFIG.yaml`` file does not exist.
    ValueError
        If the configuration is invalid or the class cannot be resolved.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Initial conditions directory does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Path must be a directory, not a file: {path}")

    config_path = path / "IC_CONFIG.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No 'IC_CONFIG.yaml' found in directory: {path}")

    try:
        with open(config_path) as f:
            config = InitialConditions.__YAML__.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse YAML configuration: {config_path}") from e

    metadata = config.get("metadata", {})
    class_name = metadata.get("class_name")
    if not class_name:
        raise ValueError(f"Configuration missing required 'metadata.class_name': {config_path}")

    # Recursive subclass lookup
    def _all_subclasses(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from _all_subclasses(sub)

    subclass_map = {cls.__name__: cls for cls in _all_subclasses(InitialConditions)}
    ics_class = subclass_map.get(class_name)
    if ics_class is None:
        raise ValueError(f"Unknown InitialConditions subclass '{class_name}' in {config_path}")

    return ics_class(path)
