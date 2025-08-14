"""
Core classes for facilitating simulation initial conditions.

This module defines the :class:`InitialConditions` class, which serves as a base for
all simulation initial conditions in Pisces. It allows users to insert models into
a 3D box, providing a framework for defining and manipulating initial conditions
in astrophysical simulations.

"""

import datetime
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import unyt

from pisces.models.core.base import BaseModel
from pisces.models.core.utils import load_model
from pisces.utilities.config import ConfigManager
from pisces.utilities.io_tools import unyt_yaml
from pisces.utilities.log import LogDescriptor

if TYPE_CHECKING:
    from logging import Logger

    from pisces.particles.base import ParticleDataset


class InitialConditions:
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
          3. Confirms that the `ndim` field exists in the metadata and that all models
             have matching position dimensionality.
          4. Validates that all referenced model and particle files exist on disk.

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

        # Check the NDIM field.
        if "ndim" not in metadata:
            raise ValueError("Metadata is missing required 'ndim' field.")

        # As long as these pass, we consider the configuration valid. We now need to
        # validate that the models in models each are valid. That is done
        # separately in the __init__ method.
        self.logger.debug(f"{self} passed configuration validation.")

    @classmethod
    def _validate_model(cls, model_name: str, model_info: dict, expected_ndim: int) -> None:
        """
        Validate a single model entry for correctness.

        This method ensures that:

          1. All required keys are present.
          2. The main model file path exists on disk.
          3. Vector-valued fields (`position`, `velocity`, `orientation`) have the
             correct dimensionality (either 1-D of length `ndim` or an `ndim × ndim`
             rotation matrix for orientation).
          4. If a particle file is referenced, it exists on disk.

        Parameters
        ----------
        model_name : str
            Name/identifier of the model in the configuration.
        model_info : dict
            Dictionary of model attributes from the IC configuration.
        expected_ndim : int
            Expected number of spatial dimensions for vector fields.

        Raises
        ------
        ValueError
            If required keys are missing or vector shapes are inconsistent with
            `expected_ndim`.
        FileNotFoundError
            If referenced model or particle files do not exist.
        """
        # Ensure that the model info contains all of the required keys. If not,
        # we need to raise an error letting the user know what's missing.
        _required_model_keys = ("path", "position", "velocity", "orientation", "spin")
        for key in _required_model_keys:
            if key not in model_info:
                raise ValueError(f"Model '{model_name}' is missing required key '{key}'.")

        # Ensure that the model path actually exists on disk.
        model_path = Path(model_info["path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file for '{model_name}' not found: {model_info['path']}")

        # Check that all of the vector values in the model configuration are
        # correctly dimensioned for this scenario. If not, we raise an error.
        _required_vector_keys = ("position", "velocity", "orientation")
        for _required_vector_key in _required_vector_keys:
            # obtain the shape of the relevant vector and ensure
            # it matches the expected dimensionality.
            vec = model_info[_required_vector_key]
            shape = getattr(vec, "shape", ())
            if shape != (expected_ndim,):
                raise ValueError(
                    f"Model '{model_name}' has invalid '{_required_vector_key}' shape {shape}; "
                    f"expected shape ({expected_ndim},)."
                )

        # Now check that the particle file (if it is specified) is actually
        # a real file.
        if "particles" in model_info and model_info["particles"] is not None:
            particle_path = Path(model_info["particles"])
            if not particle_path.exists():
                raise FileNotFoundError(f"Particle file for '{model_name}' not found: {particle_path}")

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
        self.logger.info("Loading IC object from directory: %s", self.__directory__.absolute())

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
            self._validate_model(model_name, model_info, expected_ndim=self.ndim)

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
    def ndim(self) -> int:
        """
        The number of dimensions for the initial conditions.

        Returns
        -------
        int
            The number of dimensions (e.g., 3 for 3D).
        """
        return int(self.__config__["metadata.ndim"])

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

    # ============================== #
    # MAGIC Methods                  #
    # ============================== #
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(dir='{self.__directory__}', ndim={self.ndim}, models={len(self.models)})>"

    def __str__(self) -> str:
        """Human-readable summary of the InitialConditions object."""
        return f"{self.__class__.__name__}({self.__directory__.absolute()})"

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
        for key in ("path", "particles"):
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
        model_config,
        file_processing_mode: str = "copy",
        overwrite: bool = False,
    ):
        """
        Add a new model to these initial conditions.

        Parameters
        ----------
        name : str
            Unique name/identifier for the model within the initial conditions set.
            If the provided name is not unique, then ``overwrite`` will determine if
            an error is raised or if the existing model is replaced.
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
            A tuple specifying the model configuration in the format

            .. code-block:: python

                (position, velocity[, orientation][, spin])

            Where:

            - ``position`` : sequence of length ``ndim`` or unyt array with shape (ndim,)
            - ``velocity`` : sequence of length ``ndim`` or unyt array with shape (ndim,)
            - ``orientation`` : optional; sequence or array defining orientation
            - ``spin`` : optional; scalar float

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
        if not isinstance(model_config, tuple):
            model_config = (model_config,)
        processed = self._process_models(
            self.directory, (name, model, *model_config), file_processing_mode=file_processing_mode
        )

        self.__config__["models"].update(processed)
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
            Key-value pairs of parameters to update. Supported keys include:

            - ``position`` : unyt array or sequence of length ``ndim``
            - ``velocity`` : unyt array or sequence of length ``ndim``
            - ``orientation`` : sequence or array defining orientation vector
            - ``spin`` : scalar float

        Raises
        ------
        KeyError
            If no model with the given ``model_name`` exists.
        ValueError
            If provided parameters are invalid or inconsistent with expected dimensions/units.
        """
        # Ensure model exists
        if model_name not in self.models:
            raise KeyError(f"No model named '{model_name}' found in initial conditions.")

        ndim = self.ndim
        model_info = self.__config__[f"models.{model_name}"]

        for key, value in parameters.items():
            if key == "position":
                arr = unyt.unyt_array(value, units="m") if not isinstance(value, unyt.unyt_array) else value
                if arr.shape != (ndim,):
                    raise ValueError(f"Position must have shape ({ndim},), got {arr.shape}.")
                model_info[key] = arr

            elif key == "velocity":
                arr = unyt.unyt_array(value, units="km/s") if not isinstance(value, unyt.unyt_array) else value
                if arr.shape != (ndim,):
                    raise ValueError(f"Velocity must have shape ({ndim},), got {arr.shape}.")
                model_info[key] = arr

            elif key == "orientation":
                arr = np.asarray(value, dtype=float)
                if arr.shape != (ndim,):
                    raise ValueError(f"Orientation must have shape ({ndim},), got {arr.shape}.")
                norm = np.linalg.norm(arr)
                if norm == 0:
                    raise ValueError("Orientation vector cannot be zero.")
                model_info[key] = arr / norm  # Normalize

            elif key == "spin":
                try:
                    model_info[key] = float(value)
                except (TypeError, ValueError) as exp:
                    raise ValueError(f"Spin must be a scalar float, got {value!r}.") from exp

            else:
                raise ValueError(f"Unsupported model parameter: '{key}'.")

        self.logger.info(f"Updated parameters for model '{model_name}': {list(parameters.keys())}")

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

    def get_model_grid(self, model_name: str):
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
    # Methods - Particle Interaction   #
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
        if model_name not in self.models:
            raise KeyError(f"No model named '{model_name}' found in initial conditions.")

        particle_path = Path(particle_path)
        if not particle_path.exists() or not particle_path.is_file():
            raise FileNotFoundError(f"Particle file '{particle_path}' does not exist.")

        dest_path = self.directory / f"{model_name}_p.hdf5"

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
        # noinspection PyUnresolvedReferences
        model.generate_particles(self.__directory__ / "f{model_name}_p.hdf5", num_particles, **kwargs)

        self.logger.info(f"Generated particles for model '{model_name}' with counts: {num_particles}")

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

    # ============================== #
    # Generator Methods              #
    # ============================== #
    # These methods are used to generate the initial conditions
    # from a set of models and particle datasets.
    @classmethod
    def _process_models(cls, directory, *models, **kwargs):
        """
        Process the list of models to generate initial conditions.

        Parameters
        ----------
        directory : ~pathlib.Path
            Directory where the models will be stored or processed.
        models : tuple
            Each model should be specified as:
            (name, model, position, velocity[, orientation][, spin])
        kwargs :
            Additional keyword arguments, such as:
            - ndim (int): The number of dimensions for the models (default is 3).

        Returns
        -------
        dict
            Dictionary mapping model names to their processed properties.
        """
        # Start by ensuring that the models are all specified in a valid format.
        # That means we require (at a minimum) the model name, the model (or its path),
        # its position, and its velocity. We allow the orientation and spin to be optional
        # in each model.
        processed_models = {}

        for model_tuple in models:
            # --- Unpack the model tuple --- #
            try:
                model_name, model, position, *extra_params = model_tuple
            except ValueError as exp:
                raise ValueError(
                    f"Invalid model specification: {model_tuple}.\n"
                    "Models should be specified as "
                    "(name, model, position, velocity[, orientation][, spin])."
                ) from exp

            # --- Validate and uniquify the model name --- #
            # If the name already exists in the set, append a numeric suffix.
            model_name = str(model_name)
            if model_name in processed_models:
                suffix = 1
                new_name = f"{model_name}_{suffix}"
                while new_name in processed_models:
                    suffix += 1
                    new_name = f"{model_name}_{suffix}"
                model_name = new_name

            # --- Validate and resolve the model --- #
            # If given a path, load the model from disk; if a BaseModel, use directly.
            if isinstance(model, (str, Path)):
                model_path = Path(model)
                model = load_model(model_path)
            elif isinstance(model, BaseModel):
                model_path = Path(model.__path__)
            else:
                raise TypeError(f"Model must be a string, Path, or BaseModel instance, not {type(model).__name__}.")

            if not model_path.exists():
                raise FileNotFoundError(f"Model path '{model_path}' does not exist.")

            model_ndim = model.coordinate_system.ndim
            ndim = kwargs.get("ndim", 3)
            if model_ndim != ndim:
                raise ValueError(f"Model '{model_name}' has {model_ndim} dimensions, but expected {ndim}.")

            processed_models[model_name] = {"path": model_path}

            # --- Validate the position vector --- #
            # Must be a unyt_array with length units and shape (ndim,).
            if not isinstance(position, unyt.unyt_array):
                raise TypeError(f"Position must be a unyt.unyt_array, not {type(position).__name__}.")
            if position.units.dimensions != unyt.dimensions.length:
                raise TypeError(f"Position must have units of length, not {position.units.dimensions}.")
            if position.shape != (ndim,):
                raise ValueError(f"Position must have shape ({ndim},), not {position.shape}.")

            processed_models[model_name]["position"] = position

            # --- Validate or set default velocity --- #
            if extra_params:
                velocity, *extra_params = extra_params
            else:
                velocity = unyt.unyt_array([0] * ndim, units="km/s")

            if not isinstance(velocity, unyt.unyt_array):
                raise TypeError(f"Velocity must be a unyt.unyt_array, not {type(velocity).__name__}.")
            if velocity.units.dimensions != (unyt.dimensions.length / unyt.dimensions.time):
                raise TypeError(f"Velocity must have length/time units, not {velocity.units.dimensions}.")
            if velocity.shape != (ndim,):
                raise ValueError(f"Velocity must have shape ({ndim},), not {velocity.shape}.")

            processed_models[model_name]["velocity"] = velocity

            # --- Handle the orientation vector --- #
            # Default: aligned in standard orientation (unit vector along last axis).
            if extra_params:
                orientation, *extra_params = extra_params
                orientation = np.asarray(orientation, dtype=float)
                if orientation.shape != (ndim,):
                    raise ValueError(f"Orientation must have shape ({ndim},), not {orientation.shape}.")
            else:
                orientation = np.zeros(ndim, dtype=float)
                orientation[-1] = 1.0  # Unit vector along last coordinate axis

            # Normalize the orientation vector to unit length.
            orientation /= np.linalg.norm(orientation)
            processed_models[model_name]["orientation"] = orientation

            # --- Handle the spin parameter --- #
            # Default: zero spin if not provided.
            if extra_params:
                spin, *extra_params = extra_params
                try:
                    spin = float(spin)
                except (TypeError, ValueError) as exp:
                    raise ValueError(f"Spin must be convertible to float, got {type(spin).__name__}.") from exp
            else:
                spin = 0.0

            processed_models[model_name]["spin"] = spin

            # Logging
            cls.logger.debug(f"Added model '{model_name}' at position {position}.")

        # --- Move / Manage the Model Files --- #
        # With the models processed, we can check if we need to copy or move the
        # models into the directory.
        _mode = kwargs.get("file_processing_mode", "copy")
        if _mode == "copy":
            # We now move a copy of each of the model files into the directory
            # and rename with the model name provided to use by the user.
            for mname, minfo in processed_models.items():
                model_path = minfo["path"]
                new_model_path = directory / f"{mname}.hdf5"
                shutil.copy(model_path, new_model_path)
                minfo["path"] = new_model_path

        elif _mode == "move":
            # We move the model files and then rename them.
            for mname, minfo in processed_models.items():
                model_path = minfo["path"]
                new_model_path = directory / f"{mname}.hdf5"
                shutil.move(model_path, new_model_path)
                minfo["path"] = new_model_path
        else:
            raise ValueError(f"Invalid file processing mode: {_mode}. Must be 'copy' or 'move'.")

        return processed_models

    @classmethod
    def _process_particle_files(
        cls, directory: Path, models: dict[str, dict], particle_files: dict[str, Union[str, Path]], **kwargs
    ) -> dict[str, dict]:
        # --- Setup --- #
        # Setup the procedure and fetch relevant parameters.
        mode = kwargs.get("file_processing_mode", "copy").lower()
        if mode not in ("copy", "move"):
            raise ValueError(f"Invalid file processing mode: {mode!r}. Must be 'copy' or 'move'.")

        # --- Validate Particle Files --- #
        # This method processes the particle files provided by the user.
        # It checks that the files exist, are valid, and then copies or moves them
        # into the target directory, renaming them to match the model names.
        for model_name, particle_file in particle_files.items():
            # Ensure that the model name actually exists in the models dictionary.
            if model_name not in models:
                raise ValueError(
                    f"Particle dataset provided for unknown model '{model_name}'. "
                    f"Ensure the model name matches one from the processed models."
                )

            # Ensure that the particle's path actually exists and is a file.
            src_path = Path(particle_file)
            if not src_path.exists() or not src_path.is_file():
                raise FileNotFoundError(f"Particle file '{src_path}' does not exist or is not a file.")

            # Now copy / move the file to the directory and add it as
            # the particles field in the processed models.
            dest_path = directory / f"{model_name}_p.hdf5"

            if mode == "copy":
                shutil.copy(src_path, dest_path)
            elif mode == "move":
                shutil.move(src_path, dest_path)
            else:
                raise ValueError(f"Invalid file processing mode: {mode}. Must be 'copy' or 'move'.")

            # add the particle file path to the model's info.
            models[model_name]["particles"] = dest_path

        return models

    @classmethod
    def _process_metadata(cls, directory: Path, **kwargs) -> dict:
        """
        Generate metadata for the initial conditions set.

        Parameters
        ----------
        directory : ~pathlib.Path
            The directory where ICs are being created.
        kwargs : dict
            Additional keyword arguments (e.g., ndim) that may be used.

        Returns
        -------
        dict
            Metadata dictionary to be stored alongside model data.
        """
        ndim = kwargs.get("ndim", 3)
        timestamp = datetime.datetime.now().isoformat() + "Z"

        metadata = {
            "metadata": {
                "created_at": timestamp,
                "class_name": cls.__name__,
                "directory": str(directory.resolve()),
                "ndim": ndim,
            }
        }

        return metadata

    @classmethod
    def create_ics(
        cls, directory: Union[str, Path], *models, particle_files: dict[str, Union[str, Path]] = None, **kwargs
    ):
        """
        Create a new initial conditions (IC) directory with optional particle datasets.

        This is a convenience constructor for building an `InitialConditions`
        instance from scratch. It will:

          1. Create (or overwrite) the target IC directory on disk.
          2. Process and validate the provided models via
             ``_process_models``, copying or moving model files into the IC
             directory.
          3. Optionally process particle dataset files via
             ``_process_particle_files`` if ``particle_files`` is provided.
          4. Generate the ``IC_CONFIG.yaml`` file containing metadata and model
             configuration.

        Parameters
        ----------
        directory : str or ~pathlib.Path
            Path to the directory where the new initial conditions will be created.
            If the directory already exists and is non-empty, ``overwrite=True`` must
            be provided in ``kwargs`` to remove its contents before creation.
        *models : tuple
            One or more model specifications to include in the initial conditions.
            Each model specification should be a tuple of the form:

            .. code-block:: python

                (name, model, position, velocity[, orientation][, spin])

            Where:

              - ``name`` : str
                Unique name/identifier for the model.
              - ``model`` : str, ~pathlib.Path or ~pisces.models.core.base.BaseModel
                Path to a model file on disk **or** an already loaded
                :class:`~pisces.models.core.base.BaseModel` instance.
              - ``position`` : sequence or ~unyt.array.unyt_array
                Position vector of length ``ndim`` with length units (default: meters).
              - ``velocity`` : sequence or ~unyt.array.unyt_array
                Velocity vector of length ``ndim`` with velocity units (default: km/s).
              - ``orientation`` : optional, sequence or array
                Orientation vector (shape: ``(ndim,)``) or rotation matrix
                (shape: ``(ndim, ndim)``). If omitted, the identity is used.
              - ``spin`` : optional, float
                Scalar spin value (unitless). Defaults to ``0.0``.

        particle_files : dict of {str: (str or Path)}, optional
            Mapping from model name to path to a particle dataset file.
            Only processed if provided. Files will be copied or moved into the
            IC directory with the naming scheme ``<model_name>_p.hdf5``.
        **kwargs :
            Additional keyword arguments forwarded to ``_process_models`` and
            ``_process_particle_files``. Common options include:

              - ``file_processing_mode`` : {"copy", "move"}
                Whether to copy (default) or move files into the IC directory.
              - ``overwrite`` : bool
                If ``True``, existing files or directories will be overwritten.

        Returns
        -------
        InitialConditions
            An initialized :class:`InitialConditions` instance for the newly
            created directory.

        Raises
        ------
        FileExistsError
            If ``directory`` exists and is non-empty, and ``overwrite`` is not True.
        FileNotFoundError
            If any provided model or particle file does not exist.
        ValueError
            If model definitions are invalid or missing required parameters.

        """
        # --- DIRECTORY SETUP --- #
        # Process the provided directory. We check that it is a valid directory
        # and that it doesn't contain any existing files that need to be overwritten.
        cls.logger.info("Creating initial conditions in directory: %s", directory)
        directory = Path(directory)

        if not directory.exists():
            # Path is new, we just create it.
            directory.mkdir(parents=True, exist_ok=True)
            cls.logger.debug("Created new directory: %s", directory)
        else:
            # Path exists, if we have content, then we need to check overwrite.
            _has_content = any(directory.iterdir())
            if _has_content and kwargs.get("overwrite", False):
                shutil.rmtree(directory)
                directory.mkdir(parents=True, exist_ok=True)
                cls.logger.debug("Overwriting existing directory: %s", directory)
            elif _has_content:
                raise FileExistsError(
                    f"Directory `{directory}` already exists and is not empty. To overwrite, set `overwrite=True`."
                )
            else:
                # The directory exists but is empty, we can use it.
                pass

        # --- MODEL MANAGEMENT --- #
        # Taking the models provided by the user, we process them to get the
        # processed models dictionary. This is a somewhat involved process of validation.
        processed_models = cls._process_models(directory, *models, **kwargs)

        # --- Particle Files --- #
        # If the user provides particle files, we can process them now
        # and do the same copy/move operation we did for the models.
        if particle_files:
            processed_models = cls._process_particle_files(directory, processed_models, particle_files, **kwargs)

        # --- Create the IC_CONFIG File --- #
        # We create a configuration file that contains the processed models
        # and their properties. This is a simple YAML file that can be read later.
        with open(directory / "IC_CONFIG.yaml", "w") as f:
            metadata = cls._process_metadata(directory, **kwargs)
            metadata["models"] = processed_models
            cls.__YAML__.dump(metadata, f)

        # Return the class initialized with the directory.
        cls.logger.info("Initial conditions created successfully in %s", directory)
        return cls(directory)
