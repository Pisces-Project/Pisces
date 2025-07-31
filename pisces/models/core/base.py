"""Base model abstraction for Pisces HDF5-backed physical models.

This module defines the :class:`BaseModel`, a foundational abstraction for all
Pisces-compatible models that are stored and accessed via HDF5 files. It provides
a unified interface for loading, saving, introspecting, and interacting with model
components. For more details on the Pisces model format and conventions, see :ref:`models_overview`.
"""

import datetime
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np
from unyt import unyt_array

from pisces.profiles.base import BaseProfile
from pisces.utilities.io_tools import HDF5Serializer
from pisces.utilities.log import LogDescriptor

from .hooks import _HookTools
from .utils import ModelConfig

if TYPE_CHECKING:
    from logging import Logger


class BaseModel(_HookTools, ABC):
    """Abstract base class for Pisces-compatible physical models.

    The :class:`BaseModel` defines the foundational logic and storage conventions for
    HDF5-backed physical models used throughout the Pisces framework. It provides
    a consistent interface for:

    - Loading and validating model files
    - Accessing physical fields, analytic profiles, and metadata
    - Saving models from in-memory components
    - Subclass extension via hooks and typed configuration

    Model Structure
    ---------------
    Pisces models are stored as structured HDF5 files with the following layout:

    .. code-block:: text

        /
        ├── FIELDS/        # Physical field arrays (e.g., density, temperature)
        ├── PROFILES/      # Serialized analytic profile definitions
        └── attrs/         # Model metadata and identity tags

    Each subclass is expected to match this layout and may define additional groups
    or attributes as needed. At minimum, each model must store its `__model_class__`
    identifier at the root to verify identity during loading.

    Access Patterns
    ---------------
    This class supports dictionary-like access and assignment:

    .. code-block:: python

        model["density"]  # returns field array
        model["temperature"]  # returns profile object
        model["description"]  # returns metadata string

        "pressure" in model  # checks all three namespaces
        model["mass"] = new_arr  # updates field

    Subclassing Guidelines
    ----------------------
    To implement a new model type:

    1. Subclass `BaseModel` and define the appropriate sampling or analysis methods.
    2. Optionally override `__post_init__()` to customize setup after loading.
    3. Optionally override `_construct_model_file_skeleton()` to define a custom layout.
    4. Register the model in your configuration under `[models.MyModelClass]`.
    """

    logger: "Logger" = LogDescriptor(mode="models")
    """
    Logger interface for this model.

    This logger provides structured logging functionality specific to the model's
    operation within the Pisces framework. It captures status updates, warnings,
    debug information, and other messages emitted during model construction,
    file I/O, sampling, or analysis.

    Use this logger for:

    - Reporting progress or diagnostics in custom model methods
    - Emitting debug messages during development or profile tuning
    - Logging warnings for missing data, invalid configuration, etc.

    Access the logger via:

    .. code-block:: python

        self.logger.info("Loaded profile '%s'", name)
        self.logger.warning("Missing metadata for '%s'", key)

    The logger is registered under the ``models`` subsystem. You can configure
    its behavior (e.g., log level or format) by editing the ``[logging.models]``
    section in your Pisces configuration file.
    """
    config: dict = ModelConfig()
    """
    Model-specific configuration settings.

    This dictionary provides access to optional, model-scoped settings defined
    in the Pisces configuration system. Each model class can retrieve its default
    parameters from the global config file under the section:

    .. code-block:: yaml

        models:
          MyGalaxyClusterModel:
            sampling_resolution: 1000
            enable_logging: true

    Access the config like a normal dictionary:

    .. code-block:: python

        resolution = self.config.get("sampling_resolution", 1000)
        enable_logging = self.config.get("log_fields", True)

    This allows you to customize model behavior without hardcoding values into
    your Python code. If the model class is not registered in the config file,
    accessing `self.config` will raise an error with guidance.
    """
    metadata_serializer: HDF5Serializer = HDF5Serializer
    """~pisces.utilities.io.HDF5Serializer: Serializer for model metadata.

    This serializer is used to convert complex Python objects (like `unyt` arrays,
    quantities, and custom types) into JSON-compatible formats that can be stored
    as attributes in HDF5 files. It provides methods for serializing and deserializing
    model metadata, ensuring that all necessary information can be preserved and
    reconstructed when loading models.

    The serializer for a particular model can be customized by subclassing
    `HDF5Serializer` and overriding the `metadata_serializer` property in your model
    class. This allows you to define custom serialization logic for any additional
    types or structures you need to store in the model's metadata.
    """

    # ========================================= #
    # Initialization and Configuration          #
    # ========================================= #
    def __read_fields__(self) -> dict[str, unyt_array | np.ndarray]:
        """Load the physical fields from the Pisces model file.

        This method reads all physical fields stored in the model file and returns
        them as a dictionary mapping field names to arrays (with or without units).
        It is intended for internal use during model loading and assumes compliance
        with Pisces model HDF5 layout conventions.

        Pisces Format Standard
        ----------------------
        Pisces models store all field data under the `/FIELDS` group of the HDF5 file.
        Each field is saved as an individual HDF5 dataset with the following conventions:

        - **Dataset Name**: The name of the dataset should match the field name (e.g., "density").
        - **Units Attribute** (optional): If present, the dataset must define a `UNITS` attribute
          as a string parsable by `unyt.Unit`. If no units are found, the field is treated
          as unitless.

        Field arrays should be broadcastable to the model's spatial grid and are typically
        1D, 2D, or 3D arrays depending on model resolution and dimensionality.

        Developer Guidance
        ------------------
        When implementing or modifying this method for a custom subclass:

        - Validate that `/FIELDS` exists before attempting to access datasets.
        - For each dataset:
            * Extract the raw data using slicing (`[...]`).
            * Check for a `units` attribute and apply it using `unyt_array(...)`.
        - Store each field under its corresponding name in a `dict`.

        Logging
        -------
        A debug log message will be emitted for each field successfully loaded.
        If the `/FIELDS` group is missing entirely, a warning is issued.

        Returns
        -------
        dict of str, unyt_array
            A dictionary containing all loaded field arrays, indexed by field name.

        Examples
        --------
        .. code-block:: text

            /FIELDS
                ├── density       [shape: (128,), units: "Msun/kpc**3"]
                ├── temperature   [shape: (128,), units: "keV"]
                └── pressure      [shape: (128,), units: "dyne/cm**2"]

        This would result in:

        .. code-block:: python

            {
                "density": unyt_array([...], "Msun/kpc**3"),
                "temperature": unyt_array([...], "keV"),
                "pressure": unyt_array([...], "dyne/cm**2"),
            }

        """
        # Create the buffer into which we'll dump the data.
        fields = {}

        # Now identify the FIELDS group and start rolling.
        if "FIELDS" in self.__handle__:
            for field_name, field_data in self.__handle__["FIELDS"].items():
                # fetch units.
                units = field_data.attrs.get("units", None)

                if units is None:
                    fields[field_name] = np.array(field_data[...])
                else:
                    fields[field_name] = unyt_array(field_data[...], units)
                self.logger.debug("Loaded field: '%s'. Units=%s", field_name, str(units))
        else:
            self.logger.warning("No `FIELDS` group found in model file '%s'.", self.__path__)

        return fields

    def __read_metadata__(self) -> dict[str, Any]:
        """Load model metadata from the root-level attributes of the Pisces HDF5 model file.

        This method extracts metadata stored in the root-level attributes (`attrs`) of
        the HDF5 file and returns it as a dictionary. These metadata values serve as
        persistent identifiers, descriptive tags, or build-time diagnostics that are
        relevant to the model as a whole.

        Pisces Format Standard
        ----------------------
        Metadata in Pisces models is stored at the **file root level** as plain attributes.
        Each entry should be a simple key-value pair compatible with HDF5 attribute types,
        such as:

        - Strings (e.g., `"description"`, `"author"`)
        - Numbers (int, float)
        - Dates or timestamps (typically stored as ISO strings)
        - Small arrays (e.g., version tags or configuration flags)

        Required Keys (by Convention)
        -----------------------------
        While not strictly enforced, Pisces models generally include:

        - `"__model_class__"` : Name of the class that generated the model.
        - `"date_created"` : ISO timestamp of model file creation.
        - `"source"` : Optional tag indicating model origin or provenance.
        - `"description"` : Optional summary of the model contents.

        These values are typically injected at model export time (e.g., via `from_components()`).

        Developer Guidance
        ------------------
        When extending this method or modifying metadata formats:

        - Avoid parsing or interpreting values at load time—just read and return raw values.
        - Do not hardcode expected keys unless your subclass guarantees them.
        - If the file lacks metadata entirely, this method should return an empty dictionary.

        Logging
        -------
        This method is intentionally silent (no logging), since absence of metadata
        may be normal in some intermediate outputs or synthetic models.

        Returns
        -------
        A dictionary of all root-level metadata attributes stored in the model file.

        Example Layout
        --------------
        .. code-block:: text

            /
            ├── FIELDS/
            ├── PROFILES/
            └── attrs:
                ├── "__model_class__" = "SphericalGalaxyClusterModel"
                ├── "date_created"    = "2025-07-24T15:31:10.103Z"
                ├── "description"     = "Hydrostatic mass test model."
                └── "source"          = "test_suite"

        This results in:

        .. code-block:: python

            {
                "__model_class__": "SphericalGalaxyClusterModel",
                "date_created": "2025-07-24T15:31:10.103Z",
                "description": "Hydrostatic mass test model.",
                "source": "test_suite",
            }

        """
        return self.__class__.metadata_serializer.deserialize_dict(dict(self.__handle__.attrs))

    def __read_profiles__(self) -> dict[str, "BaseProfile"]:
        """Load all analytic profiles from the `/PROFILES` group of the model file.

        This method reads all stored analytic profiles in the Pisces model file
        and reconstructs them using the standardized class method
        :meth:`BaseProfile.from_hdf5`. Each profile is loaded into memory and
        returned as part of a dictionary keyed by its group name.

        Pisces Format Standard
        ----------------------
        Analytic profiles are stored in a dedicated group at `/PROFILES` in the HDF5 model.
        Each profile is represented as a subgroup under `/PROFILES`, where:

        - The **key** is the profile name (e.g., `"density"`, `"temperature"`).
        - The **value** is an HDF5 group containing serialized data and metadata
          as required by the corresponding `BaseProfile` subclass.

        The structure and interpretation of each profile group is delegated to the
        :meth:`BaseProfile.from_hdf5` loader, which must be implemented for all
        profile subclasses.

        Example Structure
        -----------------
        .. code-block:: text

            /
            ├── FIELDS/
            ├── PROFILES/
            │   ├── density/
            │   │   ├── parameters
            │   │   ├── metadata
            │   ├── temperature/
            │   │   ├── parameters
            │   │   ├── metadata
            └── attrs/

        The resulting dictionary will contain:

        .. code-block:: python

            {
                "density": <SphericalNFWProfile(...)>,
                "temperature": <VikhlininProfile(...)>,
            }

        Developer Notes
        ---------------
        - This method is silent if the `/PROFILES` group is missing, but logs a warning.
        - If subclassed, developers may override this method to implement custom
          logic for filtering or categorizing profiles.
        - If profiles are optional for a given model class, ensure your code gracefully
          handles an empty return.

        Logging
        -------
        - Each loaded profile logs a DEBUG-level message with its name.
        - Missing `/PROFILES` group triggers a WARNING message.

        Returns
        -------
        dict of str -> BaseProfile
            Dictionary of profile names to fully instantiated profile objects.

        """
        profiles = {}
        if "PROFILES" in self.__handle__:
            for profile_name, profile_data in self.__handle__["PROFILES"].items():
                profiles[profile_name] = BaseProfile.from_hdf5(profile_data, profile_name)
                self.logger.debug("Loaded profile: '%s'.", profile_name)
        else:
            self.logger.warning("No `PROFILES` group found in model file '%s'.", self.__path__)

        return profiles

    def __init__(self, filepath: str | Path, *args, **kwargs):
        """Load a Pisces model from an existing HDF5 file.

        This loads the model from disk, verifies that the file matches the expected
        model class, and reads its core components:

        - Metadata (from root attributes)
        - Analytical profiles (from ``/PROFILES``)
        - Physical field data (from ``/FIELDS``)

        Subclasses may extend loading behavior by overriding :meth:`__post_init__` or
        modifying the ``__init__`` method directly.

        Parameters
        ----------
        filepath : str or Path
            Path to the existing HDF5 model file.
        *args, **kwargs :
            Optional arguments forwarded to :meth:`__post_init__`.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        TypeError
            If the file does not belong to this model class.

        """
        # Configure the path, ensure it is a Path object,
        # and then check that it exists before opening it.
        self.__path__ = Path(filepath)

        if not self.__path__.exists():
            raise FileNotFoundError(f"File {self.__path__} does not exist.")

        self.__handle__ = h5py.File(self.__path__, mode="r+")

        # Verify that the file was generated by this model class. This
        # ensures that we cannot open a model with the wrong model class.
        model_tag = self.metadata_serializer.deserialize_data(self.__handle__.attrs.get("__model_class__", None))
        if model_tag != self.__class__.__name__:
            raise TypeError(
                f"File {self.__path__} is not a valid model of type {self.__class__.__name__} (found tag: {model_tag})."
            )

        # --- Data Loading --- #
        # This is the meaty bit of the loading process. We read
        # the metadata, profiles, and fields before passing off to
        # the post init hook. Subclasses may override any or all of
        # these methods to customize the loading process.
        self.__metadata__: dict[str, Any] = self.__read_metadata__()
        self.__profiles__: dict[str, BaseProfile] = self.__read_profiles__()
        self.__fields__: dict[str, unyt_array] = self.__read_fields__()

        # Allow subclass extensions
        self.__post_init__(*args, **kwargs)

        # Log model load
        self.logger.info(
            "Loaded model from '%s' [%s]: %d fields, %d profiles, %d metadata entries.",
            str(self.__path__),
            self.__class__.__name__,
            len(self.__fields__),
            len(self.__profiles__),
            len(self.__metadata__),
        )

    def __post_init__(self, *args, **kwargs):
        """Add additional structure to subclass initialization.

        This method is automatically called at the end of `__init__()`.
        Subclasses may override it to perform additional setup or data processing
        without rewriting the entire constructor.

        Parameters
        ----------
        *args, **kwargs :
            Arbitrary arguments forwarded from the constructor.

        Examples
        --------
        .. code-block:: python

            class MyModel(BaseModel):
                def __post_init__(self):
                    self.mass_field = self.__fields__["mass"]
                    self.density_profile = self.__profiles__[
                        "density"
                    ]

        """
        pass

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
        overwrite : bool, optional
            Whether to overwrite the file if it already exists (default: False).

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

            # Add the class name as an attribute on the file.
            handle.attrs["__model_class__"] = cls.__name__

            # Add any other necessary groups or attributes here.

        return filepath

    @classmethod
    def from_components(
        cls,
        filepath: str | Path,
        fields: dict[str, unyt_array | np.ndarray],
        profiles: dict[str, "BaseProfile"],
        metadata: dict[str, Any],
        overwrite: bool = False,
    ) -> "BaseModel":
        """Create a Pisces model from in-memory fields, profiles, and metadata.

        Parameters
        ----------
        filepath : str or Path
            Path to the output HDF5 model file.
        fields : dict of str to array
            Dictionary of model field names to arrays (with or without units).
        profiles : dict of str to BaseProfile
            Dictionary of profile names to profile objects.
        metadata : dict
            Metadata dictionary to store in the file.
        overwrite : bool, optional
            Whether to overwrite the file if it already exists (default: False).

        Returns
        -------
        BaseModel
            An instance of the model loaded from the saved file.

        """
        filepath = cls._construct_model_file_skeleton(filepath, overwrite=overwrite)

        # -------------------------------------------------- #
        # Add required metadata fields                       #
        # -------------------------------------------------- #
        metadata["date_created"] = datetime.datetime.now().isoformat()
        metadata["__model_class__"] = cls.__name__
        metadata = cls.metadata_serializer.serialize_dict(metadata)

        with h5py.File(filepath, "r+") as f:
            # Store metadata in root attributes
            for key, val in metadata.items():
                f.attrs[key] = val

            # Write fields
            field_group = f["FIELDS"]
            for name, arr in fields.items():
                data = arr.to_base().value if hasattr(arr, "to_base") else np.asarray(arr)
                dset = field_group.create_dataset(name, data=data)
                if hasattr(arr, "units"):
                    dset.attrs["units"] = str(arr.units)

            # Write profiles
            profile_group = f["PROFILES"]
            for name, profile in profiles.items():
                subgrp = profile_group.create_group(name)
                profile.to_hdf5(subgrp, name)

        return cls(filepath)

    # ========================================= #
    # Dunder Methods                            #
    # ========================================= #
    def __str__(self):
        """Return a readable string representation of the model."""
        return f"<{self.__class__.__name__} @ {self.__path__}>"

    def __repr__(self):
        """Return the default representation of the model."""
        return super().__repr__()

    def __del__(self):
        """Ensure that the HDF5 file handle is closed when the model is deleted."""
        handle = getattr(self, "__handle__", None)
        if handle is not None and handle.id.valid:
            handle.close()

    def __getitem__(self, item):
        """Allow for dictionary-like access to the model's fields."""
        if item in self.__fields__:
            return self.__fields__[item]
        elif item in self.__profiles__:
            return self.__profiles__[item]
        elif item in self.__metadata__:
            return self.__metadata__[item]
        else:
            raise KeyError(f"Item '{item}' not found in model.")

    def __setitem__(self, key: str, value: Any):
        """Allow for dictionary-like setting of the model's fields."""
        if key in self.__fields__:
            self.__fields__[key] = value
        elif key in self.__profiles__:
            self.__profiles__[key] = value
        elif key in self.__metadata__:
            self.__metadata__[key] = value
        else:
            raise KeyError(f"Item '{key}' not found in model.")

    def __contains__(self, key):
        """Check if a key exists in the fields, profiles, or metadata."""
        return key in self.__fields__ or key in self.__profiles__ or key in self.__metadata__

    def __len__(self):
        """Return the number of physical fields in the model."""
        return len(self.__fields__)

    def __iter__(self):
        """Iterate over the field names in the model."""
        return iter(self.__fields__)

    # ========================================= #
    # Properties                                #
    # ========================================= #
    @property
    def path(self) -> Path:
        """The path to the model's HDF5 file."""
        return self.__path__

    @property
    def handle(self) -> h5py.File:
        """The open HDF5 file handle."""
        return self.__handle__

    @property
    def metadata(self) -> dict[str, Any]:
        """Model metadata dictionary."""
        return self.__metadata__

    @property
    def profiles(self) -> dict[str, "BaseProfile"]:
        """Model profile objects."""
        return self.__profiles__

    @property
    def fields(self) -> dict[str, unyt_array | np.ndarray]:
        """Model field buffers."""
        return self.__fields__
