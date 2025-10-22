"""Core classes for particle data support in Pisces.

This module defines the base class for particle datasets, providing a standard interface
for reading, writing, and interacting with particle datasets in the Pisces framework.
"""

from datetime import datetime
from pathlib import Path
from typing import overload

import h5py
import numpy as np
import unyt

from pisces.utilities.io_tools import HDF5Serializer


class ParticleDataset:
    """Base class for particle datasets in Pisces.

    This class provides the standard interface for reading, writing, and interacting with particle datasets
    in the Pisces framework. It parallels the standard HDF5 formats used in codes like AREPO, GADGET, etc.
    but includes some additional flexibility and features.

    Details regarding the expected structure of the dataset and how to work with particle
    datasets can be found at :ref:`particles_overview`.
    """

    metadata_serializer: HDF5Serializer = HDF5Serializer
    """~utilities.io_tools.HDF5Serializer: The HDF5 serializer used for reading and writing metadata.

    This serializer class is responsible for converting metadata types into
    formats which are compatible with HDF5 attributes. By default, this is the
    :py:class:`~pisces.utilities.io_tools.HDF5Serializer` class, which handles
    serialization of common Python types (e.g., dict, list, str) into HDF5 attributes
    along with unyt arrays, quantities, and units.

    Developers may extend or replace this serializer to support custom types
    if there is need.
    """

    # --------------------------------- #
    # Class Constants / Flags           #
    # --------------------------------- #
    # These flags provide standard names for built-in access to specific
    # fields in the dataset. This ensures that those areas of the code are
    # not hardcoded with string literals. These can be overridden in
    # subclasses if the dataset uses different naming conventions.

    # --- Field Name Conventions --- #
    _POSITION_FIELD_NAME: str = "particle_position"
    """str: The standard name for the position field in each particle group."""
    _VELOCITY_FIELD_NAME: str = "particle_velocity"
    """str: The standard name for the velocity field in each particle group."""
    _MASS_FIELD_NAME: str = "particle_mass"
    """str: The standard name for the mass field in each particle group."""
    _ID_FIELD_NAME: str = "particle_id"
    """str: The standard name for the unique identifier field in each particle group."""

    # --- Class Settings --- #
    _ID_POLICY: str = "global"
    """str: The policy for assigning unique particle IDs.

    This can be either "global" or "per_group":

    - "global": Particle IDs are unique across all groups in the dataset.
    - "per_group": Particle IDs are only unique within each group.

    This setting affects how particle IDs are interpreted and managed within the dataset.
    Default is "global".
    """

    # --- Metadata Requirements ---#
    __REQUIRED_GLOBAL_METADATA__: list[str] = ["CLASS_NAME", "GEN_TIME"]
    """list of str: The required global metadata attributes for this dataset.

    If these are not all present in the global metadata on load, then
    the dataset will raise a :py:class:`IOError` during validation.
    """
    __REQUIRED_GROUP_METADATA__: list[str] = ["NUMBER_OF_PARTICLES"]
    """list of str: The required group metadata attributes for each particle group.

    If these are not all present in the group metadata, then
    the dataset will raise a :py:class:`ValueError` during validation.
    """

    @classmethod
    def _serialized(cls, obj):
        """Serialize an object using the dataset's metadata serializer.

        Parameters
        ----------
        obj : Any
            The object to serialize.

        Returns
        -------
        Any
            The serialized representation of the object.

        """
        if isinstance(obj, dict):
            return cls.metadata_serializer.serialize_dict(obj)
        else:
            return cls.metadata_serializer.serialize_data(obj)

    @classmethod
    def _deserialized(cls, obj):
        """Deserialize an object using the dataset's metadata serializer.

        Parameters
        ----------
        obj : Any
            The object to deserialize.

        Returns
        -------
        Any
            The deserialized representation of the object.

        """
        if isinstance(obj, dict):
            return cls.metadata_serializer.deserialize_dict(obj)
        else:
            return cls.metadata_serializer.deserialize_data(obj)

    # -------------------------------------- #
    # Initialization and Validation Methods  #
    # -------------------------------------- #
    # These methods are responsible for initializing the dataset.
    def __validate__(self):
        """Validate that this dataset meets a minimum set of format requirements.

        The following steps are performed to check the dataset structure:

        - Check the **global metadata**:
          We look through all of the __REQUIRED_GLOBAL_METADATA__ attributes
          to ensure that they are all present in the global metadata.
        - Check the **group metadata**:
          For each of the particle groups which DOESN'T have the
          ``NOT_PARTICLE_GROUP`` attribute, we check that the __REQUIRED_GROUP_METADATA__
          attributes are present in the group metadata.
        - Check the **number of particles**:
            For each dataset in each particle group, we check that the number of particles
            matches the ``NUMBER_OF_PARTICLES`` attribute in the group metadata.

        This method can be extended in subclasses to implement additional validation logic
        or constraints specific to the dataset type. It is called automatically during
        initialization to ensure that the dataset is in a valid state before any operations
        are performed.
        """
        # CHECKING GLOBAL METADATA:
        # Ensure that all required global metadata attributes are present
        # and that the CLASS_NAME flag is set to the correct class name.
        # The global metadata is ALREADY DESERIALIZED.
        _glob_metadata = self.get_global_metadata()

        # Ensure required metadata is present.
        if any(required_key not in _glob_metadata for required_key in self.__REQUIRED_GLOBAL_METADATA__):
            missing_keys = [key for key in self.__REQUIRED_GLOBAL_METADATA__ if key not in _glob_metadata]
            raise OSError(f"Missing required global metadata keys: {', '.join(missing_keys)}")

        # Check that this is the correct loading class.
        _expected_class_name = _glob_metadata.get("CLASS_NAME")
        if _expected_class_name != self.__class__.__name__:
            raise OSError(
                f"Expected global metadata CLASS_NAME to be '{self.__class__.__name__}', "
                f"but found '{_expected_class_name}'. This file may not be a valid "
                f"{self.__class__.__name__} dataset."
            )

        # CHECKING PARTICLE GROUPS METADATA:
        # Cycle through all particle groups and validate their metadata.
        for group_name in self.__handle__.keys():
            _group_handle = self.__handle__[group_name]

            # Check if the group actually has the `NOT_PARTICLE_GROUP` attribute. If
            # so, we just skip it straight up.
            if "NOT_PARTICLE_GROUP" in _group_handle.attrs:
                continue

            # Otherwise, we need to validate the metadata.
            _group_metadata = self.get_group_metadata(group_name)

            # Ensure that all required group metadata attributes are present.
            if any(required_key not in _group_metadata for required_key in self.__REQUIRED_GROUP_METADATA__):
                missing_keys = [key for key in self.__REQUIRED_GROUP_METADATA__ if key not in _group_metadata]
                raise ValueError(f"Group '{group_name}' is missing required metadata keys: {', '.join(missing_keys)}")

            # Finally, check that the number of particles in each
            # dataset matches the number of particles specified in the metadata.
            num_particles = _group_metadata.get("NUMBER_OF_PARTICLES")

            for dataset_name in _group_handle.keys():
                dataset_handle = _group_handle[dataset_name]

                # Check that the dataset has the correct number of particles.
                if dataset_handle.shape[0] != num_particles:
                    raise ValueError(
                        f"Dataset '{dataset_name}' in group '{group_name}' has {dataset_handle.shape[0]} "
                        f"particles, but 'NUMBER_OF_PARTICLES' metadata indicates {num_particles}."
                    )

    def __init__(self, path: str | Path, mode="r+"):
        """Initialize a :class:`ParticleDataset` from a file on disk.

        This constructor opens the specified HDF5 file and validates
        the global metadata to ensure that it conforms to the expected
        format / structure.

        Parameters
        ----------
        path : str or ~pathlib.Path
            The path to the HDF5 file containing the particle data.
            This can be a string or a :class:`pathlib.Path` object.
            If the path does not exist, a `FileNotFoundError` is raised.
        mode: str, optional
            The mode in which to open the HDF5 file. Defaults to "r+" (read/write mode).

            The available modes are:

            - "r": Read-only mode. The file must exist.
            - "r+": Read/write mode. The file must exist.
            - "w": Write mode. Creates a new file or truncates an existing file.
            - "w-": Write mode, but fails if the file already exists.
            - "x": Exclusive creation mode. Fails if the file already exists.

        Notes
        -----
        At this level, initialization consists of only the following 4 steps:

        1. Set the path to the HDF5 file and check that it exists.
        2. Open the HDF5 file in the specified mode and create the handle
           reference to the file.
        3. Load the global metadata from the file using the serializer.
        4. Validate the dataset structure by calling the ``.__validate__`` method.

        Subclasses may extend this behavior to include custom behavior beyond this.
        Additionally, the ``.__post_init__`` method is called after initialization,
        allowing for further customization or setup that is specific to the subclass.
        """
        # Set the path and open the handle to the HDF5 file.
        self.__path__ = Path(path)
        if not self.__path__.exists():
            raise FileNotFoundError(f"Particle dataset file not found: {self.__path__}")

        self.__handle__ = h5py.File(self.__path__, mode=mode)

        # Load the global metadata from disk via
        # the serializer.
        self.__global_metadata__ = self.get_global_metadata()

        # Check that the file is a valid particle dataset. This defers
        # to the __validate__ method to ensure that the file structure
        # and metadata conform to the expected format. This can be overridden
        # in subclasses to implement custom validation logic.
        self.__validate__()

        # Pass on to the post init method.
        self.__post_init__()

    def __post_init__(self):
        """Post-initialization hook for the :class:`ParticleDataset` class.

        This method is called after the dataset has been initialized and validated.
        It can be overridden in subclasses to perform additional setup or
        configuration that is specific to the subclass implementation.
        """
        pass

    # ------------------------------------ #
    # Properties                           #
    # ------------------------------------ #

    # --- Basic Attributes --- #
    @property
    def path(self) -> str | Path:
        """The path to the HDF5 file containing the particle dataset.

        This property returns the path as a string or a Path object, depending on how
        it was initialized.

        Returns
        -------
        Union[str, Path]
            The path to the HDF5 file.

        """
        return self.__path__

    @property
    def handle(self) -> h5py.File:
        """The HDF5 file handle for the particle dataset.

        This property provides direct access to the underlying HDF5 file handle,
        allowing for low-level operations if needed. It is recommended to use
        higher-level methods and properties for most use cases.

        Returns
        -------
        h5py.File
            The HDF5 file handle.

        """
        return self.__handle__

    @property
    def global_metadata(self) -> dict:
        """Global metadata attributes at the root level of the HDF5 file.

        This includes attributes such as creation time, cosmological parameters,
        and dataset-wide configuration flags. Attributes marked here are
        accessible via :attr:`ParticleDataset.metadata`.

        Returns
        -------
        dict
            A dictionary of all global HDF5 attributes.

        """
        # We return a copy to prevent weird editing attempts.
        return self.__global_metadata__.copy()

    @property
    def particle_groups(self) -> list[str]:
        """Names of all particle groups present in the dataset.

        This excludes any HDF5 groups that are marked with the attribute
        ``NOT_PARTICLE_GROUP``.

        Returns
        -------
        list of str
            The names of valid particle groups.

        """
        groups = []
        for name, group in self.__handle__.items():
            if isinstance(group, h5py.Group) and "NOT_PARTICLE_GROUP" not in group.attrs:
                groups.append(name)
        return groups

    @property
    def fields(self) -> list[str]:
        """List of all fields (datasets) available in the dataset, in dot notation.

        This property returns a list of all field names across all particle groups,
        using the format ``group_name.field_name``. This allows direct access via
        indexing, e.g., ``ds["baryons.particle_velocity"]``.

        Returns
        -------
        list of str
            A sorted list of all field names in dot notation.

        """
        field_names = []
        for group_name in self.particle_groups:
            group = self.__handle__[group_name]
            field_names.extend(f"{group_name}.{field}" for field in group.keys())
        return sorted(field_names)

    @property
    def num_particles(self) -> dict[str, int]:
        """Number of particles in each particle group.

        This property returns a dictionary mapping each particle group name to the
        number of particles it contains, as specified by the ``NUMBER_OF_PARTICLES``
        attribute in each group's metadata.

        All groups must define this attribute; otherwise, a ValueError is raised.

        Returns
        -------
        dict
            A dictionary mapping group names to the number of particles in each group.

        Raises
        ------
        ValueError
            If any group is missing the ``NUMBER_OF_PARTICLES`` attribute.

        """
        counts = {}
        for group_name in self.particle_groups:
            metadata = self.get_group_metadata(group_name)
            if "NUMBER_OF_PARTICLES" not in metadata:
                raise ValueError(f"Group '{group_name}' is missing required 'NUMBER_OF_PARTICLES' attribute.")
            counts[group_name] = metadata["NUMBER_OF_PARTICLES"]
        return counts

    @property
    def total_particles(self) -> int:
        """Total number of particles across all particle groups.

        This property sums the number of particles in each group as specified by
        the ``NUMBER_OF_PARTICLES`` attribute in each group's metadata.

        Returns
        -------
        int
            The total number of particles across all groups.

        """
        return sum(self.num_particles.values())

    # ------------------------------------ #
    # Dunder Methods                       #
    # ------------------------------------ #
    def __str__(self) -> str:
        """Return a human-readable string representation of the ParticleDataset.

        This includes the file path, creation date, total number of particles, and
        a list of particle groups with their particle counts.

        Returns
        -------
        str
            A formatted string describing the dataset.

        """
        return f"<{self.__class__.__name__} @ {self.path.name} | N = {self.total_particles}>"

    def __repr__(self) -> str:
        """Return a detailed string representation for debugging.

        This includes the class name, file path, total particle count,
        and number of groups.

        Returns
        -------
        str
            A concise technical summary of the dataset.

        """
        return (
            f"<{self.__class__.__name__}("
            f"path={repr(str(self.path))}, "
            f"groups={len(self.particle_groups)}, "
            f"total_particles={self.total_particles})>"
        )

    def __del__(self):
        """Destructor for the ParticleDataset class. Closes the HDF5 file handle if it is open."""
        if hasattr(self, "__handle__") and self.__handle__ is not None:
            self.__handle__.close()
            del self.__handle__

    def __getitem__(self, key: str) -> unyt.unyt_array:
        """Get a particle field by its name in dot notation.

        Parameters
        ----------
        key : str
            The field name in the format ``"group_name.field_name"``.

        Returns
        -------
        unyt.array.unyt_array
            The data for the specified field, converted to a unyt array.

        Raises
        ------
        KeyError
            If the specified field does not exist in the dataset.

        """
        try:
            group_name, field_name = key.split(".")
        except ValueError as err:
            raise KeyError(f"Invalid field name format: '{key}'. Expected 'group_name.field_name'.") from err

        if not self.__contains__(key):
            raise KeyError(f"Field '{key}' does not exist in the dataset.")

        return self.get_particle_field(group_name, field_name)

    def __contains__(self, key: str) -> bool:
        """Check if a particle field exists in the dataset.

        Parameters
        ----------
        key : str
            The field name in the format ``"group_name.field_name"``.

        Returns
        -------
        bool
            True if the field exists, False otherwise.

        """
        try:
            group_name, field_name = key.split(".")
        except ValueError as err:
            raise KeyError(f"Invalid field name format: '{key}'. Expected 'group_name.field_name'.") from err

        return f"{group_name}/{field_name}" in self.__handle__

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__del__()

    def __len__(self) -> int:
        """Return the number of particle fields in the dataset."""
        return len(self.fields)

    def __iter__(self):
        """Iterate over all particle fields in dot notation."""
        return iter(self.fields)

    def __dir__(self):
        """Return a list of all attributes and methods of the ParticleDataset."""
        return list(super().__dir__()) + self.fields

    # ------------------------------------ #
    # Metadata Management Methods          #
    # ------------------------------------ #
    def get_global_metadata(self) -> dict:
        """Get the global metadata attributes of the dataset.

        This method retrieves all global attributes at the root level of the HDF5 file
        and returns them as a dictionary. It is used to access metadata such as creation date,
        cosmological parameters, and dataset-wide configuration flags.

        Returns
        -------
        dict
            A dictionary containing all global metadata attributes.

        """
        return self.metadata_serializer.deserialize_dict(dict(self.__handle__.attrs))

    def get_group_metadata(self, group_name: str) -> dict:
        """Get the metadata attributes for a specific particle group.

        This method retrieves all attributes associated with a given particle group
        and returns them as a dictionary. It is used to access metadata such as the
        number of particles in the group and any additional attributes defined by the user.

        Parameters
        ----------
        group_name : str
            The name of the particle group whose metadata is to be retrieved.

        Returns
        -------
        dict
            A dictionary containing all metadata attributes for the specified group.

        Raises
        ------
        KeyError
            If the specified group does not exist in the dataset.

        """
        if group_name not in self.particle_groups:
            raise KeyError(f"Particle group '{group_name}' does not exist in the dataset.")
        return self.metadata_serializer.deserialize_dict(dict(self.__handle__[group_name].attrs))

    def get_field_metadata(self, group_name: str, field_name: str) -> dict:
        """Get the metadata attributes for a specific field in a particle group.

        This method retrieves all attributes associated with a given field in a particle group
        and returns them as a dictionary. It is used to access metadata such as units, data type,
        and any additional attributes defined by the user.

        Parameters
        ----------
        group_name : str
            The name of the particle group containing the field.
        field_name : str
            The name of the field whose metadata is to be retrieved.

        Returns
        -------
        dict
            A dictionary containing all metadata attributes for the specified field.

        Raises
        ------
        KeyError
            If the specified group or field does not exist in the dataset.

        """
        dataset_handle = self.get_particle_field_handle(group_name, field_name)
        return self.metadata_serializer.deserialize_dict(dict(dataset_handle.attrs))

    def reload_global_metadata(self):
        """Reload the global metadata from the HDF5 file.

        This method refreshes the global metadata attributes by re-reading them from the
        HDF5 file. It is useful if the metadata has been modified externally or if you want
        to ensure you have the latest version of the metadata.

        """
        self.__global_metadata__ = self.get_global_metadata()

    def update_global_metadata(self, metadata: dict):
        """Update the global metadata attributes of the dataset.

        This method allows you to modify or add global metadata attributes at the root level
        of the HDF5 file. It updates the attributes in memory and writes them back to the file.

        Parameters
        ----------
        metadata : dict
            A dictionary containing the metadata attributes to update or add.

        """
        # Serialize the metadata dictionary
        serialized_meta = self.metadata_serializer.serialize_dict(metadata)

        # Now write.
        for key, value in serialized_meta.items():
            self.__handle__.attrs[key] = value

        # Reload global metadata.
        self.reload_global_metadata()

    def update_group_metadata(self, group_name: str, metadata: dict):
        """Update the metadata attributes for a specific particle group.

        This method allows you to modify or add metadata attributes for a given particle group.
        It updates the attributes in memory and writes them back to the HDF5 file.

        Parameters
        ----------
        group_name : str
            The name of the particle group whose metadata is to be updated.
        metadata : dict
            A dictionary containing the metadata attributes to update or add.

        Raises
        ------
        KeyError
            If the specified group does not exist in the dataset.

        """
        group_handle = self.get_particle_group_handle(group_name)

        # Serialize the metadata dictionary
        serialized_meta = self.metadata_serializer.serialize_dict(metadata)

        # Now write.
        for key, value in serialized_meta.items():
            group_handle.attrs[key] = value

    def update_field_metadata(self, group_name: str, field_name: str, metadata: dict):
        """Update the metadata attributes for a specific field in a particle group.

        This method allows you to modify or add metadata attributes for a given field in a particle group.
        It updates the attributes in memory and writes them back to the HDF5 file.

        Parameters
        ----------
        group_name : str
            The name of the particle group containing the field.
        field_name : str
            The name of the field whose metadata is to be updated.
        metadata : dict
            A dictionary containing the metadata attributes to update or add.

        Raises
        ------
        KeyError
            If the specified group or field does not exist in the dataset.

        """
        dataset_handle = self.get_particle_field_handle(group_name, field_name)

        # Serialize the metadata dictionary
        serialized_meta = self.metadata_serializer.serialize_dict(metadata)

        # Now write.
        for key, value in serialized_meta.items():
            dataset_handle.attrs[key] = value

    def delete_global_metadata_keys(self, *keys: str):
        """
        Delete one or more metadata keys from the global metadata.

        Parameters
        ----------
        *keys : str
            The names of the metadata keys to delete from the global metadata.

        Notes
        -----
        This method will not permit you to remove a required global metadata key.
        """
        for key in keys:
            if key in self.__REQUIRED_GLOBAL_METADATA__:
                raise ValueError("Cannot delete required global metadata key: " + key)

            if key in self.__handle__.attrs:
                del self.__handle__.attrs[key]
        self.reload_global_metadata()

    def delete_group_metadata_keys(self, group_name: str, *keys: str):
        """Delete one or more metadata keys from a specific particle group.

        Parameters
        ----------
        group_name : str
            The name of the particle group from which to delete metadata keys.
        *keys : str
            The names of the metadata keys to delete from the global metadata.

        Notes
        -----
        This method will not permit you to remove a required global metadata key.
        """
        group_handle = self.get_particle_group_handle(group_name)
        for key in keys:
            if key in self.__REQUIRED_GROUP_METADATA__:
                raise ValueError("Cannot delete required group metadata key: " + key)

            if key in group_handle.attrs:
                del group_handle.attrs[key]

    def delete_field_metadata_keys(self, group_name: str, field_name: str, *keys: str):
        """Delete one or more metadata keys from a specific field in a particle group.

        Parameters
        ----------
        group_name : str
            The name of the particle group containing the field.
        field_name : str
            The name of the field from which to delete metadata keys.

        Notes
        -----
        You may not remove ``"UNITS"`` from the field metadata, as this is required.
        """
        field_handle = self.get_particle_field_handle(group_name, field_name)
        for key in keys:
            if key == "UNITS":
                raise ValueError("Cannot delete required field metadata key: 'UNITS'")

            if key in field_handle.attrs:
                del field_handle.attrs[key]

    # ------------------------------------ #
    # Data Access Methods                  #
    # ------------------------------------ #
    # --- HDF5 Group and Dataset Accessors --- #
    def get_particle_group_handle(self, group_name: str) -> h5py.Group:
        """Get the HDF5 group handle for a specific particle group.

        Parameters
        ----------
        group_name : str
            The name of the particle group to retrieve.

        Returns
        -------
        h5py.Group
            The HDF5 group handle for the specified particle group.

        Raises
        ------
        KeyError
            If the specified group does not exist in the dataset.

        """
        if group_name not in self.particle_groups:
            raise KeyError(f"Particle group '{group_name}' does not exist in the dataset.")
        return self.__handle__[group_name]

    def get_particle_field_handle(self, group_name: str, field_name: str):
        """Get the HDF5 dataset handle for a specific field in a particle group.

        Parameters
        ----------
        group_name : str
            The name of the particle group containing the field.
        field_name : str
            The name of the field to retrieve.

        Returns
        -------
        h5py.Dataset
            The HDF5 dataset handle for the specified field.

        Raises
        ------
        KeyError
            If the specified group or field does not exist in the dataset.

        """
        group_handle = self.get_particle_group_handle(group_name)
        if field_name not in group_handle:
            raise KeyError(f"Field '{field_name}' does not exist in group '{group_name}'.")
        return group_handle[field_name]

    # --- Field Accessors --- #
    def get_particle_field(self, group_name: str, field_name: str) -> unyt.unyt_array:
        """Get the particle field data as a unyt array.

        Parameters
        ----------
        group_name : str
            The name of the particle group containing the field.
        field_name : str
            The name of the field to retrieve.

        Returns
        -------
        unyt.array.unyt_array
            The data for the specified field, converted to a unyt array.

        Raises
        ------
        KeyError
            If the specified group or field does not exist in the dataset.

        """
        dataset_handle = self.get_particle_field_handle(group_name, field_name)
        return unyt.unyt_array(dataset_handle[...], units=self.get_field_units(group_name, field_name))

    def get_particle_fields(self, fields: list[str]) -> dict[str, unyt.unyt_array]:
        """Get multiple particle fields as a dictionary of unyt arrays.

        Parameters
        ----------
        fields : List[str]
            A list of field names in the format ``"group_name.field_name"``.

        Returns
        -------
        dict
            A dictionary mapping field names to their data as unyt arrays.

        Raises
        ------
        KeyError
            If any specified field does not exist in the dataset.

        """
        field_data = {}
        for field in fields:
            group_name, field_name = field.split(".")
            field_data[field] = self.get_particle_field(group_name, field_name)
        return field_data

    def get_field_units(self, group_name: str, field_name: str) -> unyt.Unit:
        """Get the units of a specific particle field.

        Parameters
        ----------
        group_name : str
            The name of the particle group containing the field.
        field_name : str
            The name of the field whose units are to be retrieved.

        Returns
        -------
        ~unyt.unit_object.Unit
            The units of the specified field as a unyt Unit object.

        """
        dataset_handle = self.get_particle_field_handle(group_name, field_name)
        return unyt.Unit(self._deserialized(dataset_handle.attrs.get("UNITS", "")))

    # ------------------------------------ #
    # Modification Methods                 #
    # ------------------------------------ #
    def copy(self, output_path: str | Path, overwrite: bool = False, **kwargs) -> "ParticleDataset":
        """Create a full copy of this particle dataset at a new location.

        This method replicates the entire contents of the HDF5 file, including all
        particle groups, fields, field metadata, and global attributes, into a new file.
        It returns a new :class:`ParticleDataset` instance pointing to the copied file.

        Parameters
        ----------
        output_path : str or Path
            The path to the new HDF5 file to create.
        overwrite : bool, optional
            If True, overwrite the file at `output_path` if it already exists. Defaults to False.
        **kwargs
            Additional keyword arguments passed to the constructor of the copied dataset.

        Returns
        -------
        ParticleDataset
            A new dataset instance pointing to the copied file.

        Raises
        ------
        FileExistsError
            If `output_path` exists and `overwrite` is False.
        IsADirectoryError
            If `output_path` is a directory.

        """
        output_path = Path(output_path)

        if output_path.exists():
            if output_path.is_dir():
                raise IsADirectoryError(f"Cannot copy dataset to a directory: {output_path}")
            if not overwrite:
                raise FileExistsError(f"File already exists at {output_path}. Use overwrite=True to overwrite.")
            output_path.unlink()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open the output file and copy all content
        with h5py.File(output_path, "w") as f_out:
            # Copy all groups and datasets
            self.handle.copy(source="/", dest=f_out, name="/")

        return self.__class__(output_path, **kwargs)

    def add_particle_type(self, name: str, num_particles: int, metadata: dict = None, **kwargs):
        """Add a new particle group to the dataset.

        This method creates a new HDF5 group representing a particle type (e.g., ``"baryons"``, ``"dark_matter"``)
        and assigns the required metadata attribute ``NUMBER_OF_PARTICLES`` to the group. Optional metadata
        can be added via the `metadata` dictionary or keyword arguments. This method is intended to be general
        and extensible for use in simulation initialization, preprocessing, or structured data generation.

        It is safe to override in subclasses that implement additional constraints or need to attach simulation-specific
        annotations, provenance, or physical properties to each group.

        Parameters
        ----------
        name : str
            The name of the new particle group. This must be unique within the dataset and conform to
            HDF5 naming rules (alphanumeric, no slashes).
        num_particles : int
            The number of particles in the new group. This value is stored in the group's metadata
            under the ``NUMBER_OF_PARTICLES`` key and is required for downstream field shape validation.
        metadata : dict, optional
            A dictionary of metadata attributes to attach to the group. Keys must be strings and values
            must be serializable by HDF5 (e.g., int, float, str). These attributes will be written in
            addition to ``NUMBER_OF_PARTICLES``.
        **kwargs
            Additional metadata attributes provided as keyword arguments. These are merged with `metadata`
            and override any overlapping keys. Use this to quickly attach single attributes.

        Raises
        ------
        ValueError
            If a group with the specified name already exists in the dataset.

        Notes
        -----
        - All metadata is stored as HDF5 attributes on the group object.
        - This method does **not** allocate any fields or datasets; it only creates the group and metadata.
        - Subclasses may override this method to add simulation-specific metadata keys or validation logic.

        """
        if name in self.particle_groups:
            raise ValueError(f"Particle group '{name}' already exists.")

        group = self.__handle__.create_group(name)
        group.attrs["NUMBER_OF_PARTICLES"] = self._serialized(num_particles)

        # Merge metadata from both dict and kwargs, prioritizing kwargs
        metadata = metadata or {}
        merged = {**metadata, **kwargs}

        for key, value in merged.items():
            group.attrs[key] = value

    def add_particle_field(
        self,
        group_name: str,
        field_name: str,
        data: unyt.unyt_array | np.ndarray,
        metadata: dict = None,
        overwrite: bool = False,
    ):
        """Add a new field (dataset) to an existing particle group.

        This method creates and writes a new field to the specified particle group
        within the dataset. The data can be a :class:`numpy.ndarray` or a :class:`~unyt.array.unyt_array`.
        Units (if present) will be stored in the dataset metadata. Additional metadata
        may also be included via the `metadata` argument.

        Parameters
        ----------
        group_name : str
            The name of the particle group to which the field will be added. This must
            be a pre-existing group in the particle dataset.
        field_name : str
            The name of the new field to create (e.g., ``"particle_mass"``). If the `field_name` is
            already specified in the group, it will raise an error unless `overwrite` is True.
        data : ~unyt.array.unyt_array or ~numpy.ndarray
            The data to write to the new field. The leading dimension must match the number of particles
            in the group.
        metadata : dict, optional
            Optional dictionary of metadata to attach to the dataset. This is generally used
            by subclasses of the base class to add type specific metadata.
        overwrite : bool, optional
            Whether to overwrite an existing dataset with the same name. Defaults to False.

        Raises
        ------
        KeyError
            If the group does not exist, or if the field exists and `overwrite` is False.
        ValueError
            If the field's leading dimension does not match the group's particle count.

        """
        # Ensure that the group exists and that the field name is valid /
        # correctly handle the overwrite behavior.
        group = self.get_particle_group_handle(group_name)
        group_attrs = self.get_group_metadata(group_name)
        if field_name in group.keys():
            # The field already exists. Our behavior depends on the `overwrite` flag.
            if not overwrite:
                raise ValueError("Field already exists. Set `overwrite=True` to replace it.")
            else:
                del group[field_name]

        # Determine the number of particles expected and
        # ensure that the data matches this shape.
        num_particles = group_attrs["NUMBER_OF_PARTICLES"]
        data = np.atleast_1d(data)
        if data.shape[0] != num_particles:
            raise ValueError(
                f"Number of particles in field {field_name} does not match number of particles in group {group_name}"
            )

        # Validation has been completed and we can therefore now
        # proceed with writing the field to the group.
        units = getattr(data, "units", "")

        if isinstance(data, unyt.unyt_array):
            dset = group.create_dataset(field_name, data=data.d, dtype=data.dtype)
        else:
            dset = group.create_dataset(field_name, data=data, dtype=data.dtype)

        # Handle the metadata.
        dset.attrs["UNITS"] = self.metadata_serializer.serialize_data(units)
        if metadata is not None:
            dset.attrs.update(self.metadata_serializer.serialize_dict(metadata))

    def remove_particle_group(self, group_name: str):
        """Remove a particle group from the dataset.

        This method deletes the specified particle group and all its associated fields
        from the dataset. It is a destructive operation and cannot be undone.

        Parameters
        ----------
        group_name : str
            The name of the particle group to remove.

        Raises
        ------
        KeyError
            If the specified group does not exist in the dataset.

        """
        if group_name not in self.particle_groups:
            raise KeyError(f"Particle group '{group_name}' does not exist in the dataset.")
        del self.__handle__[group_name]

    def remove_particle_field(self, group_name: str, field_name: str):
        """Remove a specific field from a particle group.

        This method deletes the specified field from the given particle group.
        It is a destructive operation and cannot be undone.

        Parameters
        ----------
        group_name : str
            The name of the particle group containing the field to remove.
        field_name : str
            The name of the field to remove.

        Raises
        ------
        KeyError
            If the specified group or field does not exist in the dataset.

        """
        group_handle = self.get_particle_group_handle(group_name)
        if field_name not in group_handle:
            raise KeyError(f"Field '{field_name}' does not exist in group '{group_name}'.")
        del group_handle[field_name]

    def extend_group(
        self,
        group_name: str,
        num_particles: int,
        fields: dict[str, np.ndarray | unyt.unyt_array] = None,
    ):
        """Extend a particle group by adding new particles and updating all fields.

        This method appends `num_particles` new entries to each existing field in the group.
        For fields provided in the `fields` dictionary, the new particle values are appended.
        For fields not provided, the new values are filled with NaN (if supported).

        The group's ``NUMBER_OF_PARTICLES`` attribute is updated accordingly.

        Parameters
        ----------
        group_name : str
            The name of the group to extend.
        num_particles : int
            The number of new particles to append to the group.
        fields : dict, optional
            A dictionary mapping field names to new particle data arrays of shape
            (num_particles, ...) to append. Any fields not specified here will be
            extended with `NaN` fill values if their dtype supports it.

        Raises
        ------
        KeyError
            If the specified group does not exist.
        ValueError
            If the new data for any field has incompatible shape.
        TypeError
            If an existing field cannot be filled with NaNs and no new data is provided.

        """
        # Ensure access to the group and that the
        # number of new particles is non zero.
        group = self.get_particle_group_handle(group_name)
        group_attrs = self.get_group_metadata(group_name)
        if not isinstance(num_particles, int) or num_particles <= 0:
            raise ValueError("`num_particles` must be a positive integer.")

        # Create the field dictionary and
        # modify the group attribute.
        fields = fields or {}
        old_particle_count = group_attrs["NUMBER_OF_PARTICLES"]
        new_particle_count = old_particle_count + num_particles

        # Make corrections to the fields.
        for field_name in group.keys():
            # In order to correct the fields, the first step is
            # to extract the previously existing data and its
            # relevant metadata since we'll need to delete it to
            # continue.
            _original_field_array = group[field_name][...]
            _metadata = dict(group[field_name].attrs)

            # Start by extending the dataset to accommodate the new particles.
            # this is always performed the same way regardless whether we
            # have the field data or not.
            _new_field_array = np.zeros((new_particle_count,) + _original_field_array.shape[1:])
            _new_field_array[:old_particle_count, ...] = _original_field_array

            if field_name in fields:
                # Coerce the field data to an unyt array so that
                # we have an easier time manipulating the units.
                # This will assign dimensionless units to empty arrays.
                field_data = unyt.unyt_array(fields[field_name])

                # If we have a field that we are going to insert, we need to
                # check the shape and handle the units to ensure that everything
                # behaves correctly.
                if field_data.shape != (num_particles,) + _original_field_array.shape[1:]:
                    raise ValueError(
                        f"Field '{field_name}' data must have shape "
                        f"({num_particles}, ...) to match existing field shape."
                    )

                try:
                    field_data = field_data.to_value(self.get_field_units(group_name, field_name))
                except Exception as exp:
                    raise TypeError(f"Cannot convert field '{field_name}' data to existing units: {exp}") from exp

                # Fill the remaining data with the correct values.
                _new_field_array[old_particle_count:, ...] = field_data
            else:
                _new_field_array[old_particle_count:, ...] = np.nan

            # Now that the _new_field_array is filled, we need to
            # delete and replace the existing dataset with the new one.
            del group[field_name]
            dset = group.create_dataset(field_name, data=_new_field_array, dtype=_new_field_array.dtype)
            dset.attrs["UNITS"] = _metadata.get("UNITS", "")

            for key, value in _metadata.items():
                if key != "UNITS":
                    dset.attrs[key] = value

    def concatenate_inplace(self, *others: "ParticleDataset", groups=None):
        """Concatenate another :class:`ParticleDataset` into this one, extending specified groups.

        This method appends the particle data from `other` to this dataset for the specified groups.
        If `groups` is None, all groups in `other` are concatenated. The number of particles in each
        group is updated accordingly.

        Parameters
        ----------
        *others : list of ParticleDataset
            The datasets to concatenate into this one.
        groups : list of str, optional
            Names of groups to concatenate. If None, all groups are concatenated.

        Raises
        ------
        KeyError
            If a specified group does not exist in either dataset.
        ValueError
            If the datasets have incompatible shapes for concatenation.

        """
        for other in others:
            # Select the groups of `other` that we're going to add to
            # our own groups. If `groups` is None, we will use all of the groups.
            if groups is None:
                groups = other.particle_groups

            # Iterate through all of the groups so
            # that we can concatenate all of the groups.
            for group in groups:
                # Check if the group is already present in the new dataset
                # or if it needs to be added.
                if group in self.particle_groups:
                    # This group is already present. We need to
                    # concatenate the data. This will be a little bit
                    # trickier than the missing group case.

                    # Extract all the fields from the old group and
                    # begin the procedure of extending the group.
                    group_fields = [k for k in other.fields if k.startswith(group + ".")]
                    old_fields = {field: other.get_particle_field(group, field) for field in group_fields}
                    self.extend_group(group, other.num_particles[group], fields=old_fields)
                else:
                    # We don't already have the group so we can just
                    # copy the group directly across.
                    source_group = other.handle[group]
                    self.handle.copy(source=source_group, dest=self.handle, name=group)

    def reduce_group(
        self,
        group_name: str,
        mask: np.ndarray | unyt.unyt_array,
    ):
        """Reduce a particle group by applying a boolean mask.

        This method filters all fields in the specified group by the given mask,
        retaining only those particles where the mask is `True`. All other particles
        are discarded. The group's ``NUMBER_OF_PARTICLES`` attribute is updated accordingly.

        This is a destructive operation.

        Parameters
        ----------
        group_name : str
            Name of the particle group to apply the mask to.
        mask : array_like of bool
            Boolean array of shape (N,) where N is the number of particles in the group.
            Must be 1D and have exactly one element per particle.

        Raises
        ------
        KeyError
            If the specified group does not exist.
        ValueError
            If the mask has an incorrect shape or is not boolean.

        """
        group = self.get_particle_group_handle(group_name)
        group_attrs = self.get_group_metadata(group_name)
        n = group_attrs["NUMBER_OF_PARTICLES"]

        mask = np.asarray(mask)
        if mask.shape != (n,) or mask.dtype != bool:
            raise ValueError(f"Mask must be a 1D boolean array of shape ({n},)")

        new_count = int(np.count_nonzero(mask))

        for field_name in list(group.keys()):
            old_data = group[field_name][...]
            new_data = old_data[mask]

            # Preserve metadata and overwrite dataset
            metadata = dict(group[field_name].attrs)
            del group[field_name]
            dset = group.create_dataset(field_name, data=new_data, dtype=new_data.dtype)

            for key, value in metadata.items():
                dset.attrs[key] = value

        group.attrs["NUMBER_OF_PARTICLES"] = self.metadata_serializer.serialize_data(new_count)

    def rename_field(self, group_name: str, old_name: str, new_name: str):
        """Rename a field within a particle group.

        This method renames the dataset (field) `old_name` to `new_name` in the specified group.
        Metadata is preserved during the renaming. This operation is destructive and cannot
        be undone.

        Parameters
        ----------
        group_name : str
            The name of the particle group containing the field.
        old_name : str
            The current name of the field to rename.
        new_name : str
            The new name to assign to the field.

        Raises
        ------
        KeyError
            If the specified group or field does not exist.
        ValueError
            If the new field name already exists in the group.

        """
        group = self.get_particle_group_handle(group_name)

        if old_name not in group:
            raise KeyError(f"Field '{old_name}' does not exist in group '{group_name}'.")

        if new_name in group:
            raise ValueError(f"Field '{new_name}' already exists in group '{group_name}'.")

        # Extract existing data and metadata
        data = group[old_name][...]
        metadata = dict(group[old_name].attrs)

        # Create the new dataset with the same data and metadata
        dset = group.create_dataset(new_name, data=data, dtype=data.dtype)
        for key, value in metadata.items():
            dset.attrs[key] = value

        # Remove the old field
        del group[old_name]

    def offset_particle_positions(self, offset: unyt.unyt_array, groups: list[str] = None):
        """Apply a constant offset to particle positions in specified groups.

        The method adds the given offset vector to the particle position field.
        This is the correct way to shift particle coordinates around via translation.

        .. note::

            The name of the particle position field is assumed to be that specified by
            the class's ``_POSITION_FIELD_NAME`` attribute. If your dataset does not have
            this field, you will need to manually apply the offset to the appropriate field
            or rename the field.

        Parameters
        ----------
        offset : unyt.array.unyt_array
            A vector specifying the offset to apply. Must have units compatible
            with the ``particle_position`` field(s). The `offset` may be any 1D array; however,
            it must match the shape of the particle positions. Thus, if the particles are in 3D space,
            the `offset` must be a 3-element vector.
        groups : list of str, optional
            Names of groups to apply the offset to. If None, all particle groups are used.

        Raises
        ------
        ValueError
            If `offset` is not a 3-element vector.

        Notes
        -----
        To ensure that this method functions properly across subclasses with various naming
        conventions, we require that the position field be named according to the class
        attribute ``_POSITION_FIELD_NAME``. Subclasses may change this attribute to match
        a particular naming convention.
        """
        # Ensure that the offset gets cast to an unyt array so
        # that it at least has unit attributes. We will check for
        # unit consistency later.
        offset = unyt.unyt_array(offset)

        # Handle the groups.
        if groups is None:
            groups = self.particle_groups

        # Now for each of the groups, we're going to
        # cycle through, apply the offset, and continue.
        # If we run into a shape issue, we raise an error.
        for group in groups:
            field_key = f"{group}.{self.__class__._POSITION_FIELD_NAME}"

            if field_key not in self:
                continue

            # Obtain the handle and the units.
            handle = self.get_particle_field_handle(group, self.__class__._POSITION_FIELD_NAME)
            units = self.get_field_units(group, self.__class__._POSITION_FIELD_NAME)

            # Check the shape.
            if handle.shape[-1] != len(offset):
                raise ValueError(f"Offset must match the shape of particle positions in group '{group}'.")

            # Apply the offset.
            handle[...] += offset.to_value(units)

    def offset_particle_velocities(self, offset: unyt.unyt_array, groups: list[str] = None):
        """Apply a constant offset to particle velocities in specified groups.

        The method adds the given offset vector to the particle velocity field.
        This is the correct way to shift particle coordinates around via translation.

        .. note::

            The name of the particle velocity field is assumed to be that specified by
            the class's ``_VELOCITY_FIELD_NAME`` attribute. If your dataset does not have
            this field, you will need to manually apply the offset to the appropriate field
            or rename the field.

        Parameters
        ----------
        offset : unyt.array.unyt_array
            A vector specifying the velocity offset to apply. Must have units compatible
            with the ``particle_velocity`` field(s). The `offset` may be any 1D array; however,
            it must match the shape of the particle velocities. Thus, if the particles are in 3D space,
            the `offset` must be a 3-element vector.
        groups : list of str, optional
            Names of groups to apply the offset to. If None, all particle groups are used.

        Raises
        ------
        ValueError
            If `offset` is not the correct shape.

        """
        # Ensure the offset has unit information.
        offset = unyt.unyt_array(offset)

        # Determine the list of groups to modify.
        if groups is None:
            groups = self.particle_groups

        for group in groups:
            field_key = f"{group}.{self.__class__._VELOCITY_FIELD_NAME}"

            if field_key not in self:
                continue

            # Obtain the handle and the units.
            handle = self.get_particle_field_handle(group, self.__class__._VELOCITY_FIELD_NAME)
            units = self.get_field_units(group, self.__class__._VELOCITY_FIELD_NAME)

            # Confirm shape match.
            if handle.shape[-1] != len(offset):
                raise ValueError(f"Offset must match the shape of particle velocities in group '{group}'.")

            # Apply the velocity offset.
            handle[...] += offset.to_value(units)

    def apply_linear_transformation(
        self,
        matrix: np.ndarray,
        groups: list[str] = None,
        fields: tuple[str, ...] = None,
    ):
        r"""Apply a linear transformation matrix to vector fields in specified particle groups.

        This method performs an in-place matrix transformation on each specified vector field
        (e.g., ``particle_position``, ``particle_velocity``) in one or more particle groups. It is
        useful for performing operations such as coordinate rotation, scaling, reflection, or shear.

        Each particle's vector field :math:`\mathbf{x}_i` is updated according to:

        .. math::

            \mathbf{x}_i \rightarrow \mathbf{A} \cdot \mathbf{x}_i

        where :math:`\mathbf{A}` is the transformation matrix and :math:`\mathbf{x}_i` is the
        vector value (e.g., position or velocity) of the :math:`i`-th particle.

        Parameters
        ----------
        matrix : array_like
            A 2D NumPy array of shape :math:`(D, D)` representing the linear transformation
            to apply. The dimension :math:`D` must match the last axis of each target field.
        groups : list of str, optional
            List of particle group names to which the transformation will be applied. If None,
            all groups in the dataset are used.
        fields : tuple of str, optional
            Tuple of field names (e.g., ``particle_position``, ``particle_velocity``) to transform.
            Default is ``("particle_position", "particle_velocity")``.

        Raises
        ------
        ValueError
            If `matrix` is not square or its shape does not match the vector dimensionality
            of the fields being transformed.
        KeyError
            If a specified field is not present in the given group(s).

        Notes
        -----
        - The transformation is performed in-place and modifies the original field values.
        - Fields that are not present in a group are skipped silently.
        - This operation assumes that vector fields are stored with shape :math:`(N, D)`, where
          :math:`N` is the number of particles and :math:`D` is the number of spatial dimensions.

        """
        # Set the default fields to all of the standard vector fields we
        # expect.
        if fields is None:
            fields = (self.__class__._POSITION_FIELD_NAME, self.__class__._VELOCITY_FIELD_NAME)

        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Transformation matrix must be square (D, D), got shape {matrix.shape}.")

        if groups is None:
            groups = self.particle_groups

        for group in groups:
            for field in fields:
                field_key = f"{group}.{field}"
                if field_key not in self:
                    continue  # Skip if the field is not present

                handle = self.get_particle_field_handle(group, field)

                if handle.shape[-1] != matrix.shape[0]:
                    raise ValueError(
                        f"Field '{field}' in group '{group}' has vector dimension {handle.shape[-1]}, "
                        f"which does not match transformation matrix shape {matrix.shape}."
                    )

                # Apply transformation using Einstein summation (broadcast-safe)
                transformed = np.einsum("ij,nj->ni", matrix, handle[...])
                handle[...] = transformed

    def rotate_particles(
        self,
        norm: np.ndarray,
        angle: float,
        groups: list[str] = None,
        fields: tuple[str, ...] = None,
    ):
        r"""Rotate vector fields in specified particle groups around a given axis.

        This method rotates each specified vector field (e.g., ``particle_position``, ``particle_velocity``)
        around the axis defined by `norm` by a given angle. The rotation is applied uniformly across
        all particles in the specified groups.

        The transformation uses the **Rodrigues' rotation formula**, which constructs a rotation matrix
        for an axis–angle pair. For a unit vector :math:`\hat{n}` and angle :math:`\theta`, the formula is:

        .. math::

            R = I + \sin\theta [\hat{n}]_\times + (1 - \cos\theta) [\hat{n}]_\times^2

        where :math:`[\hat{n}]_\times` is the skew-symmetric matrix of the axis vector.
        For more details, see: `Rodrigues Formula <https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula>`__.

        Parameters
        ----------
        norm : array_like
            A 3-element vector representing the rotation axis. This does not need to be normalized;
            it will be internally converted to a unit vector.
        angle : float
            The rotation angle in radians.
        groups : list of str, optional
            The particle groups to apply the rotation to. If None, all particle groups are used.
        fields : tuple of str, optional
            The vector fields to rotate. Defaults to ("particle_position", "particle_velocity").

        Raises
        ------
        ValueError
            If the axis is not a 3-element vector, or if the angle is invalid.

        """
        # Set the default fields to all of the standard vector fields we
        # expect.
        if fields is None:
            fields = (self.__class__._POSITION_FIELD_NAME, self.__class__._VELOCITY_FIELD_NAME)

        # Normalize axis
        axis = np.asarray(norm, dtype=float)
        if axis.shape != (3,):
            raise ValueError("Rotation axis must be a 3-element vector.")
        axis /= np.linalg.norm(axis)

        # Compute Rodrigues rotation matrix
        x, y, z = axis
        K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        self.apply_linear_transformation(R, groups=groups, fields=fields)

    def reorient_particles(
        self,
        direction_vector: np.ndarray,
        spin: float,
        groups: list[str] = None,
        fields: tuple[str, ...] = None,
    ):
        """
        Reorient particles given a direction vector and spin angle.

        This method aligns the +z axis with the specified `direction_vector` and then applies
        a spin rotation about the new axis. The transformation is applied to the specified
        vector fields (e.g., ``particle_position``, ``particle_velocity``) in the specified particle groups.

        The transformation uses a two-step process:

        1. **Alignment**: Compute the minimal rotation that aligns the +z axis with the
              normalized `direction_vector`.
        2. **Spin**: Apply a rotation by `spin` radians about the new z-axis (aligned with `direction_vector`).

        The final rotation matrix is the composition of the alignment and spin rotations.

        Parameters
        ----------
        direction_vector : array_like
            Target direction to become the new +z axis. Need not be unit, but must be nonzero.
        spin : float
            Spin angle (in radians) applied about the new axis after alignment.
        groups : list of str, optional
            Particle groups to transform. If None, all groups are used.
        fields : tuple of str, optional
            Vector fields to transform. Defaults to ``("particle_position", "particle_velocity")``.

        Returns
        -------
        numpy.ndarray
            The 3x3 rotation matrix applied: ``R = R_spin @ R_align``.
        """
        # Set the default fields to all of the standard vector fields we
        # expect.
        if fields is None:
            fields = (self.__class__._POSITION_FIELD_NAME, self.__class__._VELOCITY_FIELD_NAME)

        # Validate and normalize the direction vector for
        # the target axis.
        direction_vector = np.asarray(direction_vector, dtype=float)
        if direction_vector.ndim != 1 or direction_vector.shape[0] != 3:
            raise ValueError("Direction vector must be a 3-element vector.")
        dv_norm = np.linalg.norm(direction_vector)

        if dv_norm <= 1e-8:
            raise ValueError("Direction vector must be nonzero.")
        _dv = direction_vector / dv_norm  # unit target axis

        # Construct the necessary data to construct the
        # relevant rotation matrices.
        z_axis = np.array([0.0, 0.0, 1.0])
        cross = np.cross(z_axis, _dv)
        sin_phi = np.linalg.norm(cross)  # = sin(tilt)
        cos_phi = float(np.dot(z_axis, _dv))  # = cos(tilt)

        if sin_phi < 1e-12 and cos_phi > 0.0:
            # Already aligned
            R_align = np.eye(3)
        elif sin_phi < 1e-12 and cos_phi < 0.0:
            # Exactly opposite: rotate pi about any axis ⟂ z (choose x)
            Kx = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])  # skew([1,0,0])
            R_align = np.eye(3) + np.sin(np.pi) * Kx + (1 - np.cos(np.pi)) * (Kx @ Kx)
        else:
            # General case: axis = unit(cross), angle = atan2(sin_phi, cos_phi)
            rot_axis = cross / sin_phi
            x, y, z = rot_axis
            K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            # Use sin/cos of angle directly via sin_phi/cos_phi (avoids recomputing trig)
            R_align = np.eye(3) + sin_phi * K + (1 - cos_phi) * (K @ K)

        # After alignment, the new z-axis is exactly _dv.
        # Step 2: Spin by `spin` about _dv (which is world-space axis after R_align)
        x, y, z = _dv
        Kspin = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        s = np.sin(spin)
        c = np.cos(spin)
        R_spin = np.eye(3) + s * Kspin + (1 - c) * (Kspin @ Kspin)

        # Compose to "align then spin" (active, column-vector convention)
        R = R_spin @ R_align

        # Apply to requested particle groups/fields
        self.apply_linear_transformation(R, groups=groups, fields=fields)

        return R

    def cut_particles_to_bbox(
        self,
        bbox: unyt.unyt_array,
        groups: list[str] = None,
        center: unyt.unyt_array = None,
    ):
        """
        Cut out particles outside the specified bounding box.

        This method takes a **cartesian** bounding box defined by `bbox` and
        checks the positions of all particles in all groups. Particles that lie
        outside of the bounding box are removed from the dataset.

        Parameters
        ----------
        bbox: ~unyt.array.unyt_array
            A 2D unyt array of shape (2, D) defining the bounding box in D dimensions.
            The first row specifies the minimum corner, and the second row specifies
            the maximum corner. Units must be compatible with particle positions.
        groups: list of str, optional
            List of particle group names to apply the cut to. If None, all groups are used.
        center: ~unyt.array.unyt_array, optional
            An optional center point to offset the bounding box. If provided, the bounding
            box is shifted by this center before applying the cut. Units must be compatible
            with particle positions.
        """
        # Validate the particle positions field so that
        # we can uniformly access it as a dictionary.
        groups = groups if groups is not None else self.particle_groups

        # manage center if it needs to be managed.
        if center is None:
            center = unyt.unyt_array([0.0] * bbox.shape[1], bbox.units)
        else:
            center = unyt.unyt_array(center)

        for particle_type in self.particle_groups:
            # Check if we are processing this group.
            if particle_type not in groups:
                continue

            # Extract the position array for this particle type
            # so that we can determine the dimension and eventually
            # obtain the mask.
            position_field_handle = self.get_particle_field_handle(particle_type, self._POSITION_FIELD_NAME)
            position_field_units = self.get_field_units(particle_type, self._POSITION_FIELD_NAME)
            ndim = position_field_handle.shape[-1]

            # Check that the number of dimensions is compatible with the
            # bounding box.
            if bbox.shape != (2, ndim):
                raise ValueError(
                    f"Bounding box shape {bbox.shape} is incompatible with "
                    f"particle positions of dimension {ndim} in group '{particle_type}'."
                )
            if center.shape != (ndim,):
                raise ValueError(
                    f"Center shape {center.shape} is incompatible with "
                    f"particle positions of dimension {ndim} in group '{particle_type}'."
                )

            # We now want to make a local copy of the bounding box in the correct
            # units for the position field.
            _bbox_unitless = bbox.to_value(position_field_units)
            _center_unitless = center.to_value(position_field_units)

            # We now generate the mask against the bounding box.
            mask = np.all(
                [
                    (_bbox_unitless[0, _k] <= position_field_handle[:, _k] + _center_unitless[_k])
                    & (_bbox_unitless[1, _k] >= position_field_handle[:, _k] + _center_unitless[_k])
                    for _k in range(ndim)
                ],
                axis=0,
            )

            # With the mask, we can now reduce the particle group accordingly.
            self.reduce_group(particle_type, mask)

    def add_particle_ids(self, groups: list[str] = None, policy: str = None, overwrite: bool = False, **kwargs):
        """
        Add unique particle IDs to specified groups.

        Parameters
        ----------
        groups: list of str, optional
            List of particle group names to add particle ids to. If None, all groups are used.
            This can be used both to specify the groups that should be given an ID field and
            (for ``policy='global'``) to specify the order in which the IDs are assigned to each
            group.
        policy : {"global", "per_group"}, optional
            The policy for assigning particle IDs. Options are:

            - "global": Assign unique IDs across all specified groups, ensuring no duplicates.
              IDs are assigned sequentially starting from 1, in the order of groups provided.
            - "per_group": Assign unique IDs within each group, starting from 1 for each group.
              This allows for duplicate IDs across different groups.

            Default is "global".
        overwrite: bool, optional
            Whether to overwrite existing ID fields if they already exist. Defaults to False.
        kwargs:
            Additional keyword arguments passed to the ID generation function. The following
            are recognized:

            - ``start_id``: int, optional
                The starting ID number for the first group (default is 1).
            - ``dtype``: numpy dtype, optional
                The numpy dtype to use for the ID field (default is np.uint32).

        Notes
        -----
        Behind the scenes, this method will do two things:

        1. Create a "particle id" group (named following the class's ``_ID_FIELD_NAME`` attribute) which contains
           an ordered list of all of the particle IDs for that particle type. Depending on the policy,
           these ids will either start from 1 for each group or will be globally unique across all groups.
        2. Each group will get a ``PIDOFF`` attribute, which indicates the PID of the very first particle in
           that group.

        """
        # Validate the input information, set up the groups, pull the ID
        # policy, and the starting ID.
        groups = groups if groups is not None else self.particle_groups
        if any(grp not in self.particle_groups for grp in groups):
            raise ValueError("All specified groups must exist in the dataset.")

        policy = policy if policy is not None else self.__class__._ID_POLICY
        if policy not in ("global", "per_group"):
            raise ValueError("Policy must be either 'global' or 'per_group'.")

        start_id = kwargs.get("start_id", 1)
        dtype = kwargs.get("dtype", np.uint32)

        # Now for each group in the list of groups, we go through and add the
        # relevant field. If we encounter an existing field and overwrite is False,
        # we raise an error.
        _offset = start_id
        for group in groups:
            # Extract the number of particles in this group
            num_particles = self.get_group_metadata(group)["NUMBER_OF_PARTICLES"]

            # Generate the IDs based on the policy.
            _id_array = np.arange(_offset, _offset + num_particles, dtype=dtype)
            self.add_particle_field(
                group_name=group,
                field_name=self.__class__._ID_FIELD_NAME,
                data=unyt.unyt_array(_id_array, ""),
                overwrite=overwrite,
            )
            self.update_group_metadata(group, {"PIDOFF": _offset})

            if policy == "global":
                _offset += num_particles
            else:
                pass

    def update_particle_ids(self, groups: list[str] = None, policy: str = None, **kwargs):
        """
        Update (reset) particle IDs for groups that already have an ID field.

        This method goes through each of the specified groups and checks for the existence
        of a particle ID field. If the field exists, it is overwritten with a new set of IDs
        according to the specified policy. If the field does not exist, the group is skipped
        silently.

        For ``policy='global'``, IDs are assigned sequentially across all specified groups. As such,
        even if a group does not have particle ids, it's total number of particles is still counted
        towards the global ID assignment.

        Parameters
        ----------
        groups: list of str, optional
            List of particle group names to add particle ids to. If None, all groups are used.
            This can be used both to specify the groups that should be given an ID field and
            (for ``policy='global'``) to specify the order in which the IDs are assigned to each
            group.
        policy : {"global", "per_group"}, optional
            The policy for assigning particle IDs. Options are:

            - "global": Assign unique IDs across all specified groups, ensuring no duplicates.
              IDs are assigned sequentially starting from 1, in the order of groups provided.
            - "per_group": Assign unique IDs within each group, starting from 1 for each group.
              This allows for duplicate IDs across different groups.

            Default is "global".
        kwargs:
            Additional keyword arguments passed to the ID generation function. The following
            are recognized:

            - ``start_id``: int, optional
                The starting ID number for the first group (default is 1).
            - ``dtype``: numpy dtype, optional
                The numpy dtype to use for the ID field (default is np.uint32).

        Notes
        -----
        - Groups without an existing ID field are skipped silently.
        - This operation **overwrites** the existing ID field for selected groups.
        - IDs are always assigned as **1-based** positive integers, consistent with
          Gadget and most downstream analysis tools.
        """
        # Validate the input information, set up the groups, pull the ID
        # policy, and the starting ID.
        groups = groups if groups is not None else self.particle_groups
        if any(grp not in self.particle_groups for grp in groups):
            raise ValueError("All specified groups must exist in the dataset.")

        policy = policy if policy is not None else self.__class__._ID_POLICY
        if policy not in ("global", "per_group"):
            raise ValueError("Policy must be either 'global' or 'per_group'.")

        start_id = kwargs.get("start_id", 1)
        dtype = kwargs.get("dtype", np.uint32)

        # Track running offset for global assignment
        offset = start_id
        id_field = self.__class__._ID_FIELD_NAME

        for group in groups:
            n = int(self.get_group_metadata(group)["NUMBER_OF_PARTICLES"])

            if f"{group}.{id_field}" in self:
                # Group has an existing ID field → reset it
                if policy == "global":
                    ids = np.arange(offset, offset + n, dtype=dtype)
                    self.update_group_metadata(group, {"PIDOFF": offset})
                else:  # per_group
                    ids = np.arange(start_id, start_id + n, dtype=dtype)
                    self.update_group_metadata(group, {"PIDOFF": start_id})

                self.add_particle_field(
                    group_name=group,
                    field_name=id_field,
                    data=unyt.unyt_array(ids, ""),  # dimensionless
                    overwrite=True,
                )

            # For global policy: always advance the offset,
            # even if the group did not have an ID field.
            if policy == "global":
                offset += n

    # ------------------------------------- #
    # Generation Methods                    #
    # ------------------------------------- #
    # noinspection PyMissingOrEmptyDocstring
    @overload
    @classmethod
    def build_particle_dataset(
        cls,
        path: str | Path,
        fields: dict[str, unyt.unyt_array] = None,
        overwrite: bool = False,
    ) -> "ParticleDataset": ...

    @classmethod
    def build_particle_dataset(
        cls,
        *args,
        **kwargs,
    ):
        """Create a new :class:`ParticleDataset` HDF5 file with the given fields.

        This method initializes a new HDF5 file, organizes the provided fields into
        groups based on their dot notation names (e.g., ``"baryons.particle_mass"``),
        and writes each field to the appropriate group. Each group must have fields
        with the same number of particles (i.e., matching leading dimension).
        Global metadata (such as creation date) is also written.

        This is the standard factory method for creating new Pisces-compatible particle datasets.

        Parameters
        ----------
        path : str or pathlib.Path
            The target path for the new HDF5 file.
        fields : dict of {str: unyt.array.unyt_array}, optional
            A dictionary mapping dot-notation field names to unyt arrays.
            If None or empty, a valid file with metadata but no particle data is created.
        overwrite : bool, optional
            Whether to overwrite the file if it already exists. Defaults to False.
        *args, **kwargs:
            Additional positional and keyword arguments passed to the dataset constructor.

        Returns
        -------
        ParticleDataset
            An instance of the newly created dataset.

        Raises
        ------
        FileExistsError
            If the file exists and `overwrite` is False.
        IsADirectoryError
            If `path` is a directory.
        ValueError
            If any field name is not in dot notation or group fields mismatch in particle count.

        """
        (
            path,
            *args,
        ) = args
        fields, overwrite = kwargs.pop("fields", None), kwargs.pop("overwrite", False)
        path = Path(path)

        # --- Path validation ---
        if path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {path}. Use overwrite=True to replace it.")
        elif path.exists() and overwrite:
            path.unlink()
        elif path.is_dir():
            raise IsADirectoryError(f"Path is a directory: {path}")

        path.parent.mkdir(parents=True, exist_ok=True)

        # --- Create HDF5 file ---
        with h5py.File(path, "w") as f:
            # Add required global metadata
            initial_metadata = {
                "GEN_TIME": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "CLASS_NAME": cls.__name__,
            }
            for k, v in cls.metadata_serializer.serialize_dict(initial_metadata).items():
                f.attrs[k] = v

            if fields is not None:
                group_registry: dict[str, int] = {}
                fields = dict(fields)

                for full_field_name, data in fields.items():
                    # Validate field name format
                    try:
                        group_name, field_name = full_field_name.split(".")
                    except ValueError as err:
                        raise ValueError(
                            f"Invalid field name '{full_field_name}'. Expected format 'group.field'."
                        ) from err

                    # Validate data type
                    if not isinstance(data, unyt.unyt_array):
                        raise TypeError(f"Field '{full_field_name}' must be a unyt_array.")

                    # Create or validate the group
                    if group_name not in group_registry:
                        group = f.create_group(group_name)
                        group_registry[group_name] = data.shape[0]
                        group.attrs["NUMBER_OF_PARTICLES"] = cls.metadata_serializer.serialize_data(data.shape[0])
                    else:
                        if data.shape[0] != group_registry[group_name]:
                            raise ValueError(
                                f"Inconsistent particle count for field '{full_field_name}'. "
                                f"Expected {group_registry[group_name]}, got {data.shape[0]}"
                            )

                    # Create dataset and write units
                    dset = f[group_name].create_dataset(field_name, data=data, dtype=data.dtype)
                    dset.attrs["UNITS"] = cls.metadata_serializer.serialize_data(data.units)

        # Return a validated ParticleDataset instance
        return cls(path, **kwargs)
