"""Base class for particle datasets in Pisces."""

from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import unyt


class ParticleDataset:
    """Base class for particle datasets in Pisces.

    This class uses the standard HDF5 format from Gadget-2 to store particle data. Each
    particle type has its own group in the HDF5 file, and each group contains datasets for
    each of the fields. These datasets may each have an attribute specifying the units of the
    field.

    The :class:`ParticleDataset` class provides a common interface for reading and writing particle data,
    including lazy loading of data, accessing particle properties, and iterating over particles.

    File Format
    -----------

    Particle datasets in Pisces are stored in **HDF5 format**, where each particle type
    (e.g., ``"dark_matter"``, ``"baryons"``) is stored in its own HDF5 group. Each group contains
    fields (datasets) describing per-particle quantities. All fields must have a leading shape
    of ``(N_particles, ...)``, where ``N_particles`` is the number of particles in the group.

    The file structure is as follows:

    - **HDF5 Group**: Each particle type is stored in its own group. The group contains metadata
      specifying the number of particles and other attributes.
    - **HDF5 Datasets**: Each group contains datasets for each field, such as particle mass, position, etc.
      Each dataset may contain metadata attributes, such as units.

    Standard Field Names
    ''''''''''''''''''''''''''''''

    In general, there are no restrictions on the names of particle fields in a :class:`ParticleDataset`; However,
    Pisces follows a standard convention for naming common fields to ensure compatibility across the ecosystem. Misnamed
    fields may lead to errors in analysis or data processing, as Pisces utilities expect specific field names.

    The fields for which Pisces has standard names are contained in the following table.

    +---------------------+------------------+------------------------------------------------------------+
    | **Field Name**      | **Shape**        | **Description**                                            |
    +=====================+==================+============================================================+
    | ``particle_mass``   | (N_particles,)   | Mass of each particle                                      |
    +---------------------+------------------+------------------------------------------------------------+
    | ``particle_position``| (N_particles, 3)| 3D position vector of each particle                        |
    +---------------------+------------------+------------------------------------------------------------+
    | ``particle_velocity``| (N_particles, 3)| 3D velocity vector of each particle                        |
    +---------------------+------------------+------------------------------------------------------------+
    | ``particle_id``     | (N_particles,)   | Unique identifier for each particle                        |
    +---------------------+------------------+------------------------------------------------------------+

    In most other instances, any Pisces utility which requires a specific field will allow the user to
    specify / overwrite the default expected field name. This allows for flexibility in naming conventions
    while still providing a standard set of fields for common use cases.

    Group Metadata
    ''''''''''''''''''''''''''''''

    Each particle group may include the following attribute:

    - ``NUMBER_OF_PARTICLES`` : `int`
        The total number of particles in the group.

    Additional metadata attributes may also be present and are accessible via
    :attr:`ParticleDataset.group_metadata`. These are optional and not required
    by the base class.

    Field Metadata
    ''''''''''''''''''''''''''''''

    Each dataset (field) within a particle group may include the following attribute:

    - ``UNITS`` : `str`
        A string specifying the physical units of the field. This must be a valid
        `unyt` unit string (e.g., ``"Msun"``, ``"kpc"``, ``"km/s"``).

    Custom field-level metadata may also be stored and accessed via standard HDF5
    attributes.

    Global Metadata
    ''''''''''''''''''''''''''''''

    At the root level of the HDF5 file, the following attribute is supported:

    - ``CREATION_DATE`` : `str`
        The UTC date and time when the dataset was created, formatted as an
        ISO 8601 string (e.g., ``"2025-07-23T16:45:00Z"``).

    Additional global metadata may be included and is available via
    :attr:`ParticleDataset.metadata`. These attributes are optional and can be used
    to store cosmological parameters, simulation provenance, or software versioning
    information.


    """

    def __validate__(self):
        """Validate that the HDF5 file conforms to the standard format for particle datasets.

        This method checks the structure of the HDF5 file, ensuring that it contains
        the expected groups, datasets, and metadata attributes.

        It can be overridden in subclasses to implement custom validation logic.

        Raises
        ------
        ValueError
            If required metadata is missing or inconsistencies are found in the dataset.

        Notes
        -----
        At the level of the base class, this method checks for the following:

        - The presence of the global metadata attribute ``CREATION_DATE``.
        - Each particle group must have the attribute ``NUMBER_OF_PARTICLES``.
        - Each dataset in a particle group must have a leading dimension that matches
          the number of particles specified in the group's metadata.

        """
        # Validate global metadata
        if "CREATION_DATE" not in self.global_metadata:
            raise ValueError("Missing required global metadata attribute: 'CREATION_DATE'")

        # Validate each group
        for group_name in self.particle_groups:
            metadata = self.group_metadata[group_name]
            if "NUMBER_OF_PARTICLES" not in metadata:
                raise ValueError(f"Missing 'NUMBER_OF_PARTICLES' in group '{group_name}'")

            num = metadata["NUMBER_OF_PARTICLES"]
            group = self.handle[group_name]

            for field_name, dataset in group.items():
                if not hasattr(dataset, "shape") or dataset.shape[0] != num:
                    raise ValueError(f"Field '{field_name}' in group '{group_name}' must have leading dimension {num}")

    def __init__(self, path: str | Path, mode="r+"):
        """Initialize the ParticleDataset with the given path to the HDF5 file.

        Parameters
        ----------
        path : Union[str, Path]
            The path to the HDF5 file containing the particle data. This can be a string or a Path object.
        mode: str, optional
            The mode in which to open the HDF5 file. Defaults to "r+" (read/write mode).

        """
        # Set the path and open the handle to the HDF5 file.
        self.__path__ = Path(path)
        if not self.__path__.exists():
            raise FileNotFoundError(f"Particle dataset file not found: {self.__path__}")

        # Open the HDF5 file in the specified mode.
        self.__handle__ = h5py.File(self.__path__, mode=mode)

        # Check that the file is a valid particle dataset. This defers
        # to the __validate__ method to ensure that the file structure
        # and metadata conform to the expected format. This can be overridden
        # in subclasses to implement custom validation logic.
        self.__validate__()

        # Pass on to the post init method.
        self.__post_init__()

    def __post_init__(self):
        pass

    # ------------------------------------ #
    # Properties                           #
    # ------------------------------------ #
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
        return dict(self.__handle__.attrs)

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
    def group_metadata(self) -> dict[str, dict]:
        """Metadata attributes for each particle group.

        This includes attributes such as the number of particles in each group.
        Each group is represented as a dictionary with the group name as the key.

        Returns
        -------
        dict
            A dictionary mapping group names to their metadata attributes.

        """
        metadata = {}
        for group_name in self.particle_groups:
            group = self.__handle__[group_name]
            metadata[group_name] = dict(group.attrs)
        return metadata

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
        for group_name, metadata in self.group_metadata.items():
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
    def creation_date(self) -> datetime:
        """The creation date of the particle dataset.

        This property retrieves the creation date from the global metadata attribute
        ``CREATION_DATE``. The date is returned as a `datetime` object.

        Returns
        -------
        datetime
            The creation date of the dataset.

        """
        return self.global_metadata.get("CREATION_DATE")

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
        unyt.unyt_array
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
        unyt.unyt_array
            The data for the specified field, converted to a unyt array.

        Raises
        ------
        KeyError
            If the specified group or field does not exist in the dataset.

        """
        dataset_handle = self.get_particle_field_handle(group_name, field_name)
        return unyt.unyt_array(dataset_handle[...], units=dataset_handle.attrs.get("UNITS", ""))

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
        return unyt.Unit(dataset_handle.attrs.get("UNITS", ""))

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
        group.attrs["NUMBER_OF_PARTICLES"] = num_particles

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

        if field_name in group.keys():
            # The field already exists. Our behavior depends on the `overwrite` flag.
            if not overwrite:
                raise ValueError("Field already exists. Set `overwrite=True` to replace it.")
            else:
                del group[field_name]

        # Determine the number of particles expected and
        # ensure that the data matches this shape.
        num_particles = group.attrs.get("NUMBER_OF_PARTICLES")
        data = np.atleast_1d(data)
        if data.shape[0] != num_particles:
            raise ValueError(
                f"Number of particles in field {field_name} does not match number of particles in group {group_name}"
            )

        # Validation has been completed and we can therefore now
        # proceed with writing the field to the group.
        unit_string = str(getattr(data, "units", ""))

        if isinstance(data, unyt.unyt_array):
            dset = group.create_dataset(field_name, data=data.d, dtype=data.dtype)
        else:
            dset = group.create_dataset(field_name, data=data, dtype=data.dtype)

        # Handle the metadata.
        dset.attrs["UNITS"] = unit_string
        if metadata is not None:
            for key, value in metadata.items():
                dset.attrs[key] = value

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
        if not isinstance(num_particles, int) or num_particles <= 0:
            raise ValueError("`num_particles` must be a positive integer.")

        # Create the field dictionary and
        # modify the group attribute.
        fields = fields or {}
        old_particle_count = group.attrs["NUMBER_OF_PARTICLES"]
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
                    field_data = field_data.to_value(group[field_name].attrs.get("UNITS", ""))
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

    def reduce_group(self, group_name: str, mask: np.ndarray | unyt.unyt_array):
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
        n = group.attrs["NUMBER_OF_PARTICLES"]

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

        group.attrs["NUMBER_OF_PARTICLES"] = new_count

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

        This method adds the given offset vector to the ``particle_position`` field
        of each specified group. If `groups` is not provided, the offset is applied to all groups.

        This is the correct way to shift particle coordinates around via translation.

        Parameters
        ----------
        offset : unyt.unyt_array
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
            field_key = f"{group}.particle_position"

            if field_key not in self:
                continue

            # Obtain the handle and the units.
            handle = self.get_particle_field_handle(group, "particle_position")
            units = self.get_field_units(group, "particle_position")

            # Check the shape.
            if handle.shape[-1] != len(offset):
                raise ValueError(f"Offset must match the shape of particle positions in group '{group}'.")

            # Apply the offset.
            handle[...] += offset.to_value(units)

    def offset_particle_velocities(self, offset: unyt.unyt_array, groups: list[str] = None):
        """Apply a constant offset to particle velocities in specified groups.

        This method adds the given offset vector to the ``particle_velocity`` field
        of each specified group. If `groups` is not provided, the offset is applied to all groups.

        This is the correct way to impart a bulk velocity or center-of-mass frame shift.

        Parameters
        ----------
        offset : unyt.unyt_array
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
            field_key = f"{group}.particle_velocity"

            if field_key not in self:
                continue

            # Access the velocity dataset and its units.
            handle = self.get_particle_field_handle(group, "particle_velocity")
            units = self.get_field_units(group, "particle_velocity")

            # Confirm shape match.
            if handle.shape[-1] != len(offset):
                raise ValueError(f"Offset must match the shape of particle velocities in group '{group}'.")

            # Apply the velocity offset.
            handle[...] += offset.to_value(units)

    def apply_linear_transformation(
        self,
        matrix: np.ndarray,
        groups: list[str] = None,
        fields: tuple[str, ...] = ("particle_position", "particle_velocity"),
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
        fields: tuple[str, ...] = ("particle_position", "particle_velocity"),
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

    # ------------------------------------- #
    # Generation Methods                    #
    # ------------------------------------- #
    @classmethod
    def build_particle_dataset(
        cls,
        path: str | Path,
        fields: dict[str, unyt.unyt_array] = None,
        *args,
        overwrite: bool = False,
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
        fields : dict of {str: unyt.unyt_array}, optional
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
            f.attrs["CREATION_DATE"] = datetime.utcnow().isoformat()

            if fields:
                group_registry: dict[str, int] = {}

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
                        group.attrs["NUMBER_OF_PARTICLES"] = data.shape[0]
                    else:
                        if data.shape[0] != group_registry[group_name]:
                            raise ValueError(
                                f"Inconsistent particle count for field '{full_field_name}'. "
                                f"Expected {group_registry[group_name]}, got {data.shape[0]}"
                            )

                    # Create dataset and write units
                    dset = f[group_name].create_dataset(field_name, data=data, dtype=data.dtype)
                    dset.attrs["UNITS"] = str(data.units)

        # Return a validated ParticleDataset instance
        return cls(path, *args, **kwargs)
