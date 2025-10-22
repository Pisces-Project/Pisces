"""
Gadget-style particle dataset support.

This module provides classes and methods for handling particle datasets
formatted in a Gadget-like manner. It includes functionality for reading,
writing, and manipulating such datasets, with a focus on compatibility
with the Gadget simulation code and its derivatives. The particle dataset
classes in this module maintain the Pisces standard of descending from :class:`~particles.base.ParticleDataset`
and therefore support all of the standard particle dataset operations.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Union, overload

import h5py
import numpy as np
import unyt
from particles import ParticleDataset
from utilities.io_tools import NullHDF5Serializer


class GadgetLikeParticleDataset(ParticleDataset, ABC):
    """
    Base class for Gadget/AREPO–style particle datasets.

    This class implements the conventions and file layout used by Gadget-like
    particle-based codes (e.g., Gadget-2, Gadget-4, AREPO). It provides
    standard field names, default units, and header structures so that
    simulation initial conditions and snapshots can be generated, inspected,
    or modified in a consistent way.
    """

    metadata_serializer = NullHDF5Serializer
    """Serializer for Gadget particle metadata.

    Gadget particle datasets do not require any special metadata
    serialization, so a no-op (null) serializer is used. This means that
    no specialized metadata classes are permissible. Additionally, we now
    manually handle unit string conversion on dataset attributes because it
    is not handled by the serializer.
    """

    # --------------------------------- #
    # Class Constants / Flags           #
    # --------------------------------- #
    # These flags provide standard names for built-in access to specific
    # fields in the dataset. This ensures that those areas of the code are
    # not hardcoded with string literals. These can be overridden in
    # subclasses if the dataset uses different naming conventions.

    # --- Field Name Conventions --- #
    _POSITION_FIELD_NAME = "Coordinates"
    _VELOCITY_FIELD_NAME = "Velocities"
    _MASS_FIELD_NAME = "Masses"
    _ID_FIELD_NAME = "ParticleIDs"
    _INTERNAL_ENERGY_FIELD_NAME = "InternalEnergy"
    _DENSITY_FIELD_NAME = "Density"

    # --- Class Settings --- #
    _default_ntypes: int = 6
    """int: Default number of Gadget-like particle types.

    By default, this is set to 6, corresponding to the standard Gadget
    particle types; however, some simulations may have a different number.
    This can also be specified by the user as a kwarg when building a dataset.
    """
    _default_unit_system = unyt.unit_systems.galactic_unit_system
    """Default unit system for interpreting Gadget particle fields.
    Gadget itself does not enforce units; this is provided for Pisces
    consistency and metadata export.
    """
    _header_group_name: str = "Header"
    """str: Name of the group containing the Gadget-like file header.

    This can be overridden in subclasses if the header group has a different name.
    """
    _particle_type_name_prefix: str = "PartType"
    """str: Prefix for particle type groups in the HDF5 file.

    This can be overridden in subclasses if the particle type groups have a different prefix.
    """

    # ------------------------------------ #
    # Properties                           #
    # ------------------------------------ #
    @property
    def header_handle(self) -> h5py.Group:
        """Access the ``Header`` group of the Gadget particle dataset.

        Returns
        -------
        h5py.Group
            The HDF5 group object corresponding to the ``Header`` group.

        Raises
        ------
        KeyError
            If the ``Header`` group does not exist in the file.
        """
        return self.__handle__[self.__class__._header_group_name]

    @property
    def header(self) -> dict:
        """Retrieve the attributes of the ``Header`` group as a dictionary.

        Returns
        -------
        dict
            A dictionary containing the attributes of the ``Header`` group.

        Raises
        ------
        KeyError
            If the ``Header`` group does not exist in the file.
        """
        return dict(self.header_handle.attrs)

    # ------------------------------------ #
    # Modification Methods                 #
    # ------------------------------------ #
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
        dset.attrs["UNITS"] = str(units)
        if metadata is not None:
            dset.attrs.update(self.metadata_serializer.serialize_dict(metadata))

    # ------------------------------------- #
    # Generation Methods                    #
    # ------------------------------------- #
    @classmethod
    def generate_file_skeleton(
        cls,
        path: Path,
        number_of_particles: np.ndarray,
        box_size: unyt.unyt_quantity,
        unit_system: unyt.UnitSystem,
        **kwargs,
    ):
        """
        Generate the basic file skeleton for a Gadget-like particle dataset.

        This method creates the HDF5 file and populates it with the necessary
        groups and datasets to conform to the Gadget-like format. It handles
        writing the required metadata, file header, and particle type groups.

        Parameters
        ----------
        path: ~pathlib.Path
            The file path where the HDF5 file will be created. There is no overwrite safety
            in this method, so the caller must ensure that the path is valid. If it already exists,
            then the data will be overwritten.
        number_of_particles: numpy.ndarray
            The number of particles of each type to include in the file. This should be an array
            of length equal to the number of particle types (default is 6 for Gadget-like files).

            If a particle type has zero particles, its group will not be created.
        box_size: unyt.unyt_quantity
            The size of the simulation box. This should be a unyt quantity with units of length.
        unit_system: unyt.unit_systems.UnitSystem
            The unit system in which to write the particle fields. This is used to set up the units
            of each of the fields in the file.
        kwargs:
            Additional keyword arguments to pass to the header and particle field generation methods.
            This can include any parameters that are needed to customize the file header or
            particle fields, such as cosmological parameters, time information, etc.

        Notes
        -----
        This is a 3 step process:

        1. Create the HDF5 file and write the Pisces metadata.
        2. Write the Gadget-like file header.
        3. Create the particle type groups and populate them with the required fields.

        """
        # Construct the HDF5 file object. Each of the steps in
        # the process of creating / modifying the file is handled by
        # a separate submethod to keep things clean. These can be
        # modified in subclasses as needed. Additionally, new processes
        # can be added by overriding this method, calling super() and then
        # re-opening the file to add new features.
        with h5py.File(path, "w") as stream:
            # Write the required pisces metadata into the file's
            # base directory attributes so that it can be recognized as
            # a valid pisces dataset.
            cls._generate_pisces_metadata(stream)

            # Write the Gadget-like header. This section of the code may require
            # considerable modification in subclasses to ensure that the header
            # maintains the correct structures. See notes in `_generate_file_header`.
            header_group = stream.require_group(cls._header_group_name)
            header_group.attrs["NOT_PARTICLE_GROUP"] = True
            cls._generate_file_header(header_group, number_of_particles, box_size, unit_system, **kwargs)

            # With the basic skeleton in place, we can now iterate through
            # all of the particle types and generate the necessary fields for
            # each type. This ensures that the file is fully compliant with
            # the Gadget-like format.
            for _type, _count in enumerate(number_of_particles):
                # If the count is zero or lower, we just skip and don't do anything.
                if _count <= 0:
                    continue

                # We do have particles of this type, so we ensure that the group
                # for these particles exists.
                _particle_group = stream.require_group(f"{cls._particle_type_name_prefix}{_type}")
                _particle_group.attrs["NUMBER_OF_PARTICLES"] = np.uint32(_count)

                # We now offload to ``_generate_particle_fields`` to handle generating
                # the fields individually. This permits subclasses to minimally modify the
                # the fields being generated, including taking options and manipulating the fields
                # that are present.
                cls._generate_particle_fields(_particle_group, _type, _count, unit_system, **kwargs)

            # The file skeleton is now complete save for any additional features
            # which can be added by subclasses.
        return

    @classmethod
    def _generate_pisces_metadata(cls, stream: h5py.File):
        """Write required Pisces metadata to the HDF5 file.

        In order for the file to be recognized as a valid Pisces dataset, it
        requires that some metadata be stored in the root attributes of the
        HDF5 file. This method handles writing that metadata.

        By default, we only write the GEN_TIME and the CLASS_NAME attributes,
        which are the minimal requirements for particle datasets in Pisces. This
        also ensures that users can read these particle files as datasets later.

        Parameters
        ----------
        stream : h5py.File
            The HDF5 file object to write metadata to.

        Notes
        -----
        In principle, this method can be extended in order to include additional
        required metadata, but it is generally not necessary.
        """
        # At least in this base class, we only need to keep track of the generation
        # time and the class name for the particle dataset. This is then used
        # to ensure that we can read from disk correctly later.
        initial_metadata = {
            "GEN_TIME": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "CLASS_NAME": cls.__name__,
        }
        for k, v in cls.metadata_serializer.serialize_dict(initial_metadata).items():
            stream.attrs[k] = v

    @classmethod
    @abstractmethod
    def _generate_file_header(
        cls,
        stream: h5py.Group,
        number_of_particles: np.ndarray,
        box_size: unyt.unyt_quantity,
        unit_system: unyt.UnitSystem,
        **kwargs,
    ):
        """Write the Gadget-like file header to the HDF5 file.

        In Gadget-like particle datasets, there is an additional group called
        with a specific name (generally "Header") that contains metadata about the
        simulation snapshot. This method handles writing that header group and
        populating it with the required attributes and datasets.

        Because the structure of the header can vary so much with the structure of
        the simulation code, this method is almost always overridden in subclasses. We provide
        a template here to illustrate the general structure, but it is not intended
        to be used directly.
        """
        return

    @classmethod
    @abstractmethod
    def _generate_particle_fields(
        cls,
        particle_group: h5py.Group,
        particle_type: int,
        number_of_particles: int,
        unit_system: unyt.UnitSystem,
        **kwargs,
    ):
        """Generate the required fields for a given particle type.

        This method is responsible for creating the datasets within a
        particle type group (e.g., "PartType0") that store the actual
        particle data. The specific fields and their structures can vary
        significantly between different simulation codes, so this method
        is abstract and must be implemented in subclasses.

        Parameters
        ----------
        particle_group : h5py.Group
            The HDF5 group corresponding to the particle type (e.g., "PartType0").
        particle_type : int
            The integer index of the particle type (e.g., 0 for gas, 1 for dark matter).
        number_of_particles : int
            The number of particles of this type.
        unit_system : unyt.UnitSystem
            The unit system to use for interpreting the fields.

        Notes
        -----
        This method should create datasets within `particle_group` for each of the
        required fields, using the appropriate data types and shapes. It should also
        handle any necessary unit conversions or metadata attributes.
        """
        return

    @classmethod
    def _add_particle_field_to_file(
        cls,
        particle_group: h5py.Group,
        field_name: str,
        number_of_particles: int,
        field_shape: tuple,
        field_dtype: np.dtype,
        field_units: Union[str, unyt.Unit],
        **kwargs,
    ) -> h5py.Dataset:
        """
        Add a new particle field dataset to a particle group.

        This is a low level operation which abstracts out elements of dataset
        creation that may lead to code duplication and / or errors when subclassing.

        Parameters
        ----------
        particle_group : h5py.Group
            The HDF5 group representing the particle type.
        field_name : str
            Name of the dataset to create (e.g., ``"Coordinates"``).
        number_of_particles : int
            Number of particles in the group (size of the leading dimension).
        field_shape : tuple of int
            Shape of a single particle entry (e.g., ``(3,)`` for vectors).
        field_dtype : numpy.dtype
            Data type of the dataset (e.g., ``np.float64``).
        field_units : str or ~unyt.Unit
            Units associated with the dataset. Stored in the ``UNITS`` attribute.
        **kwargs :
            Extra keyword arguments passed to :func:`h5py.Group.create_dataset`.

        Returns
        -------
        h5py.Dataset
            The newly created dataset.

        Raises
        ------
        ValueError
            If the dataset already exists and ``overwrite`` is False.
        """
        if field_name in particle_group:
            raise ValueError(f"Field '{field_name}' already exists in group '{particle_group.name}'.")

        # Create dataset
        dset = particle_group.create_dataset(
            field_name,
            shape=(number_of_particles, *field_shape),
            dtype=field_dtype,
            **kwargs,
        )

        # Attach required units attribute
        dset.attrs["UNITS"] = str(field_units)

        return dset

    @overload
    @classmethod
    def build_particle_dataset(
        cls,
        path: Union[str, Path],
        number_of_particles: Sequence,
        box_size: unyt.unyt_quantity,
        *,
        ntypes: int = 6,
        unit_system: unyt.UnitSystem = None,
        overwrite: bool = False,
        fields: dict = None,
    ) -> "GadgetLikeParticleDataset": ...

    @classmethod
    def build_particle_dataset(cls, *args, **kwargs):
        """
        Create a new Gadget-like particle dataset on disk.

        This public factory method initializes a new HDF5 file containing the
        header, particle groups, and required fields in the Gadget/AREPO format.
        It is the main entry point for generating initial condition files or
        empty snapshot skeletons.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination path for the new HDF5 file. If the file already exists
            and ``overwrite`` is False, a :class:`FileExistsError` is raised.
        number_of_particles : sequence of int
            The number of particles of each type to include in the particle dataset.
            This should be a sequence (e.g., list or tuple) of length ``ntypes``,
            each element of which is a non-negative integer. If a particle type
            has zero particles, its group will not be created.
        box_size : unyt.unyt_quantity
            The physical size of the simulation box. Must carry length units.
        ntypes : int, default=6
            Number of particle types to include. Gadget-like codes use six by
            default (0=gas, 1=DM, 2=disk, 3=bulge, 4=stars, 5=BH).
        unit_system : unyt.UnitSystem, optional
            Unit system for tagging dataset attributes. If not provided, the
            class default is used.
        overwrite : bool, default=False
            If True, overwrite any existing file at ``path``. If False, raise
            :class:`FileExistsError` if the file exists.
        fields : dict, optional
            Extra particle fields to populate. Keys must be of the form
            ``"PartTypeX.FieldName"`` where X is the particle type index.
            Values must be :class:`unyt.array.unyt_array` or :class:`numpy.ndarray`
            with the correct leading dimension.

        Returns
        -------
        GadgetLikeParticleDataset
            A dataset object representing the newly created HDF5 file.

        Raises
        ------
        FileExistsError
            If the target file exists and ``overwrite`` is False.
        ValueError
            If the number of particles does not match ``ntypes`` or if a field
            shape is inconsistent with its particle group.
        TypeError
            If ``unit_system`` is not a :class:`unyt.UnitSystem` instance.

        Notes
        -----
        - This overload describes the base API. Subclasses may add further
          keyword arguments for physics-specific fields (e.g. MHD, cooling).
        - At minimum, each populated particle type will include the required
          datasets: ``Coordinates``, ``Velocities``, ``ParticleIDs``, and
          ``Masses``. Gas cells (PartType0) also include ``Density`` and
          ``InternalEnergy``.

        """
        # --- Input Validation --- #
        # To begin, we need to sort out the various args and kwargs that get passed through. We do all of
        # this to permit a flexible / self-consistent interface for subclasses, but it comes at the cost
        # of some boiler plate code here.
        try:
            path, number_of_particles, box_size = args
        except ValueError as err:
            raise ValueError(
                "`path`, `number_of_particles`, and `box_size` must be provided as positional arguments."
            ) from err

        # Handle the path and overwrite management first. We attempt to
        # coerce the path to a Path object, then check if it exists.
        path = Path(path)
        overwrite = kwargs.pop("overwrite", False)

        if path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {path}. Use overwrite=True to replace it.")
        elif path.exists() and overwrite:
            path.unlink()
        elif path.is_dir():
            raise IsADirectoryError(f"Path is a directory: {path}")

        path.parent.mkdir(parents=True, exist_ok=True)

        # We now ensure that the number of particles is valid and of the
        # right length. We do permit users to provide us with a count of the number of
        # particle types that is different from the default, but we do need to ensure
        # that the provided number of particles matches the expected number of particle types.
        _expected_ntypes = kwargs.get("number_of_particle_types", cls._default_ntypes)

        # coerce the number of particles to a list if it is not already
        number_of_particles = np.asarray(number_of_particles, dtype=np.uint32)
        if number_of_particles.shape != (_expected_ntypes,):
            raise ValueError(
                f"`number_of_particles` must be of length {_expected_ntypes} to"
                f" match the expected number of particle types."
            )

        # Configure the unit system so that we have one available at all
        # times. This is taken from the class defaults if it is not provided.
        unit_system = kwargs.pop("unit_system", cls._default_unit_system)
        if not isinstance(unit_system, unyt.UnitSystem):
            raise TypeError("`unit_system` must be a unyt.UnitSystem instance.")

        # We now process the box size to ensure that it is a correctly specified unyt
        # quantity with units of length.
        elif isinstance(box_size, unyt.unyt_quantity):
            box_size = unyt.unyt_array(box_size.in_base(unit_system), dtype=np.float64)
        else:
            box_size = unyt.unyt_array(box_size * unit_system["length"], dtype=np.float64)

        # Pop out fields so its not contaminant in the kwargs.
        fields = kwargs.pop("fields", None)

        # --- File Generation --- #
        # At this stage, we can proceed to generate the file skeleton. In doing
        # so, we will first create the file, then write the Pisces metadata, then
        # the Gadget header, then the file skeleton, and finally any fields that
        # are provided to us.
        cls.generate_file_skeleton(
            path,
            number_of_particles,
            box_size,
            unit_system,
            **kwargs,  # pass through any additional kwargs to the generation methods.
        )

        # --- Field Population --- #
        # If fields are provided, we can now populate them.
        # These are provided as `<particle_type_name>.<field_name>`, so
        # we can just split the names, then dump them to the correct dataset
        # after some validation checks.
        if fields is not None:
            fields = dict(fields)

            with h5py.File(path, "r+") as stream:
                # Cycle through all of the provided fields and
                # populate them.
                for field_name, field_data in fields.items():
                    # Split the field name into particle type and
                    # actual field name.
                    try:
                        ptype_name, f_name = field_name.split(".")
                    except ValueError as exp:
                        raise ValueError(
                            f"Field name '{field_name}' is not in the correct format."
                            " It must be of the form '<particle_type>.<field_name>'"
                        ) from exp

                    # Validate that the particle type group exists, and that
                    # the field also exists as expected.
                    if ptype_name not in stream:
                        raise ValueError(f"Particle type group '{ptype_name}' does not exist in the file.")
                    if f_name not in stream[ptype_name]:
                        raise ValueError(f"Field '{f_name}' does not exist in particle type group '{ptype_name}'.")

                    # Validate that the shape of the provided data matches
                    # the shape of the dataset.
                    dset = stream[ptype_name][f_name]
                    if dset.shape != field_data.shape:
                        raise ValueError(
                            f"Shape of provided data for field '{field_name}' does not match dataset shape."
                            f" Provided shape: {field_data.shape}, Dataset shape: {dset.shape}"
                        )

                    # If all checks pass, we can write the data to
                    # the dataset. We still need to ensure that the data gets
                    # cast to the correct units from the dataset.
                    _units = unyt.Unit(dset.attrs.get("UNITS", "dimensionless"))
                    dset[...] = unyt.unyt_array(field_data, units=_units).to_value(_units)

        # Return a validated ParticleDataset instance
        output = cls(path)

        # Add the particle ids
        output.add_particle_ids(policy="global", overwrite=True)
        return output


class Gadget4ParticleDataset(GadgetLikeParticleDataset):
    """
    Representation of a Gadget-4 style particle dataset.

    This class specializes :class:`GadgetLikeParticleDataset` for the
    conventions of Gadget-4 HDF5 initial conditions and snapshots. It
    defines the expected header structure, field names, and dataset
    generation logic for creating valid Gadget-4–compatible files.
    """

    # --------------------------------- #
    # Class Constants / Flags           #
    # --------------------------------- #
    # These flags provide standard names for built-in access to specific
    # fields in the dataset. This ensures that those areas of the code are
    # not hardcoded with string literals. These can be overridden in
    # subclasses if the dataset uses different naming conventions.

    # --- Field Name Conventions --- #
    _POSITION_FIELD_NAME = "Coordinates"
    _VELOCITY_FIELD_NAME = "Velocities"
    _MASS_FIELD_NAME = "Masses"
    _ID_FIELD_NAME = "ParticleIDs"
    _INTERNAL_ENERGY_FIELD_NAME = "InternalEnergy"
    _ELECTRON_ABUNDANCE_FIELD_NAME = "ElectronAbundance"

    # --- Class Settings --- #
    _default_ntypes: int = 6
    """int: Default number of Gadget-like particle types.

    By default, this is set to 6, corresponding to the standard Gadget
    particle types; however, some simulations may have a different number.
    This can also be specified by the user as a kwarg when building a dataset.
    """
    _default_unit_system = unyt.unit_systems.galactic_unit_system
    """Default unit system for interpreting Gadget particle fields.
    Gadget itself does not enforce units; this is provided for Pisces
    consistency and metadata export.
    """
    _header_group_name: str = "Header"
    """str: Name of the group containing the Gadget-like file header.

    This can be overridden in subclasses if the header group has a different name.
    """
    _particle_type_name_prefix: str = "ParticleType"
    """str: Prefix for particle type groups in the HDF5 file.

    This can be overridden in subclasses if the particle type groups have a different prefix.
    """

    # ------------------------------------- #
    # Generation Methods                    #
    # ------------------------------------- #
    @classmethod
    def _generate_file_header(
        cls,
        stream: h5py.File,
        number_of_particles: np.ndarray,
        box_size: unyt.unyt_quantity,
        unit_system: unyt.UnitSystem,
        **kwargs,
    ):
        """
        Write Gadget-4 style header attributes to the HDF5 file.

        This method writes the standard Gadget-4 header attributes
        into the file’s ``Header`` group. These attributes describe
        particle counts, box size, and basic cosmology metadata, and
        must be present for Gadget-4 to interpret the file as valid.

        Parameters
        ----------
        stream : h5py.File
            The open HDF5 file handle to which attributes will be written.
            Attributes are stored at the root or within the ``Header`` group.
        number_of_particles : numpy.ndarray
            Array of length ``ntypes`` giving the number of particles per type.
            Values must fit into 32-bit integers since only single-file
            snapshots are supported here.
        box_size : unyt.unyt_quantity
            Simulation box size with length units. Converted into the
            provided unit system before being written.
        unit_system : unyt.UnitSystem
            Unit system used for box size conversion.
        **kwargs
            Optional keyword arguments. Recognized:
            - ``double_precision`` : bool (default True)
              Indicates whether floating-point particle data is stored in
              double precision. This does not change the header layout,
              but is included for API consistency.

        Raises
        ------
        ValueError
            If any particle count exceeds the 32-bit integer limit.

        Notes
        -----
        The following attributes are written:

        - ``NumPart_ThisFile`` (int32[ntypes]) — per-file particle counts.
        - ``NumPart_Total`` (uint64[ntypes]) — total particle counts.
        - ``MassTable`` (float64[ntypes]) — per-type constant masses
          (always zeros here, since individual masses are assumed).
        - ``BoxSize`` (float64) — simulation box size.
        - ``Time`` (float64) — simulation time (0.0 for ICs).
        - ``Redshift`` (float64) — redshift (0.0 for ICs).
        """
        # --- Particle Count Management --- #
        # Manage the particle count writing segment. We ensure that the counts we have
        # from the user are 64 bit unsigned integers so that we can handle large counts.
        # We then need to cast that down to 32 to ensure that we have the correct types. Because
        # we don't support multi-file snapshots here, we need to ensure that the counts
        # do not exceed the 32-bit limit.
        number_of_particles = np.asarray(number_of_particles, dtype=np.uint64)

        # Check that we don't surpass the 32-bit limit.
        int32_max = np.iinfo(np.int32).max
        if np.any(number_of_particles > int32_max):
            raise ValueError(
                "Particle count exceeds 32-bit limit for single-file snapshots. "
                "Either reduce particles per type or split snapshot into multiple files."
            )

        # Because we are under the 32-bit limit, we can safely cast to the
        # expected C-int.
        low_word = (number_of_particles & 0xFFFFFFFF).astype(np.int32)
        stream.attrs["NumPart_ThisFile"] = low_word
        stream.attrs["NumPart_Total"] = low_word.astype(np.uint64)
        stream.attrs["NumFilesPerSnapshot"] = np.int32(1)

        # --- Cosmology Parameters --- #
        # Like the physics flags, the cosmological parameters are not actually
        # used by AREPO, but are ready nonetheless. As such, we write them but
        # they are set to zero values.
        stream.attrs["Redshift"] = np.float64(0.0)
        stream.attrs["Time"] = np.float64(0.0)

        # --- Writing the Mass Table and the Box Size --- #
        # The final two relevant elements of the header are the mass table
        # and the box size. In AREPO, both of these are required. The mass table
        # is (by our convention) zeros because we ALWAYS specify particle masses.
        # The boxsize requires conversion.
        stream.attrs["MassTable"] = np.zeros_like(number_of_particles, dtype=np.float64)

        _native_boxsize = box_size.to_value(unit_system["length"])
        stream.attrs["BoxSize"] = np.float64(_native_boxsize)

    @classmethod
    def _generate_particle_fields(
        cls,
        particle_group: h5py.Group,
        particle_type: int,
        number_of_particles: int,
        unit_system: unyt.UnitSystem,
        **kwargs,
    ):
        """
        Create required Gadget-4 particle fields for a particle group.

        This method defines and initializes the standard datasets that
        must exist for a given particle type in a Gadget-4 dataset.
        Datasets are created with the correct shape, dtype, and unit
        attributes. Precision and optional fields are controlled via
        keyword arguments.

        Parameters
        ----------
        particle_group : h5py.Group
            HDF5 group corresponding to the particle type (e.g.,
            ``ParticleType0`` for gas).
        particle_type : int
            Particle type index (0=gas, 1=DM, 4=stars, 5=BH, etc.).
        number_of_particles : int
            Number of particles of this type.
        unit_system : unyt.UnitSystem
            Unit system used for unit annotations.
        **kwargs
            Configuration flags:
            - ``double_precision`` : bool, default=True
              Store float fields as float64 instead of float32.
            - ``coordinates_precision`` : {32, 64}, default=32
              Precision of ``Coordinates`` dataset.
            - ``id_precision`` : {32, 64}, default=32
              Bit-width of ``ParticleIDs`` dataset.
            - ``enable_cooling`` : bool, default=False
              If True, add ``ElectronAbundance`` field to gas.

        Notes
        -----
        **Always created for all particle types**:
        - ``Coordinates`` (N,3) float32/float64
        - ``Velocities`` (N,3) float32/float64
        - ``Masses`` (N,) float32/float64
        - ``ParticleIDs`` (N,) uint32/uint64

        **Created for gas (ParticleType0)**:
        - ``InternalEnergy`` (N,) float32/float64
        - ``ElectronAbundance`` (N,) float32/float64 if ``enable_cooling=True``

        Units are stored in the ``UNITS`` attribute of each dataset.
        """
        # --- Extract Basic Information --- #
        # Handle the coordinate dtype based on the double precision flag.
        _coordinate_precision = kwargs.get("coordinates_precision", 32)
        _id_precision = kwargs.get("id_precision", 32)
        _double_precision = kwargs.get("double_precision", True)

        if _coordinate_precision not in [32, 64]:
            raise ValueError("`coordinates_precision` must be either 32 or 64.")
        if _id_precision not in [32, 64]:
            raise ValueError("`id_precision` must be either 32 or 64.")

        coordinates_dtype = np.float64 if _double_precision or (_coordinate_precision == 64) else np.float32
        float_dtype = np.float32 if not _double_precision else np.float64
        id_dtype = np.uint64 if _id_precision == 64 else np.uint32

        # --- Specify Field Parameters --- #
        # In order to make things reasonably clean, we specify each
        # of the required fields in a dictionary, then iterate through
        # that dictionary to create the datasets. This permits subclasses
        # to minimally modify the fields that are created. We also provide
        # access to hooks here so that users can configure the additional fields
        # for particular flags.
        _required_field_data = {
            cls._POSITION_FIELD_NAME: {
                "types": "all",
                "shape": (3,),
                "dtype": coordinates_dtype,
                "units": unit_system["length"],
            },
            cls._VELOCITY_FIELD_NAME: {
                "types": "all",
                "shape": (3,),
                "dtype": float_dtype,
                "units": unit_system["velocity"],
            },
            cls._MASS_FIELD_NAME: {
                "types": "all",
                "shape": (),
                "dtype": float_dtype,
                "units": unit_system["mass"],
            },
            cls._ID_FIELD_NAME: {
                "types": "all",
                "shape": (),
                "dtype": id_dtype,
                "units": "dimensionless",
            },
            cls._INTERNAL_ENERGY_FIELD_NAME: {
                "types": [0],
                "shape": (),
                "dtype": float_dtype,
                "units": unit_system["specific_energy"],
            },
        }

        if kwargs.get("enable_cooling", False):
            _required_field_data = {
                **_required_field_data,
                cls._ELECTRON_ABUNDANCE_FIELD_NAME: {
                    "types": [0],
                    "shape": (),
                    "dtype": float_dtype,
                    "units": "dimensionless",
                },
            }

        # --- Create Fields --- #
        # We now iterate through each required fields, check the particle type,
        # and create the dataset if it is required for this particle type.
        for field_name, field_info in _required_field_data.items():
            if (field_info["types"] == "all") or (particle_type in field_info["types"]):
                cls._add_particle_field_to_file(
                    particle_group,
                    field_name,
                    number_of_particles,
                    field_shape=field_info["shape"],
                    field_dtype=field_info["dtype"],
                    field_units=field_info["units"],
                )

    @overload
    @classmethod
    def build_particle_dataset(
        cls,
        path: Union[str, Path],
        number_of_particles: Sequence,
        box_size: unyt.unyt_quantity,
        *,
        ntypes: int = 6,
        unit_system: unyt.UnitSystem = None,
        overwrite: bool = False,
        fields: dict = None,
        double_precision: bool = True,
        coordinates_precision: int = 32,
        id_precision: int = 32,
        enable_cooling: bool = False,
    ) -> "Gadget4ParticleDataset": ...

    @classmethod
    def build_particle_dataset(cls, *args, **kwargs):
        """
        Build a new Gadget-4 particle dataset.

        This is the primary public constructor for Gadget-4 style
        initial condition files. It writes the file header, particle
        groups, and required fields according to Gadget-4 conventions.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination path for the new dataset.
        number_of_particles : sequence of int
            Particle counts per type. Length must equal ``ntypes``.
        box_size : unyt.unyt_quantity
            Simulation box size with length units.
        ntypes : int, default=6
            Number of particle types (Gadget standard).
        unit_system : unyt.UnitSystem, optional
            Unit system for annotating dataset units.
            Defaults to galactic unit system.
        overwrite : bool, default=False
            Overwrite the file if it exists. If False, raise an error
            instead.
        fields : dict, optional
            Additional fields to populate after dataset creation.
            Keys must be of the form ``"ParticleTypeX.FieldName"``.
        double_precision : bool, default=True
            Use float64 for floating-point fields (except where overridden).
        coordinates_precision : {32, 64}, default=32
            Precision for ``Coordinates``.
        id_precision : {32, 64}, default=32
            Precision for ``ParticleIDs``.
        enable_cooling : bool, default=False
            If True, add ``ElectronAbundance`` to gas.

        Returns
        -------
        Gadget4ParticleDataset
            A dataset object pointing to the newly created file.

        Raises
        ------
        FileExistsError
            If the file exists and ``overwrite=False``.
        ValueError
            If particle counts or field shapes are inconsistent.
        TypeError
            If units or parameters are of invalid types.

        Examples
        --------
        Create a simple Gadget-4 IC file with gas and DM::

            import unyt as u

            ds = (
                Gadget4ParticleDataset.build_particle_dataset(
                    "ics_gadget4.hdf5",
                    number_of_particles=[
                        1000,
                        2000,
                        0,
                        0,
                        0,
                        0,
                    ],
                    box_size=1.0 * u.Mpc,
                    overwrite=True,
                )
            )

        Create a Gadget-4 IC file with cooling enabled::

            ds = (
                Gadget4ParticleDataset.build_particle_dataset(
                    "ics_cooling.hdf5",
                    number_of_particles=[
                        500,
                        500,
                        0,
                        0,
                        0,
                        0,
                    ],
                    box_size=500.0 * u.kpc,
                    enable_cooling=True,
                )
            )
        """
        return super().build_particle_dataset(*args, **kwargs)


class AREPOParticleDataset(GadgetLikeParticleDataset):
    """
    Representation of an AREPO-style particle dataset.

    This class specializes :class:`GadgetLikeParticleDataset` for
    AREPO’s HDF5 initial condition format. It implements the header
    layout, field conventions, and optional physics extensions expected
    by the AREPO code.

    See Also
    --------
    :class:`GadgetLikeParticleDataset`
        Base class for all Gadget/AREPO-compatible formats.
    :class:`Gadget4ParticleDataset`
        Implementation for Gadget-4 format.
    """

    # --------------------------------- #
    # Class Constants / Flags           #
    # --------------------------------- #
    _MAGNETIC_FIELD_NAME = "MagneticField"
    _ELECTRON_ABUNDANCE_FIELD_NAME = "ElectronAbundance"
    _PASSIVE_SCALARS_FIELD_NAME = "PassiveScalars"

    # ------------------------------------- #
    # Generation Methods                    #
    # ------------------------------------- #
    @classmethod
    def _generate_file_header(
        cls,
        stream: h5py.File,
        number_of_particles: np.ndarray,
        box_size: unyt.unyt_quantity,
        unit_system: unyt.UnitSystem,
        **kwargs,
    ):
        """
        Generate and write the AREPO-style header attributes to the HDF5 file.

        This method creates the ``Header`` attributes required for AREPO
        initial conditions. The structure is modeled on the Gadget-2 snapshot
        header format, with several additional attributes retained for
        compatibility. While many of these fields are not used directly by
        AREPO when reading ICs (e.g., star formation and cooling flags),
        they are still written to maintain full Gadget-format compliance.

        Parameters
        ----------
        stream : h5py.File
            The open HDF5 group handle into which the header attributes will
            be written. Attributes are stored at the root level (``stream.attrs``).
        number_of_particles : numpy.ndarray
            Array of shape (6,) giving the particle counts per type. Must be
            convertible to 32-bit integers for single-file snapshots.
        box_size : unyt.unyt_quantity
            The size of the simulation box, with units of length.
        unit_system : ~unyt.UnitSystem
            The unit system into which the box size will be converted before
            writing.
        **kwargs
            Additional options for header customization. Recognized keys:

            - ``double_precision`` : bool, optional
              Whether particle positions/velocities are stored in double
              precision. Default is True.

        Raises
        ------
        ValueError
            If any particle count exceeds the 32-bit signed integer limit,
            which is not permitted in single-file snapshots.

        Notes
        -----
        The following attributes are written:

        - ``NumPart_ThisFile`` (int32[6]): Per-file particle counts (must fit in 32-bit).
        - ``NumPart_Total`` (uint32[6]): Total particle counts (low word).
        - ``NumPart_Total_HighWord`` (uint32[6]): High word of total counts (all zero in single-file mode).
        - ``MassTable`` (float64[6]): Particle type masses. Always zeros here, since variable masses are used.
        - ``Time`` (float64): Simulation time (written as 0.0 for ICs).
        - ``Redshift`` (float64): Redshift of the snapshot (0.0 for ICs).
        - ``BoxSize`` (float64): Physical size of the simulation box, converted to the specified unit system.
        - ``NumFilesPerSnapshot`` (int32): Not written here (implicit = 1).
        - ``Omega0``, ``OmegaLambda``, ``OmegaBaryon`` (float64): Cosmology parameters. Written as 0.0/1.0 placeholders.
        - ``HubbleParam`` (float64): Hubble parameter (written as 1.0).
        - ``Flag_Sfr``, ``Flag_Cooling``, ``Flag_StellarAge``, ``Flag_Metals``,
          ``Flag_Feedback`` (int32): Physics flags. Always 0 for ICs.
        - ``Flag_DoublePrecision`` (int32): Precision flag for particle positions/velocities.

        Although AREPO does not actually use the physics or cosmology flags
        when reading ICs, they are included for full Gadget-format compliance.
        """
        # --- Particle Count Management --- #
        # Manage the particle count writing segment. We ensure that the counts we have
        # from the user are 64 bit unsigned integers so that we can handle large counts.
        # We then need to cast that down to 32 to ensure that we have the correct types. Because
        # we don't support multi-file snapshots here, we need to ensure that the counts
        # do not exceed the 32-bit limit.
        number_of_particles = np.asarray(number_of_particles, dtype=np.uint64)

        # Check that we don't surpass the 32-bit limit.
        int32_max = np.iinfo(np.int32).max
        if np.any(number_of_particles > int32_max):
            raise ValueError(
                "Particle count exceeds 32-bit limit for single-file snapshots. "
                "Either reduce particles per type or split snapshot into multiple files."
            )

        # Because we are under the 32-bit limit, we can safely cast to the
        # expected C-int.
        low_word = (number_of_particles & 0xFFFFFFFF).astype(np.int32)
        stream.attrs["NumPart_ThisFile"] = low_word
        stream.attrs["NumPart_Total"] = low_word.astype(np.uint32)
        stream.attrs["NumPart_Total_HighWord"] = np.zeros_like(low_word).astype(np.uint32)

        # --- IC File Flags --- #
        # An inspection of the AREPO base code indicates that for IC files, these
        # header flags are effectively ignored, so we just write them unilaterally
        # to 0 even if they are actually enabled in the parameter file / configuration.
        #
        # The double-precision flag is important, however, as it indicates to the
        # code whether the positions and velocities are stored as float32 or float64.
        stream.attrs["Flag_Sfr"] = np.int32(0)
        stream.attrs["Flag_Cooling"] = np.int32(0)
        stream.attrs["Flag_StellarAge"] = np.int32(0)
        stream.attrs["Flag_Metals"] = np.int32(0)
        stream.attrs["Flag_Feedback"] = np.int32(0)
        stream.attrs["Flag_DoublePrecision"] = np.int32(kwargs.get("double_precision", True))

        # --- Cosmology Parameters --- #
        # Like the physics flags, the cosmological parameters are not actually
        # used by AREPO, but are ready nonetheless. As such, we write them but
        # they are set to zero values.
        stream.attrs["Omega0"] = np.float64(0.0)
        stream.attrs["OmegaLambda"] = np.float64(0.0)
        stream.attrs["OmegaBaryon"] = np.float64(0.0)
        stream.attrs["HubbleParam"] = np.float64(1.0)
        stream.attrs["Redshift"] = np.float64(0.0)
        stream.attrs["Time"] = np.float64(0.0)

        # --- Writing the Mass Table and the Box Size --- #
        # The final two relevant elements of the header are the mass table
        # and the box size. In AREPO, both of these are required. The mass table
        # is (by our convention) zeros because we ALWAYS specify particle masses.
        # The boxsize requires conversion.
        stream.attrs["MassTable"] = np.zeros_like(number_of_particles, dtype=np.float64)

        _native_boxsize = box_size.to_value(unit_system["length"])
        stream.attrs["BoxSize"] = np.float64(_native_boxsize)

    @classmethod
    def _generate_particle_fields(
        cls,
        particle_group: h5py.Group,
        particle_type: int,
        number_of_particles: int,
        unit_system: unyt.UnitSystem,
        **kwargs,
    ):
        """
        Create the HDF5 datasets for a given particle type.

        This method defines and initializes the particle field datasets that
        must exist in an AREPO–compatible IC file. The base implementation
        guarantees creation of all *required* fields, and may conditionally add
        fields depending on enabled physics modules (e.g., MHD, cooling,
        passive scalars). Subclasses may extend or override this behavior
        to support additional physics or code–specific conventions.

        Parameters
        ----------
        particle_group : h5py.Group
            The open HDF5 group corresponding to the particle type
            (e.g., ``"PartType0"`` for gas).
        particle_type : int
            Particle type index (0 = gas, 1 = dark matter, 4 = stars, 5 = BHs, etc.).
        number_of_particles : int
            Number of particles of this type to allocate datasets for.
        unit_system : unyt.UnitSystem
            Unit system used to tag dataset attributes.
        **kwargs
            Configuration flags that control optional fields:

            - ``coordinates_in_double_precision`` : bool, default=True
              Store coordinates as float64 instead of float32.
            - ``double_precision`` : bool, default=True
              Store all other float fields as float64 instead of float32.
            - ``long_ids`` : bool, default=True
              Store ``ParticleIDs`` as uint64 instead of uint32.
            - ``enable_mhd`` : bool, default=False
              Add ``MagneticField`` vector (3-component) for gas.
            - ``enable_cooling`` : bool, default=False
              Add ``ElectronAbundance`` scalar field for gas.
            - ``passive_scalars`` : int, default=0
              If >0, add a ``PassiveScalars`` field with this many components.

        Notes
        -----
        **Always created for all particle types:**

        - ``Coordinates`` : float32[3] or float64[3]
        - ``Velocities`` : float32[3] or float64[3]
        - ``ParticleIDs`` : uint32 or uint64
        - ``Masses`` : float32 or float64

        **Always created for gas (PartType0):**

        - ``InternalEnergy`` : float32 or float64
        - ``Density`` : float32 or float64

        **Conditionally created (if flags set):**

        - ``MagneticField`` : float32[3] or float64[3] (gas, if ``enable_mhd``)
        - ``ElectronAbundance`` : float32 or float64 (gas, if ``enable_cooling``)
        - ``PassiveScalars`` : float32[N] or float64[N] (gas, if ``passive_scalars`` > 0)

        Returns
        -------
        None
            Datasets are created directly in ``particle_group``. Attributes
            describing units are also attached.

        Raises
        ------
        ValueError
            If required arguments are missing or inconsistent.
        """
        # --- Extract Basic Information --- #
        # Handle the coordinate dtype based on the double precision flag.
        _coords_dp = kwargs.get("coordinates_in_double_precision", True)
        _dp = kwargs.get("double_precision", True)
        _id_dp = kwargs.get("long_ids", True)

        coordinates_dtype = np.float64 if (_coords_dp or _dp) else np.float32
        float_dtype = np.float64 if _dp else np.float32
        id_dtype = np.uint64 if _id_dp else np.uint32

        # --- Specify Field Parameters --- #
        # In order to make things reasonably clean, we specify each
        # of the required fields in a dictionary, then iterate through
        # that dictionary to create the datasets. This permits subclasses
        # to minimally modify the fields that are created. We also provide
        # access to hooks here so that users can configure the additional fields
        # for particular flags.
        _required_field_data = {
            cls._POSITION_FIELD_NAME: {
                "types": "all",
                "shape": (3,),
                "dtype": coordinates_dtype,
                "units": unit_system["length"],
            },
            cls._VELOCITY_FIELD_NAME: {
                "types": "all",
                "shape": (3,),
                "dtype": float_dtype,
                "units": unit_system["velocity"],
            },
            cls._MASS_FIELD_NAME: {
                "types": "all",
                "shape": (),
                "dtype": float_dtype,
                "units": unit_system["mass"],
            },
            cls._ID_FIELD_NAME: {
                "types": "all",
                "shape": (),
                "dtype": id_dtype,
                "units": "dimensionless",
            },
            cls._INTERNAL_ENERGY_FIELD_NAME: {
                "types": [0],
                "shape": (),
                "dtype": float_dtype,
                "units": unit_system["specific_energy"],
            },
            cls._DENSITY_FIELD_NAME: {"types": [0], "shape": (), "dtype": float_dtype, "units": unit_system["density"]},
        }

        # Add MHD fields if necessary.
        if kwargs.get("enable_mhd", False):
            _required_field_data = {
                **_required_field_data,
                cls._MAGNETIC_FIELD_NAME: {
                    "types": [0],
                    "shape": (3,),
                    "dtype": float_dtype,
                    "units": unit_system["magnetic_field"],
                },
            }
        if kwargs.get("enable_cooling", False):
            _required_field_data = {
                **_required_field_data,
                cls._ELECTRON_ABUNDANCE_FIELD_NAME: {
                    "types": [0],
                    "shape": (),
                    "dtype": float_dtype,
                    "units": "dimensionless",
                },
            }
        if kwargs.get("passive_scalars", 0) > 0:
            n_scalars = kwargs.get("passive_scalars", 0)
            _required_field_data = {
                **_required_field_data,
                cls._PASSIVE_SCALARS_FIELD_NAME: {
                    "types": [0],
                    "shape": (n_scalars,),
                    "dtype": float_dtype,
                    "units": "dimensionless",
                },
            }

        # --- Create Fields --- #
        # We now iterate through each required fields, check the particle type,
        # and create the dataset if it is required for this particle type.
        for field_name, field_info in _required_field_data.items():
            if (field_info["types"] == "all") or (particle_type in field_info["types"]):
                cls._add_particle_field_to_file(
                    particle_group,
                    field_name,
                    number_of_particles,
                    field_shape=field_info["shape"],
                    field_dtype=field_info["dtype"],
                    field_units=field_info["units"],
                )

    @overload
    @classmethod
    def build_particle_dataset(
        cls,
        path: Union[str, Path],
        number_of_particles: Sequence,
        box_size: unyt.unyt_quantity,
        *,
        ntypes: int = 6,
        unit_system: unyt.UnitSystem = None,
        overwrite: bool = False,
        fields: dict = None,
        double_precision: bool = True,
        coordinates_in_double_precision: bool = True,
        long_ids: bool = True,
        enable_mhd: bool = False,
        enable_cooling: bool = False,
        passive_scalars: int = 0,
    ) -> "AREPOParticleDataset": ...

    @classmethod
    def build_particle_dataset(cls, *args, **kwargs) -> "AREPOParticleDataset":
        """
        Create a new HDF5 particle dataset in AREPO IC format.

        This is the **primary factory method** for generating particle datasets
        that can be used as initial conditions in simulation codes. It writes a
        complete, self-contained HDF5 file containing the particle fields
        required by AREPO (and optional fields depending on physics
        modules).

        Parameters
        ----------
        path : str or pathlib.Path
            Destination file path for the new dataset. If the file exists and
            ``overwrite=False``, a :class:`FileExistsError` is raised.
        number_of_particles : sequence of int
            Number of particles for each particle type. The length must match
            ``ntypes``. For example, ``[1000, 2000, 0, 0, 500, 0]`` specifies
            1000 gas cells, 2000 dark matter particles, 500 stars, and no others.
        box_size : unyt.unyt_quantity
            Physical size of the simulation box (must have length units).
            This is written into the dataset metadata and header.
        ntypes : int, default=6
            Number of particle types to include. Gadget/AREPO use 6 by default
            (0=gas, 1=DM, 2=disk, 3=bulge, 4=stars, 5=BHs), but this may be
            larger in customized builds.
        unit_system : unyt.UnitSystem, optional
            Unit system for tagging dataset attributes (e.g., cgs or astrophysical).
            If omitted, a default astrophysical unit system is used.
        overwrite : bool, default=False
            If True, overwrite an existing file at ``path``. If False, raise
            :class:`FileExistsError` if the file exists.
        fields : dict[str, unyt.unyt_array], optional
            Optional dictionary of fields to write into the dataset at creation.
            Keys must be of the form ``"<particle_type>.<field_name>"``, and each
            value must be a :class:`unyt.array.unyt_array` instance with the correct
            shape and units for the specified field.

            Field may also be added, removed, and manipulated after creation. This
            option simply permits populating fields at creation time.
        double_precision : bool, default=True
            Enable double precision initial conditions. This flag is equivalent to
            AREPO's ``INPUT_IN_DOUBLEPRECISION`` compile-time option. If True,
            all floating-point fields are stored as double precision floats as defined
            by the local system. If False, single precision (float32) is used.
        coordinates_in_double_precision : bool, default=True
            If True, ``Coordinates`` are stored in double precision. If False, they are
            stored as float32. Useful when positions need higher accuracy than
            velocities or internal energy. This corresponds to the AREPO flag
            ``INPUT_COORDINATES_IN_DOUBLEPRECISION``.
        long_ids : bool, default=True
            If True, ``ParticleIDs`` are stored as 64-bit unsigned integers.
            If False, 32-bit IDs are used. 64-bit IDs are required for runs with
            >4 billion particles. This corresponds to the AREPO flag
            ``LONGIDS``.
        enable_mhd : bool, default=False
            If True, adds ``MagneticField`` (3-component vector) to gas particles
            (PartType0). Required if running with MHD enabled.
        enable_cooling : bool, default=False
            If True, adds ``ElectronAbundance`` scalar field to gas particles.
            This initializes the ionization fraction for cooling modules.
            Defaults to zero if not supplied.
        passive_scalars : int, default=0
            If >0, adds a ``PassiveScalars`` field with the given number of
            components for gas particles. These are advected tracers that do not
            affect dynamics unless used by custom physics modules.

        Returns
        -------
        AREPOParticleDataset
            The newly created particle dataset object pointing to the file at
            ``path``.

        Raises
        ------
        FileExistsError
            If the file exists and ``overwrite=False``.
        ValueError
            If particle counts are inconsistent with field shapes or box size.
        TypeError
            If provided fields are not :class:`unyt.unyt_array` instances.

        """
        return super().build_particle_dataset(*args, **kwargs)
