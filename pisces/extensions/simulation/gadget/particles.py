"""
Specialized particle dataset class for Gadget-compatible HDF5 files.

This module defines the :class:`GadgetParticleDataset` class, which extends the
generic :class:`~pisces.particles.base.ParticleDataset` class to handle HDF5 files formatted
according to the Gadget-4 (and related Gadget-family) conventions.
It provides methods for creating, reading, and manipulating particle
datasets that follow Gadget's structure, including support for
standard particle types, fields, and metadata.
"""

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Union

import h5py
import numpy as np
import unyt

from pisces.particles.base import ParticleDataset
from pisces.utilities.io_tools import NullHDF5Serializer


class GadgetParticleDataset(ParticleDataset):
    """
    Representation of a Gadget-compatible particle dataset.

    This class provides a structured interface for building, reading,
    and manipulating HDF5 files that follow the Gadget-4 (and related
    Gadget-family) conventions for initial condition (IC) and snapshot
    particle data. It extends the generic :class:`~pisces.particles.base.ParticleDataset`
    with Gadget-specific conventions.
    """

    metadata_serializer = NullHDF5Serializer
    """Serializer for Gadget particle metadata.

    Gadget particle datasets do not require any special metadata
    serialization, so a no-op (null) serializer is used.
    """

    # --------------------------------- #
    # Class Constants / Flags           #
    # --------------------------------- #
    # These flags provide standard names for built-in access to specific
    # fields in the dataset. This ensures that those areas of the code are
    # not hardcoded with string literals. These can be overridden in
    # subclasses if the dataset uses different naming conventions.

    # --- Field Name Conventions --- #
    _POSITION_FIELD_NAME: str = "Coordinates"
    """str: The standard name for the position field in each particle group."""
    _VELOCITY_FIELD_NAME: str = "Velocities"
    """str: The standard name for the velocity field in each particle group."""
    _MASS_FIELD_NAME: str = "Masses"
    """str: The standard name for the mass field in each particle group."""
    _ID_FIELD_NAME: str = "ParticleIDs"
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
    _PARTICLE_TYPE_PREFIX = "ParticleType"
    """Prefix for particle-type groups in Gadget HDF5 files
    (e.g., ``PartType0``, ``PartType1``, ...)."""
    _HEADER_GROUP_NAME = "Header"
    """Name of the HDF5 group that stores global header attributes."""
    _DEFAULT_N_PTYPES = 6
    """Default number of Gadget particle types:

    0. Gas
    1. Dark Matter
    2. Disk Stars
    3. Bulge Stars
    4. Stellar Particles
    5. Boundary Particles
    """
    _DEFAULT_UNIT_SYSTEM = unyt.unit_systems.galactic_unit_system
    """Default unit system for interpreting Gadget particle fields.
    Gadget itself does not enforce units; this is provided for Pisces
    consistency and metadata export.
    """

    # Field Standards
    # ---------------- #
    # This is where we define the standard fields
    # that are present for each particle type. Gadget
    # has a fairly standard set of fields that are expected
    # for each particle type, so we define them here.
    __GAS_FIELD_DATA__ = {
        "Coordinates": {
            "element_shape": (3,),
            "dtype": np.float32,
            "dimensions": "length",
        },
        "Velocities": {
            "element_shape": (3,),
            "dtype": np.float32,
            "dimensions": "velocity",
        },
        "Masses": {
            "element_shape": (),
            "dtype": np.float32,
            "dimensions": "mass",
        },
        "InternalEnergy": {
            "element_shape": (),
            "dtype": np.float32,
            "dimensions": "specific_energy",
        },
        "ParticleIDs": {
            "element_shape": (),
            "dtype": np.uint32,
            "dimensions": None,
        },
    }
    """Field definitions for gas particles (Type 0).

    Includes the standard fields (Coordinates, Velocities, Masses,
    ParticleIDs) plus the hydrodynamical ``InternalEnergy`` field.
    """
    __NONGAS_FIELD_DATA__ = {
        "Coordinates": {
            "element_shape": (3,),
            "dtype": np.float32,
            "dimensions": "length",
        },
        "Velocities": {
            "element_shape": (3,),
            "dtype": np.float32,
            "dimensions": "velocity",
        },
        "Masses": {
            "element_shape": (),
            "dtype": np.float32,
            "dimensions": "mass",
        },
        "ParticleIDs": {
            "element_shape": (),
            "dtype": np.uint32,
            "dimensions": None,
        },
    }
    """Field definitions for non-gas particles (Types 1–5).

    Standard fields include Coordinates, Velocities, Masses, and
    ParticleIDs. Unlike gas particles, these do not carry
    ``InternalEnergy`` by default.
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
        return self.__handle__[self._HEADER_GROUP_NAME]

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
    def _write_pisces_metadata(cls, f: h5py.File):
        """Write Pisces metadata to the HDF5 file.

        This method writes the standard Pisces metadata to the root of the
        HDF5 file. It uses the metadata serializer defined for the class.

        Parameters
        ----------
        f : h5py.File
            The HDF5 file object to write metadata to.
        """
        # At least in this base class, we only need to keep track of the generation
        # time and the class name for the particle dataset. This is then used
        # to ensure that we can read from disk correctly later.
        initial_metadata = {
            "GEN_TIME": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "CLASS_NAME": cls.__name__,
        }
        for k, v in cls.metadata_serializer.serialize_dict(initial_metadata).items():
            f.attrs[k] = v

    @classmethod
    def _write_gadget_header(
        cls,
        f: h5py.File,
        number_of_particles: np.ndarray,
        box_size: unyt.unyt_quantity,
        unit_system: unyt.UnitSystem,
        **kwargs,
    ):
        """
        Create and populate the ``Header`` group for a Gadget-compatible initial condition (IC) HDF5 file.

        This method writes the **minimal set of header attributes** required
        by Gadget-4 (and other Gadget-family codes) for reading an IC file.
        Specifically, it encodes particle counts, per-type mass tables,
        and basic simulation box information.

        Parameters
        ----------
        f : h5py.File
            Open HDF5 file handle in write mode.
        number_of_particles : numpy.ndarray
            1D array giving the number of particles of each Gadget particle type.

            By convention Gadget defines six particle types:

            ``[Type0 (gas), Type1 (dark matter), Type2 (disk),
              Type3 (bulge), Type4 (stars), Type5 (boundary)]``

            This array is typically length 6, but in practice the value is
            coerced elsewhere in the code to match the number of supported
            particle types. Extra entries are ignored; missing entries are
            padded with zeros.

            The values are written twice in the header:

              - As ``NumPart_ThisFile`` (dtype = uint32), giving the counts
                for this file only.
              - As ``NumPart_Total`` (dtype = uint64), giving the global totals.
                In the single-file IC case, these arrays are identical.

        box_size : unyt.unyt_quantity
            The physical size of the simulation box. This will be coerced into
            the provided ``unit_system`` and written as ``BoxSize`` (float64).
            Gadget expects this to be given in code units; in practice, the
            runtime parameter file also defines ``BoxSize``. Including it here
            makes the IC file self-contained.
        unit_system : unyt.UnitSystem
            The unit system that defines the "code units" for Gadget. This is
            used to coerce ``box_size`` and (optionally) the ``MassTable`` into
            the correct base units before writing.
        **kwargs
            Optional keyword arguments:

            - ``header_mass_table`` : array_like or unyt_array, shape (6,), optional
              A mass table giving the *uniform per-particle mass* of each type.
              If a type has variable particle masses (stored in a per-particle
              ``Masses`` dataset), the corresponding entry here should be zero.
              If not provided, defaults to an array of zeros. Units may be given
              as a unyt array, or as raw code units; in either case values are
              coerced into the supplied ``unit_system``.

        Notes
        -----
        - ``NumFilesPerSnapshot`` is fixed to 1 (single-file ICs).
        - ``NumPart_ThisFile`` is stored as 32-bit integers, while
          ``NumPart_Total`` is stored as 64-bit integers. This matches
          Gadget-4’s convention, which supports large (>2³²) particle counts.
          For maximum backward compatibility with older Gadget versions,
          you could also add ``NumPart_Total_HW`` (high 32-bit words).
        - ``MassTable`` is written as a float64 array attribute on the header.
          Default is zeros, implying that masses must be present in per-particle
          datasets. If nonzero entries are provided, Gadget will use those
          uniform masses for the corresponding types instead of reading a
          ``Masses`` dataset.
        - ``BoxSize`` is stored as a float64 scalar in code units.
        - ``Redshift`` and ``Time`` are set to 0.0 here, since Gadget expects
          the starting redshift/time to be specified in the runtime parameter
          file (``TimeBegin``). Writing them as 0.0 is conventional for IC files.
        - This helper does *not* write optional cosmology attributes such as
          ``Omega0``, ``OmegaLambda``, or ``HubbleParam``. These are also
          normally specified in the runtime parameter file, but can be added
          here if desired.
        - The resulting HDF5 structure is:

          .. code-block:: text

              /Header (Group)
                  ├─ NumFilesPerSnapshot   (attr, int32, scalar)
                  ├─ NumPart_ThisFile      (attr, uint32[6])
                  ├─ NumPart_Total         (attr, uint64[6])
                  ├─ MassTable             (attr, float64[6])
                  ├─ BoxSize               (attr, float64, scalar)
                  ├─ Redshift              (attr, float64, scalar)
                  └─ Time                  (attr, float64, scalar)
        """
        # Start by creating the header group.
        header_group = f.create_group(cls._HEADER_GROUP_NAME)

        # Add the NOT_PARTICLE_GROUP attribute to the header group so
        # pisces knows its not a particle group.
        header_group.attrs["NOT_PARTICLE_GROUP"] = True

        # managing the number of particles in each group and
        # the number of particles per file.
        header_group.attrs["NumFilesPerSnapshot"] = np.asarray(1, dtype=np.int32)
        header_group.attrs["NumPart_ThisFile"] = np.asarray(number_of_particles, dtype=np.uint32)
        header_group.attrs["NumPart_Total"] = np.asarray(number_of_particles, dtype=np.uint64)

        # We now handle the mass table. This is pulled out of the
        # kwargs if it is provided to allow a user to do such a thing, but
        # we generally will set this to a full set of zeros and allow for
        # the mass to be specified on a per-particle basis.
        _mass_table = kwargs.get("header_mass_table", None)
        _mass_table = _mass_table if _mass_table is not None else np.zeros(len(number_of_particles), dtype=np.float64)
        _mass_table = unyt.unyt_array(_mass_table, units=unit_system["mass"]).in_base(unit_system).d
        header_group.attrs["MassTable"] = np.asarray(_mass_table, dtype=np.float64)

        # The boxsize, redshift, and time are all set to zero because these
        # are not generally relevant for ICs.
        if not isinstance(box_size, unyt.unyt_quantity):
            box_size = unyt.unyt_quantity(box_size, unit_system["length"])

        box_size = box_size.in_base(unit_system).d

        header_group.attrs["BoxSize"] = np.asarray(box_size, dtype=np.float64)
        header_group.attrs["Redshift"] = np.asarray(0.0, dtype=np.float64)
        header_group.attrs["Time"] = np.asarray(0.0, dtype=np.float64)

    @classmethod
    def _build_file_skeleton(
        cls,
        f: h5py.File,
        number_of_particles: np.ndarray,
        unit_system: unyt.UnitSystem,
        **kwargs,
    ):
        """
        Create the particle-type groups and empty datasets for a Gadget-compatible IC file.

        This builds the HDF5 "skeleton": one group per particle type
        (`ParticleType0`, ..., `ParticleType5` by default) and inside
        each group the standard per-particle fields with the correct
        shapes, dtypes, and Pisces unit metadata.

        Parameters
        ----------
        f : h5py.File
            Open HDF5 file handle in write mode.
        number_of_particles : numpy.ndarray
            Array giving the number of particles of each type. By Gadget
            convention this should have length 6:

            ``[Type0 (gas), Type1 (dark matter), Type2 (disk),
              Type3 (bulge), Type4 (stars), Type5 (boundary)]``

            However, this method will create one group per entry, so arrays
            of other lengths are accepted. Zero counts still create a group,
            but the datasets will have shape (0, ...).
        unit_system : unyt.UnitSystem
            Unit system to which field dimensions are coerced when writing
            Pisces-specific ``UNITS`` attributes.
        **kwargs
            Forwarded to subclass/customization hooks.

        Notes
        -----
        - Gadget itself does *not* require a ``UNITS`` attribute; this is
          purely Pisces metadata for downstream analysis.
        - If a field’s `dimensions` are not specified, the dataset is marked
          with UNIT = "dimensionless".
        - Field shapes are defined by the class-level field tables
          (``__GAS_FIELD_DATA__`` / ``__NONGAS_FIELD_DATA__``). For vector
          fields (Coordinates, Velocities), these tables should specify
          an element shape of (3,), ensuring Gadget’s expectation of
          3D arrays even in 1D/2D runs.
        - Gas particles (ptype_id = 0) include additional hydro fields
          (e.g. InternalEnergy) defined in ``__GAS_FIELD_DATA__``.
        """
        # Determine the number of particle types that the
        # user is working with. This is the number of particle
        # type groups that we will create.
        n_ptypes = number_of_particles.shape[0]

        # Iterate through each of the particle types and
        # create the relevant groups and datasets. We pull down
        # the field data for each particle type based on the class flags.
        #
        # For each field, we take the dimension specified in the class flags
        # and use it to create the relevant dataset with correct units.
        for ptype_id in range(n_ptypes):
            group_name = f"{cls._PARTICLE_TYPE_PREFIX}{ptype_id}"
            group = f.create_group(group_name)

            # Store particle count for convenience (Pisces-specific).
            group.attrs["NUMBER_OF_PARTICLES"] = int(number_of_particles[ptype_id])

            # Choose field template based on type.
            field_data = cls.__GAS_FIELD_DATA__ if ptype_id == 0 else cls.__NONGAS_FIELD_DATA__

            # Iterate through all of the fields and create them correctly
            # with the specified shapes and dtypes.
            for field_name, field_info in field_data.items():
                # Determine the shape of the dataset that
                # is being generated.
                n = int(number_of_particles[ptype_id])
                element_shape = tuple(field_info.get("element_shape", ()))
                shape = (n,) if not element_shape else (n, *element_shape)

                # Create dataset.
                dset = group.create_dataset(
                    field_name,
                    shape=shape,
                    dtype=field_info["dtype"],
                )

                # Handle the units. Gadget doesn't require these, but we want them to
                # be present for Pisces dataset interaction.
                unit_dimensions = field_info.get("dimensions", None)
                if unit_dimensions is not None:
                    dset.attrs["UNITS"] = str(unit_system[unit_dimensions])
                else:
                    dset.attrs["UNIT"] = str(unyt.Unit(""))

    @classmethod
    def build_particle_dataset(
        cls,
        path: Union[str, Path],
        *args,
        number_of_particles: Sequence = None,
        box_size: unyt.unyt_quantity = None,
        fields: dict[str, unyt.unyt_array] = None,
        unit_system: unyt.UnitSystem = None,
        overwrite: bool = False,
        **kwargs,
    ):
        # Ensure that required kwargs are provided before proceeding. We
        # do this to ensure signature compatibility with the base class.
        if number_of_particles is None:
            raise ValueError("`number_of_particles` must be provided to build a Gadget particle dataset.")
        else:
            # coerce the number of particles to a list if it is not already
            number_of_particles = np.asarray(number_of_particles, dtype=np.uint32)

            # Ensure that this matches the expected number of particle types
            expected_n_ptypes = kwargs.get("number_of_particle_types", cls._DEFAULT_N_PTYPES)
            if number_of_particles.shape != (expected_n_ptypes,):
                raise ValueError(
                    f"`number_of_particles` must be of length {expected_n_ptypes} to"
                    f" match the expected number of particle types."
                )

        # Configure the unit system so that we have one available at all
        # times. This is taken from the class defaults if it is not provided.
        unit_system = unit_system or cls._DEFAULT_UNIT_SYSTEM
        if not isinstance(unit_system, unyt.UnitSystem):
            raise TypeError("`unit_system` must be a unyt.UnitSystem instance.")

        if box_size is None:
            raise ValueError("`box_size` must be provided to build a Gadget particle dataset.")
        elif isinstance(box_size, unyt.unyt_quantity):
            box_size = unyt.unyt_array(box_size.in_base(unit_system), dtype=np.float64)
        else:
            box_size = unyt.unyt_array(box_size * unit_system["length"], dtype=np.float64)

        # Standardize and then validate the file path so that
        # we have a consistent interface.
        path = Path(path)

        if path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {path}. Use overwrite=True to replace it.")
        elif path.exists() and overwrite:
            path.unlink()
        elif path.is_dir():
            raise IsADirectoryError(f"Path is a directory: {path}")

        path.parent.mkdir(parents=True, exist_ok=True)

        # --- File Generation --- #
        # At this stage, we can proceed to generate the file skeleton. In doing
        # so, we will first create the file, then write the Pisces metadata, then
        # the Gadget header, then the file skeleton, and finally any fields that
        # are provided to us.
        with h5py.File(path, "w") as f:
            # --- Pisces Metadata --- #
            # Start with Pisces metadata as managed by the
            # _write_pisces_metadata method.
            cls._write_pisces_metadata(f)

            # --- Gadget Header --- #
            # We now write the Gadget header. This is largely a simple
            # undertaking, but does require some care to ensure that
            # everything is correct. We use ._write_gadget_header to
            # handle this.
            cls._write_gadget_header(f, number_of_particles, box_size, unit_system=unit_system, **kwargs)

            # --- Field Structure --- #
            # With the header written, we can now create the
            # groups for each of the particle types and start generating
            # the relevant fields.
            cls._build_file_skeleton(f, number_of_particles, unit_system=unit_system, **kwargs)

            # --- Field Population --- #
            # If fields are provided, we can now populate them.
            # These are provided as `<particle_type_name>.<field_name>`, so
            # we can just split the names, then dump them to the correct dataset
            # after some validation checks.
            if fields is not None:
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
                    if ptype_name not in f:
                        raise ValueError(f"Particle type group '{ptype_name}' does not exist in the file.")
                    if f_name not in f[ptype_name]:
                        raise ValueError(f"Field '{f_name}' does not exist in particle type group '{ptype_name}'.")

                    # Validate that the shape of the provided data matches
                    # the shape of the dataset.
                    dset = f[ptype_name][f_name]
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
