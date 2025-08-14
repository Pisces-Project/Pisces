.. _particles_overview:

===================================
Particle Representations in Pisces
===================================

Converting data into particles is a flexible and powerful tool in Pisces. It is particularly useful for:

- Smoothed Particle Hydrodynamics (SPH) calculations
- Generating initial conditions for simulations
- Exporting data to external analysis tools such as `yt <https://yt-project.org/>`__

Pisces provides tools for working with particle-based datasets in a simple, consistent, and extensible way.

Particle Datasets
------------------

Particulate data in Pisces is represented by the :class:`~pisces.particles.base.ParticleDataset` class, and
all particle-related functionality is provided through the :mod:`~pisces.particles` module. At their core, particle
datasets are relatively simple: they are HDF5 files which separate particles into **types** (species), each
corresponding to a different group in the file. Each particle species may then have any number of **fields**
representing physically relevant quantities, such as position, density, temperature, etc. Each field is stored as
a dataset within the particle species group. The positions of each particle are stored in a field named
``particle_position`` and correspond to the Cartesian coordinates of the particles in a simulation box.

.. note::

    Particle datasets are an inherently **Cartesian** representation of data. This means that all particle positions
    are stored in Cartesian coordinates; however, non-Euclidean coordinates may also be stored as additional fields.

Data Conventions
^^^^^^^^^^^^^^^^

In general, there is no strict convention for how particle datasets should be structured, what particles and fields
should be named, etc. Nonetheless, various Pisces tools which need to interact with particle datasets will assume a
general convention for naming particle species and fields. For example, if a user is converting a particle dataset into
initial conditions for an SPH simulation code, Pisces will assume a conventional naming scheme for the particle species
and fields. This convention can, in general, be overridden by the user by changing the settings of the process using the
particle dataset; however, it is recommended to follow the conventions to ensure compatibility with Pisces tools.

Particle Types
**************

Pisces adopts the standard Gadget-2 convention for particle type naming. Each particle type corresponds to a different
group in the HDF5 file, and the following names are used to identify the particle types:

+-------------------+-----------------------+---------------------------------------------+
| Particle Type     | Gadget-2 ID           | Description                                 |
+===================+=======================+=============================================+
| ``gas``           |                 0     | Gas particles (e.g., SPH fluid elements)    |
+-------------------+-----------------------+---------------------------------------------+
| ``dark_matter``   |                 1     | Dark matter particles.                      |
+-------------------+-----------------------+---------------------------------------------+
| ``tracer``        |                 3     | Tracer particles.                           |
+-------------------+-----------------------+---------------------------------------------+
| ``stars``         |                 4     | Stellar particles and wind.                 |
+-------------------+-----------------------+---------------------------------------------+
| ``black_holes``   |                 5     | Black hole particles.                       |
+-------------------+-----------------------+---------------------------------------------+

When generating particle datasets, it is recommended to use these names for the particle types to ensure compatibility.
Additional particle types may be added with arbitrary names; however, care should be taken to ensure that they
are handled properly by any tools that will be used to process the dataset.

Particle Fields
***************

Like particle types, Pisces adopts a general convention for naming particle fields. This convention provides
both a set of **required** fields that must be present in all particle datasets, as well as a set of expected names
for some common fields. The following fields are required for all particle datasets:

+-----------------------+---------------------------------------------+
| Field Name            | Description                                 |
+=======================+=============================================+
| ``particle_position`` | The Cartesian position of the particles.    |
+-----------------------+---------------------------------------------+
| ``particle_velocity`` | The Cartesian velocity of the particles.    |
+-----------------------+---------------------------------------------+
| ``particle_mass``     | The mass of the particles.                  |
+-----------------------+---------------------------------------------+

In addition to these required fields, the following fields are the default expected names for some common
particle properties. These fields are not required, but if they are present, they should be named as follows:

+-------------------------------------+--------------------------+---------------------------------------------+
| Field Name                          | Particle Types           | Description                                 |
+=====================================+==========================+=============================================+
| ``particle_position``               | All particle types       | The Cartesian position of the particles.    |
+-------------------------------------+--------------------------+---------------------------------------------+
| ``particle_velocity``               | All particle types       | The Cartesian velocity of the particles.    |
+-------------------------------------+--------------------------+---------------------------------------------+
| ``particle_mass``                   | All particle types       | The mass of the particles.                  |
+-------------------------------------+--------------------------+---------------------------------------------+
| ``particle_id``                     | All particle types       | Unique identifier for each particle.        |
+-------------------------------------+--------------------------+---------------------------------------------+
| ``potential``                       | All particle types       | The gravitational potential at the position |
+-------------------------------------+--------------------------+---------------------------------------------+
| ``gravitational_field``             | All particle types       | The gravitational field at the position of  |
|                                     |                          | each particle.                              |
+-------------------------------------+--------------------------+---------------------------------------------+
| ``density``                         | gas                      | The density of gas at the position of       |
|                                     |                          | each particle.                              |
+-------------------------------------+--------------------------+---------------------------------------------+
| ``temperature``                     | gas                      | The temperature of gas at the position of   |
|                                     |                          | each particle.                              |
+-------------------------------------+--------------------------+---------------------------------------------+
| ``metallicity``                     | gas                      | The metallicity of gas at the position of   |
|                                     |                          | each particle.                              |
+-------------------------------------+--------------------------+---------------------------------------------+
| ``internal_energy``                 | gas                      | The internal energy of gas at the position  |
|                                     |                          | of each particle.                           |
+-------------------------------------+--------------------------+---------------------------------------------+
| ``magnetic_field``                  | gas                      | The magnetic field at the position of each  |
|                                     |                          | particle.                                   |
+-------------------------------------+--------------------------+---------------------------------------------+
| ``smoothing_length``                | gas                      | The smoothing length for SPH calculations.  |
+-------------------------------------+--------------------------+---------------------------------------------+

Loading a Particle Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^

To load a particle dataset in Pisces, simply initialize the :class:`pisces.particles.base.ParticleDataset` class with the
path to the HDF5 file:

.. code-block:: python

   from pisces.particles import ParticleDataset

   # Load a particle dataset from an HDF5 file
   dataset = ParticleDataset.from_hdf5("path/to/pisces.particles.h5")

.. note::

    Like standard file access in python, you can also supply a ``mode='r'`` or ``mode='r+'`` argument to open the file
    in read-only or read-write mode, respectively. The default is read-write mode, which allows you to modify the
    dataset in place. If you only need to read the dataset, it is recommended to use read-only mode to avoid
    accidental modifications.

Once the dataset is loaded, all of the data will be accessible through the dataset object. You can see the available
fields using the :attr:`~pisces.particles.base.ParticleDataset.fields` attribute, and the particle types using
:attr:`~pisces.particles.base.ParticleDataset.particle_types`.

Accessing Particle Data
^^^^^^^^^^^^^^^^^^^^^^^^

Pisces provides several ways to access particle data, depending on whether you want to load the
data immediately into memory or work with it lazily. All field values are returned
as `unyt <https://unyt.readthedocs.io/>`_ arrays, with units automatically parsed from the HDF5 metadata.

There are three primary access methods:

1. **Immediate (eager) access using indexing**
2. **Lazy access via field handles**
3. **Batch access for multiple fields**

Indexing Access
******************

You can access particle fields directly using dictionary-style indexing with dot notation:

.. code-block:: python

   # Load gas density into memory as a unyt array
   rho = dataset["gas.density"]

   # Load the position field for dark matter
   pos = dataset["dark_matter.particle_position"]

This will **immediately** load the field into memory, including unit conversion via :mod:`unyt`. Equivalently,
you can use the :meth:`~pisces.particles.base.ParticleDataset.get_particle_field` method to achieve the same result.

.. note::

    Each particle dataset has a ``"UNITS"`` attribute that contains the units for each field. This is used to
    ensure that all fields are loaded with the correct units, and it is automatically handled by Pisces when
    accessing fields.


Lazy Access with Field Handles
******************************

For memory-efficient workflows, use the :meth:`~pisces.particles.base.ParticleDataset.get_particle_field_handle` method
to obtain a handle to the HDF5 dataset:

.. code-block:: python

   # Get lazy-access handle to the 'velocity' field
   handle = dataset.get_particle_field_handle("gas", "particle_velocity")

   # Access data slice-by-slice
   slice_0 = handle[0]
   chunk = handle[100:200]

This does not load the entire dataset into memory. You can use slicing, chunking, or deferred computation on the handle.
Field access handles are useful when working with very large datasets or performing parallel or chunked operations.

.. important::

    When you access data directly through the handle, it will **not** automatically convert units. You must
    convert the data to unyt arrays manually if needed.

Batch Access
*************

To load multiple fields at once, use :meth:`~pisces.particles.base.ParticleDataset.get_particle_fields` or
:meth:`~pisces.particles.base.ParticleDataset.get_particle_field_handles`. This method returns a dictionary
of unyt arrays for the specified fields:

.. code-block:: python

   fields = dataset.get_particle_fields([
       "gas.particle_position",
       "gas.particle_velocity",
       "gas.density"
   ])

   pos = fields["gas.particle_position"]
   vel = fields["gas.particle_velocity"]

This returns a dictionary of unyt arrays, and is a convenient way to prepare data for computation or export.

Checking for Field Existence
****************************

You can check if a field exists using Python's `in` operator:

.. code-block:: python

   if "gas.temperature" in dataset:
       T = dataset["gas.temperature"]


Modifying Particle Fields
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If a user accesses a particle field via indexing (or :meth:`~pisces.particles.base.ParticleDataset.get_particle_field`),
and then proceeds to edit the resulting :class:`unyt.array.unyt_array`, the changes will **NOT** be automatically
written back to the HDF5 file. This is because Pisces uses lazy loading for particle fields, meaning that the data is
not loaded into memory until it is explicitly accessed. If you want to modify a particle field and save the changes back
to the HDF5 file, you must modify the field via the field handle. For example:

.. code-block:: python

   # Get a handle to the 'density' field
   density_handle = dataset.get_particle_field_handle("gas", "density")

   # Modify the density values
   density_handle[:] *= 2.0  # Double the density of all gas particles

   density_handle.flush()  # Force write.

Alternatively, you can simply replace an entire field with a new set of data using the
:meth:`~pisces.particles.base.ParticleDataset.add_particle_field` method:

.. code-block:: python

   # Create a new field with modified data
   new_density = unyt_array([1.0, 2.0, 3.0], "g/cm**3")

   # Add the new field to the dataset
   dataset.add_particle_field("gas", "new_density", new_density)

This has the advantage of automatically handling unit conversion and metadata updates, ensuring that the
new field is properly integrated into the dataset and has the correct number of particles present. You can also rename
existing fields using the :meth:`~pisces.particles.base.ParticleDataset.rename_field` method.

Geometric Transformations
^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to direct modification of a dataset's fields, there are a number of helper methods to do various geometric
modifications. The most useful of these is :meth:`~pisces.particles.base.ParticleDataset.offset_particle_positions` and
:meth:`~pisces.particles.base.ParticleDataset.offset_particle_velocities`, which can be use to systematically shift the
positions and velocities of the particle dataset. There is also the
:meth:`~pisces.particles.base.ParticleDataset.rotate_particles`, which can be used to rotate the particle dataset around
a particular axis.

Between the three of these, one can effectively produce an arbitrary transformation of the dataset.

Combining and Reducing Particles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pisces provides robust functionality for merging, filtering, and restructuring particle data. These operations are
useful when assembling simulation initial conditions, downsampling data for analysis, or constructing composite datasets
from multiple sources.

To merge multiple particle datasets into one, use the
:meth:`~pisces.particles.base.ParticleDataset.concatenate_inplace` method. This method appends the particle groups
and fields from one or more other datasets into the current dataset. Additionally,
:func:`~pisces.particles.utils.concatenate_particles` can be used to combine particle datasets into a new
particle dataset file.

.. code-block:: python

   # Load two separate particle datasets
   ds1 = ParticleDataset("gas_particles.h5")
   ds2 = ParticleDataset("stellar_particles.h5")

   # Append all particle groups from ds2 into ds1
   ds1.concatenate_inplace(ds2)

By default, all groups are merged. You can optionally restrict the operation to specific groups.
If a group already exists in the target dataset, Pisces will **extend** the group's fields by appending new particles.
For new groups, the entire group is copied directly.

Filtering Particles with Boolean Masks
**************************************

To downsample or restrict a particle group based on some condition (e.g., density threshold), use
:meth:`~pisces.particles.base.ParticleDataset.reduce_group`. This method removes all particles from a group that do not
match the given boolean mask.

.. code-block:: python

   # Load gas density
   density = ds["gas.density"]

   # Create a mask for low-density particles
   mask = density < 1e-26 * unyt.g / unyt.cm**3

   # Retain only the particles that satisfy the mask
   ds.reduce_group("gas", mask)

This is a **destructive** operation — particles not matching the mask are permanently removed from the file. It will
apply the mask to all fields within the specified group, keeping only matching entries.

Copying a Dataset
******************

If you want to make a safe copy of a dataset (e.g., before modification), use the
:meth:`~pisces.particles.base.ParticleDataset.copy` method:

.. code-block:: python

   ds_copy = ds.copy("filtered_particles.h5", overwrite=True)

This creates a new HDF5 file with identical structure, field data, and metadata. You can then safely modify the copy
without altering the original dataset.

Extending Particle Groups
**************************

You can append new particles to a group with the :meth:`~pisces.particles.base.ParticleDataset.extend_group` method:

.. code-block:: python

   new_positions = unyt_array([[1, 2, 3], [4, 5, 6]], "kpc")
   new_masses = unyt_array([1e5, 2e5], "Msun")

   ds.extend_group("gas", 2, fields={
       "gas.particle_position": new_positions,
       "gas.particle_mass": new_masses
   })

Any fields not provided will be filled with NaNs (if possible). Unit compatibility is checked automatically.

The Particle Dataset File Structure
-----------------------------------

Pisces stores all particle data in a single **HDF5 file**, structured to group particles by type and
organize fields within those groups. This format is designed for interoperability, lazy loading,
and compatibility with simulation tools and data analysis libraries like :mod:`yt`.

The structure of a particle dataset is as follows:

.. code-block::

    /                         (HDF5 root)
    ├── PartType0/            (particle group: gas)
    │   ├── particle_position
    │   ├── particle_velocity
    │   ├── particle_mass
    │   └── ...
    ├── PartType1/            (particle group: dark matter)
    │   └── ...
    └── <attributes>          (global metadata)

Metadata
^^^^^^^^

Metadata is stored as **HDF5 attributes** at multiple levels:

- **Global (root-level) metadata** is stored as attributes of the root group and can include:

  - ``CREATION_DATE``: ISO 8601 UTC string representing the dataset creation time.
  - Arbitrary user- or subclass-defined attributes (e.g., simulation parameters, units, version info).
  - Serialized values via ``__serialize_metadata__`` and ``__deserialize_metadata__``,
    including unit-aware values using `unyt`.

Groups
^^^^^^

Each particle group corresponds to a particle type (e.g., ``gas``, ``dark_matter``, ``stars``) and is stored
as an HDF5 group. Every group **must** define:

- ``NUMBER_OF_PARTICLES``: an integer attribute specifying the number of particles in the group.

Additional group-level metadata may be added and accessed via:

.. code-block:: python

   group_metadata = dataset.group_metadata["gas"]

Fields
^^^^^^

Each field is stored as an HDF5 dataset within a particle group. Fields are expected to follow this shape:

- **Shape**: ``(n_particles, ...)`` — the first dimension must match ``NUMBER_OF_PARTICLES`` for that group.
- **Attributes**:
  - ``UNITS``: a string representing the physical units of the field (e.g., ``"Msun"``, ``"kpc"``, ``"km/s"``).

Fields can be accessed either eagerly or lazily, and unit metadata is automatically handled:

.. code-block:: python

   # Load the full field (with units)
   density = dataset["gas.density"]

   # Load HDF5 handle only (no memory load or unit conversion)
   handle = dataset.get_particle_field_handle("gas", "density")
   units = handle.attrs["UNITS"]

Pisces ensures all field shapes are consistent with the declared particle count and that units
are stored and retrieved correctly.
