.. _initial_conditions_overview:
=========================================
Initial Conditions for Simulations
=========================================

.. currentmodule:: pisces.extensions.simulation.core.initial_conditions

In many astrophysical and cosmological simulations, the **initial conditions** (ICs) define
the complete state of a system at the start of a simulation — including positions,
velocities, orientations, physical parameters, and any associated particle or grid data.
Accurate and well-structured ICs are critical for producing reliable, reproducible results.

The :class:`InitialConditions` system in *Pisces* is designed to:

- **Combine multiple models** — You can load several pre-built models (e.g., galaxy clusters,
  stellar systems, or custom configurations) into a single simulation domain.
- **Position and orient models** — Control the placement, orientation, and velocity of each
  model within the simulation volume.
- **Attach additional data** — Optionally associate particle datasets (e.g., from previous
  simulations or analytic generators) with individual models.
- **Generate simulation-ready files** — Output a fully packaged set of HDF5 files and a
  configuration file (``IC_CONFIG.yaml``) that can be read by downstream simulation codes.
- **Inspect without loading everything** — Quickly query model metadata, grids, or particle
  counts directly from disk without loading full data structures into memory.
- **Perform pre-simulation physics adjustments** — Compute mass-weighted centers of mass,
  shift all models into a COM frame, or integrate point-mass orbits to evolve the system
  before simulation.

In short, this module allows you to go from a collection of analytic or loaded models to a
coherent, reproducible IC package that can be fed directly into compatible simulation
frontends such as those in :mod:`~pisces.extensions.simulation`.

The Initial Conditions Class
----------------------------

The :class:`InitialConditions` class provides
the core infrastructure for assembling and managing simulation initial conditions.
It acts as a **container and controller** for a collection of models, handling their
placement, orientation, and associated metadata.

In practice, an initial conditions object:

- **Combines multiple models** — Each model is stored with its own position,
  velocity, orientation, and optional particle dataset.
- **Places models in space** — Models can be positioned and oriented arbitrarily
  within the simulation volume.
- **Stores complete metadata** — All parameters needed to reconstruct the IC state
  are stored in a central configuration file.
- **Prepares simulation-ready datasets** — Outputs a directory structure and
  HDF5 files that downstream simulation frontends can read directly.

This section will introduce the key methods of the :class:`InitialConditions` class
and show how to use them to build fully packaged, simulation-ready datasets.

Initial Conditions Directory Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you create a new initial conditions (IC) package, Pisces stores all
required files in a single directory. This makes the IC set **self-contained**,
portable, and easy to share between systems or simulation codes.

The structure is:

.. code-block:: text

    my_ic_directory/
    ├── IC_CONFIG.yaml           # Main configuration file describing the IC
    ├── model1.hdf5              # Model file for "model1"
    ├── model1_p.hdf5            # Particle file for "model1" (optional)
    ├── model2.hdf5              # Model file for "model2"
    ├── model2_p.hdf5            # Particle file for "model2" (optional)
    ├── extra_config.yaml        # (optional) Additional configuration files
    └── derived_data/            # (optional) Derived analysis products

Key points:

- **Model files** are named exactly after their model name in the IC configuration
  (e.g., ``model1.hdf5`` for a model named ``"model1"``).
- **Particle files** are stored alongside their parent model, using the pattern
  ``<model_name>_p.hdf5``.
- **Configuration file**: ``IC_CONFIG.yaml`` records:

  - Overall metadata (e.g., number of dimensions, IC class name)
  - All models and their attributes (position, velocity, orientation, spin)
  - File paths to associated particle datasets

- **Additional files** (e.g., analysis outputs, extra configs) can also be stored
  in the IC directory. These should not overwrite core model or particle files
  unless intentionally replaced.

This consistent layout ensures that every IC set contains all files necessary
to be loaded, inspected, or used for simulation without relying on external
dependencies.

Creating an Initial Conditions Object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`InitialConditions`
class provides a **convenience constructor** for creating an initial conditions
(IC) dataset from one or more models, with optional particle files, in a single step.

You can use the :meth:`InitialConditions.create_ics`
class method to:

- Create (or overwrite) a target directory for the ICs.
- Validate and register one or more models.
- Place (copy or move) model files into the IC directory, renaming them to match
  their assigned names.
- Optionally attach particle datasets to models, storing them alongside their
  corresponding model files.
- Generate an ``IC_CONFIG.yaml`` file that records the IC metadata, model
  configurations, and file paths.

As an example, here’s how to create a simple IC set with a single galaxy cluster
model:

.. code-block:: python

    from pathlib import Path
    import unyt
    from pisces.extensions.simulation import InitialConditions
    from pisces.models.galaxy_clusters import SphericalGalaxyClusterModel

    # Create a simple model
    model = SphericalGalaxyClusterModel.from_dens_and_temp(...)

    # Define initial position/velocity for the model
    pos = unyt.unyt_array([0.0, 0.0, 0.0], "Mpc")
    vel = unyt.unyt_array([0.0, 0.0, 0.0], "km/s")

    # Create a new IC set
    ic = InitialConditions.create_ics(
        "my_ic_directory",
        ("clusterA", model, pos, vel),
        overwrite=True
    )

As shown, each model is specified as a tuple containing its name, model object,
and a few other pieces of metadata. The base class expects the following structures
to be provided when you add a model:

.. code-block:: raw

    (name, model, position, velocity[, orientation][, spin])

where:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Description
   * - ``name``
     - Unique string identifier for the model within the IC set.
   * - ``model``
     - either a path (``str`` or :class:`~pathlib.Path`) to an existing HDF5 model file, or an
       instantiated :class:`~pisces.models.core.base.BaseModel` object.
   * - ``position``
     - :class:`~unyt.array.unyt_array` or sequence of length ``ndim`` with **length units** (default: meters).
   * - ``velocity``
     - :class:`~unyt.array.unyt_array` or sequence of length ``ndim`` with **velocity units** (default: km/s).
   * - ``orientation`` *(optional)*
     - Sequence or array of shape ``(ndim,)`` specifying the model’s orientation vector.
       Defaults to the unit vector along the last coordinate axis.
   * - ``spin`` *(optional)*
     - Scalar ``float`` (unitless) specifying the spin parameter. Defaults to ``0.0``.


**Attaching particle datasets**:

If you already have particle data for a given model, you can pass it via the
``particle_files`` keyword as a mapping from model name to file path:

.. code-block:: python

    particle_map = {
        "clusterA": "path/to/clusterA_particles.hdf5"
    }

    ic = InitialConditions.create_ics(
        "my_ic_directory",
        ("clusterA", model, pos, vel),
        particle_files=particle_map,
        file_processing_mode="copy"
    )

Particle files will be copied (or moved) into the IC directory and renamed using
the ``<model_name>_p.hdf5`` convention.

**Additional options**:

- ``file_processing_mode`` — either ``"copy"`` (default) or ``"move"``; determines
  whether input files are copied or moved into the IC directory.
- ``overwrite`` — if ``True``, an existing non-empty IC directory will be deleted
  before creating the new one.
- ``ndim`` — number of spatial dimensions (default: 3). All models must match this.

The returned :meth:`~pisces.extensions.simulation.core.initial_conditions.InitialConditions` instance is ready
for inspection, manipulation, or export to supported simulation formats.

Loading Initial Conditions from Disk
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you’ve created an initial conditions directory (for example with
:meth:`~InitialConditions.create_ics`), you can load it back into Python
by instantiating the :class:`InitialConditions` class with the path:

.. code-block:: python

    from pisces.extensions.simulation import InitialConditions

    ic = InitialConditions("/path/to/IC_directory")

This will read the ``IC_CONFIG.yaml`` file, discover all models and particle files,
and make their metadata and paths available through convenient properties such as
:attr:`~InitialConditions.models`,
:attr:`~InitialConditions.model_positions`, and
:attr:`~InitialConditions.model_velocities`.


Accessing Models, Particles, and Metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`InitialConditions` object lets you explore stored data without fully loading it into memory.

- **List available models**
  See the names of all models currently stored in the initial conditions set.
  Useful for quickly checking what objects are present before loading or modifying anything.

  .. code-block:: python

      ic.list_models()

- **Load a model** (full :class:`~pisces.models.core.base.BaseModel` object)
  Load a model’s complete data and metadata into memory for detailed analysis or modification.
  This is the most direct way to work with a model’s physical fields.

  .. code-block:: python

      model = ic.load_model("ClusterA")

- **Inspect available model fields** without loading the full dataset
  Get a quick list of the model’s stored fields (e.g., density, temperature) and their shapes.
  This is faster than loading the full model and is useful when you just need to know what’s inside.

  .. code-block:: python

      ic.get_model_fields("ClusterA")

- **View a model’s metadata** (e.g., total mass, coordinate system info)
  Access metadata attributes stored with the model, such as coordinate system,
  creation date, and physical parameters like ``total_mass``.
  Ideal for programmatically checking parameters before running calculations.

  .. code-block:: python

      ic.get_model_metadata("ClusterA")

- **Check if a model has particles**
  Determine whether a given model has an associated particle dataset.
  This is useful when preparing simulations that require both field-based and particle-based inputs.

  .. code-block:: python

      ic.has_particles("ClusterA")

- **Load particle data** into a :class:`~pisces.particles.base.ParticleDataset`
  Fully load the particle dataset for a model, enabling access to all particle positions,
  velocities, and additional fields for direct analysis or export.

  .. code-block:: python

      particles = ic.load_particles("ClusterA")

- **Inspect particle dataset without loading all data**
  Quickly explore the structure of a particle dataset before loading it in full.
  You can list species (particle groups), available fields, and particle counts for each group.

  .. code-block:: python

      ic.get_particle_species("ClusterA")
      ic.get_particle_fields("ClusterA")
      ic.get_particle_count("ClusterA")

These methods are all very useful to simulation frontends, which may need quite a bit of
information about the models you hope to simulate without loading everything into memory at once.

Adding, Removing, and Modifying Elements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can modify an existing initial conditions set by adding or removing models,
attaching particle datasets, or updating parameters.

- **Add a new model**
  Insert a new astrophysical model into the initial conditions set.
  The model can be loaded from a file or passed as an existing
  :class:`~pisces.models.core.base.BaseModel` object.
  You must also specify its position and velocity in the simulation volume.

  .. code-block:: python

      ic.add_model(
          "NewCluster",
          "/path/to/model.hdf5",
          position=[0, 0, 0] * u.kpc,
          velocity=[100, 0, 0] * u.km/u.s
      )

- **Attach a particle dataset to a model**
  Link an existing particle dataset file to a model in the initial conditions set.
  The file will be copied or moved into the IC directory and renamed to match
  the model (e.g., ``NewCluster_p.hdf5``).

  .. code-block:: python

      ic.add_particles_to_model(
          "/path/to/particles.hdf5",
          "NewCluster"
      )

- **Update a model’s position, velocity, or spin**
  Change key kinematic or orientation parameters of a model already in the set.
  This is useful for adjusting the initial placement or motion without recreating
  the entire initial conditions.

  .. code-block:: python

      ic.update_model("NewCluster", velocity=[200, 0, 0] * u.km/u.s)

- **Remove a model and its files**
  Completely delete a model from the initial conditions set,
  including its HDF5 file and any attached particle file.

  .. code-block:: python

      ic.remove_model("NewCluster")

- **Remove only the particle file from a model**
  Detach and optionally delete the particle dataset associated with a model,
  without removing the model’s main file or other configuration.

  .. code-block:: python

      ic.remove_particles_from_model("ClusterA")

Advanced IC Manipulations
------------------------------

Beyond simply placing models and assigning particle datasets, the
:class:`~pisces.extensions.simulation.core.InitialConditions` class includes
methods for performing **mass-weighted transformations** and **dynamical
analyses** on your initial setup.

Center-of-Mass (COM) Calculations & Transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In many simulations, it is useful to transform positions and velocities into
the **center-of-mass frame** — the frame in which the system’s total momentum
is zero. This can help eliminate unwanted drift or simplify orbit analysis.

Common operations include:

- **Compute the COM position**:

  .. code-block:: python

      com_pos = ic.compute_center_of_mass()

  This calculates the mass-weighted mean position of all (or selected) models,
  using model masses either supplied directly or read from each model’s
  ``total_mass`` metadata.

- **Compute the COM velocity**:

  .. code-block:: python

      com_vel = ic.compute_center_of_mass_velocity()

  Returns the mass-weighted mean velocity vector.

- **Shift to the COM frame**:

  .. code-block:: python

      ic.shift_to_COM_frame()

  Updates all stored positions and velocities so that the COM is at the origin
  and the total velocity is zero.

- **Get COM-frame positions and velocities** *without modifying the ICs*:

  .. code-block:: python

      com_frame_positions = ic.compute_center_of_mass_frame_positions()
      com_frame_velocities = ic.compute_center_of_mass_frame_velocities()

  These return dictionaries mapping model names to COM-frame vectors.

Orbit Integration for Point-Mass Equivalents
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The IC class can also integrate **point-mass equivalent orbits** for your
models using the `REBOUND <https://rebound.readthedocs.io/>`_ N-body package
(optional dependency).

This is useful for:

- Visualizing approximate system dynamics before committing to a full
  hydrodynamical simulation.
- Checking stability of orbital setups.
- Running quick parameter sweeps for binary or multi-body systems.

Example:

.. code-block:: python

    sim = ic.integrate_point_mass_orbits(
        t_end=100 * u.Myr,
        dt=0.1 * u.Myr,
        integrator="whfast"
    )

    # Inspect the final positions in parsecs
    for p in sim.particles:
        print(p.x, p.y, p.z)

The method automatically extracts positions, velocities, and masses from the
IC configuration, sets up a REBOUND simulation in physical units
(parsecs, solar masses, Myr), and runs the chosen symplectic or high-accuracy
integrator.

.. note::

   REBOUND is not installed by default. Install it with:

   .. code-block:: bash

      pip install rebound

Simulation Frontend Support
------------------------------

Most simulation frontends will take an :class:`InitialConditions` object
as input and handle the conversion to their native format internally.
This allows you to prepare your ICs once and use them across multiple
simulation codes with minimal effort. In some scenarios, it may be necessary
for the frontend to expose a specialized initial conditions class subclassed from
:class:`InitialConditions` to handle code-specific requirements. In that case,
you'll need to ensure that the frontend's IC class is what you're using.

For more details on using initial conditions with specific simulation codes,
see the documentation for the relevant frontend in
:mod:`~pisces.extensions.simulation.frontends`.
