.. _initial_conditions_overview:
=========================================
Initial Conditions for Simulations
=========================================

.. currentmodule:: pisces.extensions.simulation.core.initial_conditions

In any astrophysical simulation, the **initial conditions** (ICs) are crucial — they
set the starting point for everything that happens next. They include the positions,
velocities, and properties of all objects in your simulation.

In order to facilitate the creation, management, and inspection of initial conditions,
*Pisces* provides a dedicated module for handling ICs in a structured, efficient way. These
are housed in the :mod:`~pisces.extensions.simulation.core.initial_conditions` package, which features
its standard base class :class:`InitialConditions` along with various subclasses.

The :class:`InitialConditions` class provides a framework for assembling
and managing initial conditions for simulations, allowing you to:

- **Combine multiple models** — Easily place different pre-built models (like galaxies, star clusters,
  or custom configurations) into a single simulation domain. This lets you set up complex scenarios,
  such as galaxy mergers or multi-component systems, without having to build each component from scratch.

- **Converting Models to Additional Data Structures** - Many simulations require either particle versions
  of the initial conditions or for initial conditions to be deposited into a grid structure. Initial conditions
  provide a number of methods and structures for managing and creating these additional data
  structures, such as particle datasets or grid-based representations.

- **Perform pre-simulation physics adjustments** — Before you even start the main simulation, you can use
  this class to perform important preparatory calculations. For example, you can calculate the center of
  mass for your system and shift all models to that reference frame, or even evolve the orbits of your objects
  for a short period to get a more stable starting configuration.

In short, this module allows you to go from a collection of analytic or loaded models to a
coherent, reproducible IC package that can be fed directly into compatible simulation
frontends such as those in :mod:`~pisces.extensions.simulation`.

The Initial Conditions Class
----------------------------

The base class for managing initial conditions is :class:`InitialConditions`, which provides
all of the various methods and properties touched on above. In practice, an initial conditions object:

- **Combines multiple models** — Each model is stored in its own HDF5 file within the IC directory,
  and the IC object keeps track of all models and their metadata.
- **Places models in space** — Depending on the IC class, models can be stored with
  their positions, velocities, orientations, and spins in the simulation volume.
- **Stores complete metadata** — All parameters needed to reconstruct the IC state
  are stored in a central configuration file.
- **Prepares simulation-ready datasets** — Outputs a directory structure and
  HDF5 files that downstream simulation frontends can read directly.

This section will introduce the key methods of the :class:`InitialConditions` class
and show how to use them to build fully packaged, simulation-ready datasets.

.. important::

    The :class:`InitialConditions` class is designed to be subclassed if necessary to provide
    additional functionality specific to a simulation code or frontend. Generally, you will use
    the :class:`InitialConditions3DCartesian`, which provides the specific case of 3D Cartesian
    coordinates. However, if you need a different coordinate system or dimensionality, you can
    create your own subclass of :class:`InitialConditions` and implement the necessary methods
    or use one of the existing subclasses.

    Before setting up your initial conditions, look at the frontend for your simulation code
    in :mod:`~pisces.extensions.simulation.frontends`. If your frontend requires a specific
    initial conditions class, you should ensure that you are using that class
    instead of the base :class:`InitialConditions` class.

Initial Conditions Directory Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you create a new initial conditions (IC) package using Pisces, the system is designed to be **self-contained**.
This means all the necessary files are stored within a single, dedicated directory. This approach makes
your IC set highly portable and easy to share with colleagues or transfer between different
computing systems and simulation codes without losing any critical information.

The directory structure follows a consistent and logical layout, ensuring that every component
of your simulation setup is easy to find. Below is a typical example of what
a Pisces IC directory looks like:

.. code-block:: text

    my_ic_directory/
    ├── IC_CONFIG.yaml           # Main configuration file describing the IC
    ├── model1.hdf5              # Model file for "model1"
    ├── model1_p.hdf5            # Particle file for "model1" (optional)
    ├── model2.hdf5              # Model file for "model2"
    ├── model2_p.hdf5            # Particle file for "model2" (optional)
    ├── extra_config.yaml        # (optional) Additional configuration files
    └── derived_data/            # (optional) Derived analysis products

Central to this structure is the `IC_CONFIG.yaml` file. This is the main configuration
file that acts as the "master" record for your entire IC set. It contains all the essential metadata,
such as the overall number of dimensions and the name of the IC class used to create the package.
Most importantly, it meticulously documents all the models you've included, along with their precise attributes
like position, velocity, orientation, and spin. This file also keeps track of the file paths for any associated
particle datasets, ensuring everything remains linked.

For each model you create (e.g., ``model1`` and ``model2``), Pisces generates a dedicated HDF5 file
named exactly after the model's name (e.g., `model1.hdf5`). If your simulation requires a particle
representation of these models, an optional particle file is stored alongside its parent model,
following a clear naming convention: the model's name with a `_p.hdf5` suffix (e.g., `model1_p.hdf5`).

You can also include additional files, such as analysis outputs or extra configuration files,
within the IC directory. These can be placed directly in the main directory or in subdirectories
like `derived_data/`. It's important to be mindful that these additional files should not accidentally
overwrite the core model or particle files unless you intend to replace them.

This standardized and comprehensive layout guarantees that every IC set is a complete and reliable package.
When you or another user loads the directory, Pisces can immediately find all the necessary files to inspect,
modify, or use the ICs for a simulation without needing any external information or dependencies.

.. note::

    When the initial conditions you generate are passed off to the frontend for a particular simulation code,
    the frontend will typically create a number of additional files and directories within the IC directory
    to handle the specifics of that simulation code. These files are not created by Pisces itself, but rather
    by the frontend when it processes the ICs. Therefore, the initial conditions directory structure you create
    with Pisces may be extended by the frontend to include additional simulation-specific files, such as
    configuration files, output directories, or other necessary components for running the simulation.

Creating an Initial Conditions Object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to create an initial conditions object, you typically will start by calling
the :meth:`InitialConditions.create_ics` class method. For example,

.. code-block:: python

    from pisces.extensions.simulation import InitialConditions3DCartesian

    model_1, model_2 = ...  # Load or create your models here

    # Create model parameters
    # These are the positions and velocities of the models in the simulation volume.
    m1_pos,m2_pos = (unyt.unyt_array([0.0, 0.0, 0.0], "Mpc"),
                    unyt.unyt_array([1.0, 0.0, 0.0], "Mpc"))
    m1_vel,m2_vel = (unyt.unyt_array([0.0, 0.0, 0.0], "km/s"),
                    unyt.unyt_array([0.0, 100.0, 0.0], "km/s"))

    # Now we create the IC's via the method call.
    ic = InitialConditions3DCartesian.create_ics(
        "my_ic_directory",
        ("model1", model_1, m1_pos, m1_vel),
        ("model2", model_2, m2_pos, m2_vel),
        overwrite=True
    )

As shown, each model is specified as a tuple containing its name, model object,
and a few other pieces of metadata. The 3D Cartesian class expects the following structures
to be provided when you add a model:

.. code-block:: text

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

In addition, the method takes a few useful arguments; most notably, the ``overwrite`` keyword, which
allows you to delete an existing IC directory if it already exists and is non-empty. This is useful
when you want to recreate the ICs from scratch without worrying about leftover files from previous runs.
Another important keyword is ``file_processing_mode``, which determines whether the input files are copied
or moved into the IC directory. By default, files are copied, but you can set this to ``"move"`` if you want
to transfer files instead of duplicating them.

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
:class:`InitialConditions` class includes
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
