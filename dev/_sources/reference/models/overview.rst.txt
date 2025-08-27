.. _models_overview:
===================================
Astrophysics Models in Pisces
===================================

.. currentmodule:: pisces.models.core.base

The most important part of the Pisces ecosystem is its models. Pisces
models are used to represent astrophysical systems ranging from simple
dark matter halos to fully realized disk galaxies, stellar models, and
more.

Behind the scenes, a model is generally an abstract wrapper around
a number of physics computations to allow a user to go from a set of
input quantities (parameters, profiles, etc.) to a set of **fields**,
representing the physical data of the model. The complexity of
this process and its numerical cost is dependent on the nature of the
model.

One of the most important elements of Pisces models is their extensibility.
Recognizing that

1. Astrophysical models are constantly changing and becoming more
   sophisticated to incorporate a larger set of relevant processes, and
2. Many models require domain-specific expertise which could not possibly
   be provided by a single developer,

we have designed the :mod:`~pisces.models` module to be not only extremely flexible, but
also quite easy to develop new code from. In many cases, a user might begin
their scientific workflow by building a custom model class and relying on Pisces
model-interacting infrastructure to perform their science. For details on
writing new models, see :ref:`models_development`.

Model Basics
---------------

Pisces models are the core building blocks for representing astrophysical
systems. A model is a *self-contained file on disk* that stores all of the
data you need: spatial coordinates, physical quantities (fields), analytic
profiles, and descriptive metadata.

As a user, you can think of a model as your **system in a file**.
You set up the system you want (e.g. a galaxy cluster, a stellar halo),
save it, and then return later to analyze, plot, or compare it without
having to recompute everything from scratch.

In general, you'll select a model which captures the physics / system you're interested in modeling,
provide some set of minimal inputs (profiles, parameters, etc.), and the model will then use the
information you provide to do all the heavy lifting to compute the relevant physical fields.

.. hint::

    For example, you might create a model of a spiral galaxy by specifying the radial density of the
    bulge and the disk. Then the model could compute things like the potential, velocity field,
    ISM temperature, etc. based on those inputs.

Every model has the following set of critical components:

- A **grid** and **coordinate system** that defines the spatial layout of the model. These are
  accessible via the :attr:`BaseModel.grid` and :attr:`BaseModel.coordinate_system` attributes.
- A set of **fields** that represent the physical quantities of interest (e.g., density, potential,
  velocity). These are accessible via the :attr:`BaseModel.fields` attribute.
- **Metadata** that describes the model, including its parameters, and other relevant
  information. These are accessible via the :attr:`BaseModel.metadata` attribute.
- **Profiles** that define analytic functions or distributions used in the model. These are
  accessible via the :attr:`BaseModel.profiles` attribute.

Building a Model
^^^^^^^^^^^^^^^^

In the most simple case, the user will be able to identify an existing model that captures the physics
they are interested in, and then use that model to compute the fields they need. This is done by
providing the model with the necessary parameters and profiles, which it will then use to compute
the relevant fields. In order to do this, look at the API documentation for the model you are
interested in, which will provide details on how to create and manipulate the model.

For example, a spherical galaxy cluster model might require you to provide a density and temperature profile,
and then it will compute the gravitational potential, pressure, and other relevant fields based on those inputs.

.. code-block:: python

    from pisces.models.galaxy_clusters import SphericalGalaxyClusterModel

    # Create a temporary output file
    filename = f"basic_cluster_model.h5"

    # Define the radial range
    rmin = unyt.unyt_quantity(1.0, "kpc")
    rmax = unyt.unyt_quantity(3.0, "Mpc")

    # Create the gas density and total density profiles
    gas_density = ...
    total_density = ...

    # Build the model
    model = SphericalGalaxyClusterModel.from_density_and_total_density(
        gas_density,
        total_density,
        filename,
        min_radius=rmin,
        max_radius=rmax,
        num_points=500,
        overwrite=True,
    )

The model will then compute the relevant fields based on the provided profiles and parameters.

.. hint::

    These generation methods are usually named things like ``from_<something>``.

Loading a Model from Disk
^^^^^^^^^^^^^^^^^^^^^^^^^

Once a model has been written to disk, you can bring it back into memory
for analysis. There are two main ways to do this:

1. **Direct constructor call**
   Each model subclass can be initialized directly with the filename of
   a saved model. This goes through the class’ ``__init__`` method, which
   verifies the file on disk, checks that it matches the expected
   ``__model_class__`` tag, and then loads metadata, profiles, grid, and
   fields into memory.

   .. code-block:: python

      from pisces.models.galaxy_clusters import SphericalGalaxyClusterModel

      # Load an existing model from its file
      model = SphericalGalaxyClusterModel("basic_cluster_model.h5")

      print(model)             # <SphericalGalaxyClusterModel @ basic_cluster_model.h5>
      print(model.fields.keys())  # dict_keys(["density", "temperature", ...])

2. **Using the registry loader**
   If you don’t know the exact subclass of the model you want to open,
   you can use the convenience function
   :func:`~pisces.models.core.utils.load_model`. This function inspects the
   ``__model_class__`` attribute stored in the file and uses Pisces’
   model registry to dispatch to the correct subclass automatically.
   This is particularly helpful when loading models produced by other
   people or by automated workflows:

   .. code-block:: python

      from pisces.models.utils import load_model

      # Load a model without needing to know its class
      model = load_model("basic_cluster_model.h5")
      print(type(model))
      # <class 'pisces.models.galaxy_clusters.SphericalGalaxyClusterModel'>

.. note::

   Pisces uses a central **model registry** (``__DEFAULT_REGISTRY__``)
   to keep track of all known model subclasses. This is how the
   :func:`~pisces.models.core.utils.load_model` function knows which class to instantiate. If you are
   writing your own custom model class, you should make sure it is
   registered so that it can be loaded from disk seamlessly.

The Geometry of the Model
^^^^^^^^^^^^^^^^^^^^^^^^^

.. hint::

   For a deeper tour of grids and coordinates, see :ref:`grids_overview`
   and :ref:`coordinate_systems_overview`.

Every model carries its own **grid** (the spatial layout and resolution) and
its **coordinate system** (e.g., Cartesian, spherical). These provide the following
key features:

- The grid defines **coordinates** (radius, x/y/z, etc.).
- The grid’s shape tells you how large each field array will be.
- The grid comes with a **coordinate system**, so you always know
  whether you’re working in Cartesian, spherical, or another system.

To access the grid or the coordinate system of a model, just access the relevant attributes:
:attr:`BaseModel.grid` and :attr:`BaseModel.coordinate_system`.

The most common grid interaction is extracting the coordinates of the grid axes, which can be done
by simply indexing into the grid:

.. code-block:: python

    # Get the grid coordinates
    r = model.grid["r"]  # Radial coordinate in spherical systems
    x = model.grid["x"]  # Cartesian x-coordinate
    y = model.grid["y"]  # Cartesian y-coordinate
    z = model.grid["z"]  # Cartesian z-coordinate

There are a number of other useful methods on the grid, which can be read about in :ref:`grids_overview`.

Accessing Model Data
^^^^^^^^^^^^^^^^^^^^

Once you have created or loaded a model, the next step is to **explore and
analyze its data**. All of this goes through the model’s **fields**, which are
NumPy-like arrays (wrapped in :class:`~unyt.array.unyt_array` so that units are
preserved).

The simplest access pattern is dictionary-style indexing:

.. code-block:: python

    # Load a model from disk
    from pisces.models.galaxy_clusters import SphericalGalaxyClusterModel
    model = SphericalGalaxyClusterModel("basic_cluster_model.h5")

    # Access fields by name
    density = model["density"]
    temperature = model["temperature"]

    print(type(density))
    # <class 'unyt.array.unyt_array'>
    print(density.units)
    # g/cm**3

    # Access grid axis arrays in the same way
    r = model["r"]   # radial coordinate
    print(r[:5])

The values come out as :class:`unyt.unyt_array`, which means they behave like
NumPy arrays but automatically track physical units. You can convert them to
different units, strip units if needed, or use them directly in calculations:

.. code-block:: python

    # Convert density to solar masses per cubic kiloparsec
    dens_msun = density.to("Msun/kpc**3")

    # Strip units and get a plain NumPy array
    dens_values = density.value

    # Perform arithmetic with units carried through
    pressure = density * temperature  # unit-aware product

.. hint::

   Always use the arrays directly from the model when plotting or
   analyzing—they are guaranteed to have the right shape for the grid
   and keep their units intact.

Accessing Metadata and Profiles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to fields and grids, models also carry **metadata** and optional
**analytic profiles**. These components provide context and reproducibility.

Metadata lives at the root of the model file and includes descriptive
information such as when the model was created, what parameters were used,
and any free-form tags or notes you added.

Access it directly from the :attr:`~BaseModel.metadata` property:

.. code-block:: python

    # Inspect model metadata
    print(model.metadata["description"])
    # "Hydrostatic mass test model."

    print(model.metadata.get("date_created"))
    # 2025-07-24T15:31:10.103Z

You can store anything meaningful in metadata when you first create the
model—for example, the name of the simulation run or the person who built it.


Profiles are analytic functions (e.g., NFW, Vikhlinin) that were used to
generate or compare against fields. They are stored in the model’s
``/PROFILES`` group and reconstituted into live Python objects when you load
the model.

.. code-block:: python

    # List all profiles in the model
    print(model.profiles.keys())  # e.g., dict_keys(["density", "temperature"])

    # Grab one profile and use it
    density_profile = model.profiles["density"]
    rho_val = density_profile.evaluate(r=100, units="kpc")

Profiles make it easy to go back to the analytic form that generated your
fields, or to compare analytic predictions against numerical arrays.

.. hint::

   Profiles are optional. Some models are built entirely from raw fields,
   while others pair fields with profiles for richer analytic control.

The Structure of a Model
^^^^^^^^^^^^^^^^^^^^^^^^^

A Pisces model is always saved as a **single HDF5 file on disk**. This makes
models easy to share, archive, and reload in a fully reproducible way. The
internal structure of the file is standardized so that any Pisces model can be
recognized and loaded automatically.

At a high level, the file contains:

- ``GRID/`` – the spatial grid and coordinate system
- ``FIELDS/`` – physical field arrays (density, temperature, etc.)
- ``PROFILES/`` – analytic profiles (optional)
- root attributes – metadata and identity tags

As a user, you do not normally interact with these groups directly. Instead,
you use the :class:`~pisces.models.core.base.BaseModel` API to access the grid,
fields, profiles, and metadata in a natural Pythonic way.


Model Fields
^^^^^^^^^^^^^^^^^^^^^^^^^

Fields are the **core data products** of a model. They are stored in the
``FIELDS`` group of the HDF5 file, and each corresponds to a physical quantity
defined over the model’s grid (for example, gas density, temperature, pressure,
or velocity).

When loaded into Python, fields are returned as :class:`~unyt.array.unyt_array`
objects, meaning they behave like NumPy arrays but also carry physical units:

.. code-block:: python

    # Access fields directly from the model
    density = model["density"]
    temperature = model["temperature"]

    print(density.units)   # g/cm**3
    print(temperature.mean())

Because units are always preserved, you can safely perform calculations
without losing track of physical dimensions:

.. code-block:: python

    # Compute pressure as density * temperature
    pressure = density * temperature
    print(pressure.units)

Fields can also be converted to different units or stripped of units if
needed:

.. code-block:: python

    # Convert to solar masses per cubic kiloparsec
    dens_msun = density.to("Msun/kpc**3")

    # Strip units to get a raw NumPy array
    raw_dens = density.value

.. hint::

   All fields are guaranteed to have shapes that match the grid of the
   model. This ensures consistency when plotting, integrating, or combining
   multiple fields.

Extension Hooks
---------------

Some Pisces models come with **hooks** – optional add-ons that give the model
extra capabilities beyond its core fields and grid. Hooks are a way for models
to “plug in” special functionality without changing how the rest of the model
works.

For example, a spherical galaxy cluster model might include a hook that knows
how to generate a particle realization of the cluster. Another model might use
a hook to provide extra diagnostic tools or quick plotting helpers.

As a user, you don’t need to know the implementation details. What matters is
that hooks can give your model extra powers, and you can easily see which ones
are available.

**Checking for hooks**

Every model has a few simple methods for working with hooks:

.. code-block:: python

    # See all hooks attached to the model
    print(model.get_all_hooks())

    # See only the hooks that are active
    print(model.get_active_hooks())

    # Check if a specific hook is available
    print(model.has_hook("particle_sampler"))

If a hook is active, you can usually just call its methods directly on the
model, just like any other method.

.. hint::

   Not all models have hooks, and the hooks available will depend on the
   type of model you are using. If you’re curious, check
   :meth:`BaseModel.get_active_hooks` to quickly see what extras your model supports.
