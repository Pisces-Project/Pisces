.. _models_development:
===================================
Model Development in Pisces
===================================

.. currentmodule:: pisces.models.core.base

Overview of Model Development
------------------------------

One of the core design principles of Pisces is that it should be as easy as possible to
build an astrophysical model. This means that you should be able to focus on the
physics and the model structure, rather than the underlying infrastructure for
file handling, grids, and fields. Pisces provides a framework for building models
that is flexible and extensible, allowing you to create custom models that fit your
needs without having to reinvent the wheel.

In general, the steps for creating a model are relatively straightforward:

1. **Determine the Physics**: This stage is on you as the developer / scientist. Figure out what you're
   trying to build (galaxy, star, planet, BH merger, etc.) and what physics you want to capture. You'll need
   to be able to identify a path from some set of **inputs** to the complete set of models, coordinate systems,
   etc. that define your model.

   This often includes figuring out how you'll compute things like the potential, handling equations of state,
   or how to represent the geometry of the model (e.g., spherical, Cartesian, etc.).

2. **Place the Model in the Pisces Framework**: Before you start coding, its a good idea to figure out where your model
   belongs in the Pisces ecosystem. Each module in :mod:`~pisces.models` is specific to a type of system and the
   submodules within each of those are particular types of models. If you're going to eventually submit your model to
   the main codebase (which we encourage!), you should make sure that it fits into the existing structure, or you should
   create a new submodule that fits the existing structure.

   .. hint::

        We actually provide a skeleton / template for an entire module at ``/pisces/models/_module_template``. You can
        (and should) use this as a guide while you write your model.

3. **Build a Subclass Skeleton**: Once you've got an idea of the physics behind your model, you can start building
   a model subclass into pisces. This should inherit from :class:`BaseModel` will have a few standard / boilerplate
   elements for you to figure out as we'll discuss below.

4. **Write Generator Methods**: These are the methods that will take user-friendly inputs and produce a fully
   specified model. They should be class methods that return a new instance of your model.
   These methods will typically take parameters like mass, radius, density, temperature, etc.,
   depending on the physics of your model.

   .. hint::

        This is the hard step: we'll discuss a bunch of details for making things as simple as possible
        below.

5. **Implement Model-Specific Methods**: Once you have the basic structure of your model, you can start adding
   methods that are specific to your model. This might include methods for computing derived quantities,
   analyzing the model, or extracting specific fields.

Each of these steps is covered in the sections below. Additionally, the later parts of this document will cover
some in-depth details about how to implement specific features / handle complex scenarios.

Step 1: Determine the Physics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before you ever write a line of code, you should take time to map out the
**physics of your model**. This is the scientific backbone that everything
else in Pisces will wrap around. Ask yourself:

- **What system are you trying to represent?**
  (e.g., a polytropic star, a spherical galaxy cluster, a rotating disk). In some
  cases, you might be minimally extending an existing model, while in others
  you might be creating something entirely new.

  .. hint::

    If you’re not sure, start by looking at existing models in Pisces. They can
    provide inspiration and a template for your own work. Extending existing models
    is often easier than starting from scratch.

- **What inputs define that system?**
  Are these fundamental parameters (mass, radius, metallicity), analytic
  profiles (e.g., NFW, Vikhlinin), or equations of state? These will be the
  quantities you expect users (including yourself) to provide.

  .. important::

    There are a number of so-called **building-blocks** provided by Pisces; specifically
    the :mod:`~pisces.geometry` module which provides built-in coordinate systems, and coordinate
    grids, the :mod:`~pisces.profiles` module which provides a number of analytic profiles, and the
    :mod:`~pisces.physics` module which provides a number of physics equations and utilities.

    If you do find yourself needing to implement a new profile, coordinate system, or physics equation,
    please consider contributing it back to Pisces so that others can benefit from your work. You can find
    information about writing these building block classes in their respective
    documentation pages.

- **What physics needs to be computed?**
  For example:

  - Does your model need to solve hydrostatic equilibrium?
  - Will you be deriving pressure from density and temperature?
  - Do you need to compute a gravitational potential from a mass profile?

- **What geometry best describes the system?**
  Decide whether the model should live on a 1D spherical grid, a 2D polar
  mesh, or a 3D Cartesian lattice. The grid you choose determines both the
  coordinate system and the shape of every field.

  .. note::

    Pisces provides a number of built-in coordinate systems and grids, which you can
    use to simplify this step. If you need a custom coordinate system or grid, you can
    create one by subclassing the appropriate base classes in the :mod:`~pisces.geometry` module.

- **What outputs are essential?**
  At the end of the day, what do you want users to be able to access as
  fields? For instance:

  - A galaxy cluster might expose density, temperature, and potential.
  - A stellar model might include density, pressure, luminosity, and opacity.
  - A dark matter halo might provide only a density and mass profile.

It’s often helpful to sketch this out as a **data-flow diagram**:

.. code-block:: text

    Inputs (parameters, profiles)  --->  Physics equations  --->  Fields
         (e.g., mass, NFW profile)       (e.g., Poisson eq.)     (e.g., density, potential)

.. hint::

   Keep your initial scope narrow. Start with a minimal set of inputs and
   outputs that clearly define your model. You can always extend it later,
   but getting the “physics pipeline” right early will make everything else
   easier.

Once you have this roadmap, you are ready to place your model inside the Pisces
framework and begin building your subclass.

Build a Subclass
^^^^^^^^^^^^^^^^

Start out with a basic subclass of :class:`~pisces.models.core.base.BaseModel`. There are relatively
few things you need to do for simple models; however, there are a large number of things you *can* modify
if you need to. In general, you'll need to do the following:

- **Mark Class Flags**: Profiles often carry class flags which are class variables used either by
  a metaclass that loads the model or by the model itself to determine how to behave. The only one
  you need to worry about when modifying the base class (:class:`~pisces.models.core.base.BaseModel`) is
  ``__IS_ABSTRACT__``. This should be set to ``False`` for models that you want to instantiate directly.

  If you're making a model that serves as a template for other models (e.g., a base class for
  polytropic stars), you should set ``__IS_ABSTRACT__ = True``. This will prevent the model from being
  instantiated directly, but will allow it to be subclassed.
- **Define Entry Points**: These are the methods that users will call to create an instance of your model.
  They should be class methods that return a new instance of your model. You can have multiple entry points
  for different sets of inputs (e.g., one for mass and radius, another for density and temperature).

As an example of this, consider the following model for a polytropic star:

.. code-block:: python

    class PolytropicStarModel(BaseModel):
        # ======================================== #
        # CLASS FLAGS AND PROPERTIES               #
        # ======================================== #
        __IS_ABSTRACT__ = False # Don't treat the model as a template.

        # ========================================= #
        # Model Generator Methods                   #
        # ========================================= #
        # These methods are the generator methods for the model. They
        # should take some set of inputs (whatever is needed for the model)
        # and yield a resulting model.
        @classmethod
        def from_density_and_temperature(
            cls,
            filename: Path | str,
            core_density: unyt_quantity,
            core_temperature: unyt_quantity,
            polytropic_index: float = 1.0,
            rmin: unyt_quantity = unyt_quantity(1, "km"),
            rmax: unyt_quantity = unyt_quantity(100_000, "km"),
            num_points: int = 1000,
            overwrite: bool = False,
            **kwargs,
        ):
            ...

        @classmethod
        def from_mass_and_radius(
            cls,
            filename: Path | str,
            mass: unyt_quantity,
            radius: unyt_quantity,
            polytropic_index: float = 1.0,
            rmin: unyt_quantity = unyt_quantity(1, "km"),
            rmax: unyt_quantity = None,
            num_points: int = 1000,
            overwrite: bool = False,
            **kwargs,
        ):
            ...

Now, what does the generator method need to provide? By the time it's done, it'll need to have provided a fully
functional model file containing all of the necessary data and meeting the standards for the file format. The details
of the model structure are discussed below, but in general, you'll want to have the following components:

- **A Grid**: This is the coordinate system and the grid that defines the model. It should be a
  :class:`~pisces.geometry.grid.BaseGrid` object that defines the coordinate system and the grid points.
- **Fields**: These are the physical quantities that define the model. They should be stored as
  dictionaries of :class:`~pisces.fields.Field` objects, where each field is a 1D, 2D, or 3D array
  of values defined on the grid.
- **Metadata**: This is the information about the model, such as the model type, the coordinate system,
  the grid resolution, and any other relevant information. This should be stored as a dictionary of
  metadata fields.
- **Profiles**: If your model uses any analytic profiles (e.g., NFW, Vikhlinin), these should be stored
  as dictionaries of :class:`~pisces.profiles.Profile` objects. These profiles can be used to generate
  the fields and metadata for the model.

The base class :class:`~pisces.models.core.base.BaseModel` provides an extremely useful method called
:mod:`~pisces.models.core.base.BaseModel.from_components` which will take all of the above components and
create a fully functional model file. This method will handle the details of writing the grid, fields,
metadata, and profiles to the file, as well as validating the inputs and ensuring that everything is
correctly formatted. You can use this method to simplify the process of creating your model file.

.. important::

    In most cases, you should be able to just generate your data and feed it into the
    :meth:`~pisces.models.core.base.BaseModel.from_components` method. Nonetheless, it is
    possible to make structural changes to the model file format itself, in which case you may
    need to customize the method in order to ensure that it is compatible with your models reading
    and writing methods. This is an advanced topic, so it is covered later in the guide.


When you then load your model from disk, everything should be automatically available in the
:attr:`~pisces.models.core.base.BaseModel.grid`, :attr:`~pisces.models.core.base.BaseModel.fields`,
:attr:`~pisces.models.core.base.BaseModel.metadata`, and :attr:`~pisces.models.core.base.BaseModel.profiles`
attributes. This means that you can access the grid, fields, metadata, and profiles directly from the model instance
without having to worry about the underlying file format or structure. This is one of the key benefits of using
the Pisces framework for model development: it abstracts away the details of file handling and allows you to focus
on the physics and structure of your model.

Add Class-Specific Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have built a working model subclass and defined one or more generator
methods (``from_*`` constructors), the final step is to **add methods that are
specific to your model’s physics**. These methods make your model useful for
analysis and ensure that users don’t have to re-implement common operations
every time they work with it.

Typical examples include:

- **Analysis methods** – compute derived quantities such as enclosed mass,
  luminosity, or cooling time.
- **Extraction methods** – pull out slices, radial profiles, or line-of-sight
  quantities from the fields.
- **Post-processing methods** – generate mock observables, export the model
  to another format, or connect to simulation pipelines.

**Example: adding a mass calculator**

.. code-block:: python

    from unyt import unyt_quantity

    class PolytropicStarModel(BaseModel):
        __IS_ABSTRACT__ = False

        ...

        def enclosed_mass(self, radius: unyt_quantity) -> unyt_quantity:
            """Return the mass enclosed within a given radius."""
            r = self.grid["r"]
            rho = self["density"]

            # integrate density over volume (simplified example)
            dr = r[1:] - r[:-1]
            shell_volumes = 4 * np.pi * (r[:-1]**2) * dr
            shell_masses = rho[:-1] * shell_volumes
            return shell_masses.sum().to("Msun")

Here, ``enclosed_mass`` is a convenience method that works directly with the
model’s grid and density field. A user can simply do:

.. code-block:: python

    m_star = model.enclosed_mass(1.0 * unyt_quantity("Rsun"))
    print(m_star)

**Why add these methods?**

- They make your model *self-documenting*: anyone using your class sees the
  physics operations it supports.
- They reduce duplication: the logic for computing derived quantities is in
  one place, not scattered across user scripts.
- They encourage reproducibility: the methods use your model’s grid and fields,
  ensuring consistent results across analyses.

.. hint::

   When you add class-specific methods, think about what a typical user will
   want to do with your model right after creating it. If there’s a “first
   calculation” or “first plot” everyone writes, that’s a good candidate for
   a built-in method.

At this point, your model class is complete: it can be built from user inputs,
saved to disk, re-loaded later, and used for analysis with both built-in fields
and your own convenience methods.

The Model Class in Focus
-------------------------

Pisces models are built on top of the :class:`~pisces.models.core.base.BaseModel`.
This class provides all of the plumbing needed to read and write model files,
introspect their contents, and extend them with your own physics. As a model
developer, you’ll spend most of your time **subclassing** BaseModel and
defining entry points for creating your model, while relying on the base class
to handle I/O and validation.

A useful way to think about it:

- **What you *can* do**:
  Add new generator methods (``from_*``), define model-specific analysis
  methods, attach hooks, extend or customize metadata, and choose which fields
  and profiles your model should carry.

- **What you *shouldn’t* do**:
  Change how BaseModel manages file handles, alter the expected HDF5 group
  names, or bypass the provided read/write methods. These parts are deliberately
  standardized so that all Pisces models interoperate.

Model File Structure
^^^^^^^^^^^^^^^^^^^^^^^

Every Pisces model is stored as a **single HDF5 file**. This makes models
portable, reproducible, and easy to load. The internal structure is simple but
strictly standardized:

.. code-block:: text

    /
    ├── FIELDS/        # Physical field arrays (unit-aware)
    ├── PROFILES/      # Analytic profile definitions
    ├── GRID/          # Serialized grid object
    └── attrs          # Metadata (root-level attributes)

- **Metadata**: Descriptive tags, provenance, model identity.
- **Profiles**: Optional analytic functions that can regenerate fields or serve
  as references.
- **Grid**: The coordinate system and geometry the fields are defined on.
- **Fields**: The actual arrays representing physical quantities.

You can extend this structure to add your own groups, but all standard
components must be present if you want your model to load cleanly.

.. important::

   Do *not* rename the core groups (``FIELDS``, ``PROFILES``, ``GRID``) or
   remove required attributes like ``__model_class__``. This will break
   compatibility with the loader.

Building and Loading Model Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The lifecycle of a model file is always the same: you **create it once**, save
it to disk, and then you (or anyone else) can **load it later** and work with
its contents. Pisces takes care of the plumbing so that your model always
follows a predictable pattern.

1. **Build and save the model**
   When you first construct a model, you usually call a class method such as
   ``from_density_and_temperature`` or ``from_mass_and_radius``. These methods
   gather the required pieces:

   - a grid (geometry + coordinate system),
   - a set of physical fields (arrays with units),
   - optional profiles,
   - metadata.

   All of these are passed to
   :meth:`~pisces.models.core.base.BaseModel.from_components`, which:

   - creates a new HDF5 file with the correct skeleton (``FIELDS``, ``PROFILES``,
     ``GRID``, and root attributes),
   - writes your metadata, profiles, grid, and fields,
   - validates shapes and units to ensure everything matches.

   At the end you have a single ``.h5`` file on disk that fully represents your
   model.

2. **Open the model again later**
   To reload your model, you can either:

   - Instantiate the model class directly with the filename:
     ``model = MyModel("my_model.h5")``
   - Or use the general loader function
     :func:`pisces.models.utils.load_model`, which inspects the file’s
     ``__model_class__`` tag and dispatches to the right subclass.

   When a model file is opened, the constructor uses a set of internal readers:

   - :meth:`BaseModel.__read_metadata__` → loads root attributes into
     ``model.metadata``.
   - :meth:`BaseModel.__read_profiles__` → reconstructs analytic profiles into
     ``model.profiles``.
   - :meth:`BaseModel.__read_grid__` → loads the grid object into
     ``model.grid``.
   - :meth:`BaseModel.__read_fields__` → loads physical arrays into
     ``model.fields``.

   These methods can be overridden in subclasses if you need to customize how
   a component is read.

3. **Finalize setup**
   After all components are loaded, the constructor calls
   :meth:`BaseModel.__post_init__`. This is a “hook” for subclasses: you can
   override it to perform additional initialization (e.g., computing a cached
   derived quantity, wiring up hooks, or validating parameters). If you don’t
   override it, nothing special happens.

.. note::

   As a model developer, you usually don’t need to touch the low-level reader
   methods. In most cases, you’ll just generate your fields and metadata, call
   ``from_components`` to save, and rely on the standard constructor or
   ``load_model`` to reload.

Metadata
#########

Metadata lives in the **root attributes** of every Pisces model file. This is
where all of the high-level descriptive information about the model is stored.
Because metadata is always present and always accessible through
:attr:`BaseModel.metadata`, it serves as the “cover page” for your model.

Required Keys
**************

At a minimum, Pisces models must store the following metadata:

- ``__model_class__``
  The fully qualified class name of the model that wrote the file. This tag is
  **required** and is used to verify that a model file matches the class trying
  to open it. If the tag does not match, the loader will raise a ``TypeError``.

It is strongly recommended (though not strictly required) to also include:

- ``description`` – a short string describing the model’s purpose.
- ``source`` – where the model came from (e.g., “test suite”, “MUSIC ICs”).

Metadata Serialization
**********************
HDF5 attributes can only store basic types (numbers, strings, and small arrays).
Pisces therefore uses a **JSON-based serializer** to make metadata storage
general-purpose. The serializer lives in :class:`~pisces.utilities.io_tools.HDF5Serializer`,
and can be subclassed to add additional structure.

.. hint::

    The source code is relatively simple to parse in that module, you should be
    able to easily add new data structures to the serializer if you need to
    or want to store more complex metadata. The serializer will automatically
    handle the conversion to and from HDF5 attributes, so you can focus on
    defining the structure of your metadata.

- On save, all metadata values are passed through
  :meth:`~pisces.utilities.io_tools.HDF5Serializer.serialize_dict`, which converts complex objects into
  JSON strings.
- On load, they are reconstructed with
  :meth:`~pisces.utilities.io_tools.HDF5Serializer.deserialize_dict`.

This means you can store objects like :class:`~unyt.array.unyt_array` or
:class:`~unyt.array.unyt_quantity` directly in metadata—they will be serialized to
JSON with type tags and re-hydrated into full objects when you load the model.

.. important::

   All metadata in Pisces should **always** be passed through the serializer
   before being written to disk, and deserialized again when read back in.
   This ensures that every value is stored as a valid JSON string inside the
   HDF5 attributes and that complex Python objects are reconstructed correctly.

   In practice, this means using:

   - :meth:`~pisces.utilities.io_tools.HDF5Serializer.serialize_data` and
     :meth:`~pisces.utilities.io_tools.HDF5Serializer.deserialize_data` for
     general single values.
   - :meth:`~pisces.utilities.io_tools.HDF5Serializer.serialize_dict` and
     :meth:`~pisces.utilities.io_tools.HDF5Serializer.deserialize_dict` for
     full metadata dictionaries.

   These methods enforce consistent serialization for *all* supported types,
   including :class:`~unyt.array.unyt_array`, :class:`~unyt.array.unyt_quantity`,
   :class:`~unyt.Unit`, NumPy arrays, and more. By always using these utilities
   (rather than trying to write raw Python objects into HDF5 attributes), you
   guarantee that your metadata will remain portable, safe to load, and easy
   to extend with custom types in the future.

   The model class also has private methods ``._deserialized`` and ``._serialized``
   which can be used to access the deserialized and serialized metadata and simply
   wrap the serialization and deserialization methods. These are used internally
   by the model class to ensure that all metadata is serialized and deserialized
   correctly, but you can also use them if you need to access the raw metadata
   without going through the serializer.

Fields
#########

Fields are the **numerical core** of every model: arrays of density,
temperature, velocity, or whatever physical quantities your model is meant to
capture. In Pisces, all fields are stored in the ``/FIELDS`` group of the HDF5
file, and they are always loaded into memory as either plain NumPy arrays or
:class:`~unyt.array.unyt_array` objects (when units are attached).

Conventions
************

Each field is saved as a dataset under ``/FIELDS`` with the following rules:

- **Dataset Name**
  The name of the dataset should match the physical quantity
  (e.g., ``density``, ``temperature``, ``pressure``). Keep names
  consistent and descriptive.

- **Units Attribute**
  If the dataset has a ``units`` attribute, it is parsed by
  :mod:`unyt` and loaded as a :class:`~unyt.array.unyt_array`. If no units
  are found, the array is treated as unitless.

- **Shape Requirements**
  The **leading shape** of every field **must** match the model’s grid shape.
  This guarantees that all fields align with the grid coordinates.
  Trailing dimensions are allowed for vector or tensor components. For example:

  - scalar field: ``(Nx, Ny, Nz)``
  - vector field: ``(Nx, Ny, Nz, 3)``
  - tensor field: ``(Nx, Ny, Nz, 3, 3)``

- **Reduced Coordinates**
  Some models use reduced grids (e.g., spherical models defined only on ``r``).
  In these cases, fields must still follow the rule:
  **all fields = (grid_shape, element_shape)**, where ``element_shape`` may
  be empty (for scalars) or higher-rank (for vectors/tensors).

Validation and Loading
**********************

When a model is opened, :meth:`BaseModel.__read_fields__` handles loading:

1. It verifies that ``/FIELDS`` exists in the file. If not, it logs a warning
   and returns an empty dictionary.
2. It ensures that the grid has already been loaded so that shape checks can be
   performed.
3. For each dataset in ``/FIELDS``:

   - The dataset is read into memory.
   - If a ``units`` attribute is present, it is wrapped into a
     :class:`~unyt.array.unyt_array`; otherwise, it becomes a plain
     :class:`numpy.ndarray`.
   - The leading shape is validated against ``grid.shape``. If there is a
     mismatch, a ``ValueError`` is raised.

**Example: valid structure**

.. code-block:: text

    /FIELDS
        ├── density       [shape: (128,), units: "Msun/kpc**3"]
        ├── temperature   [shape: (128,), units: "keV"]
        └── pressure      [shape: (128,), units: "dyne/cm**2"]

Which will be loaded as:

.. code-block:: python

    {
        "density": unyt_array([...], "Msun/kpc**3"),
        "temperature": unyt_array([...], "keV"),
        "pressure": unyt_array([...], "dyne/cm**2"),
    }

.. important::

   Pisces enforces shape validation **twice**: once when saving (in
   :meth:`BaseModel.from_components`) and again when loading
   (in :meth:`BaseModel.__read_fields__`). This double-checking prevents silent
   errors and ensures that all fields remain consistent with the grid.

Developer Notes
***************

- Fields must **always** be stored under ``/FIELDS``—do not create your own
  top-level groups for physical arrays.
- Use clear, descriptive names. Avoid overloaded terms like “data” or “array”.
- Units should always be attached where possible. This allows downstream users
  to mix and match fields safely without worrying about unit conversions.
- Reduced coordinate systems are fully supported, but **all fields must follow
  the convention**: leading shape = grid shape, trailing shape = element shape.

Grids
########

The ``/GRID`` group contains the serialized grid object, which defines the
coordinate system and spacing of the model. Grids are reconstructed using
:func:`pisces.geometry.grids.utils.load_grid_from_hdf5_group`.

For a deeper dive into grid classes and options, see
:ref:`grids_overview` and :ref:`coordinate_systems_overview`.

Profiles
###########

Analytic profiles are stored under the ``/PROFILES`` group. Each subgroup
represents one profile and contains the serialized parameters and metadata
required to rebuild it.

On load, Pisces calls :meth:`~pisces.profiles.base.BaseProfile.from_hdf5` for
each subgroup, returning live profile objects that can be evaluated or
compared directly against fields.

See the :ref:`profiles_overview` documentation for details on writing and
using profiles.

Modifying the Model Structure
##############################

Pisces is designed to be flexible: you can add your own groups or attributes to
model files if your application requires it. For example, you might add a
``DIAGNOSTICS`` group for storing intermediate analysis outputs, or a
``SYNTHETICS`` group for mock observations.

When doing so:

- Keep all standard groups (``FIELDS``, ``PROFILES``, ``GRID``) intact.
- Do not overwrite required attributes like ``__model_class__``.
- Document any custom additions clearly so that collaborators (and your future
  self) know what they mean.

Do's and Don'ts of Modification
###############################

**Do:**

- Use :meth:`BaseModel.from_components` as your primary entry point.
- Add new groups for optional data, as long as they don’t collide with the
  standard ones.
- Extend metadata with descriptive tags, provenance, or configuration flags.
- Keep shapes, units, and naming consistent across fields.

**Don’t:**

- Rename or remove core groups (``FIELDS``, ``PROFILES``, ``GRID``).
- Skip the ``__model_class__`` attribute in metadata.
- Write raw arrays to the file without validating shapes or attaching units.
- Bypass the serializer for metadata; you’ll risk unreadable or broken files.

By following these guidelines, you ensure that your custom models stay fully
compatible with Pisces’ tools and can be shared or extended by others.

API Standards
--------------

Pisces models behave like lightweight, dictionary-like objects, while also
exposing a clean attribute-based API for their core components. This makes
them familiar to use while ensuring consistency across all model subclasses.

Dictionary-Like Access
^^^^^^^^^^^^^^^^^^^^^^

Models implement several “dunder” methods to provide intuitive access:

- ``__getitem__(key)``
  Retrieves either a **field** or a **grid axis**:

  .. code-block:: python

      density = model["density"]   # field array
      r = model["r"]               # coordinate array from the grid

  If the key is not found, a ``KeyError`` is raised.

- ``__setitem__(key, value)``
  Updates an existing **field** in the model:

  .. code-block:: python

      model["density"] = new_density

  .. important::
     You cannot set grid axes this way. Attempting to do so will raise a
     ``KeyError``. To modify geometry, you must rebuild the model with a new
     grid.

- ``__contains__(key)``
  Returns ``True`` if the key is present as a field:

  .. code-block:: python

      if "temperature" in model:
          print("Temperature field is available")

- ``__len__()``
  Returns the number of physical fields in the model:

  .. code-block:: python

      print(len(model))   # e.g., 3

- ``__iter__()``
  Iterates over the field names:

  .. code-block:: python

      for field in model:
          print(field)

Properties
^^^^^^^^^^

In addition to dictionary-like access, models expose several important
properties for accessing their components:

- :attr:`BaseModel.path`
  The file path of the model on disk.

- :attr:`BaseModel.handle`
  The live HDF5 file handle (advanced use).

- :attr:`BaseModel.grid`
  The grid object representing the model’s spatial geometry.

- :attr:`BaseModel.coordinate_system`
  The coordinate system of the grid (e.g., Cartesian, spherical).

- :attr:`BaseModel.active_grid_axes`
  A list of axis names active in the grid (e.g., ``["r"]`` for spherical).

- :attr:`BaseModel.fields`
  A dictionary mapping field names to their arrays.

- :attr:`BaseModel.profiles`
  A dictionary of analytic profiles used to generate or compare fields.

- :attr:`BaseModel.metadata`
  A dictionary of metadata attributes (description, provenance, etc.).

.. note::

   The intent of this API is that **all user-level interaction** with models
   goes through these dictionary methods and properties. You should never need
   to access internal attributes like ``__handle__`` or ``__fields__`` directly.

The Metaclass and Registry System
---------------------------------

Pisces models are managed through a **registry system** powered by a custom
metaclass. This system ensures that every model subclass is discoverable,
identifiable, and loadable from disk without needing to hard-code class
lookups.

How it Works
^^^^^^^^^^^^

All models inherit from :class:`~pisces.models.core.base.BaseModel`, which
uses the :class:`~pisces._generic.RegistryMeta` metaclass. When you define a
new model subclass, the metaclass automatically:

- Registers the class into a central registry.
- Records its class name so it can be matched against the ``__model_class__``
  attribute in metadata.
- Ensures that only non-abstract models (i.e., those with
  ``__IS_ABSTRACT__ = False``) are treated as valid entries.

This mechanism allows Pisces to load models purely from their on-disk metadata.
When you call :func:`pisces.models.core.utils.load_model`, it reads the
``__model_class__`` attribute from the HDF5 file and looks up the correct class
in the registry, instantiating it automatically.

The ``__DEFAULT_REGISTRY__``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every model defines a class variable called
:attr:`BaseModel.__DEFAULT_REGISTRY__`. By default, this points to the global
model registry (``__default_model_registry__``) that tracks all models in the
Pisces ecosystem.

- If you are building a model for personal or project-specific use, you can
  rely on this default registry. Your class will be automatically added when
  it is imported.
- If you are designing a specialized extension or want to keep your models
  separate, you can create and pass a **custom registry** instead. This is
  useful for plugin systems or experimental models that you don’t want mixed
  into the main namespace.

For example, you can create a custom registry like this:

.. code-block:: python

    from pisces.models.core.base import BaseModel
    from pisces._registries import Registry

    # Create a custom registry
    my_registry = Registry("MyProjectModels")

    class MyModel(BaseModel, metaclass=BaseModel.__class__):
        __IS_ABSTRACT__ = False
        __DEFAULT_REGISTRY__ = my_registry

    # Now MyModel will register into `my_registry` instead of the default
    # global registry.

Practical Implications
^^^^^^^^^^^^^^^^^^^^^^

- **For most developers:** you don’t need to think about the registry at all.
  Just subclass :class:`BaseModel` and your model will be discoverable and
  loadable automatically.

- **For advanced users:** the registry system gives you full control over
  where models are recorded. This can be useful in large projects, testing
  frameworks, or environments where multiple versions of a model might
  coexist.

.. note::

   The registry system is what makes ``load_model("filename.h5")`` possible.
   Without it, you would always need to know the exact Python class ahead of
   time. By combining the ``__model_class__`` metadata with the registry, Pisces
   can find the correct subclass for you, no matter where it is defined.


Logging
-------

Every model has a built-in logger available at :attr:`BaseModel.logger`. This
logger is part of Pisces’ structured logging system and is tied to the
``models`` subsystem. It automatically captures important messages during
model construction, file I/O, sampling, and analysis.

You can use it in your own model methods to report progress, debug information,
or warnings:

.. code-block:: python

    self.logger.info("Loaded profile '%s'", name)
    self.logger.warning("Missing metadata for '%s'", key)

.. note::

   The logging behavior (log level, output format, etc.) is configured in your
   Pisces configuration file under the ``[logging.models]`` section. This allows
   you to keep your code clean and defer control of logging verbosity to user
   preferences.

Use the logger for:

- Reporting progress or diagnostics in custom model methods
- Emitting debug messages during development or tuning
- Logging warnings for missing data, inconsistent inputs, or unusual behavior

Configuration Settings
----------------------

Every model also has access to a model-specific configuration dictionary via
:attr:`BaseModel.config`. This dictionary is populated from the Pisces global
configuration file and scoped to the model’s class name. It allows you to
parameterize model behavior without hardcoding values in your code.

For example, the global configuration might contain:

.. code-block:: yaml

    models:
      MyGalaxyClusterModel:
        sampling_resolution: 1000
        enable_logging: true

Inside your model class, you can then read these values:

.. code-block:: python

    resolution = self.config.get("sampling_resolution", 1000)
    enable_logging = self.config.get("enable_logging", True)

This provides a clean separation between *code* and *configuration*. Users can
tweak runtime settings in their config file without having to modify your model
implementation.

.. important::

   If the model class is not registered in the configuration file, accessing
   :attr:`self.config` will raise an error with guidance on how to add it. This
   ensures that models are always tied to explicit, discoverable settings.

Hooks
-----

Pisces provides a **hook system** for extending model functionality in a clean,
modular, and non-intrusive way. Hooks are mixins that you can attach to your
model class to add specialized features such as particle sampling, diagnostics,
or derived calculations. This allows you to build models that remain focused on
their **core physics**, while gaining extra behavior through reusable hook
classes.

What are Hooks?
^^^^^^^^^^^^^^^

A hook is just a Python class that inherits from
:class:`~pisces.models.core.hooks.BaseHook`. When you mix it into your model, Pisces
recognizes it as an extension and makes its methods and tools available. Hooks
can be:

- **Sampling utilities** (e.g., generating particles from a density profile).
- **Analysis helpers** (e.g., interpolating fields, computing derived data).
- **Diagnostics** (e.g., checking consistency, logging extra info).

For example:

.. code-block:: python

    from pisces.models.core.base import BaseModel
    from pisces.models.hooks import ParticleGenerationHook

    class MyModel(BaseModel, ParticleGenerationHook):
        __ParticleGenerationHook_HOOK_ENABLED__ = True

    model = MyModel("cluster_model.h5")
    print(model.get_active_hooks())
    # -> ["ParticleGenerationHook"]

Enabling and Disabling Hooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each hook is controlled by a **feature status flag** defined at the class level:

.. code-block:: python

    __<HookClassName>_HOOK_ENABLED__ = True  # or False

By default, most hooks are **enabled** when mixed into a model. You can disable
a hook for a particular subclass simply by setting this flag to ``False``:

.. code-block:: python

    class MyClusterModel(BaseModel, ParticleGenerationHook):
        __ParticleGenerationHook_HOOK_ENABLED__ = False

This makes it easy to reuse the same physics but opt out of specific
extensions.

Discovering Hooks
^^^^^^^^^^^^^^^^^

Pisces provides a set of introspection tools through the
:class:`~pisces.models.base.hooks._HookTools` mixin. Every model automatically gains
these methods:

- :meth:`~BaseModel.get_all_hooks` – list all hooks mixed into the model.
- :meth:`~BaseModel.get_active_hooks` – list only the hooks currently enabled.
- :meth:`~BaseModel.get_hook_class` – retrieve the class object for a given hook name.
- :meth:`~BaseModel.has_hook` – check if a hook is present on a model.

For example:

.. code-block:: python

    model.get_all_hooks()
    # -> ["ParticleGenerationHook", "SphericalParticleGenerationHook"]

    model.get_active_hooks()
    # -> ["ParticleGenerationHook"]

    model.has_hook("ParticleGenerationHook")
    # -> True

Customizing Hooks
^^^^^^^^^^^^^^^^^

Hooks can define **class-level settings** that you override in your model:

.. code-block:: python

    class ParticleSamplerHook(BaseHook):
        _ParticleSamplerHook_max_attempts = 1000

    class MyCluster(BaseModel, ParticleSamplerHook):
        _ParticleSamplerHook_max_attempts = 10  # override default

This pattern lets you configure sampling resolution, accuracy thresholds, or
other hook-specific parameters without editing the hook itself.

Types of Hooks
^^^^^^^^^^^^^^

The Pisces library already provides some standard hook templates:

- :class:`~pisces.models.core.hooks.ParticleGenerationHook` – defines an abstract interface for converting
  a model into a particle dataset. Subclasses implement the actual sampling
  logic.

- :class:`~pisces.models.core.hooks.SphericalParticleGenerationHook` – a template for spherical models
  that provides built-in methods for radial sampling, field interpolation, and
  velocity generation.

.. note::

   Template hooks (e.g., :class:`~pisces.models.core.hooks.ParticleGenerationHook`) are not meant to be
   mixed in directly. They are marked with
   ``__<HookClassName>_IS_TEMPLATE__ = True`` so they won’t appear in active
   hook discovery. Instead, you should subclass them to provide concrete
   implementations.

When to Use Hooks
^^^^^^^^^^^^^^^^^

Use hooks when you want to:

- Add functionality without cluttering your core model class.
- Reuse the same extension across multiple models.
- Keep optional features (like particle sampling) **opt-in** and clearly
  separated.
- Provide a consistent public API for advanced capabilities.

Summary
^^^^^^^

Hooks are one of the most powerful extension systems in Pisces:

- They are **modular**: you can attach or detach them per-model.
- They are **discoverable**: every model can list its hooks at runtime.
- They are **configurable**: per-class settings let you fine-tune behavior.

If you are developing new models, it’s worth becoming familiar with hooks—they
are the recommended way to add optional, advanced features while keeping the
core model clean and focused.
