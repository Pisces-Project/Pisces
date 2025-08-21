.. _initial_conditions_dev_guide:
==========================================
Developer Guide: Initial Conditions
==========================================

.. currentmodule:: pisces.extensions.simulation.core.initial_conditions

This page is for Pisces contributors who are extending or subclassing the
:class:`InitialConditions` base class used to define, store, and manipulate
simulation initial conditions (ICs). It provides a detailed overview of the
class's goals, responsibilities, lifecycle, extension points, and best practices for
subclassing. For a high-level user overview, see the :ref:`initial_conditions_overview` section.

Subclassing
-----------

The :class:`InitialConditions` class is explicitly designed for extension.
If you are writing a new frontend or simulation interface, you will almost
certainly want to define a subclass with custom metadata rules, additional
processing hooks, or specialized helper methods.

Best practices:

- **Module placement**
  Define your subclass alongside the rest of your simulation or frontend code.
  This keeps the implementation localized and avoids cluttering the core Pisces
  namespace. For example:

  .. code-block:: text

     pisces/
       <simulation_code>/
            __init__.py
            initial_conditions.py  # Your subclass here
            frontend.py            # Frontend implementation

- **Override minimally**
  Subclasses should override only the parts they need:

  - Validation hooks (:meth:`_validate_configuration`, :meth:`_validate_model`)
  - Processing methods (:meth:`_process_models`, :meth:`_process_metadata`,
    :meth:`_process_particle_files` or equivalents for new structures)
  - New accessors or physics helpers

  Avoid rewriting core methods like :meth:`__init__` or
  :meth:`create_ics` unless absolutely necessary. These contain structural
  invariants that guarantee compatibility with the Pisces ecosystem.

- **Preserve base contracts**
  Always call ``super()`` when overriding validation or processing methods to
  keep the base guarantees intact. This ensures that IC directories created with
  your subclass can still be loaded and validated by the base class or other
  tools.

- **Additive, not destructive**
  Focus on *adding* capabilities rather than changing the meaning of existing
  methods. If you need fundamentally different semantics (e.g., an entirely new
  file layout), consider building a separate abstraction instead of subclassing
  :class:`InitialConditions`.

.. note::

   Place your subclass in the same module as your frontend or simulation code.
   This makes the dependency explicit: anyone using your frontend should import
   your specialized IC subclass directly.

Class Creation
--------------

Class creation is mediated via the :meth:`InitialConditions.create_ics` class method, which
builds a new IC directory from scratch. This method handles:

- Validating and processing model tuples (name, model, position, velocity, etc.)
- Copying or moving model files into the IC directory
- Writing the authoritative configuration file ``IC_CONFIG.yaml`` with metadata and models
- Optionally attaching particle datasets
- Returning a ready‑to‑use :class:`InitialConditions` instance

various elements of this process are extensible / modifiable while others are fixed to
ensure consistency and compatibility with the Pisces ecosystem. Formally, the :meth:`InitialConditions.create_ics`
method proceeds as follows:

.. code-block:: python

    @classmethod
    def create_ics(cls, directory, *models, particle_files=None, **kwargs):
        """
        Create a new IC directory with the given models and optional particle files.
        Returns an initialized InitialConditions instance.
        """
        # --- DIRECTORY SETUP --- #
        # Process the provided directory. We check that it is a valid directory
        # and that it doesn't contain any existing files that need to be overwritten.
        #
        # This is a structural invariant of this class and should NOT be overwritten
        # by subclasses to ensure that the structure is contiguous.
        ...

        # --- MODEL PROCESSING --- #
        # Validate and process the models, ensuring they conform to the expected
        # structure and dimensionality. This includes copying/moving files and
        # normalizing orientations.
        #
        # This part of the process is NOT STRUCTURALLY INVARIANT and may be modified
        # by altering the ``cls._process_models`` method.
        ...

        # --- ADDITIONAL PROCESSING --- #
        # This section handles optional particle files, metadata processing,
        # and any other additional processing required for the ICs.
        # The methods called here are also NOT STRUCTURALLY INVARIANT and may be
        # modified directly by altering the create_ics method.
        ...

        # --- IC CONFIG GENERATION --- #
        # Finally, we write the IC_CONFIG.yaml file with all the metadata and models.
        ...

        return cls(directory)

In editing this procedure, the following key points should be kept in mind:

- **Structural Invariants**: The directory structure and its handling (the first half of the
  `create_ics` method) should not be altered. This ensures that the ICs can be loaded and
  validated correctly by the base class and other Pisces components. Additionally, **ALL**
  initial conditions directories must contain an ``IC_CONFIG.yaml`` file, which is the
  authoritative source of metadata and model information.

  Additional metadata may be added to the ``IC_CONFIG.yaml`` file, but the
  resulting YAML file must contain the ``models`` key in order to ensure that the
  rest of the class structure can be loaded correctly.

  .. note::

      Additional metadata may be added to the ``IC_CONFIG.yaml`` file in other groups or
      sections, but the ``models`` key must always be present in order to ensure that the
      rest of the class structure can be loaded correctly. You can also edit the metadata
      attached to the models.

- **Additional Processing**: The second half of the `create_ics` method is where
  you can extend or modify the behavior of the IC creation process. This includes
  processing models, handling particle files, and writing metadata. You can override
  methods like `_process_models`, `_process_particle_files`, and `_process_metadata`
  to customize how these elements are handled.

  - If you modify the ``_process_metadata`` method, ensure that it still returns a
    dictionary that contains the ``metadata`` key with all the contents.
  - If you modify the ``_process_models`` method, ensure that it returns a dictionary
    with the relevant metadata. The following are required pieces of metadata for each model:

    - ``path``: The path to the model file.
    - ``position``: A unyt array with length units.
    - ``velocity``: A unyt array with velocity units.
    - ``orientation``: A normalized vector of length ``(ndim,)``.
    - ``spin``: A float representing the spin of the model.

  You may choose to add additional metadata to the models, but the above keys must always be present
  in order to ensure that the rest of the class structure can be loaded correctly.

Initialization
--------------

Initialization is handled through the :meth:`InitialConditions.__init__` constructor,
which loads an **existing** IC directory and ensures it is valid. This method
performs a series of checks and setup steps to guarantee that the instance is
ready for use:

- Normalize and store the target directory path
- Verify the directory exists on disk
- Locate and load the authoritative ``IC_CONFIG.yaml`` file
- Initialize the internal :class:`~pisces.utilities.config.ConfigManager` for autosaved config access
- Run :meth:`_validate_configuration` on the loaded config
- Run :meth:`_validate_model` on each stored model

Formally, the initialization procedure is:

.. code-block:: python

    def __init__(self, directory: Union[str, Path]):
        """
        Load an existing IC directory and validate its contents.
        """
        # --- DIRECTORY VALIDATION --- #
        # Ensure the provided directory exists on disk.
        # This structural invariant guarantees we always work with a real folder.
        ...

        # --- CONFIG FILE LOADING --- #
        # Require that the IC_CONFIG.yaml file is present in the directory.
        # Load it via ConfigManager with autosave=True.
        ...

        # --- CONFIG VALIDATION --- #
        # Delegate to self._validate_configuration() to ensure metadata consistency
        # and presence of required keys.
        ...

        # --- MODEL VALIDATION --- #
        # Iterate through all stored models, calling _validate_model on each.
        ...

        return self

Key extension points:

- **Structural Invariants**:
  - The directory must exist and contain ``IC_CONFIG.yaml``.
  - The YAML must have at least two top-level keys: ``metadata`` and ``models``.
  These invariants should not be altered by subclasses, as they are critical to
  compatibility across Pisces components.

- **Subclass Validation Hooks**:

  - :meth:`_validate_configuration` performs base checks on metadata (class name,
    ndim, required sections). Subclasses may override to enforce extra metadata rules
    (e.g., requiring box size, simulation code version). Always call
    ``super()._validate_configuration()`` first to preserve base guarantees.
  - :meth:`_validate_model` enforces presence of required fields (path, position,
    velocity, orientation, spin), correct dimensionality of vectors, and particle file
    existence. Subclasses can extend this to require additional per-model metadata.

  .. note::

     When extending validation, remember that the base checks ensure structural safety.
     Your overrides should add **domain-specific constraints** without breaking
     compatibility with the rest of the Pisces ecosystem.

In short, initialization guarantees that every :class:`InitialConditions` instance
represents a **valid, fully consistent IC directory**. Developers may tighten or
extend validation rules, but they must preserve the base invariants so that all
Pisces tools can rely on consistent structure and metadata.

Method Modification
-------------------

In addition to validation and creation hooks, subclasses are free to **add new
methods** or **override existing ones** to extend the functionality of
:class:`InitialConditions`. For example, you might add a helper that computes
angular momentum from positions and velocities, or override particle handling to
support a new file format.

That said, there are important guidelines:

- **Preserve behavioral contracts**:
  Existing methods such as :meth:`add_model`, :meth:`update_model`,
  :meth:`remove_model`, and :meth:`create_ics` have clear expectations about
  their inputs, side-effects, and outputs. Subclasses should not change these
  contracts in ways that would surprise downstream users or break other Pisces
  components.

  - If you override a method, call ``super()`` unless you are intentionally
    replacing its functionality.
  - Do not alter return types (e.g., always return a dict for model listings,
    always return a ready-to-use :class:`InitialConditions` instance from
    :meth:`create_ics`).
  - Do not silently skip validation or autosaving — these are core guarantees of
    the base class.

- **Additive extensions are preferred**:
  Rather than modifying existing logic heavily, consider:

  - Adding **new methods** with specialized behavior.
  - Wrapping or composing base methods, e.g. calling
    :meth:`update_model` and then applying domain-specific adjustments.
  - Providing subclass-only helpers for new metadata or file formats.

- **Consistency and compatibility**:
  The base methods ensure IC directories remain self-contained and portable
  across the Pisces ecosystem. Any modifications should keep directories valid
  (i.e., still containing a correct ``IC_CONFIG.yaml`` and valid model files).
  This ensures that ICs created with your subclass can still be loaded by the
  base :class:`InitialConditions` or by other tools.

.. note::

   If you need radically different semantics (e.g., a completely different on-disk
   layout), it is better to create a *new abstraction* outside of
   :class:`InitialConditions`. Subclasses are intended to **extend**, not replace,
   the guarantees of the base class.

Common Extension Patterns
--------------------------

Adding new metadata
^^^^^^^^^^^^^^^^^^^

A common subclassing task is to enrich ICs with additional metadata. This can be
done at two levels:

- **Global metadata** (applies to the IC directory as a whole)
  Override :meth:`_process_metadata` to inject extra keys into the ``metadata``
  section of ``IC_CONFIG.yaml``. For example, you might record a simulation code
  identifier, cosmological parameters, or a random seed. Always call
  ``super()._process_metadata`` first to retain the required fields (``class_name``,
  ``directory``, ``ndim``, ``created_at``).

  .. code-block:: python

     class MyICs(InitialConditions):
         @classmethod
         def _process_metadata(cls, directory, **kwargs):
             md = super()._process_metadata(directory, **kwargs)
             md["metadata"]["box_size"] = kwargs.get("box_size", 100.0)  # add box size
             md["metadata"]["frontend"] = kwargs.get("frontend", "gadget")
             return md

- **Per-model metadata** (applies to individual models)
  Extend :meth:`_process_models` to add new keys into each model’s dictionary.
  For example, you might attach a bulk temperature, metallicity, or softening
  length. Make sure the returned dict for each model still contains all required
  fields (``path``, ``position``, ``velocity``, ``orientation``, ``spin``).

  .. code-block:: python

     class MyICs(InitialConditions):
         @classmethod
         def _process_models(cls, directory, *models, **kwargs):
             processed = super()._process_models(directory, *models, **kwargs)
             for name, info in processed.items():
                 info["bulk_temperature"] = kwargs.get("bulk_temperature", 1e6)
             return processed

- **Accessors for new metadata**
  After storing additional metadata, provide convenience accessors so downstream
  code can query them easily. For example:

  .. code-block:: python

     class MyICs(InitialConditions):
         @property
         def box_size(self):
             return self.metadata.get("box_size")

         def model_temperatures(self):
             return {name: info["bulk_temperature"] for name, info in self.models.items()}

.. note::

   You may add any additional metadata you need, but the **core keys must remain**
   in both ``metadata`` and ``models`` for compatibility. These include
   ``metadata.ndim`` and, per model, the fields ``path``, ``position``,
   ``velocity``, ``orientation``, and ``spin``.

Adding new data structures
^^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes models require more than particle datasets. For example, you might
want to attach merger trees, halo catalogs, or radiative transfer grids. The
recommended approach is to follow the pattern of
:meth:`_process_particle_files`:

- **Introduce a new processing method**
  Define a method like ``_process_trees`` or ``_process_catalogs`` that takes
  paths to the new data structure(s), validates them, and copies/moves them into
  the IC directory. Extend the ``models`` dictionary with a new key
  (e.g., ``trees`` or ``catalog``) that stores the path.

- **Record relational metadata**
  Ensure your metadata clearly links the new data structure to its model(s).
  For example, store it under ``models.<name>.trees`` if each model gets its own,
  or create a new top-level section if the data applies globally.

- **Provide accessors and checks**
  Add convenience methods to query and load the new data structure, similar to
  ``has_particles``, ``get_particle_path``, and ``load_particles``. This makes it
  easy for downstream users to check whether a model has an attached dataset and
  to load it consistently.

- **Support lifecycle operations**
  Offer methods to generate, add, remove, and overwrite the new data structure.
  For example:

  - ``add_trees_to_model(name, path, overwrite=False)``
  - ``remove_trees_from_model(name, delete_file=True)``
  - ``generate_trees(name, **kwargs)``

  These should mirror the semantics of particle handling: validate existence,
  manage file copy/move, update ``IC_CONFIG.yaml``, and autosave.

.. note::

   The key requirement is that your new data structure integrates **cleanly with
   the existing config system**. That means:

   - All references must be stored in ``IC_CONFIG.yaml`` under a stable key.
   - The file(s) must be physically present inside the IC directory (or linked in
     a reproducible way).
   - Any model with an attached structure should be discoverable via accessors
     and listings, not hidden behind ad hoc conventions.
