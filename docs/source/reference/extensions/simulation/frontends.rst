.. _simulation_frontends:

=========================
Simulation Frontends
=========================

.. currentmodule:: pisces.extensions.simulation

Simulation frontends in Pisces provide the bridge between internal
:class:`~core.initial_conditions.InitialConditions` objects and external
simulation codes such as Gadget-2, Arepo, or other N-body/SPH/mesh codes.
Each supported simulation code has its own dedicated frontend class,
responsible for producing the correct input files and configuration from
Pisces-generated models.

At their core, frontends do three things:

1. Manage configuration files, ensuring every run has a reproducible
   YAML configuration stored alongside the initial conditions.
2. Validate compatibility between Pisces initial conditions and the
   requirements of the target simulation code.
3. Generate simulation-ready files (parameter files, particle datasets,
   etc.) in the native format expected by the external code.

Because each simulation code has unique requirements, Pisces provides a
separate frontend for each, all of which inherit from the same abstract
base class :class:`~core.frontends.SimulationFrontend`.

The Frontend Class
------------------

Every simulation code supported by Pisces is accessed through a *frontend*.
A frontend is a Python object that knows how to take a
:class:`~core.initial_conditions.InitialConditions` and produce the
simulation-ready files needed to run that code.

As a user, you do not need to worry about how a frontend is implemented.
Instead, you just work with a small and consistent interface:

- **Initialization**: create a frontend by attaching it to an
  :class:`~core.initial_conditions.InitialConditions` object.
- **Configuration**: adjust the automatically generated YAML configuration
  file (either by editing it directly or modifying it through the frontend).
- **Generation**: call :meth:`~core.frontends.SimulationFrontend.generate_initial_conditions`
  to write out all files needed for the external simulation code.

This makes the workflow for different simulation codes essentially the same:
you always load initial conditions, attach the appropriate frontend, edit the
configuration if needed, and generate the output files. For example:

.. code-block:: python

    from pisces.extensions.simulation.frontends import Gadget2Frontend
    from pisces.extensions.simulation import InitialConditions
    import unyt

    ic = InitialConditions.from_file("cluster_ic.hdf5")

    # Attach the Gadget-2 frontend
    frontend = Gadget2Frontend(ic)

    # Adjust configuration through the Python API
    frontend["params.units.length"] = unyt.Unit("kpc")

    # Generate Gadget-2 input files
    frontend.generate_initial_conditions()

The details (e.g. which files are written, which options are available) depend
on the specific frontend, but the overall usage pattern is the same across all
simulation codes supported by Pisces.


Initializing a Frontend
^^^^^^^^^^^^^^^^^^^^^^^

To attach a frontend to an initial conditions object, simply initialize it
with the IC:

.. code-block:: python

    from pisces.extensions.simulation.frontends import Gadget2Frontend
    from pisces.extensions.simulation import InitialConditions

    # Load an initial conditions object from disk
    ic = InitialConditions.from_file("cluster_ic.hdf5")

    # Create a Gadget-2 frontend
    frontend = Gadget2Frontend(ic)

When initialized, the frontend will:

1. Validate that the provided :class:`~core.initial_conditions.InitialConditions`
   object is suitable for the simulation code.
2. Copy the default configuration template (YAML) into the IC directory.
3. Load the configuration into a
   :class:`~pisces.utilities.config.ConfigManager` instance.
4. Allow keyword overrides to update configuration values programmatically.

For example, you could override parameters at creation time:

.. code-block:: python

    frontend = Gadget2Frontend(
        ic,
        reset_configuration=True,
        makefile={"longids": 1},
        run={"TimeBegin": 0.01, "TimeMax": 1.0}
    )

The Frontend Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

Each frontend manages its own YAML configuration file, stored in the
same directory as your initial conditions. This file is a copy of the
default template for the simulation code, taken from
``pisces/extensions/simulation/frontend_configs``. The templates are named
after the simulation code (for example, ``gadget2_config.yaml``), and you
are free to edit them if you want to change the defaults for future runs.

As a user, you normally interact with the configuration in one of two ways:

- **Edit the YAML file directly** in the IC directory.
- **Access it in Python** through the frontend’s
  :attr:`~core.frontends.SimulationFrontend.config` property or by using dictionary-style access
  (``frontend["..."]``).

Example:

.. code-block:: python

    # Both styles work:
    print(frontend.config["run.TimeBegin"])
    print(frontend["run.TimeBegin"])

    # Update values in Python
    frontend["run.TimeMax"] = 2.0

    # No need to call save() — changes are written to disk automatically!

Because the configuration is tied to a specific set of initial conditions,
you can safely customize values for one run without affecting other runs
or the global defaults.

.. note::

   The configuration file created in your IC directory is named after the
   frontend class. For example, the Gadget-2 frontend writes
   ``Gadget2Frontend_config.yaml`` next to your initial conditions.


Generating Output Files
^^^^^^^^^^^^^^^^^^^^^^^

The main job of a frontend is to take your Pisces initial conditions (ICs)
and turn them into the input files needed by the simulation code. This is
done with a single call to:

.. code-block:: python

    # Generate all Gadget-2 input files from the IC
    frontend.generate_initial_conditions()

When you run this command, the frontend will:

1. Check that your configuration is valid.
2. Write all the files required by the external code into the IC directory.

Exactly which files are written depends on the simulation code. For example:

- **Gadget-2** frontends write a parameter file (``parameterfile``) and
  particle data in HDF5 format.
- Other codes may generate unit tables, runtime scripts, or submission templates.

All generated files live in the **same directory as your initial conditions**,
so you always have a self-contained record of the IC, the configuration, and
the input files used for the run.

Example workflow:

.. code-block:: python

    from pisces.extensions.simulation.frontends import Gadget2Frontend
    from pisces.extensions.simulation import InitialConditions

    # Load ICs
    ic = InitialConditions.from_file("cluster_ic.hdf5")

    # Attach the frontend
    frontend = Gadget2Frontend(ic)

    # Generate Gadget-2 input files
    frontend.generate_initial_conditions()

.. tip::

   Because each IC object has its own configuration file, you can safely
   regenerate input files multiple times with different settings. This ensures
   reproducibility without affecting other simulations or global defaults.

Developing New Frontends
------------------------

This section is aimed at developers who want to add support for a new external
simulation code. It explains the frontend lifecycle, shows how to scaffold a
new frontend, and provides practical guidance for configuration, validation,
file generation, testing, and maintainability.


A frontend should:

- **Be reproducible by default**: configuration is copied per-IC and tracked in the IC directory.
- **Fail early with clear errors**: validate input ICs and runtime configuration before writing files.
- **Write only within the IC directory**: keep runs self-contained.
- **Expose a small, consistent user API**: initialize → (optionally tweak config) → generate files.

Most importantly, frontends should **not** attempt to fully emulate the behavior of the simulation code
or idiot-proof every possible user error. Instead, they should safeguard only elements of the simulation
workflow which are directly the responsibility of Pisces.

Lifecycle Overview
^^^^^^^^^^^^^^^^^^

All frontends inherit from :class:`~core.frontends.SimulationFrontend` and follow the same
lifecycle:

1. **Initialization**

   - Validate the input :class:`~core.initial_conditions.InitialConditions` (via
     :meth:`~core.frontends.SimulationFrontend._validate_input_ic`).
   - Copy the default YAML template to the IC directory if needed.
   - Load the per-run configuration into a :class:`~pisces.utilities.config.ConfigManager`.
   - Apply any keyword overrides passed by the caller.
   - Call :meth:`~core.frontends.SimulationFrontend.__post_init__`.
2. **Modification**

   - Users can edit the configuration file directly or modify it through
     :attr:`~core.frontends.SimulationFrontend.config` or dictionary-style access.
3. **Generation**

   - Validate runtime configuration (via
     :meth:`~core.frontends.SimulationFrontend._validate_runtime_configuration`).
   - Produce native files (via
     :meth:`~core.frontends.SimulationFrontend._generate_initial_conditions`).

Effectively every element of this lifecycle can be customized by overriding
the appropriate methods in your subclass.

Scaffolding a New Frontend
^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a new module for your code (e.g. ``frontends/mysim.py``) and implement a
subclass that points to a default config template and writes the necessary files.

.. code-block:: python

    # frontends/mysim.py
    from pathlib import Path
    from pisces.extensions.simulation.core.frontend import SimulationFrontend, __frontend_bin_path__

    class MySimFrontend(SimulationFrontend):
        # 1) Point to your default YAML template inside frontend_configs
        __default_configuration_path__ = (
            __frontend_bin_path__ / "mysim_config.yaml"
        )

        def _validate_input_ic(self, ic):
            # 2) Quick checks that ICs are compatible with your code
            #    (particle types present, units available, required fields, etc.)
            if "Coordinates" not in ic.fields:
                raise ValueError("MySim requires 'Coordinates' in the IC fields.")
            return True

        def _validate_runtime_configuration(self, *args, **kwargs):
            # 3) Enforce config-level constraints before writing files
            cfg = self.config
            if cfg["run.TimeBegin"] >= cfg["run.TimeMax"]:
                raise ValueError("run.TimeBegin must be < run.TimeMax for MySim.")
            # Add other checks: units, required keys, value ranges, etc.

        def _generate_initial_conditions(self, *args, **kwargs):
            # 4) Write native input files using IC data + config
            self.logger.info("Writing MySim parameter file...")
            param_path = self.ic_directory / "mysim_parameterfile.txt"
            with open(param_path, "w") as f:
                f.write(self._render_parameterfile())

            self.logger.info("Writing MySim particle data...")
            self._write_particles()  # implement for your code

        # ----- helpers ----- #
        def _render_parameterfile(self) -> str:
            cfg = self.config
            # Example line-oriented parameter format
            lines = [
                f"TimeBegin = {cfg['run.TimeBegin']}",
                f"TimeMax   = {cfg['run.TimeMax']}",
                f"LongIDs   = {cfg['makefile.longids']}",
                # add more mappings as needed
            ]
            return "\n".join(lines) + "\n"

        def _write_particles(self):
            # Example: translate IC fields to your format
            # Always read from self.initial_conditions and write to self.ic_directory
            coords = self.initial_conditions["Coordinates"]  # ndarray or unit array
            masses = self.initial_conditions.get("Masses", None)
            # ... write out to your code’s binary/HDF5/ASCII as required
            pass

Default Configuration Template
""""""""""""""""""""""""""""""

Place a YAML template in ``pisces/extensions/simulation/frontend_configs``. Name it
after your code (e.g. ``mysim_config.yaml``). Keep it readable and well-commented.

.. code-block:: yaml

    # pisces/extensions/simulation/frontend_configs/mysim_config.yaml

    # Compile-time (or build-time) compatibility knobs
    makefile:
      longids: 1        # 0 = 32-bit IDs, 1 = 64-bit IDs (match your code build)

    # Generic run parameters
    run:
      TimeBegin: 0.01   # start of integration (code-specific meaning)
      TimeMax:   1.0    # end of integration
      OutputDir: "./output"

    # IO options (paths are resolved relative to the IC directory)
    io:
      parameterfile: "mysim_parameterfile.txt"
      particlefile:  "mysim_particles.h5"

    # Optional: per-model mappings (field names and particle types)
    models:
      [MODEL_NAME]:
        PART_TYPE_0:
          name: gas
          fields:
            Coordinates: pos
            Velocities:  vel
            Masses:      mass

.. tip::

   The :attr:`~core.frontends.SimulationFrontend.config` object is a
   :class:`~pisces.utilities.config.ConfigManager` that **automatically writes
   changes to disk**. Users can edit the YAML or set values via
   ``frontend["section.key"] = value`` without calling ``save()``.

Validation Strategy
^^^^^^^^^^^^^^^^^^^

There are two validation hooks. Use both.

1. **Input IC validation** (:meth:`~core.frontends.SimulationFrontend._validate_input_ic`)

   - Confirm the IC has the **fields** and **particle types** your code needs.
   - Confirm **units** are compatible (or convertible) if your code expects specific ones.
   - Example checks:

     - Coordinates exist and have shape ``(N, 3)`` for 3D codes.
     - Masses are present if the code requires them.
     - Temperature/entropy/metallicity present if writing hydrodynamical ICs.

2. **Runtime configuration validation**
   (:meth:`~core.frontends.SimulationFrontend._validate_runtime_configuration`)

   - Ensure required **keys** exist (e.g. ``run.TimeBegin``, ``run.TimeMax``).
   - Enforce **ranges** (e.g. positive values, monotonic constraints).
   - Check **inter-key dependencies** (e.g. if ``ComovingIntegrationOn=1`` then ``TimeBegin`` is a scale factor).

Example validation:

.. code-block:: python

    def _validate_input_ic(self, ic):
        required = ["Coordinates", "Velocities", "Masses"]
        missing = [k for k in required if k not in ic.fields]
        if missing:
            raise ValueError(f"MySim requires fields: {missing}")
        # Unit sanity check (if using unyt)
        coords = ic["Coordinates"]
        if hasattr(coords, "units") and str(coords.units.dimensions) != "(length)":
            raise ValueError("Coordinates must carry length units.")
        return True

    def _validate_runtime_configuration(self, *args, **kwargs):
        cfg = self.config
        for k in ["run.TimeBegin", "run.TimeMax"]:
            if k not in cfg:
                raise KeyError(f"Missing required config key: {k}")
        if cfg["run.TimeBegin"] >= cfg["run.TimeMax"]:
            raise ValueError("TimeBegin must be < TimeMax.")
        if cfg.get("makefile.longids") not in (0, 1):
            raise ValueError("makefile.longids must be 0 or 1.")

Generating Files Correctly
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Always write into :attr:`~core.frontends.SimulationFrontend.ic_directory`.
- Use relative, deterministic filenames (ideally configurable via ``io`` keys).
- Prefer writing to a **temporary file** and moving it into place to avoid partial writes.
- Emit clear **log messages** (use :attr:`~core.frontends.SimulationFrontend.logger`).
- Avoid side effects outside the IC directory.

Field & Particle-Type Mappings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If your code expects specific names or component structures, expose a
``models`` section in the YAML to let users map **Pisces model** fields to
**native** fields of your code.

.. code-block:: yaml

    models:
      GalaxyModel:
        PART_TYPE_0:
          name: gas
          fields:
            Coordinates: pos
            Velocities:  vel
            InternalEnergy: u

Your frontend can read this mapping and transform IC fields accordingly.

.. code-block:: python

    def _mapped_field(self, model_name, ptype, gadget_field):
        mapping = self.config.get(f"models.{model_name}.{ptype}.fields", {})
        src = mapping.get(gadget_field, gadget_field)  # default: passthrough
        return self.initial_conditions[src]

Units & Derived Quantities
^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Units**: if IC arrays carry units (e.g. via ``unyt``), decide whether your
  target format preserves them (HDF5 attributes) or requires unitless values
  plus a unit table.
- **Derived values**: If the target code needs a quantity (e.g. specific
  internal energy) not present in the IC, compute it **explicitly** and write it.
- **Time semantics**: If the code uses a special time variable (e.g. scale
  factor), document and validate it in your config.

.. warning::

   Keep consistency between configuration (e.g. ``makefile.longids``) and
   what the downstream code is built for. Mismatches are a common source
   of cryptic I/O errors.

Logging, Errors, and UX
^^^^^^^^^^^^^^^^^^^^^^^

- Use :attr:`~core.frontends.SimulationFrontend.logger` for user-facing messages.
- Prefer precise exceptions with actionable hints:
  - ``KeyError`` for missing configuration keys.
  - ``ValueError`` for invalid values or failed checks.
  - ``TypeError`` for wrong object types.
- Wrap lower-level I/O with helpful context (which file, which dataset).
