.. _frontend_dev:
======================================
Developer Guide: Simulation Frontends
======================================

.. currentmodule:: pisces.extensions.simulation

This document is designed to help walk contributors / developers through
the process of creating a new simulation frontend for Pisces. It assumes familiarity with the
Pisces codebase and the general structure of simulation frontends. If you are not familiar
with these concepts, it is recommended to first read the
:ref:`simulation_frontends` documentation.

.. hint::

    It is often useful to take a look at the existing frontends to get a sense of what they're doing
    and how they work before diving into this material. Doing so will give you a better picture of what
    elements are relevant to your use case.

Frontend Structure
------------------

In this guide, we'll assume that you are creating a new frontend for a simulation code (we'll
call it "MySim") that is not currently supported by Pisces. To begin, you'll want to create the new
package ``pisces.extensions.simulation.mysim``, which will contain all of the necessary code for the
frontend that you're writing. There are *generally* two modules within this package that you will
need to create:

- ``frontend.py``: This module will contain the frontend(s) for your simulation code. It should
  inherit from the base class :class:`~core.frontends.SimulationFrontend` and implement the necessary methods
  to interface with your simulation code. We'll walk through the required methods in detail below.
- ``initial_conditions.py``: (**optional**) This module is used if your simulation code requires a new initial
  conditions class beyond that provided by Pisces' base class (:class:`~core.initial_conditions.InitialConditions`).
  If that is the case, consult the documentation (:ref:`initial_conditions_dev`) for how to implement a
  new initial conditions class.

.. note::

    In complex cases, your code may also need a **problem initializer**, which is written directly into the
    simulation code itself. This is relatively common in AMR codes, where initial conditions need to be deposited
    into the simulation grid. If you need to write a problem initializer, consult the
    :ref:`problem_initializer_dev` documentation for details.

With the structure in place, you're ready to start implementing the frontend.

Writing the Frontend Class
--------------------------

Developing a frontend for a simulation code can be somewhat complex and often requires some
deeper-level understanding of the code itself. It is our hope that none of that complexity comes
from Pisces itself. Frontends themselves are relatively simple, they'll subclass
:class:`~core.frontends.SimulationFrontend` and provide the following core functionality:

- **Configuration**: Within your frontend package, you'll need to create a configuration skeleton
  that is, when connected to an initial conditions object, copied and modified into the IC directory.
  The functionality here is to provide a location to store core-specific settings (from compile-time flags,
  parameter files, etc.) that are required to run the simulation code and which you need to know about from
  the user in order to run the simulation. See the section below for details on this.
- **Generation**: Once the frontend is setup and connected to the user's initial conditions, your frontend
  then needs to use all of the information available (model fields, particle data, etc.) to generate
  the necessary input files for the simulation code. This is done through the
  :meth:`~core.frontends.SimulationFrontend.generate_initial_conditions` method, which is described in detail below.

Frontend Initialization
^^^^^^^^^^^^^^^^^^^^^^^

The first steps we'll cover all concern the initialization of the frontend. This is done in the
:meth:`~core.frontends.SimulationFrontend.__init__` method, which is called when the frontend is created.
You should **never need to modify** this method directly because it is broken down into several
sub-methods that you can override to customize the behavior of your frontend. The key methods to
override are:

- ``_validate_input_ic``: This method is called to validate the initial conditions object that the frontend is being
  initialized with. You should check that the initial conditions object is compatible with your simulation code
  and raise an exception if it is not.
- ``_setup_config``: This method is called to setup the configuration for the frontend. This is the step that takes
  your default configuration skeleton and modifies it to match the initial conditions object. See the section below
  for details on how to implement this.
- ``__post_init__``: This method is called after the frontend has been initialized. You can use this method to
  perform any additional setup that is required for your frontend, such as loading additional
  data or setting up internal state.


The Configuration File
^^^^^^^^^^^^^^^^^^^^^^

The most critical element of a frontend is the configuration file. This file contains all of the
necessary settings for your simulation code, including compile-time flags, parameter files, and other
settings that are required to run the simulation. The idea is that when the user initializes the
frontend, a configuration file is generated in the initial conditions directory that contains all of
the necessary settings for the simulation code. This file can then be modified by the user to
customize the simulation run. Once the simulation settings are all set, the user can run the
generation step, which then uses those configuration settings and the initial conditions data to
generate the necessary input files for the simulation code.

To implement a configuration file for your frontend, you'll need to first point the
class to the default configuration skeleton. This is done by setting the class variable

.. code-block:: python

    class MySimFrontend(SimulationFrontend):

        # --------------------------------------- #
        # Class Variables and Constants           #
        # --------------------------------------- #
        # These are class-level variables which define connections to
        # the configuration along with some other aspects of the
        # frontend's behavior.
        __default_configuration_path__: Path = None
        """Path to the default configuration file template for this frontend.

        This should be set in subclasses to point to the default YAML
        configuration file that defines the expected parameters for
        this frontend. The file should be located in the
        `pisces/extensions/simulation/core/frontend_configs` directory.
        """

The configuration file may be configured however you choose, but it is often useful to remember
that it is exposed to the user and should therefore be organized in some reasonable way. For example,
you might use the following structure:

.. code-block:: yaml

    # ================================================================ #
    # Default FRONTEND configuration for Gadget-2 simulations.         #
    # ================================================================ #
    # This configuration file sets up the frontend parameters for running
    # Gadget-2 simulations within the Pisces framework. It includes settings
    # for simulation parameters, output options, and analysis routines.

    # -------------------------- #
    # Makefile Options           #
    # -------------------------- #
    # These are compile-time settings which are necessary to
    # ensure self-consistent / valid configurations.
    makefile:
      longids: true                # See ``LONGIDS`` in Makefile


    # -------------------------- #
    # Parameter File Options     #
    # -------------------------- #
    # These are runtime configuration settings which are necessary
    # to ensure that the simulation is run in a self-consistent manner.
    parameters:

      # The simulation box size. This should be provided in the
      # same units as the length units specified below.
      boxsize: 100.0

      # Unit support. Gadget-2 includes velocity, length, and mass
      # units. The time unit is derived from L/V.
      units:
        length: !unyt_unit "kpc"            # Length unit
        mass: !unyt_unit "Msun"             # Mass unit
        velocity: !unyt_unit "km/s"         # Velocity unit

    # -------------------------- #
    # Model Parameters (default) #
    # -------------------------- #
    # Below is the default set of model configurations for
    # mapping Pisces particle data properties to Gadget-2 particle types.
    # The convention present here matches the standard convention for Pisces.
    #
    # Once initialized, the frontend will create a unique one of these
    # for each model so that it can have specific settings.
    model_DEFAULT:
      Type0:
        name: "gas"
        fields:
          Coordinates: "particle_position"
          Velocities: "particle_velocity"
          ID: "particle_id"
          Masses: "particle_mass"
          InternalEnergy: "particle_internal_energy"
      Type1:
        name: "dark_matter"
        fields:
          Coordinates: "particle_position"
          Velocities: "particle_velocity"
          ID: "particle_id"
          Masses: "particle_mass"
      Type2:
        name: "disk"
        fields:
          Coordinates: "particle_position"
          Velocities: "particle_velocity"
          ID: "particle_id"
          Masses: "particle_mass"
      Type3:
        name: "bulge"
        fields:
          Coordinates: "particle_position"
          Velocities: "particle_velocity"
          ID: "particle_id"
          Masses: "particle_mass"
      Type4:
        name: "stars"
        fields:
          Coordinates: "particle_position"
          Velocities: "particle_velocity"
          ID: "particle_id"
          Masses: "particle_mass"
      Type5:
        name: "boundary"
        fields:
          Coordinates: "particle_position"
          Velocities: "particle_velocity"
          ID: "particle_id"
          Masses: "particle_mass"

    # Model configurations which are filled dynamically
    # when initialized:
    models:

.. important::

    While you, as the developer, can choose to structure the configuration file however you like, there
    are a few design conventions to follow:

    1. **Minimal Configuration**: The configuration file should only contain the settings that are
       necessary to **correctly build the initial conditions**. This means that for any given code,
       you generally only need a small subset of the available parameters.
    2. **Default Values**: The configuration file should provide default values for all parameters.
       Whenever possible these should match the defaults used by the simulation code itself.
    3. **Documentation**: The configuration file should be well-documented, with comments explaining
       each parameter and its purpose. This will help users understand how to modify the configuration
       file to suit their needs. When possible, point to the equivalent setting in the simulation code's
       documentation (e.g., the Gadget-2 user guide) so that users can find more information about
       the parameter and its expected values.

The Configuration Manager
`````````````````````````

Within the code, the configuration file is managed by the :class:`~pisces.utilities.config.ConfigManager` class
or some subclass thereof. The purpose for this is somewhat multiplex:

- The :class:`~pisces.utilities.config.ConfigManager` class abstracts away a lot of the nitty-gritty details of
  reading and writing configuration files, allowing you to focus on the high-level logic of your frontend. Specifically,
  the keys in the configuration file are accessed similarly to a dictionary and the values are automatically saved to
  the configuration file when modified.
- The :class:`~pisces.utilities.config.ConfigManager` class provides a way to read complex classes from yaml
  files, allowing configuration files to contain more complex data structures than just simple key-value pairs.

For the most part, you will not need to modify this aspect of the frontend, but there are two useful things that can
be done if necessary:

1. **Custom Configuration Classes**: If you need to read or write a custom class from the configuration file, you can
   create a subclass of :class:`~pisces.utilities.config.ConfigManager` and override it's behavior. Then register
   it as the configuration manager for your frontend by setting the
   ``__default_configuration_manager_class__`` class variable in your frontend class.

   .. note::

        Generally, this is useful when you want to support types in YAML which are not already natively supported in
        the :class:`~pisces.utilities.config.ConfigManager` class. You can create a custom YAML type (via
        ``ruamel.yaml``) and then register it with a new configuration manager class which simply sets the ``__YAML__``
        attribute to the custom yaml type. By default, the yaml reader fully supports unyt and pathlib.

2. Following from the first element, **make use of complex data types**: instead of requiring users to provide their
   unit system in cgs or specify values in code units, you can use ``unyt`` to allow them to specify those things
   with actual units. This is done by using the ``!unyt_unit`` YAML tag in the configuration file, which allows you to
   specify a unit in the configuration file that will be automatically converted to an unyt unit when read by the
   configuration manager.

.. hint::

    The :mod:`~pisces.utilities.config` and :mod:`~pisces.utilities.io_tools` modules are the low-level utility modules
    where we have defined all of this. They are quite short and relatively simple. If you are confused by how this works
    or need to implement something more complex, you can look at the source code for these modules to see how they work.

Setting Up the Configuration
````````````````````````````

Once the default configuration file is loaded, every frontend calls the ``_setup_config`` method to
modify the configuration file to match the initial conditions object that it is being initialized with.
This method is where you will set the necessary parameters for your simulation code based on the
initial conditions object. The method is called after the configuration file has been loaded, so you can
access the configuration file through the ``self.config`` attribute.

After ``_setup_config`` is called, the configuration file will then automatically process any ``**kwargs`` passed
through ``__init__`` into the configuration file. This allows you to pass additional parameters to the frontend
that are not part of the initial conditions object, such as compile-time flags or other settings that are
specific to your simulation code. Equivalently, the user can just modify the configuration directly after the frontend
is initialized, and the changes will be automatically saved to the configuration file.

Conventions
```````````

To keep frontends consistent, portable, and easy to maintain across Pisces, follow these conventions:

- **Configuration file**

  - Keep the default YAML **minimal but runnable**. Comment each key with what it does and (if relevant) the
    simulator’s equivalent setting.
  - Prefer **units with unyt** (``!unyt_unit`` and quantities) rather than
    raw scalars; convert explicitly at write time.
  - Treat the per-IC YAML as the **source of truth**; do not rely on global
    templates after initialization.

- **Field and particle mapping**

  - Provide a clear **models to simulator types/fields** mapping in the YAML. For example,
    a per-model section that maps Pisces field names (e.g., ``particle_position``) to simulator
    expectations (e.g., ``Coordinates``).
  - Make mappings **explicit and opt-in**; do not guess silently. If a required mapping
    is missing, **fail fast** with a helpful error and log which key is missing.
  - Keep default mappings aligned with **Pisces defaults** and common
    simulator conventions.

- **Initialization & validation**

  - Do not modify the base ``__init__`` flow of :class:`~core.frontends.SimulationFrontend`. Use hooks:

    - ``_validate_input_ic`` to reject incompatible ICs early,
    - ``_setup_config`` to populate per-model sections and derive defaults,
    - ``_validate_runtime_configuration`` to perform final checks before generation.

  - Validation should be **deterministic** and **loud**: raise
    exceptions on missing/invalid config; log any normalization you perform.

.. note::

   Conventions exist to keep frontends interchangeable and maintainable.
   If you must diverge, document the reason and preserve the **behavioral contracts**
   of the base class: deterministic validation, reproducible config,
   and predictable file layout under ``ic.directory``.


Validation
^^^^^^^^^^

During the life-cycle of a frontend, there are two key validation steps:

- ``_validate_input_ic``: This method is called when the frontend is initialized with an
  :class:`~core.initial_conditions.InitialConditions` object. It should validate that the initial conditions are
  compatible with the simulation code and raise an exception if they are not.
- ``_validate_runtime_configuration``: This method is called after the configuration file has been loaded and
  before the initial conditions are generated. It should validate that the configuration file is complete and
  consistent, checking for required keys, unit consistency, and any other constraints that are specific to your
  simulation code. If anything is missing or invalid, it should raise an exception with a helpful message.

  A common and useful pattern here is to require that the user has generated things like particle files for each
  model in the initial conditions. If any of these are missing, you should raise an exception with a clear message
  indicating which model is missing the required files. This allows users to quickly identify and fix issues with
  their initial conditions before they attempt to generate the simulation inputs. It also saves having too much
  logic in the generation step that could fail later on, which would be more difficult to debug.


IC Generation
^^^^^^^^^^^^^

Frontends generate simulator-native inputs through
:meth:`~core.frontends.SimulationFrontend.generate_initial_conditions`.
The flow has **two parts**:

1. :meth:`~core.frontends.SimulationFrontend._validate_runtime_configuration`
   Last-mile checks/normalization of the YAML before any files are written.
   Verify required keys, unit consistency, ranges, and cross-field constraints.
   If anything is missing or invalid, **raise** with a helpful message
   (and log any normalization you perform).

2. :meth:`~core.frontends.SimulationFrontend._generate_initial_conditions`
   This is the meat of the generation step. It will utilize the linked initial conditions
   object (:attr:`~core.frontends.SimulationFrontend.initial_conditions`) and the configuration
   (:attr:`~core.frontends.SimulationFrontend.config`) to generate the necessary input files for the simulation code.

   This method is where the vast majority of the frontend logic lives.

Example
```````

As an extremely simplistic case, we can implement a frontend for a hypothetical simulation code "MySim" that requires
specific initial conditions and configuration settings. The following example illustrates how to implement
a frontend for "MySim" that validates the initial conditions, sets up the configuration, and generates the necessary
input files. This example assumes that "MySim" requires 3D initial conditions with specific metadata for each model.

.. code-block:: python

   from pisces.extensions.simulation.core.frontend import (
       SimulationFrontend, __frontend_bin_path__
   )

   class MySimFrontend(SimulationFrontend):
       __default_configuration_path__ = __frontend_bin_path__ / "mysim_config.yaml"

       def _validate_input_ic(self, ic) -> bool:
           if ic.ndim != 3 or len(ic) == 0:
               raise ValueError("MySim requires 3D ICs with at least one model.")
           # Example: ensure each model has total_mass metadata
           for name in ic.list_models():
               if "total_mass" not in ic.get_model_metadata(name):
                   raise ValueError(f"Model {name!r} missing 'total_mass' in metadata.")
           return True

       def _validate_runtime_configuration(self, *args, **kwargs):
           cfg = self.config
           # Enforce required runtime keys; set safe defaults if absent
           if "io.output_dir" not in cfg:
               cfg["io.output_dir"] = "mysim_out"
           nthreads = int(cfg.get("runtime.nthreads", 4))
           if nthreads < 1:
               raise ValueError("runtime.nthreads must be >= 1")
           cfg["runtime.nthreads"] = nthreads

       def _generate_initial_conditions(self, *args, **kwargs):
           ic = self.initial_conditions
           outdir = ic.directory / self.config["io.output_dir"]
           outdir.mkdir(exist_ok=True)

           # (1) Optional: transform frame when requested
           if self.config.get("runtime.shift_to_com", False):
               ic.shift_to_COM_frame()

           # (2) Write required inputs (example writer)
           import numpy as np
           rows = []
           for name in ic.list_models():
               r = ic.model_positions[name].to_value("pc")
               v = ic.model_velocities[name].to_value("km/s")
               rows.append([name, *r, *v])
           np.savetxt(outdir / "bodies.txt", rows, fmt="%s %.6e %.6e %.6e %.6e %.6e %.6e")


Putting it together: initialize and generate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A minimal end-to-end example that creates the frontend, adjusts config, validates, and generates inputs:

.. code-block:: python

   from pisces.extensions.simulation.core.initial_conditions import InitialConditions
   from pisces.extensions.simulation.mysim.frontend import MySimFrontend

   # 1) Load an existing IC directory (contains IC_CONFIG.yaml)
   ic = InitialConditions("/path/to/MyIC")

   # 2) Attach your frontend; pass overrides as kwargs or edit later
   frontend = MySimFrontend(ic, reset_configuration=False, seed=1234)

   # You can also tweak config via key access (dotted keys supported)
   frontend["io.output_dir"] = "mysim_out"
   frontend["runtime.nthreads"] = 8
   frontend["runtime.shift_to_com"] = True

   # 3) Generate simulator-native inputs
   frontend.generate_initial_conditions()
