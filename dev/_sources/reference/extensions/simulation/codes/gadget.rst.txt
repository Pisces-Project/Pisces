.. _simulations_gadget:

==========================
Simulations with Gadget-4
==========================

.. currentmodule:: pisces.extensions.simulation.gadget

`Gadget-4 <https://wwwmpa.mpa-garching.mpg.de/gadget/>`__ is a state-of-the-art code for cosmological
N-body and smoothed particle hydrodynamics (SPH) simulations. It remains one of the most widely used
and actively developed community codes and is among the easiest to configure and run with
Pisces-generated initial conditions.

Detailed installation instructions are available in the official
`Gadget-4 User Guide <https://wwwmpa.mpa-garching.mpg.de/gadget/users-guide.pdf>`__.
For Pisces compatibility, several compile-time options must be adjusted in the Gadget-4 Makefile.
See :ref:`gadget_makefile_settings` for a summary of the required changes.

Overview of Pisces Support
--------------------------

Pisces includes native support for running simulations with **Gadget-4** through the
:mod:`~pisces.extensions.simulation.gadget` extension module. This module provides the
frontend for connection to gadget: :class:`~pisces.extensions.simulation.gadget.frontends.Gadget4Frontend`. The resulting
output is a Gadget-4 compatible HDF5 file, which is represented in Pisces using the
:class:`~pisces.particles.gadget.Gadget4ParticleDataset`. Here's how a typical workflow might progress:

.. card:: 🧭 Workflow Steps
   :class-card: shadow-sm p-2

   1. **Build models** — Using Pisces, generate whatever models you'd like to use in the simulation. This
      might be galaxy mergers, cosmological models, etc. See :ref:`models_overview` for an overview of Pisces model
      generation. These are later composed into a set of initial conditions for the simulation.

   2. **Assemble initial conditions** — Once the models are generated, they can be combined and oriented in space
      to produce a set of initial conditions
      (:class:`~pisces.extensions.simulation.core.initial_conditions.InitialConditions`). For a detailed description
      of the different types of initial conditions available and how they work, see the documentation on
      initial conditions: :ref:`initial_conditions_overview`.

   3. **Generate particles** — Once you have constructed your initial conditions, it will be necessary to
      to convert the constituent models into particle datasets. This can be done through the initial conditions
      class you generated in the previous step; however, **not all models can generate particles**. See the examples
      in :ref:`examples` for examples.

   4. **Initialize the frontend** — Create a
      :class:`~pisces.extensions.simulation.gadget.frontends.Gadget4Frontend`
      instance around your initial conditions to set the stage for conversion to Gadget-4 format.

   5. **Write Gadget-4 ICs** — Use
      :meth:`~pisces.extensions.simulation.gadget.frontends.Gadget4Frontend.generate_initial_conditions`
      to output HDF5 initial condition files readable by Gadget-4.


.. admonition:: ⚠️ Compatibility Reminder
   :class: warning

   Gadget-4’s compile-time flags and runtime parameters **must match**
   those used by Pisces when generating initial conditions.

   - A 2D Gadget-4 build cannot run 3D Pisces ICs.
   - Single-precision (32-bit) particle IDs in Gadget-4 are **incompatible**
     with 64-bit IDs written by Pisces.

   Always verify that your **Makefile settings** and **frontend configuration**
   are consistent before running the simulation.

------------------------
The Gadget-4 Frontend
------------------------

All simulation codes supported by Pisces are accessed through a *frontend*.
For Gadget-4, this is implemented in
:mod:`~pisces.extensions.simulation.gadget.frontends`, primarily via the
:class:`~pisces.extensions.simulation.gadget.frontends.Gadget4Frontend` class.

To create a frontend:

.. code-block:: python

    from pisces.extensions.simulation.gadget.frontends import Gadget4Frontend
    from pisces.extensions.simulation import InitialConditions

    ic = InitialConditions("path/to/initial_conditions_file")
    frontend = Gadget4Frontend(ic)

This creates a configuration file named ``Gadget4Frontend_config.yaml`` in the same directory
as the initial conditions. The file can be edited manually or through the
frontend API (see :class:`~pisces.extensions.simulation.core.frontends.SimulationFrontend`).

The setup process consists of three main steps:

1. **Adjust compile-time settings**
2. **Specify runtime parameters**
3. **Map model fields to Gadget particle fields**

Each is described below.

Setting Up the Frontend
^^^^^^^^^^^^^^^^^^^^^^^^

Once initialized, the frontend automatically generates a default configuration describing
your Gadget-4 build and IC layout. You **must** customize this file to match your specific
Gadget-4 compilation and runtime settings.

To inspect or modify the configuration in Python:

.. code-block:: python

    >>> print(dict(frontend.config["makefile"]))
    {'simulation_type': '3D', 'number_of_particle_types': 6, 'nbits_id': 64}

To locate the YAML configuration file on disk:

.. code-block:: python

    >>> print(frontend.config_path)
    PosixPath(".../path/to/Gadget4Frontend_config.yaml")

.. _gadget_makefile_settings:
Compile-Time Settings
^^^^^^^^^^^^^^^^^^^^^

Certain Gadget-4 Makefile options determine the binary format of the simulation files
and must align with Pisces’ output. These are mirrored in the frontend configuration
under ``makefile.*``.

.. dropdown:: Relevant Compile-Time Settings

    .. list-table::
       :widths: 25 15 60
       :header-rows: 1

       * - **Pisces Setting**
         - **Makefile Flag**
         - **Description**
       * - ``makefile.simulation_type``
         - ``TWODIM`` / ``ONEDIM``
         - Sets the dimensionality (1D, 2D, or 3D). Must match the IC dimensionality.
       * - ``makefile.number_of_particle_types``
         - ``NTYPES``
         - Number of particle types (gas, DM, stars, etc.). Must be ≥ the number in the ICs.
       * - ``makefile.nbits_id``
         - ``IDS_32BIT``, ``IDS_48BIT``, ``IDS_64BIT``
         - Bit depth for particle IDs. Must match between Pisces and Gadget-4.
       * - ``physics.cooling``
         - ``COOLING``
         - Enables Gadget cooling. Requires an ``ElectronAbundance`` field at runtime.
       * -
         - ``INITIAL_CONDITIONS_CONTAIN_ENTROPY``
         - **Must be disabled** for Pisces ICs.
       * -
         - ``GAMMA`` / ``ISOTHERM_EQS``
         - Ensure consistency with the thermodynamic assumptions of your model.

.. warning::

   If you compile Gadget-4 with incompatible flags, your simulation may fail to initialize.
   Double-check that these flags match both the frontend configuration and the physical model.

Example — Adjusting Particle Types
""""""""""""""""""""""""""""""""""

.. code-block:: python

    from pisces.extensions.simulation.gadget.frontends import Gadget4Frontend
    from pisces.extensions.simulation import InitialConditions

    ic = InitialConditions("path/to/ic_file")
    frontend = Gadget4Frontend(ic)

    # Increase the number of particle types to 7
    frontend.config["makefile.number_of_particle_types"] = 7

Runtime Settings
^^^^^^^^^^^^^^^^^

At runtime, Gadget-4 requires a parameter file specifying the simulation domain, cosmology,
and physical unit system. These must also align with the ICs.

.. dropdown:: Relevant Runtime Settings

    .. list-table::
       :widths: 25 15 60
       :header-rows: 1

       * - **Pisces Setting**
         - **Gadget Parameter**
         - **Description**
       * - ``parameters.boxsize``
         - ``BoxSize``
         - Box size in code units. All ICs are clipped to this domain.
       * -
         - ``Omega0``, ``OmegaLambda``, ``OmegaBaryon``, ``HubbleParam``
         - Cosmological parameters. Should match your physical model.
       * - ``parameters.units.length``, ``mass``, ``velocity``
         - ``UnitLength_in_cm``, ``UnitMass_in_g``, ``UnitVelocity_in_cm_per_s``
         - Defines the simulation unit system. Must match the IC units.
       * -
         - ``ICFormat``
         - Must be set to ``3`` for HDF5 compatibility.

Field Configuration
^^^^^^^^^^^^^^^^^^^

Finally, the frontend must know how to map Pisces particle fields to Gadget-4 field names.
The configuration includes a ``model_template`` section, which is duplicated for each model
in the ICs. A typical example:

.. dropdown:: Example ``model_template``

    .. code-block:: yaml

        model_template:
          ParticleType0:
            name: gas
            fields:
              Coordinates: particle_position
              Velocities: particle_velocity
              ParticleIDs: particle_id
              Masses: particle_mass
              InternalEnergy: particle_internal_energy
          ParticleType1:
            name: dark_matter
            fields:
              Coordinates: particle_position
              Velocities: particle_velocity
              ParticleIDs: particle_id
              Masses: particle_mass
          ParticleType2:
            name: disk
            fields:
              Coordinates: particle_position
              Velocities: particle_velocity
              ParticleIDs: particle_id
              Masses: particle_mass
          ParticleType3:
            name: bulge
            fields:
              Coordinates: particle_position
              Velocities: particle_velocity
              ParticleIDs: particle_id
              Masses: particle_mass
          ParticleType4:
            name: stars
            fields:
              Coordinates: particle_position
              Velocities: particle_velocity
              ParticleIDs: particle_id
              Masses: particle_mass
          ParticleType5:
            name: boundary
            fields:
              Coordinates: particle_position
              Velocities: particle_velocity
              ParticleIDs: particle_id
              Masses: particle_mass

If your particle datasets use different field names (e.g., ``pos`` instead of ``particle_position``),
update the configuration accordingly so Gadget-4 can interpret the files correctly.
