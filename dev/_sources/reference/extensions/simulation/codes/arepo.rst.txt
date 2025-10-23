.. _simulations_arepo:
======================
Simulations with AREPO
======================

.. currentmodule:: pisces.extensions.simulation.arepo

`AREPO <https://arepo-code.org>`__ is a moving-mesh hydrodynamics code designed for cosmological and astrophysical
simulations. It combines the adaptive resolution of smoothed-particle hydrodynamics (SPH) with the
accuracy of grid-based methods, providing excellent performance for both cosmological and
galaxy-scale problems.

Pisces includes native support for generating AREPO-compatible initial conditions through the
:mod:`~pisces.extensions.simulation.arepo` extension module. The resulting output is an HDF5 file
formatted according to AREPO’s IC conventions and represented within Pisces by the
:class:`~pisces.particles.gadget.AREPOParticleDataset`.

Detailed installation instructions for AREPO are provided in the official
`AREPO User Guide <https://arepo-code.org/documentation>`__.
For Pisces compatibility, several compile-time and runtime options must be adjusted in your
AREPO build configuration. See :ref:`arepo_makefile_settings` for a summary.

Overview of Pisces Support
--------------------------

The :mod:`~pisces.extensions.simulation.arepo` module provides all necessary tools for
creating, managing, and validating AREPO initial conditions. The main entry point is the
:class:`~pisces.extensions.simulation.arepo.frontends.AREPOFrontend`, which wraps an existing
Pisces initial condition file and produces AREPO-readable particle datasets.

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

   4. **Initialize the frontend** — Wrap the resulting initial conditions with
      :class:`~pisces.extensions.simulation.arepo.frontends.AREPOFrontend`.

   5. **Write AREPO ICs** — Call
      :meth:`~pisces.extensions.simulation.arepo.frontends.AREPOFrontend.generate_initial_conditions`
      to produce HDF5 files compatible with AREPO’s input routines.

.. admonition:: ⚠️ Compatibility Reminder
   :class: warning

   AREPO’s compile-time and runtime flags **must exactly match** those used by Pisces when generating initial conditions.

   - If AREPO is compiled with 32-bit particle IDs, Pisces must also use 32-bit IDs (`long_ids=False`).
   - If AREPO is compiled with MHD or cooling enabled, the corresponding fields (``MagneticField``, ``ElectronAbundance``)
     must be present in the ICs.
   - Dimensionality (1D, 2D, 3D) must match exactly between Pisces and AREPO.

--------------------------------
The AREPO Frontend
--------------------------------

Like all simulation codes supported by Pisces, AREPO is accessed through a *frontend*.
The :class:`~pisces.extensions.simulation.arepo.frontends.AREPOFrontend` handles creation,
validation, and configuration of AREPO-compatible initial condition files.

To create a frontend:

.. code-block:: python

    from pisces.extensions.simulation.arepo.frontends import AREPOFrontend
    from pisces.extensions.simulation import InitialConditions

    ic = InitialConditions("path/to/initial_conditions_file")
    frontend = AREPOFrontend(ic)

This command generates a configuration file named ``AREPOFrontend_config.yaml`` in the same
directory as your initial conditions. You can edit it manually or programmatically via
the frontend’s configuration API.

To inspect or modify settings in Python:

.. code-block:: python

    >>> print(dict(frontend.config["makefile"]))
    {'simulation_type': '3D', 'number_of_particle_types': 6, 'nbits_id': 64, 'double_precision': True}

To locate the YAML configuration file on disk:

.. code-block:: python

    >>> print(frontend.config_path)
    PosixPath(".../path/to/AREPOFrontend_config.yaml")

.. _arepo_makefile_settings:
Compile-Time Settings
^^^^^^^^^^^^^^^^^^^^^

Certain AREPO Makefile options determine the binary layout and precision of the simulation files.
These are mirrored in the frontend configuration under the ``makefile.*`` section.

.. dropdown:: Relevant Compile-Time Settings

    .. list-table::
       :widths: 25 15 60
       :header-rows: 1

       * - **Pisces Setting**
         - **Makefile Flag**
         - **Description**
       * - ``makefile.simulation_type``
         - ``TWODIM`` / ``ONEDIM`` / ``ONEDIMS_SPHERICAL``
         - Sets the simulation dimensionality. Must match the IC file.
       * - ``makefile.number_of_particle_types``
         - ``NTYPES`` / ``NTYPES_ICS``
         - Number of particle types. Must be ≥ number in ICs.
       * - ``makefile.nbits_id``
         - ``IDS_32BIT`` / ``IDS_64BIT``
         - Particle ID bit depth. Must match the IC file.
       * - ``makefile.double_precision``
         - ``DOUBLEPRECISION`` / ``INPUT_IN_DOUBLEPRECISION`` / ``READ_COORDINATES_IN_DOUBLE``
         - Internal runtime precision for floating-point arithmetic. This setting determines the
           precision of the initial conditions. We do not support mismatched precision between coordinates
           and other fields.
       * - ``physics.cooling``
         - ``COOLING``
         - Enables radiative cooling. Requires ``ElectronAbundance`` field in ICs.
       * - ``physics.mhd``
         - ``MHD``
         - Enables magneto-hydrodynamics. Requires ``MagneticField`` field in ICs.
       * - ``physics.passive_scalars``
         - ``PASSIVE_SCALARS=<N>``
         - Number of advected passive scalars to track.


Runtime Parameters
^^^^^^^^^^^^^^^^^^

AREPO’s runtime parameters specify the physical and cosmological context of the simulation.
These correspond to the ``parameters.*`` section of the frontend configuration.

.. dropdown:: Relevant Runtime Settings

    .. list-table::
       :widths: 25 15 60
       :header-rows: 1

       * - **Pisces Setting**
         - **AREPO Parameter**
         - **Description**
       * - ``parameters.boxsize``
         - ``BoxSize``
         - Simulation box size in code units.
       * - ``parameters.units.length``, ``mass``, ``velocity``
         - ``UnitLength_in_cm``, ``UnitMass_in_g``, ``UnitVelocity_in_cm_per_s``
         - Physical unit definitions (time is derived from L/V).
       * - ``parameters.cooling`` / ``parameters.mhd``
         - ``CoolingOn`` / ``MHDOn``
         - Enable or disable cooling and MHD at runtime.
       * - ``parameters.ic_format``
         - ``ICFormat``
         - Must be set to ``3`` for HDF5 ICs.

Field Configuration
^^^^^^^^^^^^^^^^^^^

Finally, the frontend defines how Pisces particle fields are mapped to AREPO fields.
This mapping is provided in the ``model_template`` section of the configuration file.

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
          ParticleType4:
            name: stars
            fields:
              Coordinates: particle_position
              Velocities: particle_velocity
              ParticleIDs: particle_id
              Masses: particle_mass

Optional fields are added automatically based on enabled physics:
- ``MagneticField`` (for MHD)
- ``ElectronAbundance`` (for cooling)
- ``PassiveScalars`` (if ``passive_scalars > 0``)
