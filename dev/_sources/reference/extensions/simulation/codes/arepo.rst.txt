.. _simulations_arepo:

==========================
Simulations with AREPO
==========================

.. currentmodule:: pisces.extensions.simulation.arepo

The AREPO code :footcite:p:`springel2010` is a massively parallel code for cosmological simulations
using a moving, unstructured mesh based on a Voronoi tessellation of the simulation domain. AREPO is capable of
solving the equations of hydrodynamics using a finite-volume approach, and it includes a variety of physical
processes such as gravity, cooling, star formation, and feedback.

.. note::

    At this time, the AREPO code is only partially supported in Pisces due to the
    considerable number of different configurations and compilation options available. We are
    continuing to work on expanding support for AREPO, and we welcome contributions from the community.

Installing AREPO
-------------------

Detailed instructions for obtaining and compiling the AREPO code can be found on the
`AREPO website <https://arepo-code.org/>`_.
For Pisces compatibility, you will also need to adjust certain compile-time options in the
Makefile. See :ref:`arepo_makefile_settings` for a summary of the required changes.

AREPO Support in Pisces
------------------------

In order to support AREPO initial conditions, Pisces provides the
:mod:`~pisces.extensions.simulation.arepo`
extension module which contains two useful classes:

- :class:`~pisces.extensions.simulation.arepo.frontends.AREPOFrontend3D`: A frontend for generating
  AREPO initial conditions from a :class:`~pisces.extensions.simulation.core.initial_conditions.InitialConditions`
  object.
- :class:`~pisces.extensions.simulation.AREPO.particles.AREPOParticleDataset`: A particle dataset class for reading
  and writing AREPO HDF5 files.

.. important::

    In future releases, we plan to add support for 1D, 2D, and spherical AREPO simulations vis-a-vis
    specialized frontend classes. At this time, however, only 3D simulations are supported.

The general process for generating initial conditions in Pisces and transferring
them to AREPO is as follows:

1. **Build the constituent models**: Create whatever models you are trying to simulate using
   the Pisces ecosystem. If you are unsure of how to build the models you need, see the
   modeling documentation: :ref:`models_overview`.
2. **Build the ICs** With your models in hand, create an initial conditions object which combines the
   relevant models into a single initial conditions dataset. See :ref:`initial_conditions_overview` for details.
3. **Generate particles for each model**: AREPO reads Gadget-like ICs, so each model in your
   initial conditions must be converted to particles. Because this process can be somewhat dependent on
   exactly what you, the user, want to achieve, Pisces does not do this step automatically. For each model
   you'll need to use the
   :meth:`~pisces.extensions.simulation.core.initial_conditions.InitialConditions.generate_particles` method
   to generate particles. See :ref:`particles_overview` for details.
4. **Initialize the AREPO frontend**: With your initial conditions object fully specified, you can
   create a :class:`~pisces.extensions.simulation.arepo.frontends.ArepoFrontend3D` object. This will
   generate a configuration file which you can modify to suit your needs. This document will provide all of
   the details regarding the various configuration options. In this step, you'll provide some information about
   your AREPO installation, and you'll also specify how the models in your initial conditions should be
   mapped to AREPO particle types and fields.
5. **Write the initial conditions**: Finally, you can call
   the :meth:`~frontends.AREPOFrontend3D.generate_initial_conditions`
   method to write the initial conditions to disk in a format that AREPO can read.

.. warning::

    As is the case with all hydrodynamical simulation codes, AREPO has a rather complicated set of
    configuration options. Pisces attempts to make the process of generating compatible initial conditions
    as straightforward as possible, but it is ultimately the user's responsibility to ensure that the
    configuration options in the frontend match those used when compiling and running AREPO itself.

    For example, if you compile AREPO to perform 2D simulations but then generate 3D initial conditions,
    the simulation will likely fail to run. Similarly, if you compile AREPO with single-precision
    particle IDs but then generate initial conditions with 64-bit IDs, the simulation will likely
    fail. Always double-check that the configuration options in your frontend match those used
    when compiling and running AREPO.


The AREPO Frontend
----------------------

As with all simulation codes supported by Pisces, AREPO is accessed through a dedicated *frontend*.
The AREPO frontend is implemented in :mod:`~pisces.extensions.simulation.arepo.frontends`, and the main
entry point is the :class:`~frontends.AREPOFrontend` class.

To create a frontend, initialize it on an existing initial conditions object
(:class:`~pisces.extensions.simulation.core.initial_conditions.InitialConditions`):

.. code-block:: python

    from pisces.extensions.simulation.frontends import Gadget2Frontend
    from pisces.extensions.simulation import InitialConditions

    # Load the initial conditions object from disk.
    ic = InitialConditions("path/to/initial/conditions/file")

    # Initialize the AREPO frontend.
    frontend = Gadget2Frontend(ic)

This will generate a configuration file, ``AREPOFrontend_config.yaml``, in the same directory as the
initial conditions. The configuration file can be modified either by hand or through the frontend
API (see :class:`~pisces.extensions.simulation.core.frontends.SimulationFrontend` for details).

Frontend Configuration Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The frontend's configuration options largely mirror the options available in AREPO's
``parameterfile``. The configuration file is divided into sections, each corresponding to a
section of the AREPO parameter file. Each option is documented in the configuration file itself,
and the AREPO user guide should be consulted for further details on each option.

.. _arepo_makefile_settings:
Makefile Settings
`````````````````

These settings concern the compile-time options from AREPO and should be set to match those
set when compiling AREPO itself. They are stored in the ``makefile`` section of the
frontend configuration file.

.. list-table:: Makefile Settings
    :widths: 15 15 70
    :header-rows: 1

    * - Makefile Option
      - Frontend Option
      - Notes
    * - ``NTYPES``
      - ``makefile.number_of_particle_types``
      - The number of particle types supported by the AREPO installation. This value must be
        greater than or equal to the number of unique particle types used in your initial conditions.
        For example, if your initial conditions contain gas, dark matter, and star particles, you
        will need to set this value to at least ``3``. Typically, 6 is chosen.

In addition to the settings above, the following options are required to be set to specific values to
ensure that the generated initial conditions are compatible with AREPO:

.. list-table:: Makefile Settings
   :widths: 15 70
   :header-rows: 1

   * - Makefile Option
     - Notes
   * - ``TWODIMS``
     - Must not be active to ensure that the simulation is run in 3D.
   * - ``ONEDIMS``
     - Must not be active to ensure that the simulation is run in 3D.
   * - ``ONEDIMS_SPHERICAL``
     - Must not be active to ensure that the simulation is run in 3D.
   * - ``PASSIVE_SCALARS``
     - Must not be active. Pisces does not currently support passive scalars in AREPO.
   * - ``MHD``
     - Must not be active. Pisces does not currently support MHD in AREPO.
   * - ``COOLING``
     - Must not be active. Pisces does not currently support cooling in AREPO.
   * - ``USE_SFR``
     - Must not be active. Pisces does not currently support star formation in AREPO.
   * - ``READ_COORDINATES_IN_DOUBLE``
     - This should be active to ensure that particle coordinates are read in double precision.
       This helps to avoid precision issues when dealing with large simulation boxes or high-resolution
       initial conditions.
   * - ``LONGIDS``
     - This should be active to ensure that particle IDs are stored as 64-bit integers. This is
       important for simulations with a large number of particles to avoid ID collisions.
   * - ``INPUTS_IN_DOUBLEPRECISION``
     - This should be active to ensure that input data (e.g., particle positions, velocities) are
       read in double precision. This helps to maintain accuracy in the initial conditions.
   * - ``NTYPES_ICS``
     - This should be set equal to ``NTYPES``.

Parameter File Settings
```````````````````````

Like the makefile settings, several parameter file settings from AREPO are needed for Pisces to successfully
generate compatible initial conditions. For the most part, these settings simply need to be consistent between
your Pisces frontend configuration and your AREPO parameter file. However, a few settings are required to
be set to specific values for Pisces compatibility. They are marked as so in the below table:

.. list-table:: Makefile Settings
    :widths: 15 15 70
    :header-rows: 1

    * - Option
      - Frontend Flag
      - Notes
    * - ``ICFormat``
      -
      - Must be set to ``3`` to read HDF5 initial conditions. Pisces does not support the legacy binary format.
    * - ``BoxSize``
      - ``parameters.box_size``
      - The size of the simulation box in code units. All of the models in the initial conditions
        will be placed within this box ([0, BoxSize] in each dimension).
    * - ``UnitVelocity_in_cm_per_s``
      - ``parameters.units.velocity``
      - Provide the Pisces configuration with an unyt unit equivalent to that used in your
        AREPO parameter file. Any velocities in the initial conditions will be converted to this unit.
    * - ``UnitLength_in_cm``
      - ``parameters.units.length``
      - Provide the Pisces configuration with an unyt unit equivalent to that used in your
        AREPO parameter file. Any lengths in the initial conditions will be converted to this unit.
    * - ``UnitMass_in_g``
      - ``parameters.units.mass``
      - Provide the Pisces configuration with an unyt unit equivalent to that used in your
        AREPO parameter file. Any masses in the initial conditions will be converted to this unit.

Fields and Particle Types
``````````````````````````

Pisces follows a standard convention for naming particle types and data fields in its particle
datasets (see :ref:`particles_overview`). In most cases, these conventions are applied automatically
when generating initial conditions.

There are situations, however, where a model may not conform to the standard naming scheme, or where
you may want to explicitly control which fields are written to the AREPO initial conditions.
To support this, the frontend configuration file includes a ``models`` section. Each model defined
in the input :class:`~pisces.extensions.simulation.core.initial_conditions.InitialConditions` object will have
a corresponding entry here.

A typical model section looks like this:

.. code-block:: yaml

    [MODEL_NAME]:
        PART_TYPE_0:
            name: [name in model particle file]
            fields:
                GADGET_FIELD_1: [name in model particle file]
                GADGET_FIELD_2: [name in model particle file]

In this mapping:

- ``PART_TYPE_0`` refers to the AREPO particle type (see the AREPO user guide for available types).
- ``name`` specifies the name of the model’s particle group to be associated with this particle type.
- ``fields`` maps AREPO field names to the corresponding field names in the model’s particle file.

By default, Pisces generates these mappings according to its internal conventions. You only need to
modify them if your model uses non-standard names or if you wish to customize the exported fields. A
common scenario in which you might want to modify the field mappings is when your model includes
star particles with additional attributes (e.g., metallicity, age) that you want to include in the
AREPO initial conditions. You may also have new particle types which need to be added to the configuration.
