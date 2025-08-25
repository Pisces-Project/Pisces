.. _simulations_gadget:

==========================
Simulations with Gadget
==========================

.. currentmodule:: pisces.extensions.simulation.gadget

`Gadget-4 <https://wwwmpa.mpa-garching.mpg.de/gadget/>`__ is a state-of-the-art code for cosmological
N-body and smoothed particle hydrodynamics (SPH) simulations. It remains one of the most widely used
and actively developed tools in the community, and is also among the more straightforward codes to
set up and run with Pisces-generated initial conditions.

Installing Gadget-4
-------------------

Detailed installation instructions are provided in the official
`Gadget-4 User Guide <https://wwwmpa.mpa-garching.mpg.de/gadget/users-guide.pdf>`__.
For Pisces compatibility, you will also need to adjust certain compile-time options in the
Makefile. See :ref:`gadget_makefile_settings` for a summary of the required changes.

Gadget Support in Pisces
------------------------

In order to support Gadget-4 initial conditions, Pisces provides the :mod:`~pisces.extensions.simulation.gadget`
extension module which contains two useful classes:

- :class:`~pisces.extensions.simulation.gadget.frontends.Gadget4Frontend`: A frontend for generating
  Gadget-4 initial conditions from a :class:`~pisces.extensions.simulation.core.initial_conditions.InitialConditions`
  object.
- :class:`~pisces.extensions.simulation.gadget.particles.GadgetParticleDataset`: A particle dataset class for reading
  and writing Gadget-4 HDF5 files.

The general process for generating initial conditions in Pisces and transferring them to Gadget-4 is as follows:

1. **Build the constituent models**: Create whatever models you are trying to simulate using
   the Pisces ecosystem. If you are unsure of how to build the models you need, see the
   modeling documentation: :ref:`models_overview`.
2. **Build the ICs** With your models in hand, create an initial conditions object which combines the
   relevant models into a single initial conditions dataset. See :ref:`initial_conditions_overview` for details.
3. **Generate particles for each model**: Gadget-4 is a particle-based code, so each model in your
   initial conditions must be converted to particles. Because this process can be somewhat dependent on
   exactly what you, the user, want to achieve, Pisces does not do this step automatically. For each model
   you'll need to use the
   :meth:`~pisces.extensions.simulation.core.initial_conditions.InitialConditions.generate_particles` method
   to generate particles. See :ref:`particles_overview` for details.
4. **Initialize the Gadget-4 frontend**: With your initial conditions object fully specified, you can
   create a :class:`~pisces.extensions.simulation.gadget.frontends.Gadget4Frontend` object. This will
   generate a configuration file which you can modify to suit your needs. This document will provide all of
   the details regarding the various configuration options. In this step, you'll provide some information about
   your Gadget-4 installation, and you'll also specify how the models in your initial conditions should be
   mapped to Gadget-4 particle types and fields.
5. **Write the initial conditions**: Finally, you can call the :meth:`~frontends.Gadget4Frontend.generate_initial_conditions`
   method to write the initial conditions to disk in a format that Gadget-4 can read.

.. warning::

    As is the case with all hydrodynamical simulation codes, Gadget-4 has a rather complicated set of
    configuration options. Pisces attempts to make the process of generating compatible initial conditions
    as straightforward as possible, but it is ultimately the user's responsibility to ensure that the
    configuration options in the frontend match those used when compiling and running Gadget-4 itself.

    For example, if you compile Gadget-4 to perform 2D simulations but then generate 3D initial conditions,
    the simulation will likely fail to run. Similarly, if you compile Gadget-4 with single-precision
    particle IDs but then generate initial conditions with 64-bit IDs, the simulation will likely
    fail. Always double-check that the configuration options in your frontend match those used
    when compiling and running Gadget-4.

The Gadget-4 Frontend
----------------------

As with all simulation codes supported by Pisces, Gadget-4 is accessed through a dedicated *frontend*.
The Gadget-4 frontend is implemented in :mod:`~pisces.extensions.simulation.gadget.frontends`, and the main
entry point is the :class:`~frontends.Gadget4Frontend` class.

To create a frontend, initialize it on an existing initial conditions object
(:class:`~pisces.extensions.simulation.core.initial_conditions.InitialConditions`):

.. code-block:: python

    from pisces.extensions.simulation.frontends import Gadget2Frontend
    from pisces.extensions.simulation import InitialConditions

    # Load the initial conditions object from disk.
    ic = InitialConditions("path/to/initial/conditions/file")

    # Initialize the Gadget-4 frontend.
    frontend = Gadget2Frontend(ic)

This will generate a configuration file, ``Gadget4Frontend_config.yaml``, in the same directory as the
initial conditions. The configuration file can be modified either by hand or through the frontend
API (see :class:`~pisces.extensions.simulation.core.frontends.SimulationFrontend` for details).

Frontend Configuration Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The frontend's configuration options largely mirror the options available in Gadget-4's
``parameterfile``. The configuration file is divided into sections, each corresponding to a
section of the Gadget-4 parameter file. Each option is documented in the configuration file itself,
and the Gadget-4 user guide should be consulted for further details on each option.

.. _gadget_makefile_settings:
Makefile Settings
`````````````````

These settings concern the compile-time options from Gadget-4 and should be set to match those
set when compiling Gadget-4 itself. They are stored in the ``makefile`` section of the
frontend configuration file.

.. list-table:: Makefile Settings
    :widths: 15 15 70
    :header-rows: 1

    * - Makefile Option
      - Frontend Option
      - Notes
    * - ``IDS_NBIT``
      - ``makefile.nbits_id``
      - Determines the number of bits used for particle IDs. Common values are ``32`` and ``64``.
        The value here must match that used when compiling Gadget-4 itself. It is used to ensure
        that the particle IDs in the initial conditions are compatible with the simulation.
    * - ``NTYPES``
      - ``makefile.number_of_particle_types``
      - The number of particle types supported by the Gadget-4 installation. This value must be
        greater than or equal to the number of unique particle types used in your initial conditions.
        For example, if your initial conditions contain gas, dark matter, and star particles, you
        will need to set this value to at least ``3``. Typically, 6 is chosen.

In addition to the settings above, the following makefile options are **required** for Pisces compatibility:

.. list-table:: Makefile Settings
   :widths: 15 15 70
   :header-rows: 1

   * - Makefile Option
     - Frontend Option
     - Notes
   * - ``GADGET2_HEADER``
     - N/A
     - Must not be active.
   * - ``NGENIC``
     - N/A
     - Must not be active. This option is for the N-GenIC initial conditions generator, which
       is not compatible with Pisces.

Parameter File Settings
```````````````````````

Like the makefile settings, several parameter file settings from Gadget-4 are needed for Pisces to successfully
generate compatible initial conditions. For the most part, these settings simply need to be consistent between
your Pisces frontend configuration and your Gadget-4 parameter file. However, a few settings are required to
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
        Gadget-4 parameter file. Any velocities in the initial conditions will be converted to this unit.
    * - ``UnitLength_in_cm``
      - ``parameters.units.length``
      - Provide the Pisces configuration with an unyt unit equivalent to that used in your
        Gadget-4 parameter file. Any lengths in the initial conditions will be converted to this unit.
    * - ``UnitMass_in_g``
      - ``parameters.units.mass``
      - Provide the Pisces configuration with an unyt unit equivalent to that used in your
        Gadget-4 parameter file. Any masses in the initial conditions will be converted to this unit.

Fields and Particle Types
``````````````````````````

Pisces follows a standard convention for naming particle types and data fields in its particle
datasets (see :ref:`particles_overview`). In most cases, these conventions are applied automatically
when generating initial conditions.

There are situations, however, where a model may not conform to the standard naming scheme, or where
you may want to explicitly control which fields are written to the Gadget-4 initial conditions.
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

- ``PART_TYPE_0`` refers to the Gadget-4 particle type (see the Gadget-4 user guide for available types).
- ``name`` specifies the name of the model’s particle group to be associated with this particle type.
- ``fields`` maps Gadget-4 field names to the corresponding field names in the model’s particle file.

By default, Pisces generates these mappings according to its internal conventions. You only need to
modify them if your model uses non-standard names or if you wish to customize the exported fields. A
common scenario in which you might want to modify the field mappings is when your model includes
star particles with additional attributes (e.g., metallicity, age) that you want to include in the
Gadget-4 initial conditions. You may also have new particle types which need to be added to the configuration.
