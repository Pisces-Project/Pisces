.. _simulations_gadget2:

==========================
Simulations with Gadget-2
==========================

.. currentmodule:: pisces.extensions.simulation

`Gadget-2 <https://wwwmpa.mpa-garching.mpg.de/gadget/>`__ is a widely used code for cosmological N-body and
smoothed particle hydrodynamics (SPH) simulations. It is also one of the simpler codes to configure and run
using Pisces-generated initial conditions.

The Gadget-2 Frontend
----------------------

As with all simulation codes supported by Pisces, Gadget-2 is accessed through a dedicated *frontend*.
The Gadget-2 frontend is implemented in :mod:`frontends.gadget_2`, and the main
entry point is the :class:`~frontends.gadget_2.Gadget2Frontend` class.

To create a frontend, initialize it on an existing initial conditions object
(:class:`~core.initial_conditions.InitialConditions`):

.. code-block:: python

    from pisces.extensions.simulation.frontends import Gadget2Frontend
    from pisces.extensions.simulation import InitialConditions

    # Load the initial conditions object from disk.
    ic = InitialConditions.from_file("path/to/initial/conditions/file")

    # Initialize the Gadget-2 frontend.
    frontend = Gadget2Frontend(ic)

This will generate a configuration file, ``gadget2_config.yaml``, in the same directory as the
initial conditions. The configuration file can be modified either by hand or through the frontend
API (see :class:`core.frontends.SimulationFrontend` for details).

Frontend Configuration Options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The frontend's configuration options largely mirror the options available in Gadget-2's
``parameterfile``. The configuration file is divided into sections, each corresponding to a
section of the Gadget-2 parameter file. Each option is documented in the configuration file itself,
and the Gadget-2 user guide should be consulted for further details on each option.

Makefile Settings
`````````````````

These settings concern the compile-time options from Gadget-2 and should be set to match those
set when compiling Gadget-2 itself. They are stored in the ``makefile`` section of the
frontend configuration file.

.. list-table:: Makefile Settings
    :widths: 15 15 70
    :header-rows: 1

    * - Makefile Option
      - Frontend Option
      - Notes
    * - ``LONGIDS``
      - ``makefile.longids``
      - Set to ``1`` to use 64-bit integers for particle IDs. Ensure that
        these settings are consistent between both Gadget-2 and Pisces.

In addition to the settings above, the following makefile options are **required** for Pisces compatibility:

.. list-table:: Makefile Settings
   :widths: 15 15 70
   :header-rows: 1

   * - Makefile Option
     - Frontend Option
     - Notes
   * - ``DOUBLEPRECISION``
     - N/A
     - Must be set to ``1``. Pisces initial conditions use double precision, and mixing precisions will cause errors.
   * - ``HAVE_HDF5``
     - N/A
     - Must be set to ``1``. Enables HDF5 I/O. Pisces does not support the legacy binary format.
   * - ``TWO_DIMS``
     - N/A
     - Must be set to ``0``. Gadget-2 requires gravity to be disabled in 2D, which is not supported for
       astrophysical runs.

Parameter File Settings
```````````````````````

Like the makefile settings, several parameter file settings from Gadget-2 are needed for Pisces to successfully
generate compatible initial conditions. For the most part, these settings simply need to be consistent between
your Pisces frontend configuration and your Gadget-2 parameter file. However, a few settings are required to
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
        Gadget-2 parameter file. Any velocities in the initial conditions will be converted to this unit.
    * - ``UnitLength_in_cm``
      - ``parameters.units.length``
      - Provide the Pisces configuration with an unyt unit equivalent to that used in your
        Gadget-2 parameter file. Any lengths in the initial conditions will be converted to this unit.
    * - ``UnitMass_in_g``
      - ``parameters.units.mass``
      - Provide the Pisces configuration with an unyt unit equivalent to that used in your
        Gadget-2 parameter file. Any masses in the initial conditions will be converted to this unit.

Fields and Particle Types
``````````````````````````

Pisces follows a standard convention for naming particle types and data fields in its particle
datasets (see :ref:`particles_overview`). In most cases, these conventions are applied automatically
when generating initial conditions.

There are situations, however, where a model may not conform to the standard naming scheme, or where
you may want to explicitly control which fields are written to the Gadget-2 initial conditions.
To support this, the frontend configuration file includes a ``models`` section. Each model defined
in the input :class:`~core.initial_conditions.InitialConditions` object will have a corresponding
entry here.

A typical model section looks like this:

.. code-block:: yaml

    [MODEL_NAME]:
        PART_TYPE_0:
            name: [name in model particle file]
            fields:
                GADGET_FIELD_1: [name in model particle file]
                GADGET_FIELD_2: [name in model particle file]

In this mapping:

- ``PART_TYPE_0`` refers to the Gadget-2 particle type (see the Gadget-2 user guide for available types).
- ``name`` specifies the name of the model’s particle group to be associated with this particle type.
- ``fields`` maps Gadget-2 field names to the corresponding field names in the model’s particle file.

By default, Pisces generates these mappings according to its internal conventions. You only need to
modify them if your model uses non-standard names or if you wish to customize the exported fields.
