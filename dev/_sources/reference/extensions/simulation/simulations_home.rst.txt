.. _simulations:
=========================
Simulations in Pisces
=========================

.. currentmodule:: pisces.extensions.simulation

One of the principle applications of Pisces is the generation of initial conditions for
hydrodynamic and N-body simulations. Pisces includes support for several popular simulation codes through
dedicated initial conditions configuration and specialized *frontends*.

In this document, we'll cover all the basic functionality for running simulations with Pisces-generated data.

Overview
---------

In order to integrate with external simulation codes, Pisces provides a system with two main components:

1. The **initial conditions**, which allow multiple Pisces models to be placed into a simulation domain with various
   orientations, offsets, and velocities. These initial conditions set a "foundation" for the eventual conversion
   into simulation-code-specific formats.
2. The **frontend interfaces**, which provide code-specific configuration and execution capabilities. Each
   supported simulation code has a dedicated frontend that handles the conversion of Pisces data into the
   appropriate format, generates configuration files, and runs the simulation itself.

In most scenarios, users create one or many Pisces models representing the physical systems they wish to simulate.
These :class:`~pisces.models.core.base.BaseModel` objects are then combined into a single initial conditions object
(:class:`~core.initial_conditions.InitialConditions`), which is responsible for managing the models and their
placement in the simulation domain. Users will then need to determine which simulation code they wish to use
for their simulation. For each supported code, Pisces provides a dedicated frontend class that handles the
conversion of the initial conditions into the appropriate format, generates any necessary configuration files,
additional structures, etc. Finally, the output from that conversion process is passed to the simulation code's
execution command, which runs the simulation with the Pisces-generated initial conditions.

.. note::

    In some scenarios, simulation software requires "problem initializers" which are highly integrated
    into their source code. In these cases, Pisces provides both a frontend and a **problem initializer**,
    which is hosted on github under the Pisces project, but is not a core part of the Pisces package. Each
    problem initializer is written in the language of the simulation code (e.g., Fortran, C, etc.) and is
    responsible for directly interacting with the simulation code's internal data structures to set up the
    initial conditions. These problem initializers are not part of the Pisces package itself, but they are
    developed and maintained by the Pisces team and are available as part of the larger Pisces project.


Initial Conditions
^^^^^^^^^^^^^^^^^^

Initial conditions are a key component of any simulation, as they define the starting state of the system being modeled.
In Pisces, initial conditions are represented by the :class:`~core.initial_conditions.InitialConditions` class,
and its subclasses, which allow users to combine multiple Pisces models into a single initial conditions object.

These initial conditions can include various models, such as galaxies, galaxy clusters, or other astrophysical systems,
and can be configured with various parameters such as orientation, offsets, and velocities. The initial conditions
system in Pisces is designed to be flexible and extensible, allowing users to create complex initial conditions
that can be used with different simulation codes.

.. note::

    For a more detailed overview of the initial conditions system in Pisces, see the
    :ref:`initial_conditions_overview` section of the user guide.

Frontends
^^^^^^^^^

Frontends in Pisces are specialized classes that handle the conversion of Pisces data into the format required
by specific simulation codes. Each frontend is responsible for generating the necessary configuration files,
additional structures, and any other code-specific requirements needed to run the simulation.
These frontends are designed to be modular and extensible, allowing users to easily add support for new simulation codes
or customize existing frontends to suit their needs.

.. note::

    For a more detailed overview of the frontend system in Pisces, see the
    :ref:`simulation_frontends` section of the reference documentation.

Problem Initializers
^^^^^^^^^^^^^^^^^^^^

In some cases, simulation codes require specialized problem initializers that are tightly integrated with their
source code. These problem initializers are responsible for directly interacting with the simulation code's internal
data structures to set up the initial conditions. In Pisces, these problem initializers are developed and maintained
by the Pisces team, but they are not part of the core Pisces package. Instead, they are hosted on GitHub as part of
the larger Pisces project. Each problem initializer is written in the language of the simulation code (e.g., Fortran,
C, etc.) and is designed to work seamlessly with the corresponding frontend. If a code requires a problem initializer,
it will be specified in the documentation for that code's frontend.

Supported Simulation Codes
--------------------------

Below is a list of the simulation codes currently supported by Pisces, along with links to their respective
frontend documentation.

.. list-table:: Supported Simulation Codes
    :widths: 15 15 20 15 40
    :header-rows: 1

    * - Code Name
      - Frontend Class
      - Documentation
      - Support Level
      - Notes
    * - `Gadget-4 <https://wwwmpa.mpa-garching.mpg.de/gadget4/>`__
      - :class:`~gadget.frontends.Gadget4Frontend`
      - :ref:`simulations_gadget`
      - |complete|
      -
    * - `RAMSES <https://www.ias.u-psud.fr/ramses/>`__
      - :class:`~ramses.RAMSESFrontend`
      - :ref:`simulations_ramses`
      - |planned|
      -
    * - `AREPO <https://arepo-code.org/>`__
      - :class:`~arepo.ArepoFrontend`
      - :ref:`simulations_arepo`
      - |planned|
      -
    * - `FLASH <https://flash.uchicago.edu/site/flashcode/>`__
      - :class:`~flash.FLASHFrontend`
      - :ref:`simulations_flash`
      - |planned|
      -

.. |complete| image:: https://img.shields.io/badge/Support-Full-brightgreen.svg
.. |planned| image:: https://img.shields.io/badge/Support-Planned-orange.svg
.. |partial| image:: https://img.shields.io/badge/Support-Partial-blue.svg
.. |none| image:: https://img.shields.io/badge/Support-No_Support-black.svg
