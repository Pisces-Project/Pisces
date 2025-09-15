.. _stars:
============================
Pisces Stellar Models
============================

Pisces provides a growing suite of stellar structure models to support astrophysical modeling and simulation.
These models are designed to reproduce physically consistent stellar interiors and atmospheres using a variety
of approximations. Current models focus on polytropic stars, with future extensions planned to support
realistic equations of state, nuclear burning, and metallicity-dependent structures.

.. note::

    This documentation is a work in progress. The models described here are
    currently under development, and the API may change in future releases.
    Please refer to the latest Pisces documentation for updates.

.. contents::
    :local:
    :depth: 2

Polytropic Stellar Models
--------------------------

The simplest stellar model available in Pisces is the so-called
**polytropic sphere** :footcite:p:`kippenhahn1990stellar`.
These stars are idealized objects that assume a polytropic equation of state to model the relationship between pressure
and density:

.. math::

    P = K \rho^{1 + 1/n}

where :math:`K` is a constant of proportionality and :math:`n` is the *polytropic index*.
This leads to the well-known **Lane-Emden Equation**, a second-order differential equation
whose solution governs the structure of the star.

Polytropic models are widely used in astrophysics as simplified models of main-sequence stars,
white dwarfs, and even planetary interiors. In Pisces, we provide high-resolution numerical
solutions and support for outputting physical profiles such as temperature, density, pressure,
and gravitational potential.

These models are contained in the :mod:`~pisces.models.stars.polytropes` module.

Model Background
^^^^^^^^^^^^^^^^

In the base model (:class:`~pisces.models.stars.polytropes.PolytropicStarModel`), we assume a spherically symmetric
star in hydrostatic equilibrium with a polytropic equation of state to describe the pressure-density relationship.
The gas is assumed to be an ideal gas which is non-relativistic and non-rotating. From these assumptions, the
structure of the star may be derived from hydrostatic equilibrium and the polytropic equation of state.

The structure of a star in equilibrium is governed by the balance between inward gravitational forces
and outward pressure gradients. This balance is described by the equation of
hydrostatic equilibrium :footcite:p:`clarke2007principles`:

.. math::

    \frac{\nabla P}{\rho} = -\nabla \Phi.

where:

- :math:`P` is the pressure,
- :math:`\rho` is the density,
- :math:`\Phi` is the gravitational potential.

The gravitational potential is related to the mass distribution within the star, which can be expressed
via the Poisson equation:

.. math::

    \nabla^2 \Phi = 4\pi G \rho

where :math:`G` is the gravitational constant.

To close the system of equations, we assume a polytropic relationship between pressure and density:

.. math::

    P = K \rho^{1 + \frac{1}{n}}

where:

- :math:`K` is a dimensional constant (the polytropic constant),
- :math:`n` is the *polytropic index*, which determines the stiffness of the equation of state.


Substituting the polytropic equation of state into the hydrostatic equilibrium condition and introducing
dimensionless variables reduces the stellar structure problem to the **Lane-Emden equation**:

.. math::

    \frac{1}{\xi^2} \frac{d}{d\xi} \left( \xi^2 \frac{d\theta}{d\xi} \right) = -\theta^n

where:

- :math:`\xi` is the dimensionless radial coordinate: :math:`\xi = r / \alpha`,
- :math:`\theta(\xi)` is the dimensionless density: :math:`\rho(r) = \rho_c \theta^n`,
- :math:`\alpha` is the scale length defined by:

  .. math::

     \alpha^2 = \frac{(n + 1) K \rho_c^{(1/n - 1)}}{4\pi G}

This second-order ordinary differential equation must be solved numerically, with boundary conditions:

.. math::

    \theta(0) = 1, \quad \left.\frac{d\theta}{d\xi}\right|_{\xi=0} = 0

The solution determines the internal structure of the star up to the first zero of :math:`\theta(\xi)`,
which corresponds to the stellar surface (i.e., where the density vanishes).

In general, the Lane-Emden equation may yield values of :math:`\theta < 0`; however, these solutions are non-physical.
As such, we restrict our attention to the first zero of the solution, which corresponds to the physical surface of the
star. This allows us to define the star's radius and other global properties.

Using Polytropic Models
^^^^^^^^^^^^^^^^^^^^^^^

There are a variety of ways to build a PolytropicStarModel in Pisces, depending on the available information:

- **Central conditions**, using :meth:`~pisces.models.stars.polytropes.PolytropicStarModel.from_density_and_temperature`, or
- **Global properties**, using :meth:`~pisces.models.stars.polytropes.PolytropicStarModel.from_mass_and_radius`.

It is typically easiest to use the global properties method, as it requires only the mass and radius of the star, which
are more commonly known than the central density and temperature. Additionally, the choice of the polytropic index
is a non-trivial decision that can significantly affect the resulting stellar structure. The following is generally
suggested from the literature:

.. list-table:: Common Polytropic Indices
   :header-rows: 1
   :widths: 10 30

   * - Polytropic Index (:math:`n`)
     - Physical Interpretation
   * - 0
     - Uniform density sphere (mathematical toy model)
   * - 1.0
     - Approximate model for isothermal spheres and some neutron star envelopes
   * - 1.5
     - Fully convective low-mass stars (e.g., red dwarfs), classical ideal gas with constant entropy
   * - 3.0
     - Radiation pressure-dominated stars, massive main-sequence stars, relativistic degenerate cores (e.g., white dwarfs)
   * - >3
     - Physically unstable configurations (not in hydrostatic equilibrium)




References
----------
.. footbibliography::
