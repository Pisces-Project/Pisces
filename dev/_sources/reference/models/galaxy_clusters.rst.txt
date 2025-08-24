.. _galaxy_clusters:
============================
Pisces Galaxy Cluster Models
============================

Pisces provides a suite of models for simulating and analyzing galaxy clusters. These models
are idealized and generally rely on spherical symmetry and hydrostatic equilibrium assumptions; however,
some more exotic models are in development. Below is a table of the available galaxy cluster models in Pisces:

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 15

   * - Model
     - Geometry
     - EOS
     - Assumptions
   * - :class:`~pisces.models.galaxy_clusters.spherical.SphericalGalaxyClusterModel`
     - Spherical
     - IG
     - HSE
   * - :class:`~pisces.models.galaxy_clusters.spherical.MagnetizedSphericalGalaxyClusterModel`
     - Spherical
     - IG + MHD
     - HSE (w/ NTPS)

.. note::

   Abbreviations used in the table:

   - **EOS** = Equation of State
   - **IG** = Ideal Gas
   - **HSE** = Hydrostatic Equilibrium
   - **MHD** = Magnetohydrodynamics
   - **NTPS** = Non-Thermal Pressure Support


Galaxy Clusters Overview
------------------------

Galaxy clusters are the largest gravitationally bound structures in the
universe, containing hundreds to thousands of galaxies, vast amounts of hot gas,
and an even larger reservoir of dark matter. For a classic review of cluster
astrophysics, see :footcite:p:`sarazin1988`.

A typical cluster’s mass budget is dominated by **dark matter**, which accounts
for roughly 80–85% of the total mass
:footcite:p:`VikhlininProfile,kravtsov2012`. The second-largest component is the
**intracluster medium (ICM)**, a hot, diffuse plasma that comprises about
10–15% of the cluster mass and emits strongly in the X-ray band via
thermal bremsstrahlung and line emission
:footcite:p:`sarazin1988,voit2005,pratt2019`. The **galaxies themselves** make
up only a few percent of the mass, but they are important tracers of the
cluster potential and history of structure formation
:footcite:p:`dressler1980`.

The ICM typically has temperatures of \(10^7 - 10^8\) K, making it observable
primarily in X-rays. These observations reveal detailed information about the
density, temperature, and dynamical state of the cluster
:footcite:p:`sarazin1988,voit2005`.

Galaxy clusters are crucial for both cosmology and astrophysics: their abundance
and growth rate provide sensitive tests of cosmological parameters such as the
matter density and dark energy equation of state, while their internal structure
offers insight into the physics of galaxy formation, feedback, and plasma
processes :footcite:p:`kravtsov2012,pratt2019`.

Modeling
---------

Galaxy cluster models in Pisces are based on the assumption of **hydrostatic
equilibrium (HSE)**, in which the pressure gradient of the intracluster medium
balances the gravitational force of the cluster potential. Different model
variants can include purely thermal support or incorporate additional
non-thermal sources such as magnetic fields.


Fully Thermal Hydrostatic Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the simplest case, the intracluster medium (ICM) is modeled as an **ideal
gas** in hydrostatic equilibrium within a spherically symmetric gravitational
potential. The condition of hydrostatic equilibrium reads

.. math::

   \frac{dP}{dr} = -\rho_g(r) \frac{G M(<r)}{r^2},

where :math:`P(r)` is the gas pressure, :math:`\rho_g(r)` is the gas density,
:math:`M(<r)` is the total mass enclosed within radius :math:`r`, and
:math:`G` is Newton’s constant. It forms the basis of the
:class:`~pisces.models.galaxy_clusters.spherical.SphericalGalaxyClusterModel`.


.. dropdown:: Details:

    Assuming an **ideal gas equation of state (EOS)**,

    .. math::

       P(r) = \frac{k_B}{\mu m_p} \rho_g(r) T(r),

    with :math:`k_B` Boltzmann’s constant, :math:`m_p` the proton mass, and
    :math:`\mu` the mean molecular weight, the density may be expressed as

    .. math::

       \rho_g(r) = \frac{\mu m_p}{k_B T(r)} P(r).

    Substituting this into the hydrostatic equilibrium equation gives

    .. math::

       \frac{dP}{dr} = - \frac{\mu m_p}{k_B T(r)} P(r) \frac{G M(<r)}{r^2}.

    Rearranging for the enclosed mass yields

    .. math::

       M(<r) = - \frac{k_B T(r) r^2}{\mu m_p G P(r)} \frac{dP}{dr}.

    It is often more convenient to express this in terms of logarithmic
    derivatives. Writing

    .. math::

       \frac{dP}{dr} = \frac{d \ln P}{d \ln r} \, \frac{P(r)}{r},

    we obtain the standard **hydrostatic mass equation**:

    .. math::

       M(<r) = - \frac{k_B T(r) r}{\mu m_p G}
               \left( \frac{d \ln P}{d \ln r} \right).

    Using the ideal gas EOS, one can also separate this into terms involving
    the gas density and temperature:

    .. math::

       \frac{d \ln P}{d \ln r}
       = \frac{d \ln \rho_g}{d \ln r} + \frac{d \ln T}{d \ln r}.

    Thus,

    .. math::

       M(<r) = - \frac{k_B T(r) r}{\mu m_p G}
               \left( \frac{d \ln \rho_g}{d \ln r}
                    + \frac{d \ln T}{d \ln r} \right).

    This relation connects the cluster’s **total mass profile** directly to the
    observable radial gradients of the **gas density** and **temperature**
    profiles.

Non-Thermal Pressure Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Observations and simulations suggest that in addition to thermal pressure,
other sources of support such as turbulence, cosmic rays, and magnetic fields
can contribute significantly to the equilibrium of clusters
:footcite:p:`lau2009,biffi2016,eckert2019`.

Magnetic Fields
###############

In the **magnetized hydrostatic case**, the total pressure can be written as

.. math::

   P_{\text{tot}}(r) = P_{\text{th}}(r) + P_B(r),

where :math:`P_{\text{th}}` is the thermal gas pressure and
:math:`P_B = B^2/(8\pi)` is the magnetic pressure associated with the magnetic
field strength :math:`B(r)`.

The modified hydrostatic equilibrium equation becomes

.. math::

   \frac{d}{dr} \left[ P_{\text{th}}(r) + P_B(r) \right]
   = -\rho_g(r) \frac{G M(<r)}{r^2}.

This framework is implemented in the
:class:`~pisces.models.galaxy_clusters.spherical.MagnetizedSphericalGalaxyClusterModel`,
which extends the purely thermal model to include **magnetohydrostatic
equilibrium (MHSE)**. Such models are important for studying the role of
magnetic fields in regulating gas dynamics, stability, and heat transport in
the ICM.


References
----------

.. footbibliography::
