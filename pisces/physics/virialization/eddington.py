r"""Eddington formula module for isotropic spherical systems.

This module implements numerical routines for computing and sampling from the
isotropic distribution function :math:`f(\mathcal{E})` in spherically symmetric,
non-rotating systems using the Eddington inversion formula.

The Eddington formula :footcite:p:`BinneyTremaine` allows one to derive the distribution function from a known
density profile :math:`\rho(r)` and gravitational potential :math:`\Phi(r)`,
assuming isotropy in velocity space. The central result is:

.. math::

    f(\mathcal{E}) = \frac{1}{\sqrt{8}\pi^2}
    \frac{d}{d\mathcal{E}} \int_0^{\mathcal{E}}
    \frac{1}{\sqrt{\mathcal{E}-\Psi}} \frac{d\rho}{d\Psi} \, d\Psi,

where:

- :math:`f(\mathcal{E})` is the phase-space distribution function,
- :math:`\mathcal{E}` is the relative energy,
- :math:`\Psi(r) = -[\Phi(r) - \Phi_0]` is the relative potential.

This formulation assumes ergodicity (i.e., that the distribution depends only on energy),
which is valid in collisionless equilibrium systems where the phase-space distribution
is a function of energy alone.

Available features
------------------

- Compute relative potential :math:`\Psi(r)` from the gravitational potential.
- Compute relative energy :math:`\mathcal{E}` from total energy values.
- Evaluate the isotropic distribution function :math:`f(\mathcal{E})` numerically.
- Sample 3D Cartesian velocities for particles drawn from :math:`f(\mathcal{E})`
  using rejection sampling accelerated with Cython.

All units are handled using `unyt`, and the inputs must be appropriately shaped and ordered
(e.g., increasing radius/potential). The numerical integration uses spline interpolation
for stability and smoothness.

References
----------
.. footbibliography::

"""

import numpy as np
import unyt
from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline
from unyt import unyt_array, unyt_quantity

from pisces.utilities import __RNG__, pisces_config

from ._eddington_sampling import generate_velocities


def compute_relative_potential(
    gravitational_potential: unyt_array,
    boundary_value: unyt_quantity = unyt_quantity(0, "km**2/s**2"),
) -> unyt_array:
    r"""Compute the relative potential from a 1D gravitational potential profile.

    The relative potential is defined as:

    .. math::

        \Psi(r) = -\left[\Phi(r) - \Phi_0\right],

    where :math:`\Phi(r)` is the gravitational potential and :math:`\Phi_0` is the
    potential at the chosen boundary (often at infinity).

    Parameters
    ----------
    gravitational_potential : unyt_array
        The 1D gravitational potential profile :math:`\Phi(r)`, with units of velocity squared.
    boundary_value : ~unyt.array.unyt_quantity, optional
        The potential at the boundary, :math:`\Phi_0`. This is typically set to zero,
        but may be chosen differently depending on the problem. Default is 0.

    Returns
    -------
    unyt_array
        The relative potential :math:`\Psi(r)`, with the same shape and units as the input.

    """
    # Ensure boundary value uses the same units as the potential array
    boundary_val = boundary_value.to_value(gravitational_potential.units)

    # Compute the relative potential
    return -(gravitational_potential - boundary_val)


def compute_relative_energy(
    energy: unyt_array,
    boundary_value: unyt_quantity = unyt_quantity(0, "km**2/s**2"),
) -> unyt_array:
    r"""Compute the relative energy :math:`\mathcal{E}` from the total energy.

    The relative energy is defined as:

    .. math::

        \mathcal{E} = -\left(E - E_0\right),

    where :math:`E` is the total energy and :math:`E_0` is the reference energy
    at the chosen boundary (typically zero).

    Parameters
    ----------
    energy : unyt_array
        The total energy values, typically of the form :math:`\frac{1}{2}v^2 + \Phi(r)`.
    boundary_value : ~unyt.array.unyt_quantity, optional
        The energy at the reference boundary (default is 0, with units inferred from `energy`).

    Returns
    -------
    unyt_array
        The relative energy :math:`\mathcal{E}`, same units and shape as the input energy.

    """
    # Ensure units match for subtraction
    boundary_val = boundary_value.to(energy.units)
    return -(energy - boundary_val)


def compute_eddington_distribution(
    density: unyt_array,
    potential: unyt_array,
    num_points: int = 1000,
    boundary_value: unyt_quantity = unyt_quantity(0, "km**2/s**2"),
    scale: str = "linear",
) -> tuple[unyt_array, unyt_array]:
    r"""Compute the isotropic Eddington distribution function from a given species density and gravitational potential.

    The distribution function :math:`f(\mathcal{E})` is computed under the assumption of ergodicity and isotropy:

    .. math::

        f(\mathcal{E}) = \frac{1}{\sqrt{8}\pi^2}
        \frac{d}{d\mathcal{E}} \int_0^\mathcal{E}
        \frac{1}{\sqrt{\mathcal{E}-\Psi}} \frac{d\rho}{d\Psi} d\Psi

    Parameters
    ----------
    density : unyt_array
        The species density profile :math:`\rho(r)`. Must be ordered identically to `potential`.

    potential : unyt_array
        The gravitational potential profile :math:`\Phi(r)` corresponding to the same radii as `density`.

    num_points : int, optional
        Number of evaluation points in relative energy :math:`\mathcal{E}`. Default is 1000.

    boundary_value : ~unyt.array.unyt_quantity, optional
        Value of :math:`\Phi_0` used to define the relative potential :math:`\Psi = -(\Phi - \Phi_0)`.
        Default is 0.

    scale : {'linear', 'log'}, optional
        Whether to sample relative energy points linearly or logarithmically. Default is 'linear'.

    Returns
    -------
    unyt_array
        Relative energies :math:`\mathcal{E}`.

    unyt_array
        Distribution function values :math:`f(\mathcal{E})`, in units of
        :math:`[\rho][\mathcal{E}]^{-3/2}`.

    """
    # Compute the relative potential
    rel_potential = compute_relative_potential(potential, boundary_value)

    # Ensure increasing order
    if rel_potential[0] > rel_potential[1]:
        rel_potential = rel_potential[::-1]
        density = density[::-1]

    # Strip units for computation
    psi_vals = rel_potential.to_value("km**2/s**2")
    rho_vals = density.to_value(density.units)

    # Interpolator: 1st derivative needed for Eddington inversion
    rho_spline = InterpolatedUnivariateSpline(psi_vals, rho_vals, k=3)

    def _integrand(v, E):
        return 2 * rho_spline(E - v**2, 1)

    def _edd_integral(E):
        return quad(_integrand, 0, np.sqrt(E), args=(E,), epsabs=0, epsrel=1e-6)[0]

    # Construct energy grid
    E_max = np.max(psi_vals)

    if scale == "log":
        E_min = E_max * 1e-6
        energies = np.logspace(np.log10(E_min), np.log10(E_max), num_points)
    elif scale == "linear":
        energies = np.linspace(0, E_max, num_points)
    else:
        raise ValueError(f"Invalid scale '{scale}'. Must be 'linear' or 'log'.")

    # Perform integral
    integral_values = np.array([_edd_integral(E) for E in energies])

    # Derivative of the integral with respect to E
    spline = InterpolatedUnivariateSpline(energies, integral_values, k=3)
    df = (1 / (np.sqrt(8) * np.pi**2)) * spline(energies, 1)

    # Attach units
    energy_out = unyt_array(energies, units=rel_potential.units)
    f_out = unyt_array(df, units=density.units / (rel_potential.units**1.5))

    return energy_out, f_out


def sample_eddington_velocities(
    relative_energy_array: unyt.unyt_array,
    distribution_function_array: unyt.unyt_array,
    particle_relative_potentials: unyt.unyt_array,
    **kwargs,
) -> unyt.unyt_array:
    r"""Sample 3D Cartesian velocities for particles based on an isotropic Eddington distribution function.

    This function uses rejection sampling to draw particle speeds from a spherically symmetric,
    ergodic distribution function :math:`f(\mathcal{E})`, and then assigns a random direction to each speed to produce
    3D velocity vectors.

    Parameters
    ----------
    relative_energy_array : ~unyt.array.unyt_array
        1D array containing the tabulated relative energies :math:`\mathcal{E}` at which the distribution
        function :math:`f(\mathcal{E})` is evaluated. Units must be compatible with
        :math:`\mathrm{km}^2\,\mathrm{s}^{-2}`. The length of this array must match the length of
        `distribution_function_array`.
    distribution_function_array : ~unyt.array.unyt_array
        1D array of the distribution function values :math:`f(\mathcal{E})`, corresponding to
        each entry in `relative_energy_array`. These values describe the phase-space density of particles
        as a function of energy in a spherically symmetric, isotropic system. Units must be
        :math:`\mathrm{M}_\odot\,\mathrm{s}^3\,\mathrm{km}^{-3}\,\mathrm{kpc}^{-3}`.
    particle_relative_potentials : ~unyt.array.unyt_array
        1D array of relative gravitational potentials :math:`\Psi(r)` for each particle position, used
        to determine the maximum allowed velocity (escape velocity) at that location.
        Each value should be convertible to :math:`\mathrm{km}^2\,\mathrm{s}^{-2}`.
        The escape velocity for a particle is computed as :math:`v_\mathrm{esc} = \sqrt{2\Psi(r)}`.
    **kwargs:
        Additional keyword arguments passed to the Cython sampling routine, such as `show_progress`
        to control progress bar visibility.

    Returns
    -------
    output_velocities : ~unyt.array.unyt_array
        A ``(3, N)`` array of sampled 3D Cartesian velocities with units of `km/s`. The velocities
        are sampled from the input Eddington distribution and distributed isotropically in direction.

    Notes
    -----
    This function samples velocities for particles in a spherically symmetric, isotropic system
    governed by an ergodic distribution function :math:`f(\mathcal{E})`. For each particle,
    the escape velocity is computed from the relative potential using the expression
    :math:`v_\mathrm{esc}(r) = \sqrt{2\Psi(r)}`, where the relative potential is defined as
    :math:`\Psi(r) = -[\Phi(r) - \Phi_0]`. This gives the maximum speed a bound particle can have
    at that location.

    The distribution function :math:`f(\mathcal{E})` is provided as a one-dimensional array evaluated
    on a corresponding grid of relative energies :math:`\mathcal{E}`, and is interpolated internally
    using a cubic spline. Particle speeds are sampled via rejection sampling on the function
    :math:`f(\mathcal{E}) \cdot v^2`, which reflects the correct velocity-space weighting in
    spherical symmetry.

    For each particle, trial velocities :math:`v` are drawn uniformly in :math:`[0, v_\mathrm{esc}]`
    and converted to relative energy via :math:`\mathcal{E} = \Psi - \frac{1}{2}v^2`. A trial is accepted
    if the product :math:`f(\mathcal{E}) \cdot v^2` is less than a uniform random deviate times a
    precomputed bounding value :math:`\mathrm{likelihood\_max}[i]`. This ensures unbiased sampling
    from the desired distribution.

    Once a speed is accepted, a random isotropic direction is assigned by sampling spherical angles
    :math:`\theta` and :math:`\phi`, and converting the result into Cartesian velocity components.
    The final output is a three-component array of velocities for each particle, sampled from
    :math:`f(\mathcal{E})` and distributed isotropically in direction.

    The core rejection sampling loop is implemented in Cython for performance and lives in the
    module :file:`pisces/particles/sampling/_eddington_sampling.pyx`. All input arrays must be
    one-dimensional, consistently ordered, and convertible to the expected physical units.

    """
    # Compute escape velocities from relative potential: v_esc = sqrt(2 * Ψ)
    escape_velocities = (2 * particle_relative_potentials) ** 0.5
    escape_velocities = escape_velocities.to_value("km/s")

    # Convert all input arrays to consistent raw units
    _relative_energy = relative_energy_array.to_value("km**2 / s**2")
    _relative_potential = particle_relative_potentials.to_value("km**2 / s**2")
    _df = distribution_function_array.to_value("Msun * s**3 / km**3 / kpc**3")

    # Conservative bounding envelope for rejection sampling: f_max * v_esc²
    f_max = _df.max()
    likelihood_bounds = f_max * escape_velocities**2

    # Build cubic spline for f(𝔈)
    df_spline = InterpolatedUnivariateSpline(_relative_energy, _df, k=3)

    # Use Cython-accelerated sampling routine
    velocities = generate_velocities(
        _relative_potential,
        escape_velocities,
        likelihood_bounds,
        df_spline.get_knots(),
        df_spline.get_coeffs(),
        df_spline._eval_args[2],  # spline degree
        show_progress=~pisces_config["system.appearance.disable_progress_bars"],
        **kwargs,
    )

    # Assign random isotropic directions using spherical coordinates
    phi = __RNG__.uniform(0, 2 * np.pi, size=velocities.shape)
    theta = np.arccos(__RNG__.uniform(-1, 1, size=velocities.shape))

    # Convert speed and angles to 3D Cartesian velocity components
    output_velocities = unyt.unyt_array(
        np.vstack(
            [
                velocities * np.sin(theta) * np.cos(phi),
                velocities * np.sin(theta) * np.sin(phi),
                velocities * np.cos(theta),
            ]
        ),
        units="km/s",
    )

    return output_velocities
