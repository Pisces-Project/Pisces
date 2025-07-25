"""
Integration methods for Pisces.

This module provides numerical integration routines for computing
enclosed mass profiles and cumulative integrals of functions defined over radial profiles.
These methods are useful for astrophysical applications where analytical solutions
are not feasible or practical.

"""

import numpy as np
from scipy.integrate import quad


def integrate_mass(profile, rr, **kwargs):
    r"""
    Compute enclosed mass profile via numerical integration.

    For a density profile :math:`\rho(r)`, this evaluates:

    .. math::

        M(r) = 4 \pi \int_0^r \rho(r') \; r'^2 \; dr'

    Parameters
    ----------
    profile : callable
        Function of radius :math:`r` returning the density :math:`\rho(r)`
        as a scalar or array.
    rr : array-like
        Radii at which to compute enclosed mass. Must be ascending and positive.
    **kwargs : dict, optional
        Additional keyword arguments passed to :func:`scipy.integrate.quad`.

    Returns
    -------
    mass : ndarray
        Array of enclosed mass values at each radius.
    """
    rr = np.atleast_1d(rr)
    mass = np.zeros_like(rr)

    def _mass_integrand(r_):
        return profile(r_) * r_**2

    for i, r in enumerate(rr):
        mass[i] = quad(_mass_integrand, 0, r, **kwargs)[0]

    return 4 * np.pi * mass


def integrate(profile, rr, rmax=None, **kwargs):
    r"""
    Compute cumulative integral from each radius :math:`r` to :math:`r_{\mathrm{max}}`.

    Evaluates:

    .. math::

        I(r) = \int_{r}^{r_{\mathrm{max}}} f(r') \; dr'

    Parameters
    ----------
    profile : callable
        Function of radius :math:`r` returning the integrand :math:`f(r)`.
    rr : array-like
        Radii at which to compute cumulative integrals. Must be sorted ascending.
    rmax : float or None, optional
        Upper bound of integration. If None, defaults to ``rr[-1]``.
    **kwargs : dict, optional
        Additional keyword arguments passed to :func:`scipy.integrate.quad`.

    Returns
    -------
    integral : ndarray
        Array of integral values :math:`I(r)` for each radius in ``rr``.

    Raises
    ------
    ValueError
        If any radius in ``rr`` exceeds ``rmax``.
    """
    rr = np.atleast_1d(rr)
    rmax = rr[-1] if rmax is None else rmax

    if not np.all(rr <= rmax):
        raise ValueError("All input radii rr must lie within [min(rr), rmax].")

    integral = np.zeros_like(rr)

    for i, r in enumerate(rr):
        integral[i] = quad(profile, r, rmax, **kwargs)[0]

    return integral


def integrate_toinf(profile, rr, rmax=None, **kwargs):
    r"""
    Compute cumulative integral from each radius :math:`r` to infinity.

    Evaluates:

    .. math::

        I(r) = \int_{r}^{\infty} f(r') \; dr'

    The integration is performed in two parts:

    1. From :math:`r` to :math:`r_{\mathrm{max}}`.
    2. From :math:`r_{\mathrm{max}}` to :math:`\infty` (computed once and added to all results).

    Parameters
    ----------
    profile : callable
        Function of radius :math:`r` returning the integrand :math:`f(r)`.
    rr : array-like
        Radii at which to compute cumulative integrals. Must be sorted ascending.
    rmax : float or None, optional
        Upper bound for intermediate integration. If None, defaults to ``rr[-1]``.
    **kwargs : dict, optional
        Additional keyword arguments passed to :func:`scipy.integrate.quad`.

    Returns
    -------
    integral : ndarray
        Array of integral values :math:`I(r)` for each radius in ``rr``.

    Raises
    ------
    ValueError
        If any radius in ``rr`` exceeds ``rmax``.
    """
    rr = np.atleast_1d(rr)
    rmax = rr[-1] if rmax is None else rmax

    if not np.all(rr <= rmax):
        raise ValueError("All input radii rr must lie within [min(rr), rmax].")

    integral = np.zeros_like(rr)

    for i, r in enumerate(rr):
        integral[i] = quad(profile, r, rmax, **kwargs)[0]

    remainder = quad(profile, rmax, np.inf, **kwargs)[0]
    integral += remainder

    return integral
