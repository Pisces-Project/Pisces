"""
Integration methods for Pisces.

This module provides numerical integration routines for computing
enclosed mass profiles and cumulative integrals of functions defined over radial profiles.
These methods are useful for astrophysical applications where analytical solutions
are not feasible or practical.

"""

import numpy as np
from scipy.integrate import quad, solve_ivp


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


def compute_lame_emden_solution(
    xi_max: float, n: float = 1, xi_min: float = 1e-5, end_at_first_zero: bool = False, **kwargs
):
    r"""
    Compute a numerical solution to the Lane-Emden equation for a given polytropic index :math:`n`.

    The Lane-Emden equation models the structure of polytropic spheres such as stars, and is given by:

    .. math::

        \frac{1}{\xi^2} \frac{d}{d\xi} \left( \xi^2 \frac{d\theta}{d\xi} \right) + \theta^n = 0,

    with boundary conditions:

    .. math::

        \theta(0) = 1, \quad \theta'(0) = 0.

    Because the equation is singular at :math:`\xi = 0`, we begin integration at a small nonzero value
    :math:`\xi_{\rm min}` using a Taylor expansion to approximate the initial conditions.

    Parameters
    ----------
    xi_max : float
        Maximum value of the dimensionless radial coordinate :math:`\xi` to integrate to. Depending on the
        index and the value of the `end_at_first_zero` flag, this may not be reached if the solution crosses zero.
        If the solution does not trigger a termination event, the integration will continue to this value.

        In general, it is best to set this well beyond the expected extent of the solution, such as 10 or 20.
    n : float, optional
        Polytropic index. Default is 1.
    xi_min : float, optional
        Starting value of :math:`\xi` to avoid the singularity at zero. Default is ``1e-5``.
    end_at_first_zero : bool, optional
        If True, halt integration when :math:`\theta` first crosses below zero. This is overruled if `n` is
        a non-integer, in which case the integration will always truncate to avoid non-real solutions.
    **kwargs
        Additional keyword arguments passed to :func:`scipy.integrate.solve_ivp`. This can be used to
        control the integration method, tolerances, and other parameters.

    Returns
    -------
    solution : scipy.integrate.OdeResult
        Full solution object from :func:`scipy.integrate.solve_ivp`. Includes:

        - ``solution.t``: array of :math:`\xi` values
        - ``solution.y``: 2D array of corresponding :math:`\theta(\xi)` and auxiliary variables
        - ``solution.status``: solver status code (0=success, 1=event-triggered)
        - ``solution.t_events``: event times, if any
        - ``solution.message``: diagnostic message

    Notes
    -----
    For small :math:`\xi`, we approximate:

    .. math::

        \theta(\xi) \approx 1 - \frac{1}{6} \xi^2, \quad \theta'(\xi) \approx -\frac{1}{3} \xi,

    which are valid for small deviations from the origin. These values are used to initialize the solver
    at :math:`\xi = \xi_{\rm min}`.

    Additionally, when ``end_at_first_zero`` is True, or when `n` is not an integer, we define an event function
    that triggers when :math:`\theta` crosses below a small threshold (e.g., 0.01). This allows the integration
    to stop gracefully when the solution is no longer valid, avoiding numerical instability.

    Examples
    --------
    .. code-block:: python

        from pisces.math_utils.integration import (
            compute_lame_emden_solution,
        )

        xi, theta = compute_lame_emden_solution(
            xi_max=10.0,
            n=1.5,
            end_at_first_zero=True,
            max_step=0.1,
        )

    """

    # Define the RHS of the ODE using our formalism.
    def _lame_emden_rhs(xi, y):
        theta, psi = y
        dtheta_dxi = psi / xi**2
        dpsi_dxi = -(theta**n) * xi**2
        return [dtheta_dxi, dpsi_dxi]

    # Construct the initial values. We need to avoid the singularity at xi=0,
    # so we start at a small positive value xi_min. Taylor expansion is used
    # to propagate.
    theta_approx = 1 - (1 / 6) * xi_min**2
    dtheta_dxi_approx = -(1 / 3) * xi_min
    psi_approx = xi_min**2 * dtheta_dxi_approx
    y0 = [theta_approx, psi_approx]

    # For non-integer n, we need to catch instances where we start getting
    # close to theta = 0 because we will not have stability in that case.
    # We use an event in the integrator to achieve this.
    if not float(n).is_integer() or end_at_first_zero:

        def _theta_crossing_event_callable(xi, y):
            return y[0] - 1e-2

        _theta_crossing_event_callable.terminal = True
        _theta_crossing_event_callable.direction = -1
        events = [_theta_crossing_event_callable]
    else:
        events = None

    # Solve the ODE
    result = solve_ivp(_lame_emden_rhs, (xi_min, xi_max), y0, events=events, **kwargs)

    # Check for failure in the solver and then
    # return the correct result.
    if result.status not in [0, 1]:
        raise RuntimeError(f"ODE solver failed with status {result.status}: {result.message}")

    return result
