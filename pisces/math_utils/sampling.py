"""Module for sampling from various statistical distributions.

This module provides functions to sample particle positions from a probability density function (PDF)
and a cumulative distribution function (CDF) defined at discrete positions. It implements
inverse transform sampling using linear interpolation, allowing for efficient sampling from
discrete distributions.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

from pisces.utilities.rng import __RNG__


def sample_from_pdf(x: np.ndarray, y: np.ndarray, num_particles: int) -> np.ndarray:
    """Sample particle positions from a probability density function (PDF) defined at discrete positions.

    This function first constructs a normalized cumulative distribution function (CDF) from the
    input PDF using the trapezoidal rule, and then uses inverse transform sampling to generate
    samples.

    Parameters
    ----------
    x : np.ndarray
        A strictly increasing 1D array representing the domain of the PDF.
    y : np.ndarray
        A 1D array of the same length as `x`, containing the PDF values at each position.
        Values must be non-negative.
    num_particles : int
        The number of samples to draw from the distribution.

    Returns
    -------
    np.ndarray
        A 1D array of shape `(num_particles,)` containing particle positions sampled
        from the input distribution.

    Raises
    ------
    ValueError
        If inputs are not 1D arrays of the same length, if `x` is not strictly increasing,
        or if `y` contains negative values.

    Notes
    -----
    This function uses inverse transform sampling: it constructs a CDF via cumulative trapezoidal
    integration, normalizes it to [0, 1], and maps uniform random values through the inverse CDF
    (using linear interpolation).

    """
    # Validate inputs. We need to check the dimensionality and the lengths. Then we
    # ensure that `x` is increasing and `y` is non-negative.
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing.")
    if np.any(y < 0):
        raise ValueError("PDF values must be non-negative.")

    # Compute the cumulative integral using the trapezoidal rule
    cumulative_integral = cumulative_trapezoid(y, x, initial=0)

    return sample_from_cdf(x, cumulative_integral, num_particles)


def sample_from_cdf(x: np.ndarray, cdf: np.ndarray, num_particles: int) -> np.ndarray:
    """Sample particle positions from a cumulative distribution function (CDF) defined at discrete positions.

    This function implements inverse transform sampling using linear interpolation. It assumes the
    input CDF is monotonically increasing and defined on a 1D grid `x`.

    Parameters
    ----------
    x : np.ndarray
        A strictly increasing 1D array of positions where the CDF is defined.
    cdf : np.ndarray
        A 1D array of the same length as `x`, containing the CDF values at each position.
        Values must be non-decreasing and bounded in [0, 1] (after normalization).
    num_particles : int
        The number of samples to draw from the distribution.

    Returns
    -------
    np.ndarray
        A 1D array of shape `(num_particles,)` containing particle positions sampled
        from the input distribution.

    Raises
    ------
    ValueError
        If inputs are not 1D arrays of the same length, if `x` is not strictly increasing,
        or if `cdf` is not non-decreasing.

    Notes
    -----
    This function uses inverse transform sampling: uniformly distributed random values
    on [0, 1] are mapped through the inverse CDF (using linear interpolation) to obtain
    samples from the target distribution.

    """
    # Input validation
    if x.ndim != 1 or cdf.ndim != 1:
        raise ValueError("x and cdf must be 1D arrays.")
    if len(x) != len(cdf):
        raise ValueError("x and cdf must have the same length.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing.")
    if not np.all(np.diff(cdf) >= 0):
        raise ValueError("cdf must be non-decreasing.")

    # Normalize the CDF to [0, 1] if not already
    cdf_min = cdf[0]
    cdf_max = cdf[-1]
    if cdf_max == cdf_min:
        raise ValueError("CDF has zero dynamic range (all values are equal).")
    cdf = (cdf - cdf_min) / (cdf_max - cdf_min)

    # Draw uniform samples and map them through the inverse CDF
    uniform_samples = __RNG__.uniform(0.0, 1.0, num_particles)
    return np.interp(uniform_samples, cdf, x)
