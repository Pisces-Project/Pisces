"""
Support for generating random fields and related utilities.

This module provides base classes and helper functions for constructing
random fields on regular grids, with particular emphasis on Gaussian
Random Fields (GRFs) specified by an input power spectrum.

Overview
--------
The key abstraction is the :class:`RandomField` base class, which defines
the spatial domain, Fourier-space utilities, and interface for generating
random fields. A concrete implementation,
:class:`GaussianRandomField`, generates realizations of Gaussian random
fields by sampling complex Fourier coefficients consistent with a chosen
power spectrum, followed by an inverse FFT to real space.

This framework is useful in astrophysical and cosmological applications
(e.g. generating initial conditions for magnetic fields in galaxy
clusters), as well as in more general stochastic modeling contexts where
random fields with prescribed statistics are needed.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Optional, Union

import numpy as np
import unyt
from scipy.interpolate import InterpolatedUnivariateSpline

from ._random_fields import (
    div_clean_c_2d,
    div_clean_c_3d,
    div_clean_fd_2d,
    div_clean_fd_3d,
)


# ========================================= #
# Power Spectrum Definitions                #
# ========================================= #
def generate_white_noise_power_spectrum(amplitude: float = 1.0) -> Callable:
    r"""
    Generate a white-noise (flat) power spectrum function.

    A white-noise spectrum is defined by a constant power at all
    wavenumbers :math:`k`, i.e.

    .. math::

        P(k) = A,

    where :math:`A` is the amplitude. This corresponds to a random
    field with equal variance contributed by all Fourier modes,
    independent of scale.

    Parameters
    ----------
    amplitude : float, optional
        Constant amplitude :math:`A` of the power spectrum.
        Default is ``1.0``.

    Returns
    -------
    Callable
        A function ``P(kx, ky, ...)`` that takes arrays of wavenumber
        components and
        returns the constant power spectrum evaluated on that grid.

    Examples
    --------

    .. plot::
        :include-source:

        from pisces.math_utils.random_fields import generate_white_noise_power_spectrum
        L, n = 10.0, 128
        k_grid = np.fft.fftfreq(n, d=L/n) * 2 * np.pi
        power_spectrum = generate_white_noise_power_spectrum()
        ps_array = power_spectrum(k_grid)

        import matplotlib.pyplot as plt
        plt.loglog(k_grid, ps_array)
        plt.xlabel('Wavenumber k')
        plt.ylabel('Power Spectrum P(k)')
        plt.title('White Noise Power Spectrum')
        plt.show()

    """

    def _ps_func(*args):
        print(args)
        return amplitude * np.ones_like(args[0])

    return _ps_func


def generate_power_law_power_spectrum(amplitude: float = 1.0, k0: float = 1.0, slope: float = -11 / 3) -> Callable:
    r"""
    Generate a power-law power spectrum function.

    A power-law spectrum is defined by

    .. math::

        P(k) = A \left( \frac{k}{k_0} \right)^{\alpha},

    where :math:`A` is the amplitude, :math:`k_0` is a reference
    wavenumber for normalization, and :math:`\alpha` is the slope.
    A slope of :math:`\alpha = -11/3` corresponds to the Kolmogorov
    scaling for incompressible turbulence.

    Parameters
    ----------
    amplitude : float, optional
        Normalization amplitude :math:`A`. Default is ``1.0``.
    k0 : float, optional
        Reference wavenumber :math:`k_0`. Default is ``1.0``.
    slope : float, optional
        Power-law slope :math:`\alpha`. Default is ``-11/3``.

    Returns
    -------
    Callable
        A function ``P(kx, ky, ...)`` that takes arrays of wavenumber
        components
        and returns the power-law spectrum evaluated on that grid.

    Examples
    --------

    .. plot::
        :include-source:

        from pisces.math_utils.random_fields import generate_power_law_power_spectrum
        L, n = 10.0, 128
        k_grid = np.fft.fftfreq(n, d=L/n) * 2 * np.pi
        k_mag = np.abs(k_grid)
        power_spectrum = generate_power_law_power_spectrum(slope=-11/3)
        ps_array = power_spectrum(k_mag)

        import matplotlib.pyplot as plt
        plt.loglog(k_mag[1:], ps_array[1:])  # skip k=0
        plt.xlabel('Wavenumber k')
        plt.ylabel('Power Spectrum P(k)')
        plt.title('Power-Law Spectrum (slope = -11/3)')
        plt.show()
    """

    def _ps_func(*args):
        k_mag = np.sqrt(sum(ki**2 for ki in args))
        P = amplitude * (np.where(k_mag > 0, k_mag / k0, 1.0)) ** slope
        P[k_mag == 0] = 0.0
        return P

    return _ps_func


def generate_gaussian_cutoff_power_spectrum(amplitude: float = 1.0, k0: float = 1.0, sigma: float = 1.0) -> Callable:
    r"""
    Generate a Gaussian-cutoff power spectrum function.

    The Gaussian-cutoff spectrum is defined by

    .. math::

        P(k) = A \exp\!\left[-\frac{1}{2}
        \left(\frac{k - k_0}{\sigma}\right)^2\right],

    where :math:`A` is the amplitude, :math:`k_0` is the central
    wavenumber, and :math:`\sigma` controls the width of the Gaussian.
    This spectrum is peaked around :math:`k_0` and falls off rapidly
    away from the peak.

    Parameters
    ----------
    amplitude : float, optional
        Normalization amplitude :math:`A`. Default is ``1.0``.
    k0 : float, optional
        Central wavenumber :math:`k_0`. Default is ``1.0``.
    sigma : float, optional
        Width of the Gaussian :math:`\sigma`. Default is ``1.0``.

    Returns
    -------
    Callable
        A function ``P(kx, ky, ...)`` that takes arrays of wavenumber
        components
        and returns the Gaussian-cutoff spectrum.

    Examples
    --------

    .. plot::
        :include-source:

        from pisces.math_utils.random_fields import generate_gaussian_cutoff_power_spectrum
        L, n = 10.0, 256
        k_grid = np.fft.fftfreq(n, d=L/n) * 2 * np.pi
        k_mag = np.abs(k_grid)
        power_spectrum = generate_gaussian_cutoff_power_spectrum(k0=5.0, sigma=2.0)
        ps_array = power_spectrum(k_mag)

        import matplotlib.pyplot as plt
        plt.plot(k_mag, ps_array)
        plt.xlabel('Wavenumber k')
        plt.ylabel('Power Spectrum P(k)')
        plt.title('Gaussian-Cutoff Spectrum')
        plt.show()
    """

    def _ps_func(*args):
        k_mag = np.sqrt(sum(ki**2 for ki in args))
        return amplitude * np.exp(-0.5 * ((k_mag - k0) / sigma) ** 2)

    return _ps_func


def generate_kolmogorov_power_spectrum(
    amplitude: float = 1.0, k0: float = 1.0, k_cut: Optional[float] = None
) -> Callable:
    r"""
    Generate a Kolmogorov-like turbulence power spectrum function.

    The Kolmogorov turbulence scaling predicts an energy spectrum
    :math:`E(k) \propto k^{-5/3}`, which corresponds to a power spectrum
    :math:`P(k) \propto k^{-11/3}` in three dimensions. This function
    implements

    .. math::

        P(k) = A \left(\frac{k}{k_0}\right)^{-11/3}
        \exp\!\left[-\left(\frac{k}{k_\mathrm{cut}}\right)^2\right],

    where the exponential cutoff is optional and only applied if
    :math:`k_\mathrm{cut}` is specified.

    Parameters
    ----------
    amplitude : float, optional
        Normalization amplitude :math:`A`. Default is ``1.0``.
    k0 : float, optional
        Reference wavenumber :math:`k_0`. Default is ``1.0``.
    k_cut : float, optional
        High-wavenumber cutoff :math:`k_\mathrm{cut}`. If provided, applies
        a Gaussian suppression factor. Default is ``None``.

    Returns
    -------
    Callable
        A function ``P(kx, ky, ...)`` that takes arrays of wavenumber
        components
        and returns the Kolmogorov-like spectrum.

    Examples
    --------

    .. plot::
        :include-source:

        from pisces.math_utils.random_fields import generate_kolmogorov_power_spectrum
        L, n = 10.0, 256
        k_grid = np.fft.fftfreq(n, d=L/n) * 2 * np.pi
        k_mag = np.abs(k_grid)
        power_spectrum = generate_kolmogorov_power_spectrum(amplitude=1.0, k0=1.0, k_cut=20.0)
        ps_array = power_spectrum(k_mag)

        import matplotlib.pyplot as plt
        plt.loglog(k_mag[1:], ps_array[1:])  # skip k=0
        plt.xlabel('Wavenumber k')
        plt.ylabel('Power Spectrum P(k)')
        plt.title('Kolmogorov-like Spectrum with Cutoff')
        plt.show()
    """
    if k_cut is None:

        def _ps_func(*args):
            k_mag = np.sqrt(sum(ki**2 for ki in args))
            P = amplitude * (np.where(k_mag > 0, k_mag / k0, 1.0)) ** (-11 / 3)
            P[k_mag == 0] = 0.0
            return P
    else:

        def _ps_func(*args):
            k_mag = np.sqrt(sum(ki**2 for ki in args))
            P = amplitude * (np.where(k_mag > 0, k_mag / k0, 1.0)) ** (-11 / 3) * np.exp(-((k_mag / k_cut) ** 2))
            P[k_mag == 0] = 0.0
            return P

    return _ps_func


def generate_zh11_power_spectrum(amplitude: float = 1.0, k0: float = 1.0, k1: float = 5.0) -> Callable:
    r"""
    Generate a Kolmogorov-like power spectrum following ZuHone et al. (2011).

    The spectrum is defined as

    .. math::

        P(k) = A \, k^{-11/3} \,
        \exp\!\left[-\left(\frac{k}{k_0}\right)^2\right]
        \exp\!\left[-\frac{k_1}{k}\right],

    where

    - :math:`A` is the normalization amplitude,
    - :math:`k_0` sets the high-wavenumber (small-scale) cutoff,
    - :math:`k_1` sets the low-wavenumber (large-scale) cutoff.

    Parameters
    ----------
    amplitude : float, optional
        Normalization amplitude :math:`A`. Default is ``1.0``.
    k0 : float, optional
        High-wavenumber cutoff :math:`k_0`. Default is ``1.0``.
    k1 : float, optional
        Low-wavenumber cutoff :math:`k_1`. Default is ``5.0``.

    Returns
    -------
    Callable
        A function ``P(kx, ky, ...)`` that takes arrays of wavenumber
        components
        and returns the ZuHone+11 spectrum evaluated on that grid.

    Examples
    --------

    .. plot::
        :include-source:

        from pisces.math_utils.random_fields import generate_zh11_power_spectrum
        L, n = 10.0, 256
        k_grid = np.fft.fftfreq(n, d=L/n) * 2 * np.pi
        k_mag = np.abs(k_grid)
        ps_func = generate_zh11_power_spectrum(amplitude=1.0, k0=5.0, k1=0.5)
        ps_array = ps_func(k_mag)

        import matplotlib.pyplot as plt
        plt.loglog(k_mag[1:], ps_array[1:])  # skip k=0
        plt.xlabel('Wavenumber k')
        plt.ylabel('Power Spectrum P(k)')
        plt.title('ZuHone+11 Spectrum with Cutoffs')
        plt.show()
    """

    def _ps_func(*args):
        k_mag = np.sqrt(sum(ki**2 for ki in args))

        f1 = amplitude * (np.where(k_mag > 0, k_mag, 1.0)) ** (-11 / 3)
        f2 = np.exp(-((k_mag / k0) ** 2))  # high-k cutoff
        f3 = np.exp(-k1 / np.where(k_mag > 0, k_mag, 1.0))  # low-k cutoff

        P = f1 * f2 * f3
        P[k_mag == 0] = 0.0  # enforce zero at DC mode

        return P

    return _ps_func


class RandomField(ABC):
    r"""
    Abstract base class for defining random fields in arbitrary dimensions.

    This class provides a general framework for representing random fields
    on uniform grids. It encapsulates domain geometry, spatial and Fourier
    space utilities, and a standard interface for generating random field
    realizations.

    Subclasses should implement :meth:`generate`, which constructs a
    particular type of random field (e.g. Gaussian random field with a
    specified power spectrum).

    Parameters
    ----------
    domain_dimensions : numpy.ndarray of int
        Number of grid points along each axis of the domain. Must be a
        1D array of length equal to the spatial dimensionality.
    bounding_box : numpy.ndarray
        The lower and upper bounds of the domain in each dimension. For
        example, ``[[x_min, y_min], [x_max, y_max]]`` in 2D. The shape
        must be ``(2, ndim)``.
    rng : numpy.random.Generator, optional
        Random number generator instance. Defaults to
        ``np.random.default_rng()``.

    Notes
    -----
    - The class provides convenience methods to compute real-space grids
      (:meth:`compute_spatial_mgrid`) and Fourier-space grids
      (:meth:`compute_wavenumber_mgrid`).
    - This base class does not implement the actual random field
      generation logic; see :class:`GaussianRandomField` for an example
      subclass.
    """

    def __init__(self, domain_dimensions: np.ndarray, bounding_box: np.ndarray, **kwargs):
        # --- Validate Domain Inputs --- #
        # We set up the domain dimensions and bounding box here. This
        # requires checking the domain dimensions and bounding box
        # are consistent.
        self._dd = np.asarray(domain_dimensions, dtype=int)
        self._bbox = np.asarray(bounding_box, dtype=float)

        if self._dd.ndim != 1:
            raise ValueError("domain_dimensions must be a 1D array.")
        self._ndim = len(self._dd)

        if self._bbox.shape != (2, self._ndim):
            raise ValueError("bounding_box must have shape (2, ndim).")
        if np.any(self._bbox[1, :] <= self._bbox[0, :]):
            raise ValueError("bounding_box upper bounds must be greater than lower bounds.")

        # Now that the grid has been defined we can derive a few
        # of the relevant parameters that will be used later.
        self._grid_range = self._bbox[1, :] - self._bbox[0, :]
        self._grid_spacing = self._grid_range / self._dd

        # Create the RNG object as well.
        self._rng = kwargs.pop("rng", np.random.default_rng())

    # ========================================= #
    # Properties                                #
    # ========================================= #
    @property
    def dd(self) -> np.ndarray:
        """Domain dimensions of the random field."""
        return self._dd

    @property
    def bbox(self) -> np.ndarray:
        """Bounding box of the random field."""
        return self._bbox

    @property
    def ndim(self) -> int:
        """Number of dimensions of the random field."""
        return self._ndim

    @property
    def grid_range(self) -> np.ndarray:
        """Range of the grid in each dimension."""
        return self._grid_range

    @property
    def grid_spacing(self) -> np.ndarray:
        """Spacing of the grid in each dimension."""
        return self._grid_spacing

    @property
    def dl(self) -> np.ndarray:
        """Alias for :attr:`grid_spacing`."""
        return self._grid_spacing

    # ========================================= #
    # Space Utilities                           #
    # ========================================= #
    def compute_grid_arrays(self):
        """
        Compute the meshgrid of spatial coordinates for the random field.

        Returns
        -------
        np.ndarray
            Meshgrid of spatial coordinates.
        """
        axes = [np.linspace(self._bbox[0, i], self._bbox[1, i], self._dd[i]) for i in range(self._ndim)]
        return tuple(axes)

    def compute_spatial_mgrid(self, **kwargs):
        """
        Compute the meshgrid of spatial coordinates for the random field.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to np.meshgrid.

        Returns
        -------
        np.ndarray
            Meshgrid of spatial coordinates.
        """
        axes = self.compute_grid_arrays()
        return np.meshgrid(*axes, **kwargs)

    # ========================================= #
    # Fourier Space Utilities                   #
    # ========================================= #
    def compute_wavenumber_arrays(self, fft_type: str = "real"):
        """
        Compute the 1D wavenumber arrays for each axis of the domain.

        This method constructs the Fourier-space sampling frequencies along
        each axis of the grid, converting from discrete FFT frequencies
        (in cycles per unit length) to angular wavenumbers (radians per unit length).

        The choice of FFT convention determines whether the last axis uses
        a reduced frequency representation:

        - If ``fft_type='complex'`` (full complex FFT, via ``np.fft.fftn``),
          then all axes use ``np.fft.fftfreq`` and include both positive and
          negative frequencies.

        - If ``fft_type='real'`` (real-to-complex FFT, via ``np.fft.rfftn``),
          then the last axis uses ``np.fft.rfftfreq`` to return only the
          non-negative frequencies, while all earlier axes still use
          ``np.fft.fftfreq``.

        Parameters
        ----------
        fft_type : {'real', 'complex'}, optional
            FFT convention to use. Default is ``'real'``.

        Returns
        -------
        tuple of np.ndarray
            Tuple of 1D arrays, one per axis, giving the discrete angular
            wavenumbers (radians per unit length) along each axis. The
            arrays can be combined with :func:`numpy.meshgrid` (for example
            via :meth:`compute_wavenumber_mgrid`) to construct the full
            multidimensional Fourier grid.

        Notes
        -----
        - The wavenumber values are computed as::

            k = 2 * pi * freq

          where ``freq`` is the output of ``np.fft.fftfreq`` or
          ``np.fft.rfftfreq``.

        - The grid spacing is inferred from the domain dimensions and
          bounding box provided at initialization.
        """
        # Define the buffer for the output
        # arrays.
        output_arrays = []

        # Iterate through each of the axes and
        # pull out the length and count of the grid along
        # that axis so that we can build the K arrays.
        for axid, (n_points, length) in enumerate(zip(self._dd, self._grid_range, strict=False)):
            dx = length / n_points

            # If we are using a real FFT, then we need to
            # use the rfftfreq function for the last axis.
            if fft_type == "real" and axid == self._ndim - 1:
                freq = np.fft.rfftfreq(n_points, d=dx)
            else:
                freq = np.fft.fftfreq(n_points, d=dx)

            # Convert from cycles/length to radians/length
            k = 2 * np.pi * freq
            output_arrays.append(k)

        return tuple(output_arrays)

    def compute_wavenumber_mgrid(self, fft_type: str = "real", **kwargs):
        """
        Compute the meshgrid of wavenumber arrays for the random field.

        Parameters
        ----------
        fft_type : str, optional
            Type of FFT to consider ('real' or 'complex'), by default 'real'.
        **kwargs : dict
            Additional keyword arguments to pass to np.meshgrid.

        Returns
        -------
        np.ndarray
            Meshgrid of wavenumber arrays.
        """
        k_arrays = self.compute_wavenumber_arrays(fft_type=fft_type)
        return np.meshgrid(*k_arrays, indexing="ij", **kwargs)

    # ========================================= #
    # Generation Methods                        #
    # ========================================= #
    @abstractmethod
    def generate(self, *args, **kwargs) -> np.ndarray:
        r"""
        Generate a realization of the random field.

        This method must be implemented by subclasses. It should return a
        field defined on the spatial grid specified by
        :attr:`domain_dimensions` and :attr:`bounding_box`.

        Returns
        -------
        np.ndarray
            Array representing the generated random field. The shape
            should be ``(*domain_dimensions, *field_shape)``, where
            ``field_shape`` is defined by the subclass (e.g. scalar or
            vector field).

        Notes
        -----
        - Implementations may include additional parameters (e.g. power
          spectrum, spatial envelope) to control the statistical
          properties of the field.
        - The output should generally be real-valued, even if Fourier
          methods are used internally.
        """
        pass


class GaussianRandomField(RandomField):
    """Class for generating Gaussian random fields in arbitrary dimensions."""

    # ======================================== #
    # Class Flags                              #
    # ======================================== #
    # Default parameters for power spectrum, envelope, and mean field.
    # These can be overridden at initialization or generation time.
    # They are defined as class attributes so that they can be
    # easily modified for different use cases.
    _default_power_spectrum: Callable = generate_white_noise_power_spectrum()
    _default_envelope_function: Optional[Callable] = None
    _default_mean_field: Optional[Union[float, np.ndarray]] = None

    # ======================================== #
    # Initialization                           #
    # ======================================== #
    def __init__(
        self,
        domain_dimensions: np.ndarray,
        bounding_box: np.ndarray,
        field_shape: Optional[tuple[int, ...]] = None,
        **kwargs,
    ):
        # --- Configure the Field Shape --- #
        # The field shape defines the shape of the field at each
        # grid point. If None, we assume a scalar field.
        if field_shape is None:
            field_shape = ()  # Scalar field
        else:
            field_shape = tuple(field_shape)

        self._field_shape = field_shape

        # --- Manage Defaults --- #
        # We now set up the power spectrum, envelope, and mean field
        self._power_spectrum = kwargs.pop("power_spectrum", self.__class__._default_power_spectrum)
        self._envelope_function = kwargs.pop("envelope_function", self.__class__._default_envelope_function)
        self._mean_field = kwargs.pop("mean_field", self.__class__._default_mean_field)

        # --- Initialize the Base Class --- #
        # This must come after setting the field shape
        # so that any methods called during initialization
        # have access to it.
        super().__init__(domain_dimensions, bounding_box, **kwargs)

    # ========================================= #
    # Properties                                #
    # ========================================= #
    @property
    def field_shape(self) -> tuple[int, ...]:
        """Shape of the field at each grid point."""
        return self._field_shape

    @property
    def power_spectrum(self) -> Callable:
        """Power spectrum function defining the statistical properties of the field."""
        return self._power_spectrum

    @power_spectrum.setter
    def power_spectrum(self, ps_func: Callable):
        """
        Set the power spectrum function.

        Parameters
        ----------
        ps_func : Callable
            Function P(kx, ky, ...,) returning the power spectrum evaluated
            on the Fourier grid. Must accept arrays of shape given by
            :meth:`compute_wavenumber_mgrid`.
        """
        self._power_spectrum = ps_func

    @property
    def envelope_function(self) -> Optional[Callable]:
        """Spatial envelope function modulating the field."""
        return self._envelope_function

    @envelope_function.setter
    def envelope_function(self, env_func: Optional[Callable]):
        """
        Set the spatial envelope function.

        Parameters
        ----------
        env_func : Callable or None
            Function f(x1, x2, ..., ) defining a multiplicative spatial envelope.
            It will be evaluated on the spatial meshgrid and applied to the
            generated field in real space. If None, no modulation is applied.
        """
        self._envelope_function = env_func

    @property
    def mean_field(self) -> float:
        """Mean field value added to the generated field."""
        return self._mean_field

    @mean_field.setter
    def mean_field(self, value: Union[float, np.ndarray]):
        """
        Set the mean field value.

        Parameters
        ----------
        value : float or numpy.ndarray
            Mean field value to add to the generated field.
        """
        self._mean_field = value

    # ========================================= #
    # Generation Methods                        #
    # ========================================= #
    def _generate_fourier_space_field(self, ps_func: Callable) -> np.ndarray:
        """
        Generate Fourier-space random coefficients for the field.

        This constructs complex Gaussian random variables on the Fourier grid
        and scales them by the square root of the target power spectrum.

        Parameters
        ----------
        ps_func : Callable
            Function P(kx, ky, ...,) returning the power spectrum evaluated
            on the Fourier grid. Must accept arrays of shape given by
            :meth:`compute_wavenumber_mgrid`.

        Returns
        -------
        np.ndarray
            Complex-valued Fourier coefficients with shape
            ``(*domain_dimensions, *field_shape)``.
        """
        # Configure the grid and determine the shape of the fourier-space
        # field. This is generally different from the real-space field shape
        # because of the reduced representation along the last axis.
        _wn_arrays = self.compute_wavenumber_arrays(fft_type="real")
        _wn_meshgrid = np.meshgrid(*_wn_arrays, indexing="ij")
        grid_shape = tuple(len(arr) for arr in _wn_arrays)

        # Generate the real and imaginary parts of the Fourier coefficients.
        # This is then combined into a complex array.
        re = self._rng.normal(size=grid_shape + self._field_shape)
        im = self._rng.normal(size=grid_shape + self._field_shape)
        coeffs = (re + 1j * im) / np.sqrt(2.0)  # variance = 1 per complex DOF

        # With the wavenumber grid in hand, we can now compute the
        # relevant power spectrum array. We ignore errors because they
        # should just get mapped over for k=0 modes.
        with np.errstate(invalid="ignore", divide="ignore"):
            _power_spectrum = ps_func(*_wn_meshgrid)  # shape ~ (*domain_dimensions,)

        _power_spectrum = np.reshape(_power_spectrum, grid_shape + (1,) * len(self._field_shape))

        # Define the complex output field and then correct for the
        # DC and Nyquist modes to ensure a real-valued field when we
        # invert the FFT.
        _complex_output_field = coeffs * np.sqrt(_power_spectrum)

        # Correct the DC mode at k = 0.
        _complex_output_field[(0,) * self._ndim] = 0

        # Correct the Nyquist modes if they exist.
        if self._dd[-1] % 2 == 0:  # Nyquist plane in last axis
            slicer = (slice(None),) * (self._ndim - 1) + (self._dd[-1] // 2,)
            _complex_output_field[slicer] = _complex_output_field[slicer].real

        return _complex_output_field

    def _generate_real_field(self, ps_func: Callable) -> np.ndarray:
        """
        Generate a real-space random field with the specified power spectrum.

        This method constructs Fourier-space coefficients scaled by the
        target power spectrum, then performs an inverse FFT to return a
        real-valued field on the spatial grid.

        Parameters
        ----------
        ps_func : Callable
            Function P(kx, ky, ...,) returning the power spectrum evaluated
            on the Fourier grid.

        Returns
        -------
        np.ndarray
            Real-valued random field with shape ``(*domain_dimensions, *field_shape)``.
        """
        from numpy.fft import irfftn

        # Step 1: Fourier coefficients with power spectrum scaling
        coeffs = self._generate_fourier_space_field(ps_func)

        # Step 2: Inverse FFT back to real space
        field = irfftn(coeffs, s=self._dd, axes=range(self._ndim), norm="ortho")

        # Step 3: Real part (imag should be numerical noise ~1e-15)
        field = np.real(field)

        return field

    def generate(self) -> np.ndarray:
        """
        Generate a real-space Gaussian random field with the specified power spectrum.

        Returns
        -------
        np.ndarray
            Real-valued random field with shape
            ``(*domain_dimensions, *field_shape)``.
        """
        # Generate the real random field for the given power spectrum.
        field = self._generate_real_field(ps_func=self.power_spectrum)

        # Correct for the envelope if it is asked for.
        if self.envelope_function is not None:
            # Calculate the RMS value of the field so that we can
            # rescale as desired to set the envelope.
            field_rms = np.sqrt(np.mean(field**2))

            # Compute the spatial coordinate grid and
            # evaluate the envelope function on it.
            coords = self.compute_spatial_mgrid(indexing="ij")
            env = self.envelope_function(*coords)

            # broadcast to field_shape if needed
            if self._field_shape:
                env = env[(...,) + (None,) * len(self._field_shape)]
            field *= env / field_rms

        # Add the mean field offset if it is
        # specified.
        if self.mean_field is not None:
            field += self.mean_field

        return field


class GaussianRandomVectorField(GaussianRandomField):
    """Class for generating Gaussian random vector fields in arbitrary dimensions."""

    # ======================================== #
    # Initialization                           #
    # ======================================== #
    def __init__(self, domain_dimensions: np.ndarray, bounding_box: np.ndarray, **kwargs):
        # Initialize the base class with the appropriate
        # field shape.
        domain_dimensions = np.asarray(domain_dimensions)
        super().__init__(domain_dimensions, bounding_box, field_shape=(len(domain_dimensions),), **kwargs)

    # ========================================= #
    # Generation Methods                        #
    # ========================================= #
    def _generate_fourier_space_field(
        self, ps_func: Callable, divergence_cleaning: Union[str, bool] = False
    ) -> np.ndarray:
        """
        Generate Fourier-space random coefficients for the field.

        This constructs complex Gaussian random variables on the Fourier grid
        and scales them by the square root of the target power spectrum.

        Parameters
        ----------
        ps_func : Callable
            Function P(kx, ky, ...,) returning the power spectrum evaluated
            on the Fourier grid. Must accept arrays of shape given by
            :meth:`compute_wavenumber_mgrid`.
        divergence_cleaning: {'discrete', 'continuous', False}, optional
            If 'discrete' or 'continuous', perform divergence cleaning
            in Fourier space to ensure the field is divergence-free.
            If False or 'off', no divergence cleaning is applied.
            Default is 'discrete'.

        Returns
        -------
        np.ndarray
            Complex-valued Fourier coefficients with shape
            ``(*domain_dimensions, *field_shape)``.
        """
        # Configure the grid and determine the shape of the fourier-space
        # field. This is generally different from the real-space field shape
        # because of the reduced representation along the last axis.
        _wn_arrays = self.compute_wavenumber_arrays(fft_type="real")
        _wn_meshgrid = np.meshgrid(*_wn_arrays, indexing="ij")
        grid_shape = tuple(len(arr) for arr in _wn_arrays)

        # Generate the real and imaginary parts of the Fourier coefficients.
        # This is then combined into a complex array.
        re = self._rng.normal(size=grid_shape + self._field_shape)
        im = self._rng.normal(size=grid_shape + self._field_shape)
        coeffs = (re + 1j * im) / np.sqrt(2.0)  # variance = 1 per complex DOF

        # With the wavenumber grid in hand, we can now compute the
        # relevant power spectrum array. We ignore errors because they
        # should just get mapped over for k=0 modes.
        with np.errstate(invalid="ignore", divide="ignore"):
            _power_spectrum = ps_func(*_wn_meshgrid)  # shape ~ (*domain_dimensions,)

        _power_spectrum = np.reshape(_power_spectrum, grid_shape + (1,) * len(self._field_shape))

        # Define the complex output field and then correct for the
        # DC and Nyquist modes to ensure a real-valued field when we
        # invert the FFT.
        _complex_output_field = coeffs * np.sqrt(_power_spectrum)

        # Perform first round divergence cleaning if requested so
        # that we can ensure the field is divergence-free.
        if divergence_cleaning not in ["off", False, None]:
            _complex_output_field = self._divergence_cleaning(_complex_output_field, mode=divergence_cleaning)

        # Correct the DC mode at k = 0.
        _complex_output_field[(0,) * self._ndim] = 0

        # Correct the Nyquist modes if they exist.
        if self._dd[-1] % 2 == 0:  # Nyquist plane in last axis
            slicer = (slice(None),) * (self._ndim - 1) + (self._dd[-1] // 2,)
            _complex_output_field[slicer] = _complex_output_field[slicer].real

        return _complex_output_field

    def _generate_real_field(self, ps_func: Callable, divergence_cleaning: Union[str, bool] = False) -> np.ndarray:
        """
        Generate a real-space random field with the specified power spectrum.

        This method constructs Fourier-space coefficients scaled by the
        target power spectrum, then performs an inverse FFT to return a
        real-valued field on the spatial grid.

        Parameters
        ----------
        ps_func : Callable
            Function P(kx, ky, ...,) returning the power spectrum evaluated
            on the Fourier grid.
        divergence_cleaning: {'discrete', 'continuous', False}, optional
            If 'discrete' or 'continuous', perform divergence cleaning
            in Fourier space to ensure the field is divergence-free.
            If False or 'off', no divergence cleaning is applied.
            Default is 'discrete'.

        Returns
        -------
        np.ndarray
            Real-valued random field with shape ``(*domain_dimensions, *field_shape)``.
        """
        from numpy.fft import irfftn

        # Step 1: Fourier coefficients with power spectrum scaling
        coeffs = self._generate_fourier_space_field(ps_func, divergence_cleaning=divergence_cleaning)

        # Step 2: Inverse FFT back to real space
        field = irfftn(coeffs, s=self._dd, axes=range(self._ndim), norm="ortho")

        # Step 3: Real part (imag should be numerical noise ~1e-15)
        field = np.real(field)

        return field

    def _divergence_cleaning(self, fourier_field, mode: str = "discrete"):
        # Check that the divergence mode is valid. Otherwise, we
        # need to raise an error.
        if mode not in ("discrete", "continuous"):
            raise ValueError(f"Invalid mode '{mode}'. Use 'discrete' or 'continuous'.")

        # Distribute by dimension and by mode.
        if self.ndim == 2:
            kx, ky = self.compute_wavenumber_mgrid(fft_type="real")
            if mode == "discrete":
                div_clean_fd_2d(fourier_field[..., 0], fourier_field[..., 1], kx, ky, self.dl)
            else:
                div_clean_c_2d(fourier_field[..., 0], fourier_field[..., 1], kx, ky)

        elif self.ndim == 3:
            kx, ky, kz = self.compute_wavenumber_mgrid(fft_type="real")
            if mode == "discrete":
                div_clean_fd_3d(
                    fourier_field[..., 0], fourier_field[..., 1], fourier_field[..., 2], kx, ky, kz, self.dl
                )
            else:
                div_clean_c_3d(fourier_field[..., 0], fourier_field[..., 1], fourier_field[..., 2], kx, ky, kz)

        else:
            raise NotImplementedError(f"Divergence cleaning not implemented for ndim={self.ndim}.")

        return fourier_field

    def generate(
        self,
        divergence_cleaning: Union[str, bool] = False,
    ) -> np.ndarray:
        """
        Generate a real-space Gaussian random vector field.

        Parameters
        ----------
        divergence_cleaning: {'discrete', 'continuous', False}, optional
            If 'discrete' or 'continuous', perform divergence cleaning
            in Fourier space to ensure the field is divergence-free.
            If False or 'off', no divergence cleaning is applied.
            Default is 'discrete'.

        Returns
        -------
        np.ndarray
            Real-valued random vector field with shape
            ``(*domain_dimensions, ndim)``.
        """
        from numpy.fft import irfftn, rfftn

        # Generate the real random field for the given power spectrum.
        field = self._generate_real_field(ps_func=self.power_spectrum, divergence_cleaning=divergence_cleaning)

        # Correct for the envelope if it is asked for.
        if self.envelope_function is not None:
            # Calculate the RMS value of the field so that we can
            # rescale as desired to set the envelope.
            field_rms = np.sqrt(np.mean(field**2))

            # Compute the spatial coordinate grid and
            # evaluate the envelope function on it.
            coords = self.compute_spatial_mgrid(indexing="ij")
            env = self.envelope_function(*coords)

            # broadcast to field_shape if needed
            if self._field_shape:
                env = env[(...,) + (None,) * len(self._field_shape)]
            field *= env / field_rms

        # Add the mean field offset if it is
        # specified.
        if self.mean_field is not None:
            field += self.mean_field

        # We now perform the second cleaning. To do this, we'll calculate
        # the RMS of the field as is, perform the cleaning again, and then
        # rescale the field to restore the original RMS.
        if divergence_cleaning not in ["off", False, None]:
            # Calculate the RMS of the field before cleaning.
            field_rms = np.sqrt(np.mean(field**2))

            # Now transform the field back into its fourier components
            fourier_field = rfftn(field, axes=range(self._ndim), norm="ortho")
            fourier_field = self._divergence_cleaning(fourier_field, mode=divergence_cleaning)
            field = np.real(irfftn(fourier_field, s=self._dd, axes=range(self._ndim), norm="ortho"))

            # Rescale to restore the original RMS.
            new_rms = np.sqrt(np.mean(field**2))
            field *= field_rms / new_rms

        return field


# ========================================= #
# Physical Constructors                     #
# ========================================= #
def generate_spherical_magnetic_field(
    domain_dimensions: np.ndarray,
    r: unyt.unyt_array,
    mag_field_rms: unyt.unyt_array,
    power_spectrum: Callable,
    divergence_cleaning: str = "discrete",
) -> unyt.unyt_array:
    r"""
    Generate a random magnetic field with a spherically specified RMS profile.

    This function takes an RMS magnetic field profile :math:`B_{\rm rms}(r)` defined at
    discrete radii and constructs a 3D Gaussian random vector field whose RMS
    matches the specified profile. The field is generated within a cubic domain
    centered at the origin, with side length determined by the maximum radius
    in the input profile.

    In order to generate the random field, a Gaussian random field (GRF) generator
    is used, with the specified power spectrum function :math:`P(k)`. The GRF is
    modulated by an envelope function that scales the field amplitude according
    to match the desired RMS profile as a function of radius. Divergence cleaning
    is applied to ensure that the field has no variance.

    Parameters
    ----------
    domain_dimensions : array-like of int
        The number of cells to place in the grid along each axis ``(Nx, Ny, ...)``.
        The value of the RMS magnetic field is evaluated at each point by interpolation. Also
        determines the number of dimensions of the domain (e.g., 3 for a 3D field).
    r : unyt.unyt_array
        The spherical radii at which the RMS magnetic field values are defined. These
        should have physical length units (e.g., 'kpc' or 'Mpc') and match the length of
        `mag_field_rms`. Additionally, we require that ``r`` be strictly increasing. The maximum
        value of ``r`` defines the size of the cubic domain.
    mag_field_rms : unyt.unyt_array
        The RMS magnetic field profile defined at the radii in `r`. These should have
        physical magnetic field units (e.g., 'μG' or 'G') and match the length of `r`.
    power_spectrum : Callable
        The power spectrum function :math:`P(k_1,k_2,\\ldots,k_N)` defining the statistical properties
        of the Gaussian random field. This function should accept arrays of wavenumber
        components and return the power spectrum evaluated on that grid. In most cases,
        this will be a function of the wavenumber magnitude only; however, it must still accept
        each of the components in keeping with the convention for specifying power spectra in the GRF generator.
    divergence_cleaning : {'discrete', 'continuous', False}, optional
        Divergence cleaning mode passed to the GRF generator.
        Default is 'discrete'.

    Returns
    -------
    list of numpy.npdarray
        A list of 1D arrays representing the grid coordinates along each axis.
    unyt.unyt_array
        A 4D array of shape ``(Nx, Ny, Nz, N)`` containing the magnetic
        field vectors with physical magnetic field units.
    """
    # Validate the inputs and ensure that everything has valid types.
    r, mag_field_rms = np.atleast_1d(r), np.atleast_1d(mag_field_rms)
    if not isinstance(r, unyt.unyt_array):
        raise ValueError(f"`r` must be an unyt array, not {type(r).__name__}.")
    if not isinstance(mag_field_rms, unyt.unyt_array):
        raise ValueError(f"`mag_field_rms` must be an unyt array, not {type(mag_field_rms).__name__}.")

    # Ensure that these are 1D arrays of matching length and that r is increasing. We'll also
    # ensure the units are reasonable.
    if (r.ndim != 1) or (mag_field_rms.ndim != 1):
        raise ValueError("Inputs `r` and `mag_field_rms` must be 1D arrays.")
    if r.units.dimensions != unyt.dimensions.length:
        raise ValueError(f"Input `r` must have physical length units (e.g., kpc, Mpc). Received units: {r.units}")
    if mag_field_rms.units.dimensions != unyt.dimensions.magnetic_field_cgs:
        raise ValueError(
            f"Input `mag_field_rms` must have magnetic field units (e.g., μG, G). Received units: {mag_field_rms.units}"
        )
    if np.any(np.diff(r) < 0):
        raise ValueError("Radii `r` must be strictly increasing.")
    if np.any(mag_field_rms < 0):
        raise ValueError("RMS magnetic field values must be non-negative.")

    # Now build the interpolation of the BRMS function so that we can
    # evaluate it at arbitrary radii.
    rarr, barr = r.to_value(), mag_field_rms.to_value()

    # We'll assume a cubic domain centered at 0 with edge 2 * r_max
    bounding_box = [3 * [-np.max(rarr)], 3 * [np.max(rarr)]]

    # Create the interpolator for B_rms(r)
    b_interpolator = InterpolatedUnivariateSpline(rarr, barr)

    # Now define the envelope function that will be used to
    # scale the GRF to have the desired RMS profile.
    def _envelope_function(*args):
        R = np.sqrt(sum(arg**2 for arg in args))
        return b_interpolator(R)

    # Generate the gaussian random field.
    grf = GaussianRandomVectorField(
        domain_dimensions=domain_dimensions,
        bounding_box=bounding_box,
        power_spectrum=power_spectrum,
        envelope_function=_envelope_function,
    )
    B_field = grf.generate(divergence_cleaning=divergence_cleaning)

    # ------------------------
    # 5. Reapply physical units
    # ------------------------
    B_field = unyt.unyt_array(B_field, mag_field_rms.units)

    return grf.compute_grid_arrays(), B_field
