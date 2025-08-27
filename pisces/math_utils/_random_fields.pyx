#cython: language_level=3, boundscheck=False
"""
Cython utilities for random field generation.

This module provides optimized functions for manipulating
random fields, particularly removing non-zero divergence from
Gaussian random vector fields.

This module was adapted from the `cluster_generator` code written
by John ZuHone.
"""

import numpy as np

cimport cython
cimport numpy as cnp


cdef extern from "math.h":
    double sqrt(double x) nogil
    double log10(double x) nogil
    double fmod(double number, double denom) nogil
    double sin(double x) nogil

cdef extern from "stdlib.h":
    double drand48() nogil
    void srand48(long int seedval) nogil

DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

CTYPE = np.complex128
ctypedef cnp.complex128_t CTYPE_t

# -------------------------------------------- #
# Divergence Cleaning Low-Level Functions      #
# -------------------------------------------- #
# These functions are all efficient Cython implementations of
# the divergence cleaning algorithm in various dimensions. We have
# two classes:
#
# div_clean_fd_{N}d: Uses the finite difference form of the divergence.
# div_clean_c_{N}d: Uses the continuous (Fourier) form of the divergence.
#
# Below these, there are Python-level wrappers that handle
# input validation and type-casting.
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def div_clean_fd_2d(cnp.ndarray[CTYPE_t, ndim=2] gx,
                    cnp.ndarray[CTYPE_t, ndim=2] gy,
                    cnp.ndarray[DTYPE_t, ndim=2] kx,
                    cnp.ndarray[DTYPE_t, ndim=2] ky,
                    cnp.ndarray[DTYPE_t, ndim=1] deltas):
    """
    2D divergence cleaning using the finite-difference form
    of the divergence operator.

    Parameters
    ----------
    gx, gy : np.ndarray[complex, ndim=2]
        Fourier coefficients of the vector field components.
    kx, ky : np.ndarray[float, ndim=2]
        Fourier-space wavenumber grids for x and y directions.
    deltas : np.ndarray[float, ndim=1]
        Real-space grid spacings (dx, dy).
    """
    cdef int i, j
    cdef int nx, ny
    cdef DTYPE_t kxd, kyd, kkd
    cdef CTYPE_t ggx, ggy, kg

    nx = gx.shape[0]
    ny = gx.shape[1]

    # Iterate through all of the elements in the
    # Fourier-space grid so that we can perform the
    # projection operation.
    for i in range(nx):
        for j in range(ny):
            ggx = gx[i, j]
            ggy = gy[i, j]

            # effective FD wavevector components
            kxd = sin(kx[i, j] * deltas[0]) / deltas[0]
            kyd = sin(ky[i, j] * deltas[1]) / deltas[1]

            kkd = sqrt(kxd * kxd + kyd * kyd)
            if kkd > 0:
                kxd /= kkd
                kyd /= kkd

                # project onto plane orthogonal to k
                kg = kxd * ggx + kyd * ggy
                gx[i, j] = ggx - kxd * kg
                gy[i, j] = ggy - kyd * kg

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def div_clean_c_2d(cnp.ndarray[CTYPE_t, ndim=2] gx,
                   cnp.ndarray[CTYPE_t, ndim=2] gy,
                   cnp.ndarray[DTYPE_t, ndim=2] kx,
                   cnp.ndarray[DTYPE_t, ndim=2] ky):
    """
    2D divergence cleaning using the *continuous Fourier* form
    of the divergence operator.

    Parameters
    ----------
    gx, gy : np.ndarray[complex, ndim=2]
        Fourier coefficients of the vector field components.
    kx, ky : np.ndarray[float, ndim=2]
        Continuous Fourier-space wavenumber grids for x and y directions.
    """
    cdef int i, j
    cdef int nx, ny
    cdef DTYPE_t kxd, kyd, kkd
    cdef CTYPE_t ggx, ggy, kg

    nx = gx.shape[0]
    ny = gx.shape[1]

    # Iterate through all Fourier-space grid points
    for i in range(nx):
        for j in range(ny):
            ggx = gx[i, j]
            ggy = gy[i, j]

            # continuous Fourier wavevector components
            kxd = kx[i, j]
            kyd = ky[i, j]

            kkd = sqrt(kxd * kxd + kyd * kyd)
            if kkd > 0:
                kxd /= kkd
                kyd /= kkd

                # project onto plane orthogonal to k
                kg = kxd * ggx + kyd * ggy
                gx[i, j] = ggx - kxd * kg
                gy[i, j] = ggy - kyd * kg



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def div_clean_fd_3d(cnp.ndarray[CTYPE_t, ndim=3] gx,
                    cnp.ndarray[CTYPE_t, ndim=3] gy,
                    cnp.ndarray[CTYPE_t, ndim=3] gz,
                    cnp.ndarray[DTYPE_t, ndim=3] kx,
                    cnp.ndarray[DTYPE_t, ndim=3] ky,
                    cnp.ndarray[DTYPE_t, ndim=3] kz,
                    cnp.ndarray[DTYPE_t, ndim=1] deltas):
    """
    3D divergence cleaning using the finite-difference form
    of the divergence operator.

    This routine enforces the discrete solenoidal constraint
    (∇·B = 0) in Fourier space by projecting each Fourier
    coefficient of the vector field onto the subspace orthogonal
    to the effective finite-difference wavevector. The effective
    wavenumber components are defined as

    .. math::

        k_{\\mathrm{eff}, i} = \\frac{\\sin(k_i \\Delta_i)}{\\Delta_i},

    where :math:`k_i` is the continuous Fourier wavenumber and
    :math:`\\Delta_i` is the grid spacing in the :math:`i`th direction.
    This ensures consistency with the finite-difference stencil
    used in real-space divergence operators.

    Parameters
    ----------
    gx, gy, gz : np.ndarray[complex, ndim=3]
        Fourier coefficients of the three vector field components.
    kx, ky, kz : np.ndarray[float, ndim=3]
        Fourier-space wavenumber grids in x, y, and z.
    deltas : np.ndarray[float, ndim=1]
        Real-space grid spacings (dx, dy, dz).

    """
    cdef int i, j, k
    cdef int nx, ny, nz
    cdef DTYPE_t kxd, kyd, kzd, kkd
    cdef CTYPE_t ggx, ggy, ggz, kg

    nx = gx.shape[0]
    ny = gx.shape[1]
    nz = gx.shape[2]

    # These k's are different because we are
    # using the finite difference form of the
    # divergence operator.
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                ggx = gx[i,j,k]
                ggy = gy[i,j,k]
                ggz = gz[i,j,k]
                kxd = sin(kx[i,j,k] * deltas[0]) / deltas[0]
                kyd = sin(ky[i,j,k] * deltas[1]) / deltas[1]
                kzd = sin(kz[i,j,k] * deltas[2]) / deltas[2]
                kkd = sqrt(kxd*kxd + kyd*kyd + kzd*kzd)
                if kkd > 0:
                    kxd /= kkd
                    kyd /= kkd
                    kzd /= kkd
                kg = kxd * ggx + kyd * ggy + kzd * ggz
                gx[i,j,k] = ggx - kxd * kg
                gy[i,j,k] = ggy - kyd * kg
                gz[i,j,k] = ggz - kzd * kg

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def div_clean_c_3d(cnp.ndarray[CTYPE_t, ndim=3] gx,
                   cnp.ndarray[CTYPE_t, ndim=3] gy,
                   cnp.ndarray[CTYPE_t, ndim=3] gz,
                   cnp.ndarray[DTYPE_t, ndim=3] kx,
                   cnp.ndarray[DTYPE_t, ndim=3] ky,
                   cnp.ndarray[DTYPE_t, ndim=3] kz):
    """
    3D divergence cleaning using the *continuous Fourier* form
    of the divergence operator.

    This routine enforces the solenoidal constraint (∇·B = 0)
    in Fourier space by projecting each Fourier coefficient of
    the vector field onto the subspace orthogonal to the wavevector
    :math:`\\mathbf{k}`. The projection is defined as

    .. math::

        \\tilde{\\mathbf{g}}' = \\tilde{\\mathbf{g}}
        - (\\hat{\\mathbf{k}} \\cdot \\tilde{\\mathbf{g}}) \\, \\hat{\\mathbf{k}},

    where :math:`\\hat{\\mathbf{k}} = \\mathbf{k}/|\\mathbf{k}|` is the
    normalized continuous Fourier wavevector.

    Parameters
    ----------
    gx, gy, gz : np.ndarray[complex, ndim=3]
        Fourier coefficients of the three vector field components.
    kx, ky, kz : np.ndarray[float, ndim=3]
        Continuous Fourier-space wavenumber grids in x, y, and z.

    Notes
    -----
    - Unlike :func:`div_clean_fd_3d`, this version uses the raw Fourier
      wavenumbers without finite-difference corrections.
    - The DC mode (:math:`k = 0`) is left unchanged.
    """
    cdef int i, j, k
    cdef int nx, ny, nz
    cdef DTYPE_t kxd, kyd, kzd, kkd
    cdef CTYPE_t ggx, ggy, ggz, kg

    nx = gx.shape[0]
    ny = gx.shape[1]
    nz = gx.shape[2]

    # Iterate through all Fourier-space grid points
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                ggx = gx[i, j, k]
                ggy = gy[i, j, k]
                ggz = gz[i, j, k]

                # continuous Fourier wavevector components
                kxd = kx[i, j, k]
                kyd = ky[i, j, k]
                kzd = kz[i, j, k]

                kkd = sqrt(kxd * kxd + kyd * kyd + kzd * kzd)
                if kkd > 0:
                    kxd /= kkd
                    kyd /= kkd
                    kzd /= kkd

                    # project onto plane orthogonal to k
                    kg = kxd * ggx + kyd * ggy + kzd * ggz
                    gx[i, j, k] = ggx - kxd * kg
                    gy[i, j, k] = ggy - kyd * kg
                    gz[i, j, k] = ggz - kzd * kg
