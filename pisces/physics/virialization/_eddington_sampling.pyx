#cython: language_level=3, boundscheck=False
"""
Cython functions for optimized sampling of particle velocities from an
ergodic / eddington distribution function.
"""
import numpy as np
cimport cython
cimport numpy as cnp
from scipy.interpolate import dfitpack
from tqdm.auto import tqdm
from libc.stdlib cimport free, malloc

# C-Level imports

cdef extern from "math.h":
    double sqrt(double x) nogil
    double log10(double x) nogil
    double fmod(double number, double denom) nogil
    double sin(double x) nogil

cdef extern from "stdlib.h":
    double drand48() nogil
    void srand48(long int seedval) nogil

# Type definitions

DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

CTYPE = np.complex128
ctypedef cnp.complex128_t CTYPE_t


# -------------------------------------------- #
# Pure-C Layer                                 #
# ------------                                 #
# This layer defines the entire velocity       #
# generator at the C-level using access to the #
# C-level numpy spline API.                    #
# -------------------------------------------- #
cdef unsigned int _generate_velocities(
        cnp.ndarray[DTYPE_t, ndim=1] relpot,
        cnp.ndarray[DTYPE_t, ndim=1] velesc,
        cnp.ndarray[DTYPE_t, ndim=1] likelihood_max,
        DTYPE_t[:] velocity_result_buffer,
        cnp.ndarray[DTYPE_t, ndim=1] t,
        cnp.ndarray[DTYPE_t, ndim=1] c,
        int k,
        bint show_progress = True,
        unsigned long max_tries = 100_000,
        ):
    """
    Generate particle velocities for a sample of particles subject to
    an Eddington distribution function.

    Parameters
    ----------
    relpot : np.ndarray of float, shape (N,)
        Array containing the relative potential of each of the sampled particles. The relative potential
        is `-phi + phi_0`. These should be obtained for each particle from the specific model being used.
    velesc: np.ndarray of float, shape (N,)
        Array containing the escape velocities for each particle. These should be computed for each particle ahead
        of time using the position of each of the particles.
    likelihood_max : np.ndarray of float, shape (N,)
        Precomputed maximum on the f*v^2 product for each particle. This is then used as the sampling bound
        on the rejection-sampler to ensure efficiency. The `likelihood_max` may be generated in any number of ways
        and has an impact on the efficiency of the sampling. The simplest prescription is to simply set it to
        f_max * v_esc^2, where f_max is the maximum value of the distribution function across all energies.
    velocity_result_buffer : 1D buffer (memoryview or NumPy array), shape (N,)
        The output array into which sampled velocities will be written.
    t : np.ndarray of float
        Knot positions for the spline (required by `dfitpack.splev`).
    c : np.ndarray of float
        Spline coefficients for the distribution function (required by `dfitpack.splev`).
    k : int
        Spline degree for the distribution function.
    show_progress : bool, default True
        If True, a progress bar is shown (using `tqdm`).
    max_tries : unsigned long, default 100_000
        Maximum number of attempts to generate a valid velocity for each particle before
        returning an error code.

    Returns
    -------
    int
        A status code: returns 0 on success if all particle velocities are generated
        within `max_tries`, otherwise returns 1 if any particle exceeds `max_tries`.

    Notes
    -----
    The eddington distribution function is some f(E), where E is the relative energy of the particle. For a given
    proposed speed v, the relative energy is relative_potential - 0.5*v^2, and the distribution function may then
    be evaluated to determine the likelihood.

    This function samples velocities between 0 and the escape velocity for each particle and then
    checks if f(relative_potential - 0.5*v^2) * v^2 is less than a random number uniformly sampled from [0, likelihood_max].
    If it is, the velocity is accepted; otherwise, a new velocity is sampled until either a valid one is found or
    `max_tries` is exceeded for that particle.

    The `likelihood_max` may be generated in any number of ways and has an impact on the efficiency of the sampling. The
    simplest prescription is to simply set it to f_max * v_esc^2, where f_max is the maximum value of the distribution function
    across all energies.

    """
    cdef Py_ssize_t num_particles = relpot.shape[0]            # The number of particles being sampled.
    cdef DTYPE_t v2, f                                         # The square velocity and DF value.
    cdef double * _e_ref = <double *> malloc(sizeof(double))   # Allocate memory for the relative energy given a proposal.
    cdef double[:] _e = <double[:1]> _e_ref                    # Create a memoryview for the relative energy.
    cdef unsigned long n_tries = 0                              # The number of tries for the current particle.
    cdef Py_ssize_t i = 0                                      # Loop index for the particles.


    # Create the progress bar.
    #
    # By default, we don't include a progress bar, but if `show_progress` is True,
    # we create a tqdm progress bar that will show the progress of the velocity generation.
    cdef object pbar = None
    if show_progress:
        pbar = tqdm(leave=False,
                    total=num_particles,
                    desc="Generating particle velocities...",
                    disable= not show_progress)

    # Begin the rejection sampling loop. For each particle `i`, we will attempt
    # the sampling procedure up to `max_tries` times. If we fail to sample a valid
    # velocity within that many tries, we will return an error code.
    for i in range(num_particles):
        n_tries = 0 # Set the trial counter to zero for each particle.

        # Start the acceptance-rejection sampling loop.
        while n_tries < max_tries:
            # Iterate the number of tries for this particle.
            n_tries += 1

            # Sample the velocity up to the escape velocity for
            # this particle.
            v2 = drand48()*velesc[i]
            v2 *= v2

            # Compute the relative energy and compute the value of the
            # distribution function for the relative energy. This relies on
            # the dfitpack C-level interface into scipy.
            _e[0] = relpot[i]-0.5*v2
            f = dfitpack.splev(t,c, k, _e, 0)[0][0]

            # Check if we can exit.
            if f*v2 < drand48()*likelihood_max[i]:
                break

        # Check if we ran out of trials for this particle. If we did, the
        # error code needs to be returned.
        if n_tries > max_tries:
            return 1

        # Write the particle data into the buffer before
        # proceeding to the next particle in the list.
        velocity_result_buffer[i] = sqrt(v2)
        if show_progress:
            pbar.update()

    # Close the progress bar.
    if show_progress:
        pbar.close()

    # Free memory
    free(_e_ref)

    # return
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_velocities(
    cnp.ndarray[DTYPE_t, ndim=1] relpot,
    cnp.ndarray[DTYPE_t, ndim=1] velesc,
    cnp.ndarray[DTYPE_t, ndim=1] likelihood_max,
    cnp.ndarray[DTYPE_t, ndim=1] t,
    cnp.ndarray[DTYPE_t, ndim=1] c,
    int k,
    bint show_progress=True,
    unsigned long max_tries=100000
):
    """
    Python-accessible wrapper around the Cython-level generate_velocities function.

    Parameters
    ----------
    relpot : np.ndarray of float, shape (N,)
        Relative potential for each particle.
    velesc : np.ndarray of float, shape (N,)
        Escape velocity for each particle.
    likelihood_max : np.ndarray of float, shape (N,)
        Maximum of f*v^2 for each particle.
    t : np.ndarray of float
        Knot positions for the spline (required by dfitpack.splev).
    c : np.ndarray of float
        Spline coefficients for the distribution function.
    k : int
        Spline degree for the distribution function.
    show_progress : bool, default True
        Whether to show a progress bar (tqdm).
    max_tries : unsigned long, default 100000
        Maximum number of attempts to sample a velocity for each particle
        before throwing an error.

    Returns
    -------
    velocity : np.ndarray of float, shape (N,)
        Generated velocities for each particle.

    Raises
    ------
    ValueError
        If the velocity generation fails for any particle within the
        allowed number of attempts.
    """
    cdef Py_ssize_t num_particles = relpot.shape[0]

    # Allocate a NumPy array for the velocity results.
    # We'll pass its data pointer into the lower-level function.
    cdef cnp.ndarray[DTYPE_t, ndim=1] velocity = np.zeros(num_particles, dtype=np.float64)

    # Call the lower-level function.
    cdef int status = _generate_velocities(
        relpot,
        velesc,
        likelihood_max,
        velocity,      # pass in velocity buffer
        t,
        c,
        k,
        show_progress,
        max_tries
    )

    # Interpret the status code:
    # 0 -> success; 1 -> too many tries
    if status != 0:
        raise ValueError(
            f"Failed to generate velocities for one or more particles "
            f"within {max_tries} attempts."
        )

    return velocity
