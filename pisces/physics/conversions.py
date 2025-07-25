"""Conversion functions for physical quantities.

These functions provide conversions between different physical quantities
such as mean molecular weights, which are commonly used in astrophysical calculations.

"""


def compute_mean_molecular_weight(hydrogen_fraction: float):
    r"""Compute the mean molecular weight assuming all non-hydrogen species are Helium.

    Parameters
    ----------
    hydrogen_fraction: float
        The relevant hydrogen fraction (:math:`\chi_H<1`).

    Returns
    -------
    float
        The mean molecular weight.

    Notes
    -----
    The mean molecular weight is the mass per particle in a fluid. Thus,

    .. math::

        \mu = \frac{\sum_k n_k m_k + (n_{k,e^-} m_{e})}{\sum_k n_k + n_{k,e^-}} \approx
         \frac{\sum_k n_k m_k}{\sum_k n_k + n_{k,e^-}},

    where :math:`k` denotes each species. In the most typical case where hydrogen and helium dominate the
    calculation, we have
    1 proton and 1 electron from the hydrogen and 1 He nucleus and 2 electrons for the helium. Thus, we have

    .. math::

        \mu = \frac{n_{\rm H} + 4n_{\rm He}}{2n_{\rm H} + 3n_{\rm He}}.

    If all non-hydrogen species are Helium, then :math:`n_{\rm He} = N(1-\chi_H)/4`, so

    .. math::

        \mu = \frac{1}{2\chi_H + (3/4)(1-\chi_H)}.

    """
    return 1.0 / (2.0 * hydrogen_fraction + 0.75 * (1.0 - hydrogen_fraction))


def compute_mean_molecular_weight_per_electron(hydrogen_fraction: float):
    r"""Compute the mean molecular weight (:math:`\mu`) per free electron for a given primordial Hydrogen fraction.

    Parameters
    ----------
    hydrogen_fraction: float
        The relevant hydrogen fraction (:math:`\chi_H<1`).

    Returns
    -------
    float
        The mean molecular weight.

    Notes
    -----
    The mean molecular weight per electron is

    .. math::

        \mu = \frac{\sum_k n_k m_k + (n_{k,e^-} m_{e})}{n_{k,e^-}} \approx \frac{\sum_k n_k m_k}{n_{k,e^-}},

    where :math:`k` denotes each species. In the most typical case where hydrogen and helium dominate the
    calculation, we have
    1 proton and 1 electron from the hydrogen and 1 He nucleus and 2 electrons for the helium. Thus, we have

    .. math::

        \mu = \frac{n_{\rm H} + 4n_{\rm He}}{1n_{\rm H} + 2n_{\rm He}}.

    If all non-hydrogen species are Helium, then :math:`n_{\rm He} = N(1-\chi_H)/4`, so

    .. math::

        \mu = \frac{1}{1\chi_H + (1/2)(1-\chi_H)}.

    """
    return 1 / (hydrogen_fraction + 0.5 * (1 - hydrogen_fraction))
