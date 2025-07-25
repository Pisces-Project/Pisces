"""
This is the core testing module for Pisces profiles. We use this module to
ensure that all of the profiles can be initialized correctly and that they are
all callable. We also check units as needed and perform a number of other tests.
"""

import pytest

from .utils import __ALL_PROFILES__


@pytest.mark.parametrize("profile_class", list(__ALL_PROFILES__.values()))
def test_profiles_initialize(profile_class):
    """
    Test that all profiles are able to instantiate with their default
    parameters properly and that they all have a set of required profile
    parameters.

    Parameters
    ----------
    profile_class: The profile class that is being tested.
    """
    # Check that we can instantiate the profile:
    profile = profile_class()

    # Ensure that we have units on the output.
    profile.__units_
