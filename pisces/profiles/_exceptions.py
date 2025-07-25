"""Exception classes for the :mod:`profiles` module."""


class ProfileClassSetupError(Exception):
    """
    Error during class setup.

    Error raised specifically during profile class setup in
    scenarios where the class itself has failed to generate for
    some reason.
    """
