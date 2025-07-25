"""RNG Management for Pisces."""

import numpy as np

from .config import pisces_config

# Create the RNG object for the codebase. This
# is based on pisces_config. If ``numeric.random.seed`` is
# set, then that is the seed to use. Otherwise, we use no
# seed.
if pisces_config["numeric.random.seed"] is not None:
    __RNG__ = np.random.default_rng(pisces_config["numeric.random.seed"])
else:
    __RNG__ = np.random.default_rng()
