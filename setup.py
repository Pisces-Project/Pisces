"""Setup script for building Cython extensions in the pisces package.

Most of the build process is handled by setuptools via the `pyproject.toml` file;
however, this script is required to compile the Cython modules that are part of the
pisces package. It defines the Cython extensions and their compilation settings.
"""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = cythonize(
    [
        Extension(
            name="pisces.physics.virialization._eddington_sampling",
            sources=["pisces/physics/virialization/_eddington_sampling.pyx"],
            include_dirs=[np.get_include()],
        ),
        Extension(
            name="pisces.math_utils._random_fields",
            sources=["pisces/math_utils/_random_fields.pyx"],
            include_dirs=[np.get_include()],
        ),
    ],
    language_level="3",
)

setup(
    ext_modules=extensions,
)
