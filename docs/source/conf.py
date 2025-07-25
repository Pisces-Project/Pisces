# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

import matplotlib  # noqa: F401

sys.path.insert(0, os.path.abspath("../.."))
sys.path.insert(0, os.path.abspath("../../pisces"))
# -- Project information -----------------------------------------------------

project = "Pisces"
copyright = "2024, Eliza Diggins"
author = "Eliza Diggins"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx_design",
    "sphinx.ext.doctest",
    "matplotlib.sphinxext.plot_directive",
    "sphinxcontrib.bibtex",
    "sphinx_gallery.gen_gallery",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["../_templates"]
bibtex_bibfiles = ["docs_bib.bib"]
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "yt": ("https://yt-project.org/doc/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
    "unyt": ("https://unyt.readthedocs.io/en/latest/", None),
    "h5py": ("https://docs.h5py.org/en/latest/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

napoleon_use_param = True
napoleon_preprocess_types = True

# Configure autodoc mock imports to ensure that we don't need to
# actually compile C level tools.
autodoc_mock_imports = [
    "pisces.physics.virialization._eddington_sampling",
]


# Configure the sphinx galleries. These are contained in the
# /examples gallery.
sphinx_gallery_conf = {
    "examples_dirs": [
        "./galleries/examples",
    ],  # path to your example scripts
    "gallery_dirs": [
        "auto_examples",
    ],  # path to where to save gallery generated output
}
