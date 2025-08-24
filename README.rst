.. image:: docs/source/images/pisces.svg
   :width: 200px
   :align: center

Pisces
======

+-------------------+----------------------------------------------------------+
| **Code**          | |RUFF| |PRE-COMMIT| |YT-PROJECT|                         |
+-------------------+----------------------------------------------------------+
| **Documentation** | |docs| |docs-stable| |numpydoc|                          |
+-------------------+----------------------------------------------------------+
| **GitHub**        | |CONTRIBUTORS| |LAST-COMMIT|                             |
|                   | |COMMITIZEN| |CONVENTIONAL-COMMITS|                      |
+-------------------+----------------------------------------------------------+
| **PyPi**          | Coming Soon                                              |
+-------------------+----------------------------------------------------------+

**Pisces (or Py-ICs)** is an open-source toolkit for constructing models of astrophysical systems and generating data for
simulations. Designed for flexibility and ease of use, Pisces enables researchers to create a wide range of astrophysical
models—including galaxies, galaxy clusters, and early universe perturbations—whether for direct analysis or as input for
simulation software.

Rather than being limited to initial condition generation, Pisces provides a modular and extensible framework that
supports a variety of scientific use cases. It is built to integrate seamlessly with major astrophysical simulation
tools through dedicated frontends, ensuring broad compatibility and interoperability.

The core package is structured to facilitate customization and development, allowing users to adapt it to their specific
research needs. All essential tools for model construction are included, with an emphasis on accessibility and ease of extension.

Development takes place on `GitHub <https://www.github.com/Pisces-Project/Pisces>`_. If you encounter any issues, documentation
gaps, or have feature suggestions, we encourage you to submit them via the repository's issues page.

Getting Started
---------------


Acknowledgment
--------------

If you use Pisces for academic work, please include a statement in your publication similar to:

    Initial conditions were generated using Pisces (version), an initial conditions generator for astrophysical
    simulations, developed by Eliza Diggins and available at https://github.com/Pisces-Project/pisces.


.. |yt-project| image:: https://img.shields.io/static/v1?label=yt&message=compatible&color=blueviolet
   :target: https://yt-project.org
   :alt: Works with yt

.. |docs| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg
   :target: https://eliza-diggins.github.io/Pisces
   :alt: Latest Docs

.. |docs-stable| image:: https://img.shields.io/badge/docs-stable-brightgreen.svg
   :target: https://eliza-diggins.github.io/Pisces/stable/
   :alt: Latest Docs

.. |ruff| image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff

.. |pre-commit| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit

.. |numpydoc| image:: https://img.shields.io/badge/docstyle-numpydoc-459db9
   :target: https://numpydoc.readthedocs.io/en/latest/
   :alt: Docstring style: numpydoc

.. |commitizen| image:: https://img.shields.io/badge/commitizen-friendly-brightgreen.svg
   :target: https://commitizen-tools.github.io/commitizen/
   :alt: Commit style: Conventional + Gitmoji

.. |conventional-commits| image:: https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white
   :target: https://www.conventionalcommits.org/en/v1.0.0/
   :alt: Commit style: Conventional Commits

.. |contributors| image:: https://img.shields.io/github/contributors/eliza-diggins/Pisces
   :target: https://github.com/eliza-diggins/Pisces/graphs/contributors
   :alt: GitHub Contributors

.. |last-commit| image:: https://img.shields.io/github/last-commit/eliza-diggins/Pisces
   :target: https://github.com/eliza-diggins/Pisces
   :alt: Last Commit
