.. _documentation:
====================
Documentation Guide
====================

Pisces maintains a high standard of documentation to ensure clarity, accessibility, and reproducibility.
This guide is intended for developers and contributors who want to understand or contribute to the Pisces documentation system.

Documentation Structure
------------------------

The documentation is organized using `Sphinx <https://www.sphinx-doc.org/>`__ and consists of several main sections:

- **Quickstart / Getting Started**: An entry-level guide for new users covering installation,
  basic usage, and common workflows.
- **User Guide**: High-level conceptual overviews of each part of Pisces, including models,
  configuration, and usage patterns. Best for users who want to understand
  what Pisces does and how to use it effectively.
- **Examples Gallery**: A gallery of runnable Python scripts showing common tasks, use cases,
  and visualizations. These examples are auto-generated using `sphinx-gallery`.
- **API Reference**: Automatically generated from docstrings in the codebase.
  This includes detailed signatures, type annotations, and module-level documentation.
- **Developer Guide**: Contributor documentation, workflows,
  testing, CI/CD practices, and the philosophy behind the codebase.

Building the Documentation
---------------------------

To build the documentation locally:

.. code-block:: bash

   pip install .[docs]
   sphinx-build -b html -j auto docs/source docs/_build/html

This will generate the documentation in the ``docs/_build/html`` directory.
You can open ``index.html`` in your browser to view it.

Alternatively, use the Makefile:

.. code-block:: bash

   make -C docs html

If you only want to test whether documentation builds without rendering:

.. code-block:: bash

   sphinx-build -n -W -b dummy docs/source docs/_build/dummy

Docstring Standards
--------------------

Pisces follows the `numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`__ format for all
docstrings. This format is required for compatibility with `sphinx.ext.napoleon`.

Each function, class, and module must be documented with:

- A one-line summary
- A longer description if appropriate
- Parameters and their types
- Return values and types
- Raises section (for exceptions)
- Examples using doctest or plain text

Example Gallery (sphinx-gallery)
----------------------------------

Pisces uses `sphinx-gallery <https://sphinx-gallery.github.io/>`__ to automatically
generate an example gallery from scripts in ``docs/source/galleries/examples``. The galleries
are broken down by subject area and can be extended to include new examples as needed. See the
documentation for `sphinx-gallery <https://sphinx-gallery.github.io/stable/index.html>`__ for more details on
how to structure these examples.

These examples will be automatically built into the ``auto_examples`` section of the documentation.

Autodoc and Intersphinx
------------------------

Pisces uses the following Sphinx extensions for API documentation:

- ``sphinx.ext.autodoc``: Automatically generates documentation from docstrings.
- ``sphinx.ext.intersphinx``: Enables linking to external projects like NumPy, Astropy, and SymPy.
- ``sphinx.ext.napoleon``: Parses Numpy/Google style docstrings.
- ``sphinx.ext.viewcode``: Links to source code in the documentation.
- ``sphinx.ext.mathjax``: Renders LaTeX math expressions in HTML.

The ``intersphinx_mapping`` in ``conf.py`` includes:

.. code-block:: python

   intersphinx_mapping = {
       "python": ("https://docs.python.org/3/", None),
       "numpy": ("https://numpy.org/doc/stable/", None),
       "astropy": ("https://docs.astropy.org/en/stable/", None),
       "sympy": ("https://docs.sympy.org/latest/", None),
       ...
   }

This allows referencing external docs like:

.. code-block:: rst

   See also: :external:py:mod:`numpy`

Please ensure that all external references are valid and up-to-date. If a
reference is not present in the intersphinx mapping, it will not resolve correctly in the documentation.

Contribution Guidelines
------------------------

When contributing to the documentation:

- All user-facing modules must have docstrings.
- All new features must include updated documentation and examples.
- Example gallery scripts should be added for major use cases.
- All changes must pass doc build checks in CI.

You can verify the build with:

.. code-block:: bash

   sphinx-build -n -W docs/source docs/_build/html

.. note::

   Warnings are treated as errors during documentation builds, so incomplete docstrings will fail CI.

Summary
-------

Pisces treats documentation as a first-class component of development. All contributors
are expected to maintain and extend documentation as part of their changes. Keeping
the docs up-to-date ensures accessibility for users and maintainability for future developers.
