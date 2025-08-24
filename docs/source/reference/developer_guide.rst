.. image:: ../images/pisces.svg
   :width: 200px
   :align: center

.. _developer_guide:
========================
Pisces Developer Guide
========================

Welcome to the **Pisces Developer Guide**! This section is intended for contributors, maintainers, and advanced users
who want to understand the internal architecture of Pisces, develop new features, or contribute improvements to the
codebase.

Development Structure
---------------------
Before diving into the specifics of contributing to Pisces, it's important to understand the overall structure of the git repository
and how development is organized.

Pisces uses a dual versioning system to manage both stable releases and ongoing development. There are three types
of branches to be familiar with in the repository:

- ``main``: This is the primary branch that contains the latest stable release of Pisces.
  Each release is tagged in ``main`` with a version number following the `Semantic Versioning <https://semver.org/>`__ scheme.

  For example, a stable release might be tagged as ``v1.2.0``.
- ``dev_vX.Y.Z``: This is the active development branch for the next version of Pisces.
  There is only one active development branch at a time, which corresponds to the next version to be released. This is
  the branch where active development should occur and where pull requests for new features or bug fixes should be directed.
- ``stable_vX.Y.Z``: On each release of the code, a static branch is also generated from ``main`` at the point of
  the tag so that there is also a stable branch for each version of Pisces.

In the case of a hotfix to a release, branches should be made from the stable version tag and named ``hotfix_...``.

Version Tagging Workflow
````````````````````````

1. New development should occur on a branch / fork of the bleeding edge development branch. There
   is only one active development branch at a time, which is ``dev_vX.Y.Z``. This is the next version
   to be released.
2. When a feature is complete, a pull request is created against the development branch itself and
   the feature is merged into the active development branch.
3. When the development branch is ready for release, the branch is merged into `main`.
   A new stable version tag is created from the `main` branch, following the
   Semantic Versioning scheme. The changelog and release notes are updated accordingly.
4. Changelog and release notes are updated accordingly.


Setting Up the Development Environment
--------------------------------------
In order to successfully contribute to Pisces, you will need to set up a development environment that allows you to run the code,
test changes, and build documentation. This section provides a step-by-step guide to getting started. Note that
this will require some external tools to be installed on your system to provide linting, testing, etc.

Cloning the Repository
^^^^^^^^^^^^^^^^^^^^^^^

To get started, you will need to clone the Pisces repository from GitHub. You can do this using the following command:

.. code-block:: bash

    git clone https://github.com/Pisces-Project/pisces

This will create a local copy of the Pisces repository on your machine. To get a particular branch you will need to
checkout the branch you want to work on:

.. code-block:: bash

    cd pisces
    git checkout dev_vX.Y.Z  # Replace with the desired development version

.. hint::

    Since you won't be able to push changes to the main repository (it is protected), you will need
    to fork the repository on GitHub and clone your fork instead. You can do this by clicking the "Fork" button.
    on the top right of the repository page on GitHub, then cloning your forked repository.

    Once a fork has been created, you can clone and checkout the branch you want to work on just as described above.

Installing Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^

A number of dependencies are required to run Pisces and to develop it effectively. In addition to the basic
requirements of the library, you will also need to install some development tools to help with linting, testing, and documentation generation.
These are all managed through the ``pyproject.toml`` file. To install the package and all of its development dependencies,
run the following command in the root directory of the cloned repository:

.. code-block:: bash

    pip install -e .[dev]

This will install Pisces in editable mode, allowing you to make changes to the code and see them reflected immediately. Additionally,
it will install all the development dependencies required for testing, linting, and documentation generation. Before proceeding,
ensure that hooks are installed by running:

.. code-block:: bash

    pre-commit install

Once this is done, you can run the following command to check that all dependencies are installed correctly:

.. code-block:: bash

    pre-commit run --all-files

This should run all of the pre-commit hooks. Details on testing, linting, and documentation generation can be found in the relevant sections below.

Contributing Code
-----------------

As you develop new features or fix bugs, you will need to follow the contribution guidelines to ensure that your changes
are accepted into the main codebase. This includes writing tests, following the coding style, and ensuring that your code passes all
linting checks. The following sections provide detailed information on how to contribute code effectively.

Commit Messages and Changelogs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pisces uses the `Commitizen <https://commitizen-tools.github.io/commitizen/>`__ tool to enforce structured commit messages
following the `Conventional Commits <https://www.conventionalcommits.org/en/v1.0.0/>`__ standard. This system enables
automatic changelog generation and helps keep versioning consistent and predictable.

Commit Message Format
```````````````````````````````

Each commit message should follow the general pattern:

.. code-block:: text

   <type>(optional scope): <description>

   [optional body]

   [optional footer(s)]

Examples:

.. code-block:: bash

   feat(core): add new SersicProfile subclass for galaxy modeling
   fix(fields): correct unit parsing for temperature datasets
   docs(config): clarify YAML structure for model defaults
   refactor: simplify profile evaluation logic
   test(profiles): add test coverage for entropy interpolation

The allowed types include (but are not limited to):

- ``feat`` — for new features
- ``fix`` — for bug fixes
- ``docs`` — for documentation-only changes
- ``style`` — for formatting or whitespace changes (no logic changes)
- ``refactor`` — for code restructuring that does not fix a bug or add a feature
- ``test`` — for adding or updating tests
- ``chore`` — for maintenance tasks (build config, CI setup, etc.)

Scoping (e.g., ``feat(core):``) is optional but **recommended** to indicate the module or component affected.

.. important::

    Commitizens is implemented through the pre-commit hooks and is automatically enforced. If you fail to use
    the correct commit message format, your commit will be rejected. You can run the following command to
    check your commit messages before committing:

    .. code-block:: bash

        git commit --dry-run

Pull Requests
^^^^^^^^^^^^^^

When you are ready to submit your changes, create a pull request (PR) against the ``dev_vX.Y.Z`` branch of the main repository.
Make sure to include a clear description of the changes you have made, the motivation behind them, and any relevant
issue numbers (e.g., ``Fixes #123``). This will help maintainers understand the context of your changes and facilitate
the review process.

By following the commit message guidelines and providing a clear PR description, you will help ensure that your contributions
are reviewed and merged efficiently. The maintainers will review your PR, provide feedback, and may request changes before
merging it into the development branch.

Github Actions will be used to automatically run tests and check your PR formatting and linting. If any of these checks fail,
you will need to address the issues before your PR can be merged.

Pisces Standards, Conventions, and Structures
----------------------------------------------

Pisces is developed with the explicit intention of allowing easy extensibility and maintainability. A big part of this
is the modular style of various parts of the codebase with the use of well-defined interfaces and abstract base classes.
While the details of these APIs is not relevant for the user, as developers, it is important to understand the
structure and conventions used throughout the codebase. This section provides an overview of the key standards and
conventions used in Pisces development.

.. toctree::
    :maxdepth: 1

    models/writing_models
    profiles/writing_profiles
    extensions/simulation/ics/initial_conditions_dev
    extensions/simulation/frontends/frontends_dev

Linting and Formatting
-----------------------

To ensure a consistent code style and reduce potential errors, Pisces enforces a set of linting and formatting rules
across the entire codebase. These tools are automatically run in continuous integration and should also be run locally
before submitting any pull requests.

Formatting: ``black``
^^^^^^^^^^^^^^^^^^^^^^

Pisces uses `black <https://black.readthedocs.io/en/stable/>`__ as the standard code formatter. ``black`` automatically
rewrites Python code to follow a consistent style.

To apply formatting:

.. code-block:: bash

   black .

Key configuration options (set in ``pyproject.toml``):

- ``line-length = 88``
- ``target-version = ["py311"]``

Linting: ``ruff``
^^^^^^^^^^^^^^^^^

For fast and comprehensive linting, Pisces uses `ruff <https://docs.astral.sh/ruff/>`__. Ruff checks for unused imports,
undefined variables, formatting errors, docstring issues, and more.

To run ruff manually:

.. code-block:: bash

   ruff check .

Configuration is handled in ``pyproject.toml``. For example, to ignore specific error codes:

.. code-block:: toml

   [tool.ruff]
   exclude = ["E731", "D105"]


Documentation
-------------

See :ref:`documentation` for details on how to contribute to the Pisces documentation, including
docstring standards, building the documentation, and contributing examples.

Testing
-------

See :ref:`testing` for details on how to contribute to Pisces testing, including
test structure, writing new tests, and running tests locally.
