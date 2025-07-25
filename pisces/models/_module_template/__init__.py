"""[TEMPLATE] Module template for Pisces models subpackage.

This module serves as a template for creating new model subpackages within the Pisces project. It
is provided as a starting point for developers to implement their own models following the
Pisces conventions and standards.

Submodules of :mod:`models` should be specific to a **particular type of physical system**. This may be
stellar models, cosmological models, etc. Each submodule has a standardized format for defining models, implementing
hooks for connection to extensions / external libraries, etc.

Components
-----------

Each submodule should contain the following components:

- ``._hooks.py``: Contains a number of mixin hook classes which can be inherited by model classes to
  enable extension functionality. These should all be subclassed themselves from the base classes defined
  in :mod:`models.core.hooks`. For a new extension, a new hook base class should be defined and then
  implemented in the submodule's ``._hooks.py`` file.

- Model modules: Organization of the model containing modules is left to the developer, but
  it is recommended that similar models share a common submodule. For example, if you are implementing
  a new stellar model, you might create a submodule called ``pisces.models.stars`` and implement polytropic models
  in a submodule called ``pisces.models.stars.polytropic`` while more complex stellar models might be in a different
  submodule like ``pisces.models.stars.complex``. Each model submodule should contain a class that implements
  the model itself, following the conventions of Pisces models.
"""

# Write the __all__ variable. This should DIRECTLY include all models, skipping the submodule level in
# which they are defined. This ensures that all models are directly accessible from the model module.
__all__ = []

# Import from the modules.
