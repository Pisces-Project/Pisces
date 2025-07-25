"""Utility tools for accessing Pisces model configuration.

This module provides infrastructure for binding model classes to their
corresponding configuration entries in the Pisces global config registry.
It is used internally by Pisces to expose user-defined or default configuration
options at the class level without hardcoding values.

"""

from pisces.utilities.config import pisces_config


class ModelConfig:
    """Descriptor for accessing the registered configuration of a model class.

    This descriptor provides dynamic, read-only access to the model’s
    configuration as stored in the global `pisces_config` registry. It is intended
    to be used as a class-level attribute within model classes.

    Usage
    -----
    Define a `config` attribute using this descriptor:

    .. code-block:: python

        class MyModel(BaseModel):
            config = ModelConfig()

    Then access it via the class or an instance:

    .. code-block:: python

        MyModel.config  # Access class-level configuration
        my_model.config  # Also works via instance

    The corresponding entry in the configuration registry should be:

    .. code-block:: yaml

        models:
          MyModel:
            param1: ...
            param2: ...

    Raises
    ------
    AttributeError
        If the configuration entry for the model class is missing.

    Returns
    -------
    dict
        A dictionary of configuration options for the model.

    """

    def __get__(self, instance: object, owner: type) -> dict:
        """Fetch the model configuration."""
        class_name = owner.__name__
        try:
            return pisces_config[f"models.{class_name}"]
        except KeyError as err:
            raise AttributeError(
                f"Configuration entry not found for model '{class_name}'. "
                f"Expected key 'models.{class_name}' in `pisces_config`."
            ) from err
