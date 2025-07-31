"""Logging utilities for the Pisces ecosystem.

This module defines and configures the global logger (:attr:`pisces_logger`),
which serves as the main entry point for logging messages across the Pisces
codebase. Logging behavior—such as log level, formatting, and enable/disable
switches—is controlled by the central configuration (:attr:`~utilities.config.pisces_config`) under
the ``logging.main`` namespace.

"""

import logging
from typing import TypeVar

from .config import pisces_config

Instance = TypeVar("Instance")


class LogDescriptor:
    """A descriptor for dynamically creating and managing loggers for a class.

    At its core, the :py:class:`LogDescriptor` is used for classes like :py:class:`~models.base.BaseModel` to
    create a class-specific logger.
    """

    def __init__(self, mode="models"):
        """Initialize the LogDescriptor with a specific mode.

        Parameters
        ----------
        mode : str
            The mode of logging to use. This corresponds to the configuration
            section in `pisces_config` (e.g., "models", "profiles", etc.).
            Defaults to "models".

        """
        self.mode = mode

    def __get__(self, instance: Instance, owner: type[Instance]) -> logging.Logger:
        """Retrieve or create a logger for the class.

        Parameters
        ----------
        instance: Instance
            The instance of the class (not used, but required for descriptor protocol).
        owner: Type[Instance]
            The class for which the logger is being created.

        Returns
        -------
        ~logging.Logger
            A logger instance configured according to the Pisces configuration
            for the specified mode.

        """
        # Retrieve or create a logger for the class
        logger = logging.getLogger(owner.__name__)

        if not logger.handlers:
            # Create and configure the logger if not already set up
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(pisces_config[f"system.logging.{self.mode}.format"]))
            logger.addHandler(handler)
            logger.setLevel(pisces_config[f"system.logging.{self.mode}.level"])
            logger.propagate = False
            logger.disabled = not pisces_config[f"system.logging.{self.mode}.enabled"]

        return logger


# ======================== #
# Configure the global log #
# ======================== #
pisces_logger = logging.getLogger("Pisces")
pisces_logger.setLevel(
    getattr(logging, pisces_config["system.logging.main.level"])
)  # Allow DEBUG, handlers filter final output
pisces_logger.propagate = False  # Avoid duplicate logs to root logger

# Don't permit double handler adding.
if not pisces_logger.hasHandlers():
    # Console handler with minimal formatting
    console_handler = logging.StreamHandler()
    console_fmt = pisces_config["system.logging.main.format"]
    console_handler.setFormatter(logging.Formatter(console_fmt))
    pisces_logger.addHandler(console_handler)
