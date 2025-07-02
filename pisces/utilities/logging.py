""" Logging utilities for the Pisces ecosystem.
"""
import logging

from .config import pisces_config

# ======================== #
# Configure the global log #
# ======================== #
pisces_logger = logging.getLogger("Pisces")
pisces_logger.setLevel(
    getattr(logging, pisces_config["logging.main.level"])
)  # Allow DEBUG, handlers filter final output
pisces_logger.propagate = False  # Avoid duplicate logs to root logger

# Don't permit double handler adding.
if not pisces_logger.hasHandlers():
    # Console handler with minimal formatting
    console_handler = logging.StreamHandler()
    console_fmt = pisces_config["logging.main.format"]
    console_handler.setFormatter(logging.Formatter(console_fmt))
    pisces_logger.addHandler(console_handler)
