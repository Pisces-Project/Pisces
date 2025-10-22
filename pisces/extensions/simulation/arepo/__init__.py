"""
Frontend support for the AREPO MHD code.

This module provides frontend classes for exporting initial conditions to AREPO for
simulation using the infrastructure of the Pisces framework. The AREPO code website can
be found at https://arepo-code.org/.
"""

__all__ = ["AREPOFrontend"]
from .frontends import AREPOFrontend
