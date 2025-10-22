"""
Simulation frontends for using Pisces models with the Gadget simulation code.

For more information on Gadget, see the `official documentation <https://wwwmpa.mpa-garching.mpg.de/gadget4/>`_. For
documentation on using this frontend with Pisces, see the :ref:`simulations_gadget` page.
"""

__all__ = ["Gadget4Frontend"]

# Provide access to the frontend.
from .frontends import Gadget4Frontend
