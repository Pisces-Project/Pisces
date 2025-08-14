"""
Extension modules for the Pisces framework.

This package contains optional add-on components that extend Pisces beyond its
core modeling capabilities. These modules can:

- Integrate Pisces with external frameworks (e.g., the `yt` analysis and
    visualization toolkit)
- Provide specialized functionality not included in the core library
- Enable additional data formats, I/O backends, or simulation workflows

Extensions are designed to be modular and can be imported only when their
dependencies are available, keeping the core installation lightweight while
allowing advanced features for specific use cases.
"""

__all__ = ["simulation"]

from . import simulation
