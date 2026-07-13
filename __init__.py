"""Gerchberg–Saxton phase-mask generation for multimode fibre / SLM control."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("gs")
except PackageNotFoundError:
    __version__ = "1.0.0"

from .pyGS import GS
from .toolbox.setup import GS_mask_setup, optical_setup

__all__ = [
    "GS",
    "GS_mask_setup",
    "optical_setup",
    "__version__",
]
