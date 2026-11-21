"""ForgeML — reproducible tabular machine-learning runs from a single JSON spec."""

from __future__ import annotations

__version__ = "0.1.0"

from forgeml.errors import ForgeMLError
from forgeml.spec import RunSpec

__all__ = ["ForgeMLError", "RunSpec", "__version__"]
