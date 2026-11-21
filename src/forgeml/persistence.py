"""Model artifacts in the `skops` format.

Unlike pickle/joblib, a skops file cannot execute code when loaded. On load, ForgeML trusts only
scikit-learn and NumPy types and refuses files that need anything else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import skops.io as sio

from forgeml.errors import ForgeMLError

TRUSTED_PREFIXES = ("sklearn.", "numpy.")


def save_model(model: Any, path: Path) -> None:
    sio.dump(model, path)


def load_model(path: Path) -> Any:
    untrusted = sio.get_untrusted_types(file=path)
    refused = [t for t in untrusted if not t.startswith(TRUSTED_PREFIXES)]
    if refused:
        raise ForgeMLError(
            "untrusted_model", f"The model file needs types ForgeML does not trust: {refused}"
        )
    return sio.load(path, trusted=untrusted)
