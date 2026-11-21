from __future__ import annotations


class ForgeMLError(Exception):
    """An expected failure (bad data, bad spec). `code` is stable for programmatic use."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
