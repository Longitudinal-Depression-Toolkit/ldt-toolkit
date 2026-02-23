from __future__ import annotations


class LibraryError(Exception):
    """Base error for library execution failures."""


class InputValidationError(LibraryError):
    """Raised when operation input parameters fail validation."""


class OperationNotFoundError(LibraryError):
    """Raised when an operation key is not registered."""
