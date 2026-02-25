from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ldt.utils.errors import OperationNotFoundError

OperationHandler = Callable[[Mapping[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class RegisteredOperation:
    """Descriptor for one registered operation handler.

    Attributes:
        name (str): Operation key used for dispatch.
        handler (OperationHandler): Callable that executes the operation.
        description (str): Human-readable description.
    """

    name: str
    handler: OperationHandler
    description: str = ""


class OperationRegistry:
    """Store and execute named operation handlers."""

    def __init__(self) -> None:
        self._operations: dict[str, RegisteredOperation] = {}

    def register(
        self,
        name: str,
        handler: OperationHandler,
        *,
        description: str = "",
    ) -> None:
        """Register a new operation handler.

        Args:
            name (str): Unique operation key.
            handler (OperationHandler): Callable used to execute the workflow step.
            description (str): Optional operation description for catalogs.
        """
        operation = name.strip()
        if not operation:
            raise ValueError("Operation name cannot be empty.")
        if operation in self._operations:
            raise ValueError(f"Operation already registered: {operation}")
        self._operations[operation] = RegisteredOperation(
            name=operation,
            handler=handler,
            description=description.strip(),
        )

    def execute(self, operation: str, params: Mapping[str, Any]) -> dict[str, Any]:
        """Execute a registered operation.

        Args:
            operation (str): Technique or operation identifier.
            params (Mapping[str, Any]): Parameter mapping provided by the caller.

        Returns:
            dict[str, Any]: Output payload returned by the operation handler.
        """
        descriptor = self._operations.get(operation.strip())
        if descriptor is None:
            raise OperationNotFoundError(f"Unknown operation: {operation}")
        return descriptor.handler(params)

    def names(self) -> list[str]:
        """Return all registered operation names in sorted order.

        Returns:
            list[str]: Sorted operation keys.
        """
        return sorted(self._operations.keys())
