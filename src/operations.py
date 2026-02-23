from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.data_preparation.operations import (
    register_operations as register_data_preparation_operations,
)
from src.data_preprocessing.operations import (
    register_operations as register_data_preprocessing_operations,
)
from src.machine_learning.operations import (
    register_operations as register_machine_learning_operations,
)
from src.utils.operation_registry import OperationRegistry


def _build_registry() -> OperationRegistry:
    registry = OperationRegistry()
    register_data_preprocessing_operations(registry)
    register_data_preparation_operations(registry)
    register_machine_learning_operations(registry)
    registry.register(
        "core.catalog",
        lambda _: {"operations": registry.names()},
        description="List registered bridge operations.",
    )
    return registry


_REGISTRY = _build_registry()


def execute_operation(operation: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one registered operation with JSON-friendly output.

    Args:
        operation (str): Technique or operation identifier.
        params (Mapping[str, Any]): Parameter mapping provided by the caller.

    Returns:
        dict[str, Any]: Dictionary containing tool results.
    """

    return _REGISTRY.execute(operation, params)
