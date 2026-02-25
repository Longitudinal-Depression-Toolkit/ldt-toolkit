from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from ldt.bridge.operation_registry import OperationRegistry
from ldt.utils.errors import InputValidationError

from .catalog import (
    list_data_conversion_techniques,
    list_data_preparation_presets,
    list_synthetic_techniques,
)


def register_operations(registry: OperationRegistry) -> None:
    """Register data-preparation operation handlers on a registry.

    Args:
        registry (OperationRegistry): Operation registry instance to configure.
    """
    registry.register(
        "data_preparation.synthetic_data_generation.catalog",
        _op_synthetic_catalog,
        description="List synthetic data-generation techniques.",
    )
    registry.register(
        "data_preparation.synthetic_data_generation.run",
        _op_synthetic_run,
        description="Run one synthetic generation technique.",
    )
    registry.register(
        "data_preparation.data_conversion.catalog",
        _op_conversion_catalog,
        description="List data conversion techniques.",
    )
    registry.register(
        "data_preparation.data_conversion.run",
        _op_conversion_run,
        description="Run one data conversion technique.",
    )
    registry.register(
        "data_preparation.presets.catalog",
        _op_presets_catalog,
        description="List data-preparation reproducibility presets.",
    )
    registry.register(
        "data_preparation.presets.prepare_mcs_by_leap.profile",
        _op_prepare_mcs_profile,
        description="Return defaults and supported waves for Prepare MCS by LEAP.",
    )
    registry.register(
        "data_preparation.presets.prepare_mcs_by_leap.run",
        _op_prepare_mcs_run,
        description="Run Prepare MCS by LEAP preset pipeline from Go bridge params.",
    )


def _op_synthetic_catalog(_: Mapping[str, Any]) -> dict[str, Any]:
    return {"techniques": list_synthetic_techniques()}


def _op_synthetic_run(params: Mapping[str, Any]) -> dict[str, Any]:
    technique, raw_params = _extract_technique_and_params(params)
    return _resolve_runner(
        "ldt.data_preparation.tools.synthetic_data_generation.run",
        "run_synthetic_generation",
    )(technique=technique, params=raw_params)


def _op_conversion_catalog(_: Mapping[str, Any]) -> dict[str, Any]:
    return {"techniques": list_data_conversion_techniques()}


def _op_conversion_run(params: Mapping[str, Any]) -> dict[str, Any]:
    technique, raw_params = _extract_technique_and_params(params)
    return _resolve_runner(
        "ldt.data_preparation.tools.data_conversion.run",
        "run_data_conversion",
    )(technique=technique, params=raw_params)


def _op_presets_catalog(_: Mapping[str, Any]) -> dict[str, Any]:
    return {"presets": list_data_preparation_presets()}


def _op_prepare_mcs_profile(_: Mapping[str, Any]) -> dict[str, Any]:
    return _resolve_runner(
        "ldt.data_preparation.presets.prepare_mcs_by_leap.run",
        "prepare_mcs_by_leap_profile",
    )()


def _op_prepare_mcs_run(params: Mapping[str, Any]) -> dict[str, Any]:
    raw_params = _as_object(params, "params")
    return _resolve_runner(
        "ldt.data_preparation.presets.prepare_mcs_by_leap.run",
        "run_prepare_mcs_by_leap",
    )(params=raw_params)


def _resolve_runner(module_path: str, function_name: str) -> Any:
    module = importlib.import_module(module_path)
    return getattr(module, function_name)


def _extract_technique_and_params(
    params: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    technique = _as_required_string(params, "technique")
    raw_params = _as_object(params, "params")
    return technique, raw_params


def _as_required_string(params: Mapping[str, Any], key: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"Missing required string parameter: {key}")
    return value.strip()


def _as_object(params: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = params.get(key, {})
    if not isinstance(value, dict):
        raise InputValidationError(f"`{key}` must be an object.")
    return value
