from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from ldt.bridge.operation_registry import OperationRegistry
from ldt.utils.errors import InputValidationError

from .catalog import (
    list_bench_standard_and_longitudinal_ml_techniques,
    list_benchmark_longitudinal_ml_techniques,
    list_benchmark_standard_ml_techniques,
    list_longitudinal_machine_learning_techniques,
    list_machine_learning_presets,
    list_shap_analysis_techniques,
    list_standard_machine_learning_techniques,
)


def register_operations(registry: OperationRegistry) -> None:
    """Register machine-learning operation handlers on a registry.

    Args:
        registry (OperationRegistry): Operation registry instance to configure.
    """
    registry.register(
        "machine_learning.standard_machine_learning.catalog",
        _op_standard_catalog,
        description="List standard-machine-learning techniques.",
    )
    registry.register(
        "machine_learning.standard_machine_learning.run",
        _op_standard_run,
        description="Run one standard-machine-learning technique.",
    )
    registry.register(
        "machine_learning.longitudinal_machine_learning.catalog",
        _op_longitudinal_catalog,
        description="List longitudinal-machine-learning techniques.",
    )
    registry.register(
        "machine_learning.longitudinal_machine_learning.run",
        _op_longitudinal_run,
        description="Run one longitudinal-machine-learning technique.",
    )
    registry.register(
        "machine_learning.benchmark_standard_ml.catalog",
        _op_benchmark_standard_catalog,
        description="List benchmark-standard-ml techniques.",
    )
    registry.register(
        "machine_learning.benchmark_standard_ml.run",
        _op_benchmark_standard_run,
        description="Run one benchmark-standard-ml technique.",
    )
    registry.register(
        "machine_learning.benchmark_longitudinal_ml.catalog",
        _op_benchmark_longitudinal_catalog,
        description="List benchmark-longitudinal-ml techniques.",
    )
    registry.register(
        "machine_learning.benchmark_longitudinal_ml.run",
        _op_benchmark_longitudinal_run,
        description="Run one benchmark-longitudinal-ml technique.",
    )
    registry.register(
        "machine_learning.bench_standard_and_longitudinal_ml.catalog",
        _op_benchmark_mixed_catalog,
        description="List mixed benchmark techniques.",
    )
    registry.register(
        "machine_learning.bench_standard_and_longitudinal_ml.run",
        _op_benchmark_mixed_run,
        description="Run one mixed benchmark technique.",
    )
    registry.register(
        "machine_learning.shap_analysis.catalog",
        _op_shap_catalog,
        description="List SHAP-analysis techniques.",
    )
    registry.register(
        "machine_learning.shap_analysis.run",
        _op_shap_run,
        description="Run one SHAP-analysis technique.",
    )
    registry.register(
        "machine_learning.presets.catalog",
        _op_presets_catalog,
        description="List machine-learning reproducibility presets.",
    )


def _op_standard_catalog(_: Mapping[str, Any]) -> dict[str, Any]:
    return {"techniques": list_standard_machine_learning_techniques()}


def _op_standard_run(params: Mapping[str, Any]) -> dict[str, Any]:
    return _run_tool_operation(
        params=params,
        runner=_resolve_runner(
            "ldt.machine_learning.tools.standard_machine_learning.run",
            "run_standard_machine_learning_tool",
        ),
    )


def _op_longitudinal_catalog(_: Mapping[str, Any]) -> dict[str, Any]:
    return {"techniques": list_longitudinal_machine_learning_techniques()}


def _op_longitudinal_run(params: Mapping[str, Any]) -> dict[str, Any]:
    return _run_tool_operation(
        params=params,
        runner=_resolve_runner(
            "ldt.machine_learning.tools.longitudinal_machine_learning.run",
            "run_longitudinal_machine_learning_tool",
        ),
    )


def _op_benchmark_standard_catalog(_: Mapping[str, Any]) -> dict[str, Any]:
    return {"techniques": list_benchmark_standard_ml_techniques()}


def _op_benchmark_standard_run(params: Mapping[str, Any]) -> dict[str, Any]:
    return _run_tool_operation(
        params=params,
        runner=_resolve_runner(
            "ldt.machine_learning.tools.benchmark_standard_ml.run",
            "run_benchmark_standard_ml_tool",
        ),
    )


def _op_benchmark_longitudinal_catalog(_: Mapping[str, Any]) -> dict[str, Any]:
    return {"techniques": list_benchmark_longitudinal_ml_techniques()}


def _op_benchmark_longitudinal_run(params: Mapping[str, Any]) -> dict[str, Any]:
    return _run_tool_operation(
        params=params,
        runner=_resolve_runner(
            "ldt.machine_learning.tools.benchmark_longitudinal_ml.run",
            "run_benchmark_longitudinal_ml_tool",
        ),
    )


def _op_benchmark_mixed_catalog(_: Mapping[str, Any]) -> dict[str, Any]:
    return {"techniques": list_bench_standard_and_longitudinal_ml_techniques()}


def _op_benchmark_mixed_run(params: Mapping[str, Any]) -> dict[str, Any]:
    return _run_tool_operation(
        params=params,
        runner=_resolve_runner(
            "ldt.machine_learning.tools.bench_standard_and_longitudinal_ml.run",
            "run_bench_standard_and_longitudinal_ml_tool",
        ),
    )


def _op_shap_catalog(_: Mapping[str, Any]) -> dict[str, Any]:
    return {"techniques": list_shap_analysis_techniques()}


def _op_shap_run(params: Mapping[str, Any]) -> dict[str, Any]:
    return _run_tool_operation(
        params=params,
        runner=_resolve_runner(
            "ldt.machine_learning.tools.explainability.shap_analysis.run",
            "run_shap_analysis_tool",
        ),
    )


def _op_presets_catalog(_: Mapping[str, Any]) -> dict[str, Any]:
    return {"presets": list_machine_learning_presets()}


def _run_tool_operation(*, params: Mapping[str, Any], runner: Any) -> dict[str, Any]:
    technique = _as_required_string(params, "technique")
    raw_params = _as_object(params, "params")
    return runner(technique=technique, params=raw_params)


def _resolve_runner(module_path: str, function_name: str) -> Any:
    module = importlib.import_module(module_path)
    return getattr(module, function_name)


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
