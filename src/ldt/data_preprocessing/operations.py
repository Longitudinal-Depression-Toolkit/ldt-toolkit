from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from ldt.bridge.operation_registry import OperationRegistry
from ldt.utils.errors import InputValidationError

from .catalog import (
    list_aggregate_long_to_cross_sectional_techniques,
    list_build_trajectories_techniques,
    list_clean_dataset_techniques,
    list_combine_dataset_with_trajectories_techniques,
    list_harmonise_categories_techniques,
    list_missing_imputation_techniques,
    list_pivot_long_to_wide_techniques,
    list_remove_columns_techniques,
    list_rename_feature_techniques,
    list_show_table_techniques,
    list_trajectories_viz_techniques,
)


def register_operations(registry: OperationRegistry) -> None:
    """Register data-preprocessing operation handlers on a registry."""

    registry.register(
        "data_preprocessing.remove_columns",
        _op_remove_columns,
        description="Remove selected columns from a tabular dataset.",
    )
    registry.register(
        "data_preprocessing.remove_columns.catalog",
        lambda _: {"techniques": list_remove_columns_techniques()},
        description="List remove-columns techniques.",
    )
    registry.register(
        "data_preprocessing.remove_columns.run",
        lambda params: _run_tool_operation_by_path(
            params=params,
            module_path="ldt.data_preprocessing.tools.remove_columns.run",
            function_name="run_remove_columns",
        ),
        description="Run one remove-columns technique.",
    )

    registry.register(
        "data_preprocessing.build_trajectories.catalog",
        lambda _: {"techniques": list_build_trajectories_techniques()},
        description="List build-trajectories techniques.",
    )
    registry.register(
        "data_preprocessing.build_trajectories.run",
        lambda params: _run_tool_operation_by_path(
            params=params,
            module_path="ldt.data_preprocessing.tools.build_trajectories.run",
            function_name="run_build_trajectories",
        ),
        description="Run one build-trajectories action.",
    )

    registry.register(
        "data_preprocessing.combine_dataset_with_trajectories.catalog",
        lambda _: {"techniques": list_combine_dataset_with_trajectories_techniques()},
        description="List dataset-combination techniques.",
    )
    registry.register(
        "data_preprocessing.combine_dataset_with_trajectories.run",
        lambda params: _run_tool_operation_by_path(
            params=params,
            module_path="ldt.data_preprocessing.tools.combine_dataset_with_trajectories.run",
            function_name="run_combine_dataset_with_trajectories",
        ),
        description="Run one dataset-combination action.",
    )

    registry.register(
        "data_preprocessing.clean_dataset.catalog",
        lambda _: {"techniques": list_clean_dataset_techniques()},
        description="List clean-dataset techniques.",
    )
    registry.register(
        "data_preprocessing.clean_dataset.run",
        lambda params: _run_tool_operation_by_path(
            params=params,
            module_path="ldt.data_preprocessing.tools.clean_dataset.run",
            function_name="run_clean_dataset",
        ),
        description="Run one clean-dataset action.",
    )

    registry.register(
        "data_preprocessing.missing_imputation.catalog",
        lambda _: {"techniques": list_missing_imputation_techniques()},
        description="List missing-imputation techniques.",
    )
    registry.register(
        "data_preprocessing.missing_imputation.run",
        lambda params: _run_tool_operation_by_path(
            params=params,
            module_path="ldt.data_preprocessing.tools.missing_imputation.run",
            function_name="run_missing_imputation",
        ),
        description="Run one missing-imputation action.",
    )

    registry.register(
        "data_preprocessing.harmonise_categories.catalog",
        lambda _: {"techniques": list_harmonise_categories_techniques()},
        description="List harmonise-categories techniques.",
    )
    registry.register(
        "data_preprocessing.harmonise_categories.run",
        lambda params: _run_tool_operation_by_path(
            params=params,
            module_path="ldt.data_preprocessing.tools.harmonise_categories.run",
            function_name="run_harmonise_categories",
        ),
        description="Run one harmonise-categories action.",
    )

    registry.register(
        "data_preprocessing.show_table.catalog",
        lambda _: {"techniques": list_show_table_techniques()},
        description="List show-table techniques.",
    )
    registry.register(
        "data_preprocessing.show_table.run",
        lambda params: _run_tool_operation_by_path(
            params=params,
            module_path="ldt.data_preprocessing.tools.show_table.run",
            function_name="run_show_table",
        ),
        description="Run one show-table action.",
    )

    registry.register(
        "data_preprocessing.aggregate_long_to_cross_sectional.catalog",
        lambda _: {"techniques": list_aggregate_long_to_cross_sectional_techniques()},
        description="List aggregate-long-to-cross-sectional techniques.",
    )
    registry.register(
        "data_preprocessing.aggregate_long_to_cross_sectional.run",
        lambda params: _run_tool_operation_by_path(
            params=params,
            module_path="ldt.data_preprocessing.tools.aggregate_long_to_cross_sectional.run",
            function_name="run_aggregate_long_to_cross_sectional",
        ),
        description="Run one aggregate-long-to-cross-sectional action.",
    )

    registry.register(
        "data_preprocessing.rename_feature.catalog",
        lambda _: {"techniques": list_rename_feature_techniques()},
        description="List rename-feature techniques.",
    )
    registry.register(
        "data_preprocessing.rename_feature.run",
        lambda params: _run_tool_operation_by_path(
            params=params,
            module_path="ldt.data_preprocessing.tools.rename_feature.run",
            function_name="run_rename_feature",
        ),
        description="Run one rename-feature action.",
    )

    registry.register(
        "data_preprocessing.pivot_long_to_wide.catalog",
        lambda _: {"techniques": list_pivot_long_to_wide_techniques()},
        description="List pivot-long-to-wide techniques.",
    )
    registry.register(
        "data_preprocessing.pivot_long_to_wide.run",
        lambda params: _run_tool_operation_by_path(
            params=params,
            module_path="ldt.data_preprocessing.tools.pivot_long_to_wide.run",
            function_name="run_pivot_long_to_wide",
        ),
        description="Run one pivot-long-to-wide action.",
    )

    registry.register(
        "data_preprocessing.trajectories_viz.catalog",
        lambda _: {"techniques": list_trajectories_viz_techniques()},
        description="List trajectories-viz techniques.",
    )
    registry.register(
        "data_preprocessing.trajectories_viz.run",
        lambda params: _run_tool_operation_by_path(
            params=params,
            module_path="ldt.data_preprocessing.tools.trajectories_viz.run",
            function_name="run_trajectories_viz",
        ),
        description="Run one trajectories-viz action.",
    )


def _op_remove_columns(params: Mapping[str, Any]) -> dict[str, Any]:
    """Run the default remove-columns technique directly."""

    return _resolve_runner(
        "ldt.data_preprocessing.tools.remove_columns.run",
        "run_remove_columns",
    )(technique="remove_columns", params=dict(params))


def _run_tool_operation_by_path(
    *,
    params: Mapping[str, Any],
    module_path: str,
    function_name: str,
) -> dict[str, Any]:
    runner = _resolve_runner(module_path, function_name)
    return _run_tool_operation(params=params, runner=runner)


def _run_tool_operation(*, params: Mapping[str, Any], runner: Any) -> dict[str, Any]:
    technique, raw_params = _extract_technique_and_params(params)
    return runner(technique=technique, params=raw_params)


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
