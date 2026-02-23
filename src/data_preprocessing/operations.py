from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.utils.errors import InputValidationError
from src.utils.operation_registry import OperationRegistry

from . import (
    RemoveColumnsRequest,
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
    resolve_technique_with_defaults,
    run_aggregate_long_to_cross_sectional,
    run_build_trajectories,
    run_clean_dataset,
    run_combine_dataset_with_trajectories,
    run_harmonise_categories,
    run_missing_imputation,
    run_pivot_long_to_wide,
    run_remove_columns,
    run_rename_feature,
    run_show_table,
    run_trajectories_viz,
)


def register_operations(registry: OperationRegistry) -> None:
    """Register data-preprocessing operation handlers on a registry.

    Args:
        registry (OperationRegistry): Operation registry instance to configure.
    """
    registry.register(
        "data_preprocessing.remove_columns",
        _op_remove_columns,
        description="Remove a set of columns from a tabular dataset.",
    )
    registry.register(
        "data_preprocessing.remove_columns.catalog",
        _op_remove_columns_catalog,
        description="List remove-columns techniques.",
    )
    registry.register(
        "data_preprocessing.remove_columns.run",
        _op_remove_columns_run,
        description="Run one remove-columns technique.",
    )
    registry.register(
        "data_preprocessing.build_trajectories.catalog",
        _op_build_trajectories_catalog,
        description="List build-trajectories techniques.",
    )
    registry.register(
        "data_preprocessing.build_trajectories.run",
        _op_build_trajectories_run,
        description="Run one build-trajectories action.",
    )
    registry.register(
        "data_preprocessing.combine_dataset_with_trajectories.catalog",
        _op_combine_dataset_catalog,
        description="List dataset-combination techniques.",
    )
    registry.register(
        "data_preprocessing.combine_dataset_with_trajectories.run",
        _op_combine_dataset_run,
        description="Run one dataset-combination action.",
    )
    registry.register(
        "data_preprocessing.clean_dataset.catalog",
        _op_clean_dataset_catalog,
        description="List clean-dataset techniques.",
    )
    registry.register(
        "data_preprocessing.clean_dataset.run",
        _op_clean_dataset_run,
        description="Run one clean-dataset action.",
    )
    registry.register(
        "data_preprocessing.missing_imputation.catalog",
        _op_missing_imputation_catalog,
        description="List missing-imputation techniques.",
    )
    registry.register(
        "data_preprocessing.missing_imputation.run",
        _op_missing_imputation_run,
        description="Run one missing-imputation action.",
    )
    registry.register(
        "data_preprocessing.harmonise_categories.catalog",
        _op_harmonise_categories_catalog,
        description="List harmonise-categories techniques.",
    )
    registry.register(
        "data_preprocessing.harmonise_categories.run",
        _op_harmonise_categories_run,
        description="Run one harmonise-categories action.",
    )
    registry.register(
        "data_preprocessing.show_table.catalog",
        _op_show_table_catalog,
        description="List show-table techniques.",
    )
    registry.register(
        "data_preprocessing.show_table.run",
        _op_show_table_run,
        description="Run one show-table action.",
    )
    registry.register(
        "data_preprocessing.aggregate_long_to_cross_sectional.catalog",
        _op_aggregate_catalog,
        description="List aggregate-long-to-cross-sectional techniques.",
    )
    registry.register(
        "data_preprocessing.aggregate_long_to_cross_sectional.run",
        _op_aggregate_run,
        description="Run one aggregate-long-to-cross-sectional action.",
    )
    registry.register(
        "data_preprocessing.rename_feature.catalog",
        _op_rename_feature_catalog,
        description="List rename-feature techniques.",
    )
    registry.register(
        "data_preprocessing.rename_feature.run",
        _op_rename_feature_run,
        description="Run one rename-feature action.",
    )
    registry.register(
        "data_preprocessing.pivot_long_to_wide.catalog",
        _op_pivot_long_to_wide_catalog,
        description="List pivot-long-to-wide techniques.",
    )
    registry.register(
        "data_preprocessing.pivot_long_to_wide.run",
        _op_pivot_long_to_wide_run,
        description="Run one pivot-long-to-wide action.",
    )
    registry.register(
        "data_preprocessing.trajectories_viz.catalog",
        _op_trajectories_viz_catalog,
        description="List trajectories-viz techniques.",
    )
    registry.register(
        "data_preprocessing.trajectories_viz.run",
        _op_trajectories_viz_run,
        description="Run one trajectories-viz action.",
    )


def _op_remove_columns(params: Mapping[str, Any]) -> dict[str, Any]:
    return _run_remove_columns_operation("remove_columns", dict(params))


def _op_remove_columns_catalog(_: Mapping[str, Any]) -> dict[str, Any]:
    return {"techniques": list_remove_columns_techniques()}


def _op_remove_columns_run(params: Mapping[str, Any]) -> dict[str, Any]:
    technique, raw_params = _extract_technique_and_params(params)
    return _run_remove_columns_operation(technique, raw_params)


def _run_remove_columns_operation(
    technique: str,
    raw_params: Mapping[str, Any],
) -> dict[str, Any]:
    _, resolved_params = resolve_technique_with_defaults(
        section_key="remove_columns",
        technique_id=technique,
        provided_params=dict(raw_params),
    )

    input_path = Path(_as_required_string(resolved_params, "input_path")).expanduser()
    output_path_raw = _as_optional_string(resolved_params, "output_path")
    output_path = (
        Path(output_path_raw).expanduser()
        if output_path_raw is not None
        else _default_remove_columns_output(input_path)
    )
    columns = tuple(_as_required_string_list_or_csv(resolved_params, "columns"))

    request = RemoveColumnsRequest(
        input_path=input_path,
        output_path=output_path,
        columns=columns,
    )
    result = run_remove_columns(request)
    return {
        "output_path": str(result.output_path),
        "row_count": result.row_count,
        "column_count": result.column_count,
        "removed_columns": list(result.removed_columns),
    }


def _op_build_trajectories_catalog(_: Mapping[str, Any]) -> dict[str, Any]:
    return {"techniques": list_build_trajectories_techniques()}


def _op_build_trajectories_run(params: Mapping[str, Any]) -> dict[str, Any]:
    return _run_tool_operation(params=params, runner=run_build_trajectories)


def _op_combine_dataset_catalog(_: Mapping[str, Any]) -> dict[str, Any]:
    return {"techniques": list_combine_dataset_with_trajectories_techniques()}


def _op_combine_dataset_run(params: Mapping[str, Any]) -> dict[str, Any]:
    return _run_tool_operation(
        params=params, runner=run_combine_dataset_with_trajectories
    )


def _op_clean_dataset_catalog(_: Mapping[str, Any]) -> dict[str, Any]:
    return {"techniques": list_clean_dataset_techniques()}


def _op_clean_dataset_run(params: Mapping[str, Any]) -> dict[str, Any]:
    return _run_tool_operation(params=params, runner=run_clean_dataset)


def _op_missing_imputation_catalog(_: Mapping[str, Any]) -> dict[str, Any]:
    return {"techniques": list_missing_imputation_techniques()}


def _op_missing_imputation_run(params: Mapping[str, Any]) -> dict[str, Any]:
    return _run_tool_operation(params=params, runner=run_missing_imputation)


def _op_harmonise_categories_catalog(_: Mapping[str, Any]) -> dict[str, Any]:
    return {"techniques": list_harmonise_categories_techniques()}


def _op_harmonise_categories_run(params: Mapping[str, Any]) -> dict[str, Any]:
    return _run_tool_operation(params=params, runner=run_harmonise_categories)


def _op_show_table_catalog(_: Mapping[str, Any]) -> dict[str, Any]:
    return {"techniques": list_show_table_techniques()}


def _op_show_table_run(params: Mapping[str, Any]) -> dict[str, Any]:
    return _run_tool_operation(params=params, runner=run_show_table)


def _op_aggregate_catalog(_: Mapping[str, Any]) -> dict[str, Any]:
    return {"techniques": list_aggregate_long_to_cross_sectional_techniques()}


def _op_aggregate_run(params: Mapping[str, Any]) -> dict[str, Any]:
    return _run_tool_operation(
        params=params, runner=run_aggregate_long_to_cross_sectional
    )


def _op_rename_feature_catalog(_: Mapping[str, Any]) -> dict[str, Any]:
    return {"techniques": list_rename_feature_techniques()}


def _op_rename_feature_run(params: Mapping[str, Any]) -> dict[str, Any]:
    return _run_tool_operation(params=params, runner=run_rename_feature)


def _op_pivot_long_to_wide_catalog(_: Mapping[str, Any]) -> dict[str, Any]:
    return {"techniques": list_pivot_long_to_wide_techniques()}


def _op_pivot_long_to_wide_run(params: Mapping[str, Any]) -> dict[str, Any]:
    return _run_tool_operation(params=params, runner=run_pivot_long_to_wide)


def _op_trajectories_viz_catalog(_: Mapping[str, Any]) -> dict[str, Any]:
    return {"techniques": list_trajectories_viz_techniques()}


def _op_trajectories_viz_run(params: Mapping[str, Any]) -> dict[str, Any]:
    return _run_tool_operation(params=params, runner=run_trajectories_viz)


def _run_tool_operation(*, params: Mapping[str, Any], runner: Any) -> dict[str, Any]:
    technique, raw_params = _extract_technique_and_params(params)
    return runner(technique=technique, params=raw_params)


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


def _as_optional_string(params: Mapping[str, Any], key: str) -> str | None:
    value = params.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed if trimmed else None
    raise InputValidationError(f"`{key}` must be a string when provided.")


def _as_required_string_list(params: Mapping[str, Any], key: str) -> list[str]:
    value = params.get(key)
    if not isinstance(value, list) or not value:
        raise InputValidationError(f"Missing required string-list parameter: {key}")

    parsed: list[str] = []
    for entry in value:
        if not isinstance(entry, str):
            raise InputValidationError(f"Parameter `{key}` entries must be strings.")
        candidate = entry.strip()
        if candidate:
            parsed.append(candidate)
    if not parsed:
        raise InputValidationError(f"Parameter `{key}` must include at least one item.")
    return parsed


def _as_required_string_list_or_csv(params: Mapping[str, Any], key: str) -> list[str]:
    value = params.get(key)
    if isinstance(value, list):
        return _as_required_string_list(params, key)
    if isinstance(value, str):
        parsed = [entry.strip() for entry in value.split(",") if entry.strip()]
        if parsed:
            return parsed
    raise InputValidationError(f"Missing required string-list parameter: {key}")


def _default_remove_columns_output(input_path: Path) -> Path:
    stem = input_path.stem
    suffix = input_path.suffix if input_path.suffix else ".csv"
    return input_path.with_name(f"{stem}_columns_removed{suffix}")
