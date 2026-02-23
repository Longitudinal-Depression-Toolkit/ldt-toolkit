from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_preprocessing.catalog import resolve_technique_with_defaults
from src.data_preprocessing.support.inputs import (
    as_choice,
    as_optional_string_list_or_csv,
    as_required_string,
    ensure_columns,
    run_with_validation,
)
from src.utils.errors import InputValidationError


def run_combine_dataset_with_trajectories(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Run a combine-dataset-with-trajectories technique.

    Resolve parameters, execute the technique, and return structured output.

    Args:
        technique (str): Technique or operation identifier.
        params (Mapping[str, Any]): Parameter mapping provided by the caller.

    Returns:
        dict[str, Any]: Dictionary containing tool results.

    Examples:
        ```python
        from ldt.data_preprocessing.tools.combine_dataset_with_trajectories.run import run_combine_dataset_with_trajectories
        result = run_combine_dataset_with_trajectories(technique=..., params=...)
        ```
    """

    return run_with_validation(
        lambda: _run_combine_dataset_with_trajectories(
            technique=technique, params=params
        )
    )


def _run_combine_dataset_with_trajectories(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    _, resolved = resolve_technique_with_defaults(
        section_key="combine_dataset_with_trajectories",
        technique_id=technique,
        provided_params=dict(params),
    )

    original_data_path = Path(
        as_required_string(resolved, "input_original_data_path")
    ).expanduser()
    trajectories_data_path = Path(
        as_required_string(resolved, "input_trajectories_data_path")
    ).expanduser()
    output_path = Path(as_required_string(resolved, "output_path")).expanduser()

    original_id_col = as_required_string(resolved, "original_id_col")
    trajectory_id_col = as_required_string(resolved, "trajectory_id_col")
    merge_type = as_choice(
        as_required_string(resolved, "merge_type"),
        choices=("left", "inner"),
        field_name="merge_type",
    )

    original_data = pd.read_csv(original_data_path)
    trajectories_data = pd.read_csv(trajectories_data_path)
    ensure_columns(original_data, [original_id_col])
    ensure_columns(trajectories_data, [trajectory_id_col])

    trajectory_columns = as_optional_string_list_or_csv(resolved, "trajectory_columns")
    if not trajectory_columns:
        trajectory_columns = [
            column
            for column in trajectories_data.columns
            if column != trajectory_id_col
        ]
    if not trajectory_columns:
        raise InputValidationError("No trajectory columns available to merge.")
    ensure_columns(trajectories_data, trajectory_columns)

    subset = trajectories_data[[trajectory_id_col, *trajectory_columns]].copy()
    subset = subset.dropna(subset=[trajectory_id_col])
    if subset.empty:
        raise InputValidationError("Trajectory data has no non-null IDs.")

    collapsed = subset.sort_values(by=[trajectory_id_col]).drop_duplicates(
        subset=[trajectory_id_col],
        keep="first",
    )

    combined = original_data.merge(
        collapsed,
        left_on=original_id_col,
        right_on=trajectory_id_col,
        how=merge_type,
        suffixes=("", "_trajectory"),
    )
    if original_id_col != trajectory_id_col and trajectory_id_col in combined.columns:
        combined = combined.drop(columns=[trajectory_id_col])

    unmatched_rows = 0
    if merge_type == "left" and trajectory_columns:
        unmatched_rows = int(combined[trajectory_columns].isna().all(axis=1).sum())

    drop_columns = as_optional_string_list_or_csv(resolved, "drop_columns")
    if drop_columns:
        ensure_columns(combined, drop_columns)
        combined = combined.drop(columns=drop_columns)

    if combined.empty:
        raise InputValidationError(
            "Combined dataset is empty after merge. Check IDs and merge_type."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    return {
        "output_path": str(output_path.resolve()),
        "row_count": int(len(combined)),
        "column_count": int(combined.shape[1]),
        "merge_type": merge_type,
        "trajectory_columns": list(trajectory_columns),
        "dropped_columns": list(drop_columns),
        "unmatched_rows": unmatched_rows,
    }
