from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from beartype import beartype

from ldt.data_preprocessing.catalog import resolve_technique_with_defaults
from ldt.utils.errors import InputValidationError
from ldt.utils.metadata import ComponentMetadata
from ldt.utils.templates.tools.data_preprocessing import DataPreprocessingTool


@beartype
@dataclass(frozen=True)
class CombineDatasetWithTrajectoriesResult:
    """Structured output for dataset and trajectory merge."""

    output_path: Path
    row_count: int
    column_count: int
    merge_type: str
    trajectory_columns: tuple[str, ...]
    dropped_columns: tuple[str, ...]
    unmatched_rows: int


@beartype
class CombineDatasetWithTrajectories(DataPreprocessingTool):
    """Merge a source dataset with trajectory assignments.

    Runtime parameters:
        - `input_original_data_path`: Source feature dataset CSV path.
        - `input_trajectories_data_path`: Trajectory assignment CSV path.
        - `output_path`: Combined output CSV path.
        - `original_id_col`: Subject ID column in the source dataset.
        - `trajectory_id_col`: Subject ID column in the trajectory dataset.
        - `merge_type`: `left` or `inner`.
        - `trajectory_columns`: Optional subset of trajectory columns to merge.
        - `drop_columns`: Optional columns to remove after merging.

    Examples:
        ```python
        from ldt.data_preprocessing import CombineDatasetWithTrajectories

        tool = CombineDatasetWithTrajectories()
        result = tool.fit_preprocess(
            input_original_data_path="./data/features.csv",
            input_trajectories_data_path="./outputs/trajectory_assignments.csv",
            output_path="./outputs/features_with_trajectories.csv",
            original_id_col="subject_id",
            trajectory_id_col="subject_id",
            merge_type="left",
            trajectory_columns=["trajectory_id", "trajectory_name"],
            drop_columns=[],
        )
        ```
    """

    metadata = ComponentMetadata(
        name="combine_dataset_with_trajectories",
        full_name="Combine Dataset with Trajectories",
        abstract_description="Merge a tabular dataset with trajectory assignments.",
    )

    def __init__(self) -> None:
        self._config: dict[str, Any] | None = None

    @beartype
    def fit(self, **kwargs: Any) -> CombineDatasetWithTrajectories:
        """Validate and store merge configuration.

        Args:
            **kwargs (Any): Configuration keys:
                - `input_original_data_path` (str | Path): Source dataset CSV.
                - `input_trajectories_data_path` (str | Path): Trajectory CSV.
                - `output_path` (str | Path): Output merged CSV.
                - `original_id_col` (str): Source subject ID column.
                - `trajectory_id_col` (str): Trajectory subject ID column.
                - `merge_type` (str): `left` or `inner`.
                - `trajectory_columns` (str | list[str] | None): Optional
                  columns to bring from trajectory data.
                - `drop_columns` (str | list[str] | None): Optional columns to
                  remove after merging.

        Returns:
            CombineDatasetWithTrajectories: The fitted tool instance.
        """

        original_path_raw = kwargs.get("input_original_data_path")
        trajectories_path_raw = kwargs.get("input_trajectories_data_path")
        output_path_raw = kwargs.get("output_path")

        if original_path_raw is None:
            raise InputValidationError(
                "Missing required parameter: input_original_data_path"
            )
        if trajectories_path_raw is None:
            raise InputValidationError(
                "Missing required parameter: input_trajectories_data_path"
            )
        if output_path_raw is None:
            raise InputValidationError("Missing required parameter: output_path")

        original_data_path = Path(str(original_path_raw)).expanduser()
        trajectories_data_path = Path(str(trajectories_path_raw)).expanduser()
        output_path = Path(str(output_path_raw)).expanduser()

        _validate_csv_path(original_data_path, field_name="input_original_data_path")
        _validate_csv_path(
            trajectories_data_path,
            field_name="input_trajectories_data_path",
        )
        _validate_output_csv_path(output_path)

        original_id_col = _as_required_string(
            kwargs.get("original_id_col"),
            field_name="original_id_col",
        )
        trajectory_id_col = _as_required_string(
            kwargs.get("trajectory_id_col"),
            field_name="trajectory_id_col",
        )
        merge_type = _as_choice(
            kwargs.get("merge_type", "left"),
            choices=("left", "inner"),
            field_name="merge_type",
        )

        trajectory_columns = _as_optional_string_list(kwargs.get("trajectory_columns"))
        drop_columns = _as_optional_string_list(kwargs.get("drop_columns"))

        self._config = {
            "input_original_data_path": original_data_path,
            "input_trajectories_data_path": trajectories_data_path,
            "output_path": output_path,
            "original_id_col": original_id_col,
            "trajectory_id_col": trajectory_id_col,
            "merge_type": merge_type,
            "trajectory_columns": tuple(trajectory_columns),
            "drop_columns": tuple(drop_columns),
        }
        return self

    @beartype
    def preprocess(self, **kwargs: Any) -> CombineDatasetWithTrajectoriesResult:
        """Merge configured datasets and write the combined CSV output.

        Args:
            **kwargs (Any): Optional configuration keys identical to `fit(...)`.
                If provided, they override any existing fitted configuration.

        Returns:
            CombineDatasetWithTrajectoriesResult: Typed summary of merge outputs
            and key diagnostics.
        """

        if kwargs:
            self.fit(**kwargs)
        if self._config is None:
            raise InputValidationError(
                "No configuration provided. Call `fit(...)` first or pass kwargs to `preprocess(...)`."
            )

        original_data_path = _as_path(
            self._config.get("input_original_data_path"),
            key="input_original_data_path",
        )
        trajectories_data_path = _as_path(
            self._config.get("input_trajectories_data_path"),
            key="input_trajectories_data_path",
        )
        output_path = _as_path(self._config.get("output_path"), key="output_path")
        original_id_col = str(self._config.get("original_id_col"))
        trajectory_id_col = str(self._config.get("trajectory_id_col"))
        merge_type = str(self._config.get("merge_type"))

        original_data = pd.read_csv(original_data_path)
        trajectories_data = pd.read_csv(trajectories_data_path)
        _ensure_columns(original_data, [original_id_col])
        _ensure_columns(trajectories_data, [trajectory_id_col])

        trajectory_columns = list(self._config.get("trajectory_columns", ()))
        if not trajectory_columns:
            trajectory_columns = [
                column
                for column in trajectories_data.columns
                if column != trajectory_id_col
            ]
        if not trajectory_columns:
            raise InputValidationError("No trajectory columns available to merge.")
        _ensure_columns(trajectories_data, trajectory_columns)

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
        if (
            original_id_col != trajectory_id_col
            and trajectory_id_col in combined.columns
        ):
            combined = combined.drop(columns=[trajectory_id_col])

        unmatched_rows = 0
        if merge_type == "left" and trajectory_columns:
            unmatched_rows = int(combined[trajectory_columns].isna().all(axis=1).sum())

        drop_columns = list(self._config.get("drop_columns", ()))
        if drop_columns:
            _ensure_columns(combined, drop_columns)
            combined = combined.drop(columns=drop_columns)

        if combined.empty:
            raise InputValidationError(
                "Combined dataset is empty after merge. Check IDs and merge_type."
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)

        return CombineDatasetWithTrajectoriesResult(
            output_path=output_path.resolve(),
            row_count=int(len(combined)),
            column_count=int(combined.shape[1]),
            merge_type=merge_type,
            trajectory_columns=tuple(trajectory_columns),
            dropped_columns=tuple(drop_columns),
            unmatched_rows=unmatched_rows,
        )


@beartype
def run_combine_dataset_with_trajectories(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one dataset-combination technique for the Go CLI bridge.

    !!! warning

        This `run_*` function is primarily for the Go CLI bridge and should not be
        treated as the Python library API. In Python scripts or notebooks, import
        and call `CombineDatasetWithTrajectories` directly from `ldt.data_preprocessing`.

    Args:
        technique (str): Combination technique key from the catalogue.
        params (Mapping[str, Any]): Wrapper parameters forwarded to
            `CombineDatasetWithTrajectories.fit_preprocess(...)`, including
            dataset paths, ID columns, merge options, and optional column lists.

    Returns:
        dict[str, Any]: Serialised merge summary for the Go CLI bridge.
    """

    _, resolved = resolve_technique_with_defaults(
        section_key="combine_dataset_with_trajectories",
        technique_id=technique,
        provided_params=dict(params),
    )

    result = CombineDatasetWithTrajectories().fit_preprocess(
        input_original_data_path=Path(
            _as_required_string(
                resolved.get("input_original_data_path"),
                field_name="input_original_data_path",
            )
        ).expanduser(),
        input_trajectories_data_path=Path(
            _as_required_string(
                resolved.get("input_trajectories_data_path"),
                field_name="input_trajectories_data_path",
            )
        ).expanduser(),
        output_path=Path(
            _as_required_string(resolved.get("output_path"), field_name="output_path")
        ).expanduser(),
        original_id_col=_as_required_string(
            resolved.get("original_id_col"),
            field_name="original_id_col",
        ),
        trajectory_id_col=_as_required_string(
            resolved.get("trajectory_id_col"),
            field_name="trajectory_id_col",
        ),
        merge_type=_as_choice(
            resolved.get("merge_type", "left"),
            choices=("left", "inner"),
            field_name="merge_type",
        ),
        trajectory_columns=_as_optional_string_list(resolved.get("trajectory_columns")),
        drop_columns=_as_optional_string_list(resolved.get("drop_columns")),
    )

    return {
        "output_path": str(result.output_path),
        "row_count": result.row_count,
        "column_count": result.column_count,
        "merge_type": result.merge_type,
        "trajectory_columns": list(result.trajectory_columns),
        "dropped_columns": list(result.dropped_columns),
        "unmatched_rows": result.unmatched_rows,
    }


@beartype
def _as_required_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"Missing required string parameter: {field_name}")
    return value.strip()


@beartype
def _as_optional_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        parsed: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise InputValidationError("List parameters must contain strings.")
            stripped = item.strip()
            if stripped:
                parsed.append(stripped)
        return parsed
    raise InputValidationError("Expected a CSV string or string list.")


@beartype
def _as_choice(value: Any, *, choices: tuple[str, ...], field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"Missing required string parameter: {field_name}")
    candidate = value.strip().lower()
    if candidate not in choices:
        raise InputValidationError(
            f"`{field_name}` must be one of: {', '.join(choices)}."
        )
    return candidate


@beartype
def _as_path(value: Any, *, key: str) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value.strip():
        return Path(value).expanduser()
    raise InputValidationError(f"Missing required path parameter: {key}")


@beartype
def _ensure_columns(data: pd.DataFrame, required: list[str]) -> None:
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise InputValidationError(
            f"Missing required columns: {', '.join(sorted(missing))}"
        )


@beartype
def _validate_csv_path(path: Path, *, field_name: str) -> None:
    if not path.exists() or not path.is_file():
        raise InputValidationError(f"{field_name} does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise InputValidationError(f"{field_name} must point to a .csv file.")


@beartype
def _validate_output_csv_path(path: Path) -> None:
    if path.suffix.lower() != ".csv":
        raise InputValidationError("output_path must point to a .csv file.")
