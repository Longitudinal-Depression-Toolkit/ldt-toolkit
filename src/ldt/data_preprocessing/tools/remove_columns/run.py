from __future__ import annotations

from collections.abc import Mapping, Sequence
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
class RemoveColumnsResult:
    """Structured output for column removal."""

    output_path: Path
    row_count: int
    column_count: int
    removed_columns: tuple[str, ...]


@beartype
class RemoveColumns(DataPreprocessingTool):
    """Remove selected columns from a CSV dataset.

    This tool validates requested columns, ensures at least one column remains,
    writes a new CSV, and returns output metadata.

    Initialisation parameters:
        input_path (Path | str | None): Optional default input CSV path.
        output_path (Path | str | None): Optional default output CSV path.
        columns (Sequence[str] | None): Optional default columns to remove.

    Examples:
        ```python
        from ldt.data_preprocessing import RemoveColumns

        tool = RemoveColumns()
        result = tool.fit_preprocess(
            input_path="./data/cleaned.csv",
            output_path="./data/cleaned_selected.csv",
            columns=["redundant_flag", "temp_feature"],
        )
        ```
    """

    metadata = ComponentMetadata(
        name="remove_columns",
        full_name="Remove Columns",
        abstract_description="Remove one or more selected columns from a CSV dataset.",
    )

    def __init__(
        self,
        *,
        input_path: Path | str | None = None,
        output_path: Path | str | None = None,
        columns: Sequence[str] | None = None,
    ) -> None:
        self._input_path = (
            Path(input_path).expanduser() if input_path is not None else None
        )
        self._output_path = (
            Path(output_path).expanduser() if output_path is not None else None
        )
        self._columns = tuple(
            str(column).strip() for column in (columns or ()) if str(column).strip()
        )

    @beartype
    def fit(self, **kwargs: Any) -> RemoveColumns:
        """Validate and store column-removal configuration.

        Args:
            **kwargs (Any): Configuration overrides:
                - `input_path` (str | Path): Input CSV path.
                - `output_path` (str | Path): Output CSV path.
                - `columns` (str | Sequence[str]): Columns to remove.

        Returns:
            RemoveColumns: The fitted tool instance.
        """

        input_path = kwargs.get("input_path", self._input_path)
        output_path = kwargs.get("output_path", self._output_path)
        raw_columns = kwargs.get("columns", self._columns)

        if input_path is None:
            raise InputValidationError("Missing required parameter: input_path")
        if output_path is None:
            raise InputValidationError("Missing required parameter: output_path")

        parsed_columns = _as_columns(raw_columns)

        self._input_path = Path(str(input_path)).expanduser()
        self._output_path = Path(str(output_path)).expanduser()
        self._columns = parsed_columns

        _validate_input_csv_path(self._input_path)
        _validate_output_csv_path(self._output_path, input_path=self._input_path)
        return self

    @beartype
    def preprocess(self, **kwargs: Any) -> RemoveColumnsResult:
        """Remove configured columns and write the transformed CSV file.

        Args:
            **kwargs (Any): Optional configuration keys identical to `fit(...)`.
                If provided, they override any existing fitted configuration.

        Returns:
            RemoveColumnsResult: Typed summary of the transformed output.
        """

        if kwargs:
            self.fit(**kwargs)
        if self._input_path is None or self._output_path is None:
            raise InputValidationError(
                "No configuration provided. Call `fit(...)` first or pass kwargs to `preprocess(...)`."
            )

        data = pd.read_csv(self._input_path)
        columns_to_drop = _resolve_columns_to_drop(
            available_columns=list(data.columns),
            columns=self._columns,
        )
        updated = data.drop(columns=list(columns_to_drop))
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        updated.to_csv(self._output_path, index=False)

        return RemoveColumnsResult(
            output_path=self._output_path.resolve(),
            row_count=len(updated),
            column_count=int(updated.shape[1]),
            removed_columns=columns_to_drop,
        )


@beartype
def parse_columns_csv(raw_columns: str) -> tuple[str, ...]:
    """Parse comma-separated column names into a normalised tuple."""

    parsed = tuple(item.strip() for item in raw_columns.split(",") if item.strip())
    if not parsed:
        raise InputValidationError("At least one column must be provided.")
    return parsed


@beartype
def run_remove_columns(*, technique: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Run one remove-columns technique for the Go CLI bridge.

    !!! warning

        This `run_*` function is primarily for the Go CLI bridge and should not be
        treated as the Python library API. In Python scripts or notebooks, import
        and call `RemoveColumns` directly from `ldt.data_preprocessing`.

    Args:
        technique (str): Remove-columns technique key from the catalogue.
        params (Mapping[str, Any]): Wrapper parameters forwarded to
            `RemoveColumns.fit_preprocess(...)`: `input_path`, optional
            `output_path`, and `columns`.

    Returns:
        dict[str, Any]: Serialised column-removal summary for the Go CLI bridge.
    """

    _, resolved = resolve_technique_with_defaults(
        section_key="remove_columns",
        technique_id=technique,
        provided_params=dict(params),
    )

    input_path = Path(_as_required_string(resolved, "input_path")).expanduser()
    output_path_raw = _as_optional_string(resolved.get("output_path"))
    output_path = (
        Path(output_path_raw).expanduser()
        if output_path_raw is not None
        else _default_remove_columns_output(input_path)
    )
    columns = _as_columns(resolved.get("columns"))

    result = RemoveColumns().fit_preprocess(
        input_path=input_path,
        output_path=output_path,
        columns=columns,
    )
    return {
        "output_path": str(result.output_path),
        "row_count": result.row_count,
        "column_count": result.column_count,
        "removed_columns": list(result.removed_columns),
    }


@beartype
def _as_required_string(params: Mapping[str, Any], key: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"Missing required string parameter: {key}")
    return value.strip()


@beartype
def _as_optional_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


@beartype
def _as_columns(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return parse_columns_csv(value)

    if isinstance(value, Sequence):
        parsed = tuple(
            str(item).strip()
            for item in value
            if isinstance(item, str) and str(item).strip()
        )
        if parsed:
            return parsed
    raise InputValidationError("Missing required string-list parameter: columns")


@beartype
def _resolve_columns_to_drop(
    *,
    available_columns: list[str],
    columns: tuple[str, ...],
) -> tuple[str, ...]:
    duplicates = _find_duplicates(columns)
    if duplicates:
        raise InputValidationError(
            f"Columns to remove contain duplicates: {', '.join(duplicates)}"
        )

    missing = [column for column in columns if column not in available_columns]
    if missing:
        raise InputValidationError(
            f"Requested columns do not exist in input CSV: {', '.join(missing)}"
        )
    if len(columns) == len(available_columns):
        raise InputValidationError(
            "Refusing to remove all columns. Keep at least one column in output."
        )
    return columns


@beartype
def _find_duplicates(values: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    duplicates_seen: set[str] = set()
    for value in values:
        if value in seen and value not in duplicates_seen:
            duplicates.append(value)
            duplicates_seen.add(value)
        seen.add(value)
    return duplicates


@beartype
def _default_remove_columns_output(input_path: Path) -> Path:
    stem = input_path.stem
    suffix = input_path.suffix if input_path.suffix else ".csv"
    return input_path.with_name(f"{stem}_columns_removed{suffix}")


@beartype
def _validate_input_csv_path(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise InputValidationError(f"Input CSV path does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise InputValidationError("Input path must point to a .csv file.")


@beartype
def _validate_output_csv_path(path: Path, *, input_path: Path) -> None:
    if path.suffix.lower() != ".csv":
        raise InputValidationError("Output path must point to a .csv file.")
    if path.resolve() == input_path.resolve():
        raise InputValidationError("Output path must be different from input path.")
    if path.exists():
        raise InputValidationError(
            f"Output file already exists: {path}. Provide a new output path."
        )
