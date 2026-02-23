from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.errors import InputValidationError


@dataclass(frozen=True)
class RemoveColumnsRequest:
    """Request payload for remove-columns execution.

    Attributes:
        input_path (Path): Input CSV path.
        output_path (Path): Output CSV path.
        columns (tuple[str, ...]): Column names to remove.
    """

    input_path: Path
    output_path: Path
    columns: tuple[str, ...]


@dataclass(frozen=True)
class RemoveColumnsResult:
    """Result payload produced by remove-columns execution.

    Attributes:
        output_path (Path): Output CSV path.
        row_count (int): Number of rows in output dataset.
        column_count (int): Number of columns in output dataset.
        removed_columns (tuple[str, ...]): Removed column names.
    """

    output_path: Path
    row_count: int
    column_count: int
    removed_columns: tuple[str, ...]


def parse_columns_csv(raw_columns: str) -> tuple[str, ...]:
    """Parse comma-separated column names into a normalised tuple.

    This helper is used by higher-level workflows and keeps input/output handling consistent.

    Args:
        raw_columns (str): Comma-separated column names.

    Returns:
        tuple[str, ...]: Tuple of resolved values.
    """

    parsed = tuple(item.strip() for item in raw_columns.split(",") if item.strip())
    if not parsed:
        raise InputValidationError("At least one column must be provided.")
    return parsed


def run_remove_columns(request: RemoveColumnsRequest) -> RemoveColumnsResult:
    """Remove selected columns from a CSV dataset.

    This utility drops the requested columns, validates that none are missing,
    and prevents removing every column.

    Fictional input:

    | subject_id | mood_score | sleep_score | redundant_flag |
    | --- | --- | --- | --- |
    | 101 | 4.0 | 6.0 | 1 |
    | 102 | 7.0 | 8.0 | 1 |

    Requested `columns=("redundant_flag",)` produces:

    | subject_id | mood_score | sleep_score |
    | --- | --- | --- |
    | 101 | 4.0 | 6.0 |
    | 102 | 7.0 | 8.0 |

    Args:
        request (RemoveColumnsRequest): Input/output paths and the columns to
            remove.

    Returns:
        RemoveColumnsResult: Output path, output dimensions, and removed column
            names.
    """

    _validate_input_csv_path(request.input_path)
    _validate_output_csv_path(request.output_path, input_path=request.input_path)

    data = pd.read_csv(request.input_path)
    columns_to_drop = _resolve_columns_to_drop(
        available_columns=list(data.columns),
        columns=request.columns,
    )
    updated = data.drop(columns=list(columns_to_drop))
    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(request.output_path, index=False)

    return RemoveColumnsResult(
        output_path=request.output_path.resolve(),
        row_count=len(updated),
        column_count=int(updated.shape[1]),
        removed_columns=columns_to_drop,
    )


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


def _validate_input_csv_path(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise InputValidationError(f"Input CSV path does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise InputValidationError("Input path must point to a .csv file.")


def _validate_output_csv_path(path: Path, *, input_path: Path) -> None:
    if path.suffix.lower() != ".csv":
        raise InputValidationError("Output path must point to a .csv file.")
    if path.resolve() == input_path.resolve():
        raise InputValidationError("Output path must be different from input path.")
    if path.exists():
        raise InputValidationError(
            f"Output file already exists: {path}. Provide a new output path."
        )
