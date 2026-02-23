from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_preprocessing.catalog import resolve_technique_with_defaults
from src.data_preprocessing.support.inputs import (
    as_bool,
    as_optional_float,
    as_optional_string,
    as_required_string,
    ensure_distinct_paths,
    run_with_validation,
)
from src.data_preprocessing.support.skrub_compat import import_skrub_symbol
from src.utils.errors import InputValidationError

Cleaner = import_skrub_symbol("Cleaner")


@dataclass(frozen=True)
class CleanDatasetRequest:
    """Request payload for dataset cleaning.

    Attributes:
        input_path (Path): Input CSV path.
        output_path (Path): Output CSV path.
        drop_null_fraction (float | None): Optional missingness threshold for
            dropping sparse columns.
        drop_if_constant (bool): Whether to remove constant-valued columns.
        drop_if_unique (bool): Whether to remove high-cardinality identifier-like
            columns.
        datetime_format (str | None): Optional datetime parsing format.
        cast_to_str (bool): Whether to cast compatible values to strings.
        numeric_to_float32 (bool): Whether to downcast numerics to `float32`.
    """

    input_path: Path
    output_path: Path
    drop_null_fraction: float | None
    drop_if_constant: bool
    drop_if_unique: bool
    datetime_format: str | None
    cast_to_str: bool
    numeric_to_float32: bool


@dataclass(frozen=True)
class CleanDatasetResult:
    """Result payload produced after dataset cleaning.

    Attributes:
        output_path (Path): Output CSV path.
        row_count (int): Number of rows in cleaned dataset.
        column_count (int): Number of columns in cleaned dataset.
        dropped_columns (tuple[str, ...]): Columns removed by cleaning stages.
    """

    output_path: Path
    row_count: int
    column_count: int
    dropped_columns: tuple[str, ...]


def run_clean_dataset(request: CleanDatasetRequest) -> CleanDatasetResult:
    """Clean a CSV dataset with `skrub.Cleaner`.

    The cleaner standardises table structure and optionally removes low-signal
    columns based on nullness or cardinality heuristics.

    Available cleaning stages:

    | Stage | Parameter(s) | Effect |
    | --- | --- | --- |
    | Drop sparse columns | `drop_null_fraction` | Removes columns whose missing-value proportion exceeds the threshold. |
    | Drop constant columns | `drop_if_constant` | Removes columns with only one repeated value. |
    | Drop near-identifier columns | `drop_if_unique` | Removes columns where almost every row is unique (often IDs). |
    | Parse datetime strings | `datetime_format` | Applies datetime parsing to compatible text columns. |
    | Cast values to string | `cast_to_str` | Casts supported values to string representation. |
    | Downcast numeric type | `numeric_to_float32` | Stores numeric columns as `float32` when possible. |

    Args:
        request (CleanDatasetRequest): Cleaning configuration and input/output
            paths.

    Returns:
        CleanDatasetResult: Output dataset summary and dropped-column list.

    Examples:
        ```python
        from ldt.data_preprocessing.tools.clean_dataset.run import (
            CleanDatasetRequest,
            run_clean_dataset,
        )

        result = run_clean_dataset(
            CleanDatasetRequest(
                input_path="data/raw.csv",
                output_path="outputs/cleaned.csv",
                drop_null_fraction=0.95,
                drop_if_constant=True,
                drop_if_unique=True,
                datetime_format=None,
                cast_to_str=False,
                numeric_to_float32=True,
            )
        )
        ```
    """

    _validate_input_csv_path(request.input_path)
    _validate_output_csv_path(request.output_path, input_path=request.input_path)
    if request.drop_null_fraction is not None and not (
        0 <= request.drop_null_fraction <= 1
    ):
        raise InputValidationError(
            "drop_null_fraction must be within [0, 1] when provided."
        )

    original = pd.read_csv(request.input_path)
    candidate_cleaner_kwargs = {
        "drop_null_fraction": request.drop_null_fraction,
        "drop_if_constant": request.drop_if_constant,
        "drop_if_unique": request.drop_if_unique,
        "datetime_format": request.datetime_format,
        "numeric_dtype": "float32" if request.numeric_to_float32 else None,
        "cast_to_str": request.cast_to_str,
    }
    supported_params = inspect.signature(Cleaner.__init__).parameters
    cleaner_kwargs = {
        key: value
        for key, value in candidate_cleaner_kwargs.items()
        if key in supported_params
    }

    cleaner = Cleaner(**cleaner_kwargs)
    cleaned = _to_pandas_frame(cleaner.fit_transform(original))
    dropped_columns = tuple(sorted(set(original.columns) - set(cleaned.columns)))

    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(request.output_path, index=False)

    return CleanDatasetResult(
        output_path=request.output_path.resolve(),
        row_count=len(cleaned),
        column_count=int(cleaned.shape[1]),
        dropped_columns=dropped_columns,
    )


def run_clean_dataset_tool(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Run dataset-cleaning techniques from catalog payloads.

    Supported techniques:

    | Technique | What it does |
    | --- | --- |
    | `clean` | Applies configured `skrub.Cleaner` stages and writes a cleaned CSV. |

    Args:
        technique (str): Technique identifier in the clean-dataset catalog.
        params (Mapping[str, Any]): Parameters mapped to
            `CleanDatasetRequest`.

    Returns:
        dict[str, Any]: Output path, row/column counts, and dropped columns.
    """

    return run_with_validation(
        lambda: _run_clean_dataset_tool(technique=technique, params=params)
    )


def _run_clean_dataset_tool(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    _, resolved = resolve_technique_with_defaults(
        section_key="clean_dataset",
        technique_id=technique,
        provided_params=dict(params),
    )

    input_path = Path(as_required_string(resolved, "input_path")).expanduser()
    output_path = Path(as_required_string(resolved, "output_path")).expanduser()
    ensure_distinct_paths(input_path, output_path, field_name="output_path")

    drop_null_fraction = as_optional_float(resolved, "drop_null_fraction")
    drop_if_constant = as_bool(
        resolved.get("drop_if_constant", False),
        field_name="drop_if_constant",
    )
    drop_if_unique = as_bool(
        resolved.get("drop_if_unique", False),
        field_name="drop_if_unique",
    )
    datetime_format = as_optional_string(resolved, "datetime_format")
    cast_to_str = as_bool(resolved.get("cast_to_str", False), field_name="cast_to_str")
    numeric_to_float32 = as_bool(
        resolved.get("numeric_to_float32", False),
        field_name="numeric_to_float32",
    )

    result = run_clean_dataset(
        CleanDatasetRequest(
            input_path=input_path,
            output_path=output_path,
            drop_null_fraction=drop_null_fraction,
            drop_if_constant=drop_if_constant,
            drop_if_unique=drop_if_unique,
            datetime_format=datetime_format,
            cast_to_str=cast_to_str,
            numeric_to_float32=numeric_to_float32,
        )
    )

    return {
        "output_path": str(result.output_path),
        "row_count": result.row_count,
        "column_count": result.column_count,
        "dropped_columns": list(result.dropped_columns),
    }


def _to_pandas_frame(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    if hasattr(data, "to_pandas"):
        return data.to_pandas()
    return pd.DataFrame(data)


def _validate_input_csv_path(path: Path) -> None:
    if not path.exists():
        raise InputValidationError(f"CSV path does not exist: {path}")
    if not path.is_file():
        raise InputValidationError(f"CSV path is not a file: {path}")
    if path.suffix.lower() != ".csv":
        raise InputValidationError("CSV path must point to a .csv file.")


def _validate_output_csv_path(path: Path, *, input_path: Path) -> None:
    if path.suffix.lower() != ".csv":
        raise InputValidationError("Output path must point to a .csv file.")
    if path.resolve() == input_path.resolve():
        raise InputValidationError("Output path must be different from input path.")
