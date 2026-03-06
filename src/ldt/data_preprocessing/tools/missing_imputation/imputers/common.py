from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from beartype import beartype

from ldt.utils.errors import InputValidationError


@beartype
def validate_input_csv_path(path: Path) -> None:
    """Validate that an input path points to an existing CSV file."""

    if not path.exists():
        raise InputValidationError(f"CSV path does not exist: {path}")
    if not path.is_file():
        raise InputValidationError(f"CSV path is not a file: {path}")
    if path.suffix.lower() != ".csv":
        raise InputValidationError("CSV path must point to a .csv file.")


@beartype
def validate_output_csv_path(path: Path, *, input_path: Path) -> None:
    """Validate that an output path is a distinct CSV target."""

    if path.suffix.lower() != ".csv":
        raise InputValidationError("Output path must point to a .csv file.")
    if path.resolve() == input_path.resolve():
        raise InputValidationError("Output path must be different from input path.")


@beartype
def load_csv(path: Path) -> pd.DataFrame:
    """Load one CSV dataset into a pandas dataframe."""

    return pd.read_csv(path)


@beartype
def write_csv(data: pd.DataFrame, *, output_path: Path) -> Path:
    """Write one dataframe to CSV and return the resolved path."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return output_path.resolve()


@beartype
def count_missing_values(data: pd.DataFrame) -> int:
    """Count all missing cells in one dataframe."""

    return int(data.isna().sum().sum())


@beartype
def numeric_columns(data: pd.DataFrame) -> tuple[str, ...]:
    """Return the dataframe columns eligible for numeric statistics."""

    return tuple(data.select_dtypes(include=["number"]).columns.tolist())


@beartype
def as_int(value: Any, *, field_name: str) -> int:
    """Parse one integer-like runtime value."""

    if isinstance(value, bool):
        raise InputValidationError(f"`{field_name}` must be an integer value.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise InputValidationError(f"`{field_name}` must be an integer value.")
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except ValueError as exc:
            raise InputValidationError(
                f"`{field_name}` must be an integer value."
            ) from exc
    raise InputValidationError(f"`{field_name}` must be an integer value.")


@beartype
def as_optional_int(value: Any, *, field_name: str) -> int | None:
    """Parse one optional integer-like runtime value."""

    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return as_int(value, field_name=field_name)


@beartype
def as_bool(value: Any, *, field_name: str) -> bool:
    """Parse one boolean-like runtime value."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "no", "n", "off"}:
            return False
    raise InputValidationError(f"`{field_name}` must be a boolean value.")


@beartype
def as_scalar_fill_value(value: Any, *, field_name: str) -> str | int | float | bool:
    """Parse one scalar fill value accepted by constant imputation."""

    if isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return value
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return value
        if isinstance(parsed, bool | int | float | str):
            return parsed
        raise InputValidationError(
            f"`{field_name}` must decode to a scalar JSON literal when provided as a string."
        )
    raise InputValidationError(
        f"`{field_name}` must be a scalar string, integer, float, or boolean."
    )


@beartype
def default_constant_fill_value(series: pd.Series) -> str | int:
    """Return the dtype-aware fallback fill value for constant imputation."""

    if pd.api.types.is_numeric_dtype(series):
        return 0
    return "missing"
