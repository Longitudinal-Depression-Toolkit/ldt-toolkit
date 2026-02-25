from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from beartype import beartype

from ldt.data_preprocessing.catalog import resolve_technique_with_defaults
from ldt.data_preprocessing.support.skrub_compat import import_skrub_symbol
from ldt.utils.errors import InputValidationError
from ldt.utils.metadata import ComponentMetadata
from ldt.utils.templates.tools.data_preprocessing import DataPreprocessingTool

Cleaner = import_skrub_symbol("Cleaner")


@beartype
@dataclass(frozen=True)
class CleanDatasetResult:
    """Structured output after dataset cleaning."""

    output_path: Path
    row_count: int
    column_count: int
    dropped_columns: tuple[str, ...]


@beartype
class CleanDataset(DataPreprocessingTool):
    """Clean a CSV dataset with `skrub.Cleaner`.

    The cleaner standardises table structure and can optionally remove low-signal
    columns based on nullness and cardinality heuristics.

    Initialisation parameters:
        input_path (Path | str | None): Optional default input CSV path.
        output_path (Path | str | None): Optional default output CSV path.
        drop_null_fraction (float | None): Optional threshold in `[0, 1]` for
            dropping high-null columns.
        drop_if_constant (bool): Drop columns with one unique value.
        drop_if_unique (bool): Drop ID-like columns with all unique values.
        datetime_format (str | None): Optional datetime parsing format.
        cast_to_str (bool): Cast textual columns to string.
        numeric_to_float32 (bool): Request float32 numeric casting where
            supported by installed `skrub` version.

    Examples:
        ```python
        from ldt.data_preprocessing import CleanDataset

        tool = CleanDataset()
        result = tool.fit_preprocess(
            input_path="./data/raw.csv",
            output_path="./outputs/cleaned.csv",
            drop_null_fraction=0.95,
            drop_if_constant=True,
            drop_if_unique=True,
            datetime_format=None,
            cast_to_str=False,
            numeric_to_float32=True,
        )
        ```
    """

    metadata = ComponentMetadata(
        name="clean_dataset",
        full_name="Clean Dataset",
        abstract_description="Clean a CSV dataset with skrub Cleaner.",
    )

    def __init__(
        self,
        *,
        input_path: Path | str | None = None,
        output_path: Path | str | None = None,
        drop_null_fraction: float | None = None,
        drop_if_constant: bool = False,
        drop_if_unique: bool = False,
        datetime_format: str | None = None,
        cast_to_str: bool = False,
        numeric_to_float32: bool = False,
    ) -> None:
        self._input_path = (
            Path(input_path).expanduser() if input_path is not None else None
        )
        self._output_path = (
            Path(output_path).expanduser() if output_path is not None else None
        )
        self._drop_null_fraction = drop_null_fraction
        self._drop_if_constant = drop_if_constant
        self._drop_if_unique = drop_if_unique
        self._datetime_format = datetime_format
        self._cast_to_str = cast_to_str
        self._numeric_to_float32 = numeric_to_float32

    @beartype
    def fit(self, **kwargs: Any) -> CleanDataset:
        """Validate and store cleaning configuration.

        Args:
            **kwargs (Any): Configuration overrides:
                - `input_path` (str | Path): Input CSV path.
                - `output_path` (str | Path): Output CSV path.
                - `drop_null_fraction` (float | None): Null-threshold in
                  `[0, 1]`.
                - `drop_if_constant` (bool): Drop constant columns.
                - `drop_if_unique` (bool): Drop all-unique columns.
                - `datetime_format` (str | None): Optional datetime format.
                - `cast_to_str` (bool): Cast textual values to string.
                - `numeric_to_float32` (bool): Request float32 numeric dtype.

        Returns:
            CleanDataset: The fitted tool instance.
        """

        input_path = kwargs.get("input_path", self._input_path)
        output_path = kwargs.get("output_path", self._output_path)

        if input_path is None:
            raise InputValidationError("Missing required parameter: input_path")
        if output_path is None:
            raise InputValidationError("Missing required parameter: output_path")

        self._input_path = Path(str(input_path)).expanduser()
        self._output_path = Path(str(output_path)).expanduser()
        self._drop_null_fraction = _as_optional_float(
            kwargs.get("drop_null_fraction", self._drop_null_fraction)
        )
        self._drop_if_constant = _as_bool(
            kwargs.get("drop_if_constant", self._drop_if_constant),
            field_name="drop_if_constant",
        )
        self._drop_if_unique = _as_bool(
            kwargs.get("drop_if_unique", self._drop_if_unique),
            field_name="drop_if_unique",
        )
        self._datetime_format = _as_optional_string(
            kwargs.get("datetime_format", self._datetime_format)
        )
        self._cast_to_str = _as_bool(
            kwargs.get("cast_to_str", self._cast_to_str),
            field_name="cast_to_str",
        )
        self._numeric_to_float32 = _as_bool(
            kwargs.get("numeric_to_float32", self._numeric_to_float32),
            field_name="numeric_to_float32",
        )

        _validate_input_csv_path(self._input_path)
        _validate_output_csv_path(self._output_path, input_path=self._input_path)
        if self._drop_null_fraction is not None and not (
            0 <= self._drop_null_fraction <= 1
        ):
            raise InputValidationError(
                "drop_null_fraction must be within [0, 1] when provided."
            )

        return self

    @beartype
    def preprocess(self, **kwargs: Any) -> CleanDatasetResult:
        """Run dataset cleaning with the configured cleaner options.

        Args:
            **kwargs (Any): Optional configuration keys identical to `fit(...)`.
                If provided, they override any existing fitted configuration.

        Returns:
            CleanDatasetResult: Typed summary of the cleaned output dataset.
        """

        if kwargs:
            self.fit(**kwargs)
        if self._input_path is None or self._output_path is None:
            raise InputValidationError(
                "No configuration provided. Call `fit(...)` first or pass kwargs to `preprocess(...)`."
            )

        original = pd.read_csv(self._input_path)
        candidate_cleaner_kwargs = {
            "drop_null_fraction": self._drop_null_fraction,
            "drop_if_constant": self._drop_if_constant,
            "drop_if_unique": self._drop_if_unique,
            "datetime_format": self._datetime_format,
            "numeric_dtype": "float32" if self._numeric_to_float32 else None,
            "cast_to_str": self._cast_to_str,
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

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(self._output_path, index=False)

        return CleanDatasetResult(
            output_path=self._output_path.resolve(),
            row_count=len(cleaned),
            column_count=int(cleaned.shape[1]),
            dropped_columns=dropped_columns,
        )


@beartype
def run_clean_dataset(*, technique: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Run one clean-dataset technique for the Go CLI bridge.

    !!! warning

        This `run_*` function is primarily for the Go CLI bridge and should not be
        treated as the Python library API. In Python scripts or notebooks, import
        and call `CleanDataset` directly from `ldt.data_preprocessing`.

    Args:
        technique (str): Cleaning technique key from the catalogue.
        params (Mapping[str, Any]): Wrapper parameters forwarded to
            `CleanDataset.fit_preprocess(...)`:
            `input_path`, `output_path`, `drop_null_fraction`,
            `drop_if_constant`, `drop_if_unique`, `datetime_format`,
            `cast_to_str`, and `numeric_to_float32`.

    Returns:
        dict[str, Any]: Serialised cleaning summary for the Go CLI bridge.
    """

    _, resolved = resolve_technique_with_defaults(
        section_key="clean_dataset",
        technique_id=technique,
        provided_params=dict(params),
    )

    result = CleanDataset().fit_preprocess(
        input_path=Path(_as_required_string(resolved, "input_path")).expanduser(),
        output_path=Path(_as_required_string(resolved, "output_path")).expanduser(),
        drop_null_fraction=_as_optional_float(resolved.get("drop_null_fraction")),
        drop_if_constant=_as_bool(
            resolved.get("drop_if_constant", False),
            field_name="drop_if_constant",
        ),
        drop_if_unique=_as_bool(
            resolved.get("drop_if_unique", False),
            field_name="drop_if_unique",
        ),
        datetime_format=_as_optional_string(resolved.get("datetime_format")),
        cast_to_str=_as_bool(
            resolved.get("cast_to_str", False), field_name="cast_to_str"
        ),
        numeric_to_float32=_as_bool(
            resolved.get("numeric_to_float32", False),
            field_name="numeric_to_float32",
        ),
    )

    return {
        "output_path": str(result.output_path),
        "row_count": result.row_count,
        "column_count": result.column_count,
        "dropped_columns": list(result.dropped_columns),
    }


@beartype
def _to_pandas_frame(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    if hasattr(data, "to_pandas"):
        return data.to_pandas()
    return pd.DataFrame(data)


@beartype
def _as_required_string(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"Missing required string parameter: {key}")
    return value.strip()


@beartype
def _as_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed if trimmed else None
    raise InputValidationError("datetime_format must be a string when provided.")


@beartype
def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, bool):
        raise InputValidationError("drop_null_fraction must be a float when provided.")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError as exc:
            raise InputValidationError(
                "drop_null_fraction must be a float when provided."
            ) from exc
    raise InputValidationError("drop_null_fraction must be a float when provided.")


@beartype
def _as_bool(value: Any, *, field_name: str) -> bool:
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
def _validate_input_csv_path(path: Path) -> None:
    if not path.exists():
        raise InputValidationError(f"CSV path does not exist: {path}")
    if not path.is_file():
        raise InputValidationError(f"CSV path is not a file: {path}")
    if path.suffix.lower() != ".csv":
        raise InputValidationError("CSV path must point to a .csv file.")


@beartype
def _validate_output_csv_path(path: Path, *, input_path: Path) -> None:
    if path.suffix.lower() != ".csv":
        raise InputValidationError("Output path must point to a .csv file.")
    if path.resolve() == input_path.resolve():
        raise InputValidationError("Output path must be different from input path.")
