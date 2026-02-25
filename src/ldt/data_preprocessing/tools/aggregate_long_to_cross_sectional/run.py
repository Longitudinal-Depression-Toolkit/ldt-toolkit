from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from beartype import beartype

from ldt.data_preprocessing.catalog import resolve_technique_with_defaults
from ldt.utils.errors import InputValidationError
from ldt.utils.metadata import ComponentMetadata
from ldt.utils.templates.tools.data_preprocessing import DataPreprocessingTool

_NUMERIC_AGGREGATIONS: tuple[str, ...] = ("mean", "median", "min", "max", "std")
_CATEGORICAL_AGGREGATIONS: tuple[str, ...] = ("mode", "first", "last")


@beartype
@dataclass(frozen=True)
class AggregateLongToCrossSectionalResult:
    """Structured output from long-to-cross-sectional aggregation."""

    output_path: Path
    row_count: int
    column_count: int
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numeric_agg: str
    categorical_agg: str


@beartype
class AggregateLongToCrossSectional(DataPreprocessingTool):
    """Aggregate longitudinal rows into one cross-sectional row per subject.

    This tool groups records by a subject identifier and summarises selected
    numeric and categorical features.

    Runtime parameters:
        - `input_path`: Path to the input long-format CSV.
        - `output_path`: Path where the aggregated CSV will be written.
        - `subject_id_col`: Subject identifier column used for grouping.
        - `numeric_columns`: Optional numeric columns to aggregate. If omitted,
          all numeric columns except `subject_id_col` are used.
        - `categorical_columns`: Optional categorical columns to aggregate.
        - `numeric_agg`: Numeric aggregation (`mean`, `median`, `min`, `max`,
          `std`).
        - `categorical_agg`: Categorical aggregation (`mode`, `first`, `last`).

    Examples:
        ```python
        from ldt.data_preprocessing import AggregateLongToCrossSectional

        tool = AggregateLongToCrossSectional()
        result = tool.fit_preprocess(
            input_path="./data/longitudinal.csv",
            output_path="./outputs/cross_sectional.csv",
            subject_id_col="subject_id",
            numeric_columns=["mood_score"],
            categorical_columns=["smoking_status"],
            numeric_agg="mean",
            categorical_agg="mode",
        )
        ```
    """

    metadata = ComponentMetadata(
        name="aggregate_long_to_cross_sectional",
        full_name="Aggregate Long to Cross Sectional",
        abstract_description="Aggregate repeated subject rows into one row per subject.",
    )

    def __init__(self) -> None:
        self._config: dict[str, Any] | None = None

    @beartype
    def fit(self, **kwargs: Any) -> AggregateLongToCrossSectional:
        """Validate and store aggregation configuration.

        Args:
            **kwargs (Any): Configuration keys:
                - `input_path` (str | Path): Input long-format CSV path.
                - `output_path` (str | Path): Output cross-sectional CSV path.
                - `subject_id_col` (str): Subject identifier column.
                - `numeric_columns` (str | list[str] | None): Numeric columns.
                - `categorical_columns` (str | list[str] | None): Categorical
                  columns.
                - `numeric_agg` (str): One of `mean`, `median`, `min`, `max`,
                  `std`.
                - `categorical_agg` (str): One of `mode`, `first`, `last`.

        Returns:
            AggregateLongToCrossSectional: The fitted tool instance.
        """

        input_path_raw = kwargs.get("input_path")
        output_path_raw = kwargs.get("output_path")
        subject_id_col = kwargs.get("subject_id_col")
        if input_path_raw is None:
            raise InputValidationError("Missing required parameter: input_path")
        if output_path_raw is None:
            raise InputValidationError("Missing required parameter: output_path")
        if not isinstance(subject_id_col, str) or not subject_id_col.strip():
            raise InputValidationError("Missing required parameter: subject_id_col")

        input_path = Path(str(input_path_raw)).expanduser()
        output_path = Path(str(output_path_raw)).expanduser()
        _validate_input_csv_path(input_path)
        _validate_output_csv_path(output_path, input_path=input_path)

        numeric_columns = _as_optional_string_list(kwargs.get("numeric_columns"))
        categorical_columns = _as_optional_string_list(
            kwargs.get("categorical_columns")
        )
        numeric_agg = _normalise_choice(
            str(kwargs.get("numeric_agg", "mean")),
            allowed=_NUMERIC_AGGREGATIONS,
            field_name="numeric_agg",
        )
        categorical_agg = _normalise_choice(
            str(kwargs.get("categorical_agg", "mode")),
            allowed=_CATEGORICAL_AGGREGATIONS,
            field_name="categorical_agg",
        )

        self._config = {
            "input_path": input_path,
            "output_path": output_path,
            "subject_id_col": subject_id_col.strip(),
            "numeric_columns": tuple(numeric_columns),
            "categorical_columns": tuple(categorical_columns),
            "numeric_agg": numeric_agg,
            "categorical_agg": categorical_agg,
        }
        return self

    @beartype
    def preprocess(self, **kwargs: Any) -> AggregateLongToCrossSectionalResult:
        """Aggregate configured columns and write a subject-level CSV.

        Args:
            **kwargs (Any): Optional configuration keys identical to `fit(...)`.
                If provided, they override any existing fitted configuration.

        Returns:
            AggregateLongToCrossSectionalResult: Typed summary of the written
            cross-sectional output.
        """

        if kwargs:
            self.fit(**kwargs)
        if self._config is None:
            raise InputValidationError(
                "No configuration provided. Call `fit(...)` first or pass kwargs to `preprocess(...)`."
            )

        input_path = _as_path(self._config.get("input_path"), key="input_path")
        output_path = _as_path(self._config.get("output_path"), key="output_path")
        subject_id_col = str(self._config.get("subject_id_col"))

        data = pd.read_csv(input_path)
        if subject_id_col not in data.columns:
            raise InputValidationError(f"Subject ID column not found: {subject_id_col}")

        numeric_columns = list(self._config.get("numeric_columns", ()))
        if not numeric_columns:
            numeric_columns = [
                col
                for col in data.select_dtypes(include=["number"]).columns
                if col != subject_id_col
            ]
        _ensure_columns(data, numeric_columns)
        invalid_numeric = [col for col in numeric_columns if col == subject_id_col]
        if invalid_numeric:
            raise InputValidationError("Numeric columns cannot include subject_id_col.")

        categorical_columns = list(self._config.get("categorical_columns", ()))
        _ensure_columns(data, categorical_columns)
        invalid_categorical = [
            col
            for col in categorical_columns
            if col == subject_id_col or col in set(numeric_columns)
        ]
        if invalid_categorical:
            raise InputValidationError(
                "Categorical columns cannot include subject_id_col or numeric columns."
            )

        if not numeric_columns and not categorical_columns:
            raise InputValidationError("No columns selected for aggregation.")

        numeric_agg = str(self._config.get("numeric_agg"))
        categorical_agg = str(self._config.get("categorical_agg"))

        aggregation_map: dict[str, str | object] = {}
        for column in numeric_columns:
            aggregation_map[column] = numeric_agg
        for column in categorical_columns:
            aggregation_map[column] = _resolve_categorical_aggregator(categorical_agg)

        aggregated = data.groupby(subject_id_col, dropna=False).agg(aggregation_map)
        aggregated = aggregated.reset_index()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        aggregated.to_csv(output_path, index=False)

        return AggregateLongToCrossSectionalResult(
            output_path=output_path.resolve(),
            row_count=len(aggregated),
            column_count=int(aggregated.shape[1]),
            numeric_columns=tuple(numeric_columns),
            categorical_columns=tuple(categorical_columns),
            numeric_agg=numeric_agg,
            categorical_agg=categorical_agg,
        )


@beartype
def run_aggregate_long_to_cross_sectional(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one long-to-cross-sectional technique for the Go CLI bridge.

    !!! warning

        This `run_*` function is primarily for the Go CLI bridge and should not be
        treated as the Python library API. In Python scripts or notebooks, import
        and call `AggregateLongToCrossSectional` directly from `ldt.data_preprocessing`.

    Args:
        technique (str): Aggregation technique key from the catalogue.
        params (Mapping[str, Any]): Wrapper parameters forwarded to
            `AggregateLongToCrossSectional.fit_preprocess(...)`:
            `input_path`, `output_path`, `subject_id_col`, `numeric_columns`,
            `categorical_columns`, `numeric_agg`, and `categorical_agg`.

    Returns:
        dict[str, Any]: Serialised aggregation summary for the Go CLI bridge.
    """

    _, resolved = resolve_technique_with_defaults(
        section_key="aggregate_long_to_cross_sectional",
        technique_id=technique,
        provided_params=dict(params),
    )

    result = AggregateLongToCrossSectional().fit_preprocess(
        input_path=Path(_as_required_string(resolved, "input_path")).expanduser(),
        output_path=Path(_as_required_string(resolved, "output_path")).expanduser(),
        subject_id_col=_as_required_string(resolved, "subject_id_col"),
        numeric_columns=_as_optional_string_list(resolved.get("numeric_columns")),
        categorical_columns=_as_optional_string_list(
            resolved.get("categorical_columns")
        ),
        numeric_agg=_as_required_string(resolved, "numeric_agg"),
        categorical_agg=_as_required_string(resolved, "categorical_agg"),
    )

    return {
        "output_path": str(result.output_path),
        "row_count": result.row_count,
        "column_count": result.column_count,
        "numeric_columns": list(result.numeric_columns),
        "categorical_columns": list(result.categorical_columns),
        "numeric_agg": result.numeric_agg,
        "categorical_agg": result.categorical_agg,
    }


@beartype
def _as_required_string(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"Missing required string parameter: {key}")
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
def _as_path(value: Any, *, key: str) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value.strip():
        return Path(value).expanduser()
    raise InputValidationError(f"Missing required path parameter: {key}")


@beartype
def _resolve_categorical_aggregator(name: str) -> object:
    if name == "mode":
        return _mode_aggregate
    if name == "first":
        return "first"
    if name == "last":
        return "last"
    raise InputValidationError(f"Unsupported categorical aggregation: {name}")


@beartype
def _mode_aggregate(values: pd.Series) -> object:
    non_null = values.dropna()
    if non_null.empty:
        return np.nan
    modes = non_null.mode(dropna=True)
    if modes.empty:
        return non_null.iloc[0]
    return modes.iloc[0]


@beartype
def _normalise_choice(raw: str, *, allowed: tuple[str, ...], field_name: str) -> str:
    value = raw.strip().lower()
    if value not in allowed:
        raise InputValidationError(
            f"{field_name} must be one of: {', '.join(allowed)}."
        )
    return value


@beartype
def _ensure_columns(data: pd.DataFrame, columns: list[str]) -> None:
    if not columns:
        return
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise InputValidationError(
            f"Missing columns in dataset: {', '.join(sorted(missing))}"
        )


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
