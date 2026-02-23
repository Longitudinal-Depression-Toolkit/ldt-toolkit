from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
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

_NUMERIC_AGGREGATIONS: tuple[str, ...] = ("mean", "median", "min", "max", "std")
_CATEGORICAL_AGGREGATIONS: tuple[str, ...] = ("mode", "first", "last")


@dataclass(frozen=True)
class AggregateLongToCrossSectionalRequest:
    """Request payload for long-to-cross-sectional aggregation.

    Attributes:
        input_path (Path): Input CSV path containing repeated subject rows.
        output_path (Path): Output CSV path for aggregated subject-level rows.
        subject_id_col (str): Subject identifier column used for grouping.
        numeric_columns (tuple[str, ...]): Numeric columns to aggregate.
        categorical_columns (tuple[str, ...]): Categorical columns to
            aggregate.
        numeric_agg (str): Numeric aggregation method.
        categorical_agg (str): Categorical aggregation method.
    """

    input_path: Path
    output_path: Path
    subject_id_col: str
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numeric_agg: str
    categorical_agg: str


@dataclass(frozen=True)
class AggregateLongToCrossSectionalResult:
    """Result payload produced by long-to-cross-sectional aggregation.

    Attributes:
        output_path (Path): Output CSV path.
        row_count (int): Number of aggregated subject rows.
        column_count (int): Number of columns in the output CSV.
        numeric_columns (tuple[str, ...]): Numeric columns that were
            aggregated.
        categorical_columns (tuple[str, ...]): Categorical columns that were
            aggregated.
        numeric_agg (str): Numeric aggregation method used.
        categorical_agg (str): Categorical aggregation method used.
    """

    output_path: Path
    row_count: int
    column_count: int
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numeric_agg: str
    categorical_agg: str


def run_aggregate_long_to_cross_sectional(
    request: AggregateLongToCrossSectionalRequest,
) -> AggregateLongToCrossSectionalResult:
    """Aggregate longitudinal rows into one cross-sectional row per subject.

    This operation groups records by `subject_id_col` and summarises selected
    numeric and categorical features so each subject appears once in the output.
    It is useful when downstream models require one record per subject rather
    than repeated measures.

    Supported aggregations:

    | Feature type | Supported aggregation values |
    | --- | --- |
    | Numeric | `mean`, `median`, `min`, `max`, `std` |
    | Categorical | `mode`, `first`, `last` |

    Fictional input (long format):

    | subject_id | wave | mood_score | smoking_status |
    | --- | --- | --- | --- |
    | 101 | 1 | 4.0 | never |
    | 101 | 2 | 3.0 | former |
    | 102 | 1 | 7.0 | current |
    | 102 | 2 | 8.0 | current |

    Fictional output (cross-sectional; `numeric_agg=mean`, `categorical_agg=mode`):

    | subject_id | mood_score | smoking_status |
    | --- | --- | --- |
    | 101 | 3.5 | former |
    | 102 | 7.5 | current |

    Args:
        request (AggregateLongToCrossSectionalRequest): Input/output paths and
            aggregation configuration.

    Returns:
        AggregateLongToCrossSectionalResult: Output dataset summary and selected
            aggregation metadata.

    Examples:
        ```python
        from ldt.data_preprocessing.tools.aggregate_long_to_cross_sectional.run import (
            AggregateLongToCrossSectionalRequest,
            run_aggregate_long_to_cross_sectional,
        )

        result = run_aggregate_long_to_cross_sectional(
            AggregateLongToCrossSectionalRequest(
                input_path="data/longitudinal.csv",
                output_path="outputs/cross_sectional.csv",
                subject_id_col="subject_id",
                numeric_columns=("mood_score",),
                categorical_columns=("smoking_status",),
                numeric_agg="mean",
                categorical_agg="mode",
            )
        )
        ```
    """

    _validate_input_csv_path(request.input_path)
    _validate_output_csv_path(request.output_path, input_path=request.input_path)
    numeric_agg = _normalise_choice(
        request.numeric_agg, allowed=_NUMERIC_AGGREGATIONS, field_name="numeric_agg"
    )
    categorical_agg = _normalise_choice(
        request.categorical_agg,
        allowed=_CATEGORICAL_AGGREGATIONS,
        field_name="categorical_agg",
    )

    data = pd.read_csv(request.input_path)
    if request.subject_id_col not in data.columns:
        raise InputValidationError(
            f"Subject ID column not found: {request.subject_id_col}"
        )

    numeric_columns = list(request.numeric_columns)
    if not numeric_columns:
        numeric_columns = [
            col
            for col in data.select_dtypes(include=["number"]).columns
            if col != request.subject_id_col
        ]
    _ensure_columns(data, numeric_columns)
    invalid_numeric = [col for col in numeric_columns if col == request.subject_id_col]
    if invalid_numeric:
        raise InputValidationError("Numeric columns cannot include subject_id_col.")

    categorical_columns = list(request.categorical_columns)
    _ensure_columns(data, categorical_columns)
    invalid_categorical = [
        col
        for col in categorical_columns
        if col == request.subject_id_col or col in set(numeric_columns)
    ]
    if invalid_categorical:
        raise InputValidationError(
            "Categorical columns cannot include subject_id_col or numeric columns."
        )

    if not numeric_columns and not categorical_columns:
        raise InputValidationError("No columns selected for aggregation.")

    aggregation_map: dict[str, str | object] = {}
    for column in numeric_columns:
        aggregation_map[column] = numeric_agg
    for column in categorical_columns:
        aggregation_map[column] = _resolve_categorical_aggregator(categorical_agg)

    aggregated = data.groupby(request.subject_id_col, dropna=False).agg(aggregation_map)
    aggregated = aggregated.reset_index()

    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated.to_csv(request.output_path, index=False)

    return AggregateLongToCrossSectionalResult(
        output_path=request.output_path.resolve(),
        row_count=len(aggregated),
        column_count=int(aggregated.shape[1]),
        numeric_columns=tuple(numeric_columns),
        categorical_columns=tuple(categorical_columns),
        numeric_agg=numeric_agg,
        categorical_agg=categorical_agg,
    )


def run_aggregate_long_to_cross_sectional_tool(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Run long-to-cross-sectional aggregation from catalog technique payloads.

    Supported techniques:

    | Technique | What it does |
    | --- | --- |
    | `aggregate` | Aggregates repeated subject rows into one row per subject. |

    Args:
        technique (str): Technique identifier in the aggregation catalog.
        params (Mapping[str, Any]): Parameters mapped to
            `AggregateLongToCrossSectionalRequest`.

    Returns:
        dict[str, Any]: Output path, row/column counts, selected columns, and
            aggregation methods applied.
    """

    return run_with_validation(
        lambda: _run_aggregate_long_to_cross_sectional_tool(
            technique=technique,
            params=params,
        )
    )


def _run_aggregate_long_to_cross_sectional_tool(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    _, resolved = resolve_technique_with_defaults(
        section_key="aggregate_long_to_cross_sectional",
        technique_id=technique,
        provided_params=dict(params),
    )

    input_path = Path(as_required_string(resolved, "input_path")).expanduser()
    output_path = Path(as_required_string(resolved, "output_path")).expanduser()
    subject_id_col = as_required_string(resolved, "subject_id_col")

    data = pd.read_csv(input_path)
    if subject_id_col not in data.columns:
        raise InputValidationError(f"Subject ID column not found: {subject_id_col}")

    numeric_cols = as_optional_string_list_or_csv(resolved, "numeric_columns")
    if not numeric_cols:
        numeric_cols = [
            col
            for col in data.select_dtypes(include=["number"]).columns
            if col != subject_id_col
        ]

    categorical_cols = as_optional_string_list_or_csv(resolved, "categorical_columns")

    if numeric_cols:
        ensure_columns(data, numeric_cols)
        invalid_numeric = [col for col in numeric_cols if col == subject_id_col]
        if invalid_numeric:
            raise InputValidationError("Numeric columns cannot include subject_id_col.")

    if categorical_cols:
        ensure_columns(data, categorical_cols)
        invalid_categorical = [
            col
            for col in categorical_cols
            if col == subject_id_col or col in set(numeric_cols)
        ]
        if invalid_categorical:
            raise InputValidationError(
                "Categorical columns cannot include subject_id_col or numeric columns."
            )

    if not numeric_cols and not categorical_cols:
        raise InputValidationError("No columns selected for aggregation.")

    numeric_agg = as_choice(
        as_required_string(resolved, "numeric_agg"),
        choices=("mean", "median", "min", "max", "std"),
        field_name="numeric_agg",
    )
    categorical_agg = as_choice(
        as_required_string(resolved, "categorical_agg"),
        choices=("mode", "first", "last"),
        field_name="categorical_agg",
    )

    result = run_aggregate_long_to_cross_sectional(
        AggregateLongToCrossSectionalRequest(
            input_path=input_path,
            output_path=output_path,
            subject_id_col=subject_id_col,
            numeric_columns=tuple(numeric_cols),
            categorical_columns=tuple(categorical_cols),
            numeric_agg=numeric_agg,
            categorical_agg=categorical_agg,
        )
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


def _resolve_categorical_aggregator(name: str) -> object:
    if name == "mode":
        return _mode_aggregate
    if name == "first":
        return "first"
    if name == "last":
        return "last"
    raise InputValidationError(f"Unsupported categorical aggregation: {name}")


def _mode_aggregate(values: pd.Series) -> object:
    non_null = values.dropna()
    if non_null.empty:
        return np.nan
    modes = non_null.mode(dropna=True)
    if modes.empty:
        return non_null.iloc[0]
    return modes.iloc[0]


def _normalise_choice(raw: str, *, allowed: tuple[str, ...], field_name: str) -> str:
    value = raw.strip().lower()
    if value not in allowed:
        raise InputValidationError(
            f"{field_name} must be one of: {', '.join(allowed)}."
        )
    return value


def _ensure_columns(data: pd.DataFrame, columns: list[str]) -> None:
    if not columns:
        return
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise InputValidationError(
            f"Missing columns in dataset: {', '.join(sorted(missing))}"
        )


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
