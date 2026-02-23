from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_preprocessing.catalog import resolve_technique_with_defaults
from src.data_preprocessing.support.inputs import (
    as_optional_string_list_or_csv,
    as_required_string,
    as_required_string_list_or_csv,
    run_with_validation,
)
from src.utils.errors import InputValidationError

_ALLOWED_SEPARATORS = {"_", "-"}


@dataclass(frozen=True)
class PivotLongToWideRequest:
    """Request payload for long-to-wide pivoting.

    Attributes:
        input_path (Path): Input CSV path in long format.
        output_path (Path): Output CSV path in wide format.
        id_cols (tuple[str, ...]): Subject identifier columns.
        time_col (str): Time/wave column.
        longitudinal_columns (tuple[str, ...] | None): Columns expanded across
            time points.
        non_longitudinal_columns (tuple[str, ...] | None): Subject-level static
            columns copied once.
        time_label (str): Prefix text used for generated time suffixes.
        separator (str): Separator between feature name and time label.
    """

    input_path: Path
    output_path: Path
    id_cols: tuple[str, ...]
    time_col: str
    longitudinal_columns: tuple[str, ...] | None
    non_longitudinal_columns: tuple[str, ...] | None
    time_label: str
    separator: str


@dataclass(frozen=True)
class PivotLongToWideResult:
    """Result payload produced by long-to-wide pivoting.

    Attributes:
        output_path (Path): Output CSV path.
        row_count (int): Number of subject rows in wide output.
        column_count (int): Number of columns in wide output.
        id_cols (tuple[str, ...]): Identifier columns used for pivoting.
        time_col (str): Time/wave column used for pivoting.
        time_label (str): Time label prefix used in generated columns.
        separator (str): Separator used in generated column names.
        longitudinal_columns (tuple[str, ...]): Longitudinal columns that were
            expanded across time.
        non_longitudinal_columns (tuple[str, ...]): Static columns retained at
            subject level.
    """

    output_path: Path
    row_count: int
    column_count: int
    id_cols: tuple[str, ...]
    time_col: str
    time_label: str
    separator: str
    longitudinal_columns: tuple[str, ...]
    non_longitudinal_columns: tuple[str, ...]


def run_pivot_long_to_wide(request: PivotLongToWideRequest) -> PivotLongToWideResult:
    """Pivot longitudinal records from long format to wide format.

    This operation reshapes repeated measurements into one row per subject by
    expanding each longitudinal variable across time points. Non-longitudinal
    columns stay as static subject-level columns.

    Fictional input (long format):

    | subject_id | wave | mood | sleep | sex |
    | --- | --- | --- | --- | --- |
    | 101 | 1 | 4.0 | 6.0 | F |
    | 101 | 2 | 3.0 | 5.0 | F |
    | 102 | 1 | 7.0 | 8.0 | M |
    | 102 | 2 | 8.0 | 7.0 | M |

    Fictional output (wide format; `time_label=w`, `separator=_`):

    | subject_id | sex | mood_w1 | mood_w2 | sleep_w1 | sleep_w2 |
    | --- | --- | --- | --- | --- | --- |
    | 101 | F | 4.0 | 3.0 | 6.0 | 5.0 |
    | 102 | M | 7.0 | 8.0 | 8.0 | 7.0 |

    Args:
        request (PivotLongToWideRequest): Input/output paths and pivot
            configuration (ID columns, time column, and feature-role choices).

    Returns:
        PivotLongToWideResult: Output dataset summary and resolved pivot
            configuration.

    Examples:
        ```python
        from ldt.data_preprocessing.tools.pivot_long_to_wide.run import (
            PivotLongToWideRequest,
            run_pivot_long_to_wide,
        )

        result = run_pivot_long_to_wide(
            PivotLongToWideRequest(
                input_path="data/longitudinal.csv",
                output_path="outputs/wide.csv",
                id_cols=("subject_id",),
                time_col="wave",
                longitudinal_columns=("mood", "sleep"),
                non_longitudinal_columns=("sex",),
                time_label="w",
                separator="_",
            )
        )
        ```
    """

    _validate_input_csv_path(request.input_path)
    _validate_output_csv_path(request.output_path, input_path=request.input_path)
    _validate_time_label(request.time_label)
    _validate_separator(request.separator)

    if not request.id_cols:
        raise InputValidationError("At least one ID column is required.")

    data = pd.read_csv(request.input_path)
    _ensure_columns_exist(
        data,
        [*request.id_cols, request.time_col],
        dataset_name="input long-format data",
    )

    if data[[*request.id_cols, request.time_col]].isna().any().any():
        raise InputValidationError(
            "ID/time columns contain missing values. Pure pivoting requires complete ID/time keys."
        )

    longitudinal_cols, non_longitudinal_cols = _resolve_feature_roles(
        data=data,
        id_cols=list(request.id_cols),
        time_col=request.time_col,
        longitudinal_cols=list(request.longitudinal_columns or ()),
        non_longitudinal_cols=list(request.non_longitudinal_columns or ()),
        longitudinal_auto=request.longitudinal_columns is None,
        non_longitudinal_auto=request.non_longitudinal_columns is None,
    )

    subject_static = _build_subject_static_frame(
        data=data,
        id_cols=list(request.id_cols),
        non_longitudinal_cols=non_longitudinal_cols,
    )
    subject_longitudinal = _build_subject_longitudinal_frame(
        data=data,
        id_cols=list(request.id_cols),
        time_col=request.time_col,
        longitudinal_cols=longitudinal_cols,
        time_label=request.time_label,
        separator=request.separator,
    )

    if subject_static is None and subject_longitudinal is None:
        raise InputValidationError(
            "No feature columns were selected. Choose longitudinal and/or non-longitudinal columns."
        )
    if subject_static is None:
        wide = subject_longitudinal
    elif subject_longitudinal is None:
        wide = subject_static
    else:
        wide = subject_static.merge(
            subject_longitudinal, on=list(request.id_cols), how="outer"
        )

    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(request.output_path, index=False)

    return PivotLongToWideResult(
        output_path=request.output_path.resolve(),
        row_count=len(wide),
        column_count=int(wide.shape[1]),
        id_cols=tuple(request.id_cols),
        time_col=request.time_col,
        time_label=request.time_label,
        separator=request.separator,
        longitudinal_columns=tuple(longitudinal_cols),
        non_longitudinal_columns=tuple(non_longitudinal_cols),
    )


def run_pivot_long_to_wide_tool(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Run long-to-wide pivoting from catalog technique payloads.

    Supported techniques:

    | Technique | What it does |
    | --- | --- |
    | `pivot` | Reshapes repeated time rows into subject-level wide columns. |

    Args:
        technique (str): Technique identifier in the pivot catalog.
        params (Mapping[str, Any]): Parameters mapped to
            `PivotLongToWideRequest`.

    Returns:
        dict[str, Any]: Output path, row/column counts, and resolved pivot
            settings.
    """

    return run_with_validation(
        lambda: _run_pivot_long_to_wide_tool(technique=technique, params=params)
    )


def _run_pivot_long_to_wide_tool(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    _, resolved = resolve_technique_with_defaults(
        section_key="pivot_long_to_wide",
        technique_id=technique,
        provided_params=dict(params),
    )

    input_path = Path(as_required_string(resolved, "input_path")).expanduser()
    output_path = Path(as_required_string(resolved, "output_path")).expanduser()
    id_cols = as_required_string_list_or_csv(resolved, "id_cols")
    time_col = as_required_string(resolved, "time_col")
    longitudinal_cols = as_optional_string_list_or_csv(resolved, "longitudinal_columns")
    non_longitudinal_cols = as_optional_string_list_or_csv(
        resolved,
        "non_longitudinal_columns",
    )
    time_label = as_required_string(resolved, "time_label")
    separator = as_required_string(resolved, "separator")

    result = run_pivot_long_to_wide(
        PivotLongToWideRequest(
            input_path=input_path,
            output_path=output_path,
            id_cols=tuple(id_cols),
            time_col=time_col,
            longitudinal_columns=(
                tuple(longitudinal_cols) if longitudinal_cols else None
            ),
            non_longitudinal_columns=(
                tuple(non_longitudinal_cols) if non_longitudinal_cols else None
            ),
            time_label=time_label,
            separator=separator,
        )
    )
    return {
        "output_path": str(result.output_path),
        "row_count": result.row_count,
        "column_count": result.column_count,
        "id_cols": list(result.id_cols),
        "time_col": result.time_col,
        "time_label": result.time_label,
        "separator": result.separator,
    }


def _resolve_feature_roles(
    *,
    data: pd.DataFrame,
    id_cols: list[str],
    time_col: str,
    longitudinal_cols: list[str],
    non_longitudinal_cols: list[str],
    longitudinal_auto: bool,
    non_longitudinal_auto: bool,
) -> tuple[list[str], list[str]]:
    candidate_features = [
        col for col in data.columns if col not in {time_col, *id_cols}
    ]
    candidate_set = set(candidate_features)

    if not non_longitudinal_auto:
        _ensure_columns_exist(
            data,
            non_longitudinal_cols,
            dataset_name="non-longitudinal column selection",
        )

    if longitudinal_auto:
        non_longitudinal_set = set(non_longitudinal_cols)
        longitudinal_cols = [
            col for col in candidate_features if col not in non_longitudinal_set
        ]
    else:
        _ensure_columns_exist(
            data,
            longitudinal_cols,
            dataset_name="longitudinal column selection",
        )

    if non_longitudinal_auto:
        longitudinal_set = set(longitudinal_cols)
        non_longitudinal_cols = [
            col for col in candidate_features if col not in longitudinal_set
        ]

    invalid_longitudinal = sorted(set(longitudinal_cols) - candidate_set)
    invalid_non_longitudinal = sorted(set(non_longitudinal_cols) - candidate_set)
    if invalid_longitudinal:
        raise InputValidationError(
            "Longitudinal columns cannot include ID/time columns: "
            + ", ".join(invalid_longitudinal)
        )
    if invalid_non_longitudinal:
        raise InputValidationError(
            "Non-longitudinal columns cannot include ID/time columns: "
            + ", ".join(invalid_non_longitudinal)
        )

    overlap = sorted(set(longitudinal_cols) & set(non_longitudinal_cols))
    if overlap:
        raise InputValidationError(
            "Columns cannot be both longitudinal and non-longitudinal: "
            + ", ".join(overlap)
        )

    return longitudinal_cols, non_longitudinal_cols


def _build_subject_static_frame(
    *,
    data: pd.DataFrame,
    id_cols: list[str],
    non_longitudinal_cols: list[str],
) -> pd.DataFrame | None:
    if not non_longitudinal_cols:
        return None

    subject_static = data[[*id_cols, *non_longitudinal_cols]].copy()
    inconsistent = (
        subject_static.groupby(id_cols, dropna=False)[non_longitudinal_cols]
        .nunique(dropna=False)
        .gt(1)
    )
    inconsistent_cols = sorted(
        {col for col in non_longitudinal_cols if bool(inconsistent[col].any())}
    )
    if inconsistent_cols:
        raise InputValidationError(
            "Non-longitudinal columns must be invariant within each subject. "
            "Found varying values for: " + ", ".join(inconsistent_cols)
        )
    return subject_static.drop_duplicates(subset=id_cols, keep="first")


def _build_subject_longitudinal_frame(
    *,
    data: pd.DataFrame,
    id_cols: list[str],
    time_col: str,
    longitudinal_cols: list[str],
    time_label: str,
    separator: str,
) -> pd.DataFrame | None:
    if not longitudinal_cols:
        return None

    long_data = data[[*id_cols, time_col, *longitudinal_cols]].copy()
    duplicate_mask = long_data.duplicated(subset=[*id_cols, time_col], keep=False)
    if duplicate_mask.any():
        dup_count = int(duplicate_mask.sum())
        raise InputValidationError(
            "Pure pivoting requires unique (ID, time) rows for longitudinal "
            f"features. Found {dup_count} duplicated rows."
        )

    pivoted = long_data.set_index([*id_cols, time_col])[longitudinal_cols].unstack(
        time_col
    )
    pivoted = pivoted.sort_index(axis=1, level=1)
    pivoted.columns = _flatten_longitudinal_columns(
        columns=pivoted.columns,
        time_label=time_label,
        separator=separator,
    )
    return pivoted.reset_index()


def _flatten_longitudinal_columns(
    *,
    columns: pd.MultiIndex,
    time_label: str,
    separator: str,
) -> list[str]:
    flattened: list[str] = []
    for feature_name, time_value in columns.to_list():
        time_token = _format_time_token(time_value)
        flattened.append(f"{feature_name}{separator}{time_label}{time_token}")
    return flattened


def _format_time_token(value: object) -> str:
    if pd.isna(value):
        return "missing"
    as_numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(as_numeric):
        rounded = int(as_numeric)
        if float(rounded) == float(as_numeric):
            return str(rounded)
        return str(float(as_numeric)).replace(".", "p")
    return str(value).strip().replace(" ", "_")


def _validate_input_csv_path(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise InputValidationError(f"Input CSV path does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise InputValidationError("Input path must point to a .csv file.")


def _validate_output_csv_path(path: Path, *, input_path: Path) -> None:
    if not path.name:
        raise InputValidationError("Output path must include a filename.")
    if path.suffix.lower() != ".csv":
        raise InputValidationError("Output path must point to a .csv file.")
    if path.resolve() == input_path.resolve():
        raise InputValidationError("Output path must be different from input path.")
    parent = path.parent
    if not parent.exists() or not parent.is_dir():
        raise InputValidationError(f"Output directory does not exist: {parent}")
    if path.exists():
        raise InputValidationError(
            f"Output file already exists: {path}. Provide a new path to avoid overwriting."
        )


def _validate_time_label(time_label: str) -> None:
    if not time_label:
        raise InputValidationError("Time token label cannot be empty.")
    if not time_label[0].islower() or not all(
        c.islower() or c.isdigit() for c in time_label
    ):
        raise InputValidationError(
            "Time token label must start with lowercase letter and contain lowercase letters/digits only."
        )
    if len(time_label) > 16:
        raise InputValidationError("Time token label length must be <= 16.")


def _validate_separator(separator: str) -> None:
    if not separator:
        raise InputValidationError("Column-name separator cannot be empty.")
    if len(separator) != 1:
        raise InputValidationError("Column-name separator must be a single character.")
    if separator not in _ALLOWED_SEPARATORS:
        allowed = ", ".join(sorted(_ALLOWED_SEPARATORS))
        raise InputValidationError(f"Column-name separator must be one of: {allowed}")


def _ensure_columns_exist(
    data: pd.DataFrame,
    columns: Sequence[str],
    *,
    dataset_name: str,
) -> None:
    missing = [col for col in columns if col not in data.columns]
    if missing:
        raise InputValidationError(
            f"Missing required columns in {dataset_name}: {', '.join(missing)}"
        )
