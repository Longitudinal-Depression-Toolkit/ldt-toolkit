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

_ALLOWED_SEPARATORS = {"_", "-"}


@beartype
@dataclass(frozen=True)
class PivotLongToWideResult:
    """Structured output produced by long-to-wide pivoting."""

    output_path: Path
    row_count: int
    column_count: int
    id_cols: tuple[str, ...]
    time_col: str
    time_label: str
    separator: str
    longitudinal_columns: tuple[str, ...]
    non_longitudinal_columns: tuple[str, ...]


@beartype
class PivotLongToWide(DataPreprocessingTool):
    """Pivot longitudinal records from long format to wide format.

    This tool reshapes repeated measurements into one row per subject by
    expanding each longitudinal variable across time points.

    Runtime parameters:
        - `input_path`: Input long-format CSV path.
        - `output_path`: Output wide-format CSV path.
        - `id_cols`: Identifier columns that define one subject/entity row.
        - `time_col`: Time or wave column.
        - `longitudinal_columns`: Optional columns to expand across time.
        - `non_longitudinal_columns`: Optional static columns to keep once per
          subject.
        - `time_label`: Prefix before time values in generated wide column
          names.
        - `separator`: Separator between feature names and time labels.

    Examples:
        ```python
        from ldt.data_preprocessing import PivotLongToWide

        tool = PivotLongToWide()
        result = tool.fit_preprocess(
            input_path="./data/longitudinal.csv",
            output_path="./outputs/wide.csv",
            id_cols=["subject_id"],
            time_col="wave",
            longitudinal_columns=["mood", "sleep"],
            non_longitudinal_columns=["sex"],
            time_label="w",
            separator="_",
        )
        ```
    """

    metadata = ComponentMetadata(
        name="pivot_long_to_wide",
        full_name="Pivot Long to Wide",
        abstract_description="Pivot repeated long-format rows into one wide row per subject.",
    )

    def __init__(self) -> None:
        self._config: dict[str, Any] | None = None

    @beartype
    def fit(self, **kwargs: Any) -> PivotLongToWide:
        """Validate and store long-to-wide pivot configuration.

        Args:
            **kwargs (Any): Configuration keys:
                - `input_path` (str | Path): Input CSV path.
                - `output_path` (str | Path): Output CSV path.
                - `id_cols` (str | list[str]): Identifier columns.
                - `time_col` (str): Time/wave column.
                - `longitudinal_columns` (str | list[str] | None): Optional
                  longitudinal columns.
                - `non_longitudinal_columns` (str | list[str] | None): Optional
                  static columns.
                - `time_label` (str): Prefix for generated time suffixes.
                - `separator` (str): One-character separator (`_` or `-`).

        Returns:
            PivotLongToWide: The fitted tool instance.
        """

        input_path_raw = kwargs.get("input_path")
        output_path_raw = kwargs.get("output_path")
        if input_path_raw is None:
            raise InputValidationError("Missing required parameter: input_path")
        if output_path_raw is None:
            raise InputValidationError("Missing required parameter: output_path")

        input_path = Path(str(input_path_raw)).expanduser()
        output_path = Path(str(output_path_raw)).expanduser()
        _validate_input_csv_path(input_path)
        _validate_output_csv_path(output_path, input_path=input_path)

        id_cols = _as_required_string_list(kwargs.get("id_cols"), field_name="id_cols")
        time_col = _as_required_string(kwargs.get("time_col"), field_name="time_col")
        longitudinal_columns = _as_optional_string_list(
            kwargs.get("longitudinal_columns")
        )
        non_longitudinal_columns = _as_optional_string_list(
            kwargs.get("non_longitudinal_columns")
        )
        time_label = _as_required_string(
            kwargs.get("time_label"), field_name="time_label"
        )
        separator = _as_required_string(kwargs.get("separator"), field_name="separator")

        _validate_time_label(time_label)
        _validate_separator(separator)

        self._config = {
            "input_path": input_path,
            "output_path": output_path,
            "id_cols": tuple(id_cols),
            "time_col": time_col,
            "longitudinal_columns": (
                tuple(longitudinal_columns) if longitudinal_columns else None
            ),
            "non_longitudinal_columns": (
                tuple(non_longitudinal_columns) if non_longitudinal_columns else None
            ),
            "time_label": time_label,
            "separator": separator,
        }
        return self

    @beartype
    def preprocess(self, **kwargs: Any) -> PivotLongToWideResult:
        """Execute long-to-wide pivoting and write the wide CSV.

        Args:
            **kwargs (Any): Optional configuration keys identical to `fit(...)`.
                If provided, they override any existing fitted configuration.

        Returns:
            PivotLongToWideResult: Typed summary of the generated wide dataset.
        """

        if kwargs:
            self.fit(**kwargs)
        if self._config is None:
            raise InputValidationError(
                "No configuration provided. Call `fit(...)` first or pass kwargs to `preprocess(...)`."
            )

        input_path = _as_path(self._config.get("input_path"), key="input_path")
        output_path = _as_path(self._config.get("output_path"), key="output_path")
        id_cols = tuple(self._config.get("id_cols", ()))
        time_col = str(self._config.get("time_col"))
        time_label = str(self._config.get("time_label"))
        separator = str(self._config.get("separator"))
        longitudinal_columns = self._config.get("longitudinal_columns")
        non_longitudinal_columns = self._config.get("non_longitudinal_columns")

        data = pd.read_csv(input_path)
        _ensure_columns_exist(
            data,
            [*id_cols, time_col],
            dataset_name="input long-format data",
        )

        if data[[*id_cols, time_col]].isna().any().any():
            raise InputValidationError(
                "ID/time columns contain missing values. Pure pivoting requires complete ID/time keys."
            )

        longitudinal_cols, non_longitudinal_cols = _resolve_feature_roles(
            data=data,
            id_cols=list(id_cols),
            time_col=time_col,
            longitudinal_cols=list(longitudinal_columns or ()),
            non_longitudinal_cols=list(non_longitudinal_columns or ()),
            longitudinal_auto=longitudinal_columns is None,
            non_longitudinal_auto=non_longitudinal_columns is None,
        )

        subject_static = _build_subject_static_frame(
            data=data,
            id_cols=list(id_cols),
            non_longitudinal_cols=non_longitudinal_cols,
        )
        subject_longitudinal = _build_subject_longitudinal_frame(
            data=data,
            id_cols=list(id_cols),
            time_col=time_col,
            longitudinal_cols=longitudinal_cols,
            time_label=time_label,
            separator=separator,
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
                subject_longitudinal, on=list(id_cols), how="outer"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        wide.to_csv(output_path, index=False)

        return PivotLongToWideResult(
            output_path=output_path.resolve(),
            row_count=len(wide),
            column_count=int(wide.shape[1]),
            id_cols=id_cols,
            time_col=time_col,
            time_label=time_label,
            separator=separator,
            longitudinal_columns=tuple(longitudinal_cols),
            non_longitudinal_columns=tuple(non_longitudinal_cols),
        )


@beartype
def run_pivot_long_to_wide(
    *, technique: str, params: Mapping[str, Any]
) -> dict[str, Any]:
    """Run one long-to-wide technique for the Go CLI bridge.

    !!! warning

        This `run_*` function is primarily for the Go CLI bridge and should not be
        treated as the Python library API. In Python scripts or notebooks, import
        and call `PivotLongToWide` directly from `ldt.data_preprocessing`.

    Args:
        technique (str): Pivot technique key from the catalogue.
        params (Mapping[str, Any]): Wrapper parameters forwarded to
            `PivotLongToWide.fit_preprocess(...)`:
            `input_path`, `output_path`, `id_cols`, `time_col`,
            optional feature-role lists, `time_label`, and `separator`.

    Returns:
        dict[str, Any]: Serialised pivot summary for the Go CLI bridge.
    """

    _, resolved = resolve_technique_with_defaults(
        section_key="pivot_long_to_wide",
        technique_id=technique,
        provided_params=dict(params),
    )

    result = PivotLongToWide().fit_preprocess(
        input_path=Path(
            _as_required_string(resolved.get("input_path"), field_name="input_path")
        ).expanduser(),
        output_path=Path(
            _as_required_string(resolved.get("output_path"), field_name="output_path")
        ).expanduser(),
        id_cols=_as_required_string_list(resolved.get("id_cols"), field_name="id_cols"),
        time_col=_as_required_string(resolved.get("time_col"), field_name="time_col"),
        longitudinal_columns=_as_optional_string_list(
            resolved.get("longitudinal_columns")
        ),
        non_longitudinal_columns=_as_optional_string_list(
            resolved.get("non_longitudinal_columns")
        ),
        time_label=_as_required_string(
            resolved.get("time_label"), field_name="time_label"
        ),
        separator=_as_required_string(
            resolved.get("separator"), field_name="separator"
        ),
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


@beartype
def _as_required_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"Missing required string parameter: {field_name}")
    return value.strip()


@beartype
def _as_required_string_list(value: Any, *, field_name: str) -> list[str]:
    parsed = _as_optional_string_list(value)
    if not parsed:
        raise InputValidationError(
            f"Missing required string-list parameter: {field_name}"
        )
    return parsed


@beartype
def _as_optional_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Sequence):
        parsed: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise InputValidationError("List values must be strings.")
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


@beartype
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


@beartype
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


@beartype
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


@beartype
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


@beartype
def _validate_input_csv_path(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise InputValidationError(f"Input CSV path does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise InputValidationError("Input path must point to a .csv file.")


@beartype
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


@beartype
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


@beartype
def _validate_separator(separator: str) -> None:
    if not separator:
        raise InputValidationError("Column-name separator cannot be empty.")
    if len(separator) != 1:
        raise InputValidationError("Column-name separator must be a single character.")
    if separator not in _ALLOWED_SEPARATORS:
        allowed = ", ".join(sorted(_ALLOWED_SEPARATORS))
        raise InputValidationError(f"Column-name separator must be one of: {allowed}")


@beartype
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
