from __future__ import annotations

from collections.abc import Mapping, Sequence
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

deduplicate = import_skrub_symbol("deduplicate")


@beartype
@dataclass(frozen=True)
class HarmoniseCategoriesResult:
    """Structured output from category harmonisation."""

    output_path: Path
    row_count: int
    column_count: int
    target_columns: tuple[str, ...]
    mapping_path: Path | None
    mapping_rows: int


@beartype
class HarmoniseCategories(DataPreprocessingTool):
    """Harmonise inconsistent category labels with `skrub.deduplicate`.

    This tool reduces spelling and formatting variants in categorical columns by
    clustering similar strings and replacing them with harmonised values.

    Initialisation parameters:
        input_path (Path | str | None): Optional default input CSV path.
        output_path (Path | str | None): Optional default output CSV path.
        target_columns (Sequence[str] | None): Optional default categorical
            columns to harmonise.
        n_clusters (int | None): Optional fixed cluster count per column.
        mapping_path (Path | str | None): Optional path to export original to
            harmonised value mappings.

    Examples:
        ```python
        from ldt.data_preprocessing import HarmoniseCategories

        tool = HarmoniseCategories()
        result = tool.fit_preprocess(
            input_path="./data/raw.csv",
            output_path="./outputs/harmonised.csv",
            target_columns=["diagnosis_label", "employment_status"],
            n_clusters=12,
            mapping_path="./outputs/harmonisation_mapping.csv",
        )
        ```
    """

    metadata = ComponentMetadata(
        name="harmonise_categories",
        full_name="Harmonise Categories",
        abstract_description=(
            "Harmonise inconsistent category labels across selected columns."
        ),
    )

    def __init__(
        self,
        *,
        input_path: Path | str | None = None,
        output_path: Path | str | None = None,
        target_columns: Sequence[str] | None = None,
        n_clusters: int | None = None,
        mapping_path: Path | str | None = None,
    ) -> None:
        self._input_path = (
            Path(input_path).expanduser() if input_path is not None else None
        )
        self._output_path = (
            Path(output_path).expanduser() if output_path is not None else None
        )
        self._target_columns = tuple(
            str(column).strip()
            for column in (target_columns or ())
            if str(column).strip()
        )
        self._n_clusters = n_clusters
        self._mapping_path = (
            Path(mapping_path).expanduser() if mapping_path is not None else None
        )

    @beartype
    def fit(self, **kwargs: Any) -> HarmoniseCategories:
        """Validate and store harmonisation configuration.

        Args:
            **kwargs (Any): Configuration overrides:
                - `input_path` (str | Path): Input CSV path.
                - `output_path` (str | Path): Output CSV path.
                - `target_columns` (str | Sequence[str]): Columns to harmonise.
                - `n_clusters` (int | None): Optional fixed cluster count.
                - `mapping_path` (str | Path | None): Optional mapping export
                  CSV path.

        Returns:
            HarmoniseCategories: The fitted tool instance.
        """

        input_path = kwargs.get("input_path", self._input_path)
        output_path = kwargs.get("output_path", self._output_path)
        target_columns = kwargs.get("target_columns", self._target_columns)

        if input_path is None:
            raise InputValidationError("Missing required parameter: input_path")
        if output_path is None:
            raise InputValidationError("Missing required parameter: output_path")

        self._input_path = Path(str(input_path)).expanduser()
        self._output_path = Path(str(output_path)).expanduser()
        self._target_columns = _as_required_columns(target_columns)
        self._n_clusters = _as_optional_int(kwargs.get("n_clusters", self._n_clusters))
        mapping_value = kwargs.get("mapping_path", self._mapping_path)
        self._mapping_path = (
            Path(str(mapping_value)).expanduser() if mapping_value is not None else None
        )

        _validate_csv_path(self._input_path)
        _validate_output_csv_path(self._output_path, input_path=self._input_path)

        if self._n_clusters is not None and self._n_clusters < 2:
            raise InputValidationError("n_clusters must be >= 2 when provided.")

        return self

    @beartype
    def preprocess(self, **kwargs: Any) -> HarmoniseCategoriesResult:
        """Run category harmonisation and optional mapping export.

        Args:
            **kwargs (Any): Optional configuration keys identical to `fit(...)`.
                If provided, they override any existing fitted configuration.

        Returns:
            HarmoniseCategoriesResult: Typed summary including output path,
            processed columns, and optional mapping export details.
        """

        if kwargs:
            self.fit(**kwargs)
        if self._input_path is None or self._output_path is None:
            raise InputValidationError(
                "No configuration provided. Call `fit(...)` first or pass kwargs to `preprocess(...)`."
            )

        data = pd.read_csv(self._input_path)
        missing_columns = [
            column for column in self._target_columns if column not in data.columns
        ]
        if missing_columns:
            raise InputValidationError(
                f"Missing requested columns: {', '.join(missing_columns)}"
            )

        mapping_frames: list[pd.DataFrame] = []
        for column in self._target_columns:
            original_series = data[column]
            mask = original_series.notna()
            if not mask.any():
                continue

            original_values = original_series.loc[mask].astype(str)
            unique_count = int(original_values.nunique(dropna=True))
            if unique_count < 2:
                harmonised_series = original_values
            else:
                effective_n_clusters = (
                    min(self._n_clusters, unique_count)
                    if self._n_clusters is not None
                    else None
                )
                try:
                    harmonised_values = deduplicate(
                        original_values.tolist(),
                        n_clusters=effective_n_clusters,
                    )
                    harmonised_series = pd.Series(
                        harmonised_values,
                        index=original_values.index,
                        dtype="object",
                    )
                except ValueError:
                    harmonised_series = original_values

            data.loc[mask, column] = harmonised_series
            mapping_frame = pd.DataFrame(
                {
                    "column": column,
                    "original_value": original_values.values,
                    "harmonised_value": harmonised_series.values,
                }
            )
            mapping_frames.append(
                mapping_frame.drop_duplicates().sort_values(
                    by=["original_value", "harmonised_value"]
                )
            )

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(self._output_path, index=False)

        mapping_output_path: Path | None = None
        mapping_rows = 0
        if self._mapping_path is not None and mapping_frames:
            mapping_output = pd.concat(mapping_frames, ignore_index=True)
            self._mapping_path.parent.mkdir(parents=True, exist_ok=True)
            mapping_output.to_csv(self._mapping_path, index=False)
            mapping_output_path = self._mapping_path.resolve()
            mapping_rows = len(mapping_output)

        return HarmoniseCategoriesResult(
            output_path=self._output_path.resolve(),
            row_count=len(data),
            column_count=int(data.shape[1]),
            target_columns=self._target_columns,
            mapping_path=mapping_output_path,
            mapping_rows=mapping_rows,
        )


@beartype
def run_harmonise_categories(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one harmonise-categories technique for the Go CLI bridge.

    !!! warning

        This `run_*` function is primarily for the Go CLI bridge and should not be
        treated as the Python library API. In Python scripts or notebooks, import
        and call `HarmoniseCategories` directly from `ldt.data_preprocessing`.

    Args:
        technique (str): Harmonisation technique key from the catalogue.
        params (Mapping[str, Any]): Wrapper parameters forwarded to
            `HarmoniseCategories.fit_preprocess(...)`:
            `input_path`, `output_path`, `target_columns`, `n_clusters`,
            optional mapping controls (`save_mapping`, `mapping_path`).

    Returns:
        dict[str, Any]: Serialised harmonisation summary for the Go CLI bridge.
    """

    _, resolved = resolve_technique_with_defaults(
        section_key="harmonise_categories",
        technique_id=technique,
        provided_params=dict(params),
    )

    input_path = Path(_as_required_string(resolved, "input_path")).expanduser()
    output_path = Path(_as_required_string(resolved, "output_path")).expanduser()

    target_columns = _as_required_columns(resolved.get("target_columns"))
    n_clusters = _as_optional_int(resolved.get("n_clusters"))

    save_mapping = _as_bool(
        resolved.get("save_mapping", True), field_name="save_mapping"
    )
    mapping_path: Path | None = None
    if save_mapping:
        mapping_raw = _as_optional_string(resolved.get("mapping_path"))
        if mapping_raw:
            mapping_path = Path(mapping_raw).expanduser()
        else:
            mapping_path = input_path.with_name(
                f"{input_path.stem}_harmonisation_mapping.csv"
            )

    result = HarmoniseCategories().fit_preprocess(
        input_path=input_path,
        output_path=output_path,
        target_columns=target_columns,
        n_clusters=n_clusters,
        mapping_path=mapping_path,
    )

    return {
        "output_path": str(result.output_path),
        "row_count": result.row_count,
        "column_count": result.column_count,
        "target_columns": list(result.target_columns),
        "mapping_path": (
            str(result.mapping_path) if result.mapping_path is not None else None
        ),
        "mapping_rows": result.mapping_rows,
    }


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
    raise InputValidationError("Expected a string value.")


@beartype
def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            parsed = int(value.strip())
        except ValueError as exc:
            raise InputValidationError("Expected an integer value.") from exc
        return parsed
    if isinstance(value, bool):
        raise InputValidationError("Expected an integer value.")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise InputValidationError("Expected an integer value.")


@beartype
def _as_required_columns(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        parsed = tuple(item.strip() for item in value.split(",") if item.strip())
        if parsed:
            return parsed
    if isinstance(value, Sequence):
        parsed = tuple(
            str(item).strip()
            for item in value
            if isinstance(item, str) and str(item).strip()
        )
        if parsed:
            return parsed
    raise InputValidationError("At least one target column must be provided.")


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
def _validate_csv_path(path: Path) -> None:
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
