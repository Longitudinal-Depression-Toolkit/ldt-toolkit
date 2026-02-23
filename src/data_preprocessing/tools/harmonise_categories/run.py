from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_preprocessing.catalog import resolve_technique_with_defaults
from src.data_preprocessing.support.inputs import (
    as_bool,
    as_optional_int,
    as_optional_string,
    as_required_string,
    as_required_string_list_or_csv,
    ensure_distinct_paths,
    run_with_validation,
)
from src.data_preprocessing.support.skrub_compat import import_skrub_symbol
from src.utils.errors import InputValidationError

deduplicate = import_skrub_symbol("deduplicate")


@dataclass(frozen=True)
class HarmoniseCategoriesRequest:
    """Request payload for category harmonisation.

    Attributes:
        input_path (Path): Input CSV path.
        output_path (Path): Output CSV path with harmonised categories.
        target_columns (tuple[str, ...]): Categorical columns to harmonise.
        n_clusters (int | None): Optional number of string-similarity clusters.
        mapping_path (Path | None): Optional CSV path for value-level mapping.
    """

    input_path: Path
    output_path: Path
    target_columns: tuple[str, ...]
    n_clusters: int | None
    mapping_path: Path | None


@dataclass(frozen=True)
class HarmoniseCategoriesResult:
    """Result payload from category harmonisation.

    Attributes:
        output_path (Path): Output CSV path.
        row_count (int): Number of rows in the harmonised dataset.
        column_count (int): Number of columns in the harmonised dataset.
        target_columns (tuple[str, ...]): Target columns that were harmonised.
        mapping_path (Path | None): Optional mapping CSV path.
        mapping_rows (int): Number of rows written to mapping CSV.
    """

    output_path: Path
    row_count: int
    column_count: int
    target_columns: tuple[str, ...]
    mapping_path: Path | None
    mapping_rows: int


def run_harmonise_categories(
    request: HarmoniseCategoriesRequest,
) -> HarmoniseCategoriesResult:
    """Harmonise inconsistent category labels with `skrub.deduplicate`.

    This operation reduces spelling/format variants in categorical columns
    (for example `Depressd`, `depressed`, `depression`) by clustering similar
    strings and replacing them with harmonised representatives.

    Harmonisation stages:

    | Stage | What happens |
    | --- | --- |
    | Extract non-null category values | Missing values are skipped. |
    | Estimate harmonised labels | `skrub.deduplicate` groups similar strings (`n_clusters` controls granularity). |
    | Rewrite dataset values | Original values are replaced by harmonised labels per target column. |
    | Optional mapping export | Writes `original_value -> harmonised_value` pairs to CSV. |

    Args:
        request (HarmoniseCategoriesRequest): Harmonisation configuration and
            input/output paths.

    Returns:
        HarmoniseCategoriesResult: Output dataset summary and optional mapping
            metadata.

    Examples:
        ```python
        from ldt.data_preprocessing.tools.harmonise_categories.run import (
            HarmoniseCategoriesRequest,
            run_harmonise_categories,
        )

        result = run_harmonise_categories(
            HarmoniseCategoriesRequest(
                input_path="data/raw.csv",
                output_path="outputs/harmonised.csv",
                target_columns=("diagnosis_label", "employment_status"),
                n_clusters=12,
                mapping_path="outputs/harmonisation_mapping.csv",
            )
        )
        ```
    """

    _validate_csv_path(request.input_path)
    _validate_output_csv_path(request.output_path, input_path=request.input_path)
    if not request.target_columns:
        raise InputValidationError("At least one target column must be provided.")
    if request.n_clusters is not None and request.n_clusters < 2:
        raise InputValidationError("n_clusters must be >= 2 when provided.")

    data = pd.read_csv(request.input_path)
    missing_columns = [
        column for column in request.target_columns if column not in data.columns
    ]
    if missing_columns:
        raise InputValidationError(
            f"Missing requested columns: {', '.join(missing_columns)}"
        )

    mapping_frames: list[pd.DataFrame] = []
    for column in request.target_columns:
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
                min(request.n_clusters, unique_count)
                if request.n_clusters is not None
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

    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(request.output_path, index=False)

    mapping_output_path: Path | None = None
    mapping_rows = 0
    if request.mapping_path is not None and mapping_frames:
        mapping_output = pd.concat(mapping_frames, ignore_index=True)
        request.mapping_path.parent.mkdir(parents=True, exist_ok=True)
        mapping_output.to_csv(request.mapping_path, index=False)
        mapping_output_path = request.mapping_path.resolve()
        mapping_rows = len(mapping_output)

    return HarmoniseCategoriesResult(
        output_path=request.output_path.resolve(),
        row_count=len(data),
        column_count=int(data.shape[1]),
        target_columns=request.target_columns,
        mapping_path=mapping_output_path,
        mapping_rows=mapping_rows,
    )


def run_harmonise_categories_tool(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Run category-harmonisation techniques from catalog payloads.

    Supported techniques:

    | Technique | What it does |
    | --- | --- |
    | `harmonise` | Harmonises selected categorical columns and optionally exports mapping tables. |

    Args:
        technique (str): Technique identifier in the harmonisation catalog.
        params (Mapping[str, Any]): Parameters mapped to
            `HarmoniseCategoriesRequest`.

    Returns:
        dict[str, Any]: Output path, row/column counts, target columns, and
            optional mapping output details.
    """

    return run_with_validation(
        lambda: _run_harmonise_categories_tool(technique=technique, params=params)
    )


def _run_harmonise_categories_tool(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    _, resolved = resolve_technique_with_defaults(
        section_key="harmonise_categories",
        technique_id=technique,
        provided_params=dict(params),
    )

    input_path = Path(as_required_string(resolved, "input_path")).expanduser()
    output_path = Path(as_required_string(resolved, "output_path")).expanduser()
    ensure_distinct_paths(input_path, output_path, field_name="output_path")

    target_columns = as_required_string_list_or_csv(resolved, "target_columns")
    n_clusters = as_optional_int(resolved, "n_clusters")
    if n_clusters is not None and n_clusters < 2:
        raise InputValidationError("n_clusters must be >= 2 when provided.")

    save_mapping = as_bool(
        resolved.get("save_mapping", True), field_name="save_mapping"
    )
    mapping_path: Path | None = None
    if save_mapping:
        mapping_raw = as_optional_string(resolved, "mapping_path")
        if mapping_raw:
            mapping_path = Path(mapping_raw).expanduser()
        else:
            mapping_path = input_path.with_name(
                f"{input_path.stem}_harmonisation_mapping.csv"
            )

    result = run_harmonise_categories(
        HarmoniseCategoriesRequest(
            input_path=input_path,
            output_path=output_path,
            target_columns=tuple(target_columns),
            n_clusters=n_clusters,
            mapping_path=mapping_path,
        )
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


def _validate_csv_path(path: Path) -> None:
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
