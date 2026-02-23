from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_preprocessing.catalog import resolve_technique_with_defaults
from src.data_preprocessing.support.inputs import (
    as_required_string,
    ensure_distinct_paths,
    run_with_validation,
)
from src.utils.errors import InputValidationError


@dataclass(frozen=True)
class RenameFeatureRequest:
    """Request payload for feature renaming.

    Attributes:
        input_path (Path): Input CSV path.
        output_path (Path): Output CSV path.
        feature_name (str): Existing feature name.
        new_feature_name (str): New feature name.
    """

    input_path: Path
    output_path: Path
    feature_name: str
    new_feature_name: str


@dataclass(frozen=True)
class RenameFeatureResult:
    """Result payload produced by feature renaming.

    Attributes:
        output_path (Path): Output CSV path.
        row_count (int): Number of rows in output dataset.
        column_count (int): Number of columns in output dataset.
        renamed_from (str): Original feature name.
        renamed_to (str): New feature name.
    """

    output_path: Path
    row_count: int
    column_count: int
    renamed_from: str
    renamed_to: str


def run_rename_feature(request: RenameFeatureRequest) -> RenameFeatureResult:
    """Rename one feature column in a CSV dataset.

    This operation validates that the source column exists and that the new
    name is not already used, then writes an updated dataset.

    Fictional input:

    | subject_id | dep_score_w1 | dep_score_w2 |
    | --- | --- | --- |
    | 101 | 4.0 | 3.0 |
    | 102 | 7.0 | 8.0 |

    Renaming `dep_score_w1` to `depression_score_wave1` produces:

    | subject_id | depression_score_wave1 | dep_score_w2 |
    | --- | --- | --- |
    | 101 | 4.0 | 3.0 |
    | 102 | 7.0 | 8.0 |

    Args:
        request (RenameFeatureRequest): Input/output paths plus source and
            destination column names.

    Returns:
        RenameFeatureResult: Output path, output dimensions, and rename mapping.
    """

    _validate_input_csv_path(request.input_path)
    _validate_output_csv_path(request.output_path, input_path=request.input_path)
    _validate_feature_names(
        feature_name=request.feature_name,
        new_feature_name=request.new_feature_name,
    )

    data = pd.read_csv(request.input_path)
    _validate_rename_is_possible(
        available_columns=list(data.columns),
        feature_name=request.feature_name,
        new_feature_name=request.new_feature_name,
    )

    updated = data.rename(columns={request.feature_name: request.new_feature_name})
    request.output_path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(request.output_path, index=False)

    return RenameFeatureResult(
        output_path=request.output_path.resolve(),
        row_count=len(updated),
        column_count=int(updated.shape[1]),
        renamed_from=request.feature_name,
        renamed_to=request.new_feature_name,
    )


def run_rename_feature_tool(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Run feature-renaming techniques from catalog payloads.

    Supported techniques:

    | Technique | What it does |
    | --- | --- |
    | `rename` | Renames exactly one feature column in a CSV dataset. |

    Args:
        technique (str): Technique identifier in the rename-feature catalog.
        params (Mapping[str, Any]): Parameters mapped to `RenameFeatureRequest`.

    Returns:
        dict[str, Any]: Output path, output dimensions, and rename mapping.
    """

    return run_with_validation(
        lambda: _run_rename_feature_tool(technique=technique, params=params)
    )


def _run_rename_feature_tool(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    _, resolved = resolve_technique_with_defaults(
        section_key="rename_feature",
        technique_id=technique,
        provided_params=dict(params),
    )

    input_path = Path(as_required_string(resolved, "input_path")).expanduser()
    output_path = Path(as_required_string(resolved, "output_path")).expanduser()
    ensure_distinct_paths(input_path, output_path, field_name="output_path")

    feature_name = as_required_string(resolved, "feature_name")
    new_feature_name = as_required_string(resolved, "new_feature_name")

    result = run_rename_feature(
        RenameFeatureRequest(
            input_path=input_path,
            output_path=output_path,
            feature_name=feature_name,
            new_feature_name=new_feature_name,
        )
    )
    return {
        "output_path": str(result.output_path),
        "row_count": result.row_count,
        "column_count": result.column_count,
        "renamed_from": result.renamed_from,
        "renamed_to": result.renamed_to,
    }


def _validate_rename_is_possible(
    *,
    available_columns: list[str],
    feature_name: str,
    new_feature_name: str,
) -> None:
    if feature_name not in available_columns:
        suggestion = _build_missing_column_suggestion(
            missing_column=feature_name,
            available_columns=available_columns,
        )
        raise InputValidationError(
            f"Feature name not found in input CSV: {feature_name}.{suggestion}"
        )
    if new_feature_name in available_columns:
        raise InputValidationError(
            f"New feature name already exists in input CSV: {new_feature_name}"
        )


def _build_missing_column_suggestion(
    *,
    missing_column: str,
    available_columns: list[str],
) -> str:
    candidates = get_close_matches(
        missing_column,
        available_columns,
        n=3,
        cutoff=0.55,
    )
    if candidates:
        return f" Did you mean: {', '.join(candidates)}"
    preview = ", ".join(available_columns[:5])
    suffix = "..." if len(available_columns) > 5 else ""
    return f" Available columns include: {preview}{suffix}"


def _validate_feature_names(*, feature_name: str, new_feature_name: str) -> None:
    if not feature_name:
        raise InputValidationError("Feature name to rename cannot be empty.")
    if not new_feature_name:
        raise InputValidationError("New feature name cannot be empty.")
    if feature_name == new_feature_name:
        raise InputValidationError(
            "Feature name and new feature name must be different."
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
