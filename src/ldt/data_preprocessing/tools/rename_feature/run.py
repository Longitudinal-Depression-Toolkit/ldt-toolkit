from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import pandas as pd
from beartype import beartype

from ldt.data_preprocessing.catalog import resolve_technique_with_defaults
from ldt.utils.errors import InputValidationError
from ldt.utils.metadata import ComponentMetadata
from ldt.utils.templates.tools.data_preprocessing import DataPreprocessingTool


@beartype
@dataclass(frozen=True)
class RenameFeatureResult:
    """Structured output produced by feature renaming."""

    output_path: Path
    row_count: int
    column_count: int
    renamed_from: str
    renamed_to: str


@beartype
class RenameFeature(DataPreprocessingTool):
    """Rename one feature column in a CSV dataset.

    Runtime parameters:
        - `input_path`: Input CSV path.
        - `output_path`: Output CSV path.
        - `feature_name`: Existing column name to rename.
        - `new_feature_name`: Replacement column name.

    Examples:
        ```python
        from ldt.data_preprocessing import RenameFeature

        tool = RenameFeature()
        result = tool.fit_preprocess(
            input_path="./data/features.csv",
            output_path="./data/features_renamed.csv",
            feature_name="dep_score_w1",
            new_feature_name="depression_score_wave1",
        )
        ```
    """

    metadata = ComponentMetadata(
        name="rename_feature",
        full_name="Rename Feature",
        abstract_description="Rename one selected feature column in a CSV dataset.",
    )

    def __init__(self) -> None:
        self._config: dict[str, Any] | None = None

    @beartype
    def fit(self, **kwargs: Any) -> RenameFeature:
        """Validate and store feature-renaming configuration.

        Args:
            **kwargs (Any): Configuration keys:
                - `input_path` (str | Path): Input CSV path.
                - `output_path` (str | Path): Output CSV path.
                - `feature_name` (str): Existing column to rename.
                - `new_feature_name` (str): New column name.

        Returns:
            RenameFeature: The fitted tool instance.
        """

        input_path_raw = kwargs.get("input_path")
        output_path_raw = kwargs.get("output_path")
        feature_name = kwargs.get("feature_name")
        new_feature_name = kwargs.get("new_feature_name")

        if input_path_raw is None:
            raise InputValidationError("Missing required parameter: input_path")
        if output_path_raw is None:
            raise InputValidationError("Missing required parameter: output_path")
        if not isinstance(feature_name, str) or not feature_name.strip():
            raise InputValidationError("Missing required parameter: feature_name")
        if not isinstance(new_feature_name, str) or not new_feature_name.strip():
            raise InputValidationError("Missing required parameter: new_feature_name")

        input_path = Path(str(input_path_raw)).expanduser()
        output_path = Path(str(output_path_raw)).expanduser()

        _validate_input_csv_path(input_path)
        _validate_output_csv_path(output_path, input_path=input_path)
        _validate_feature_names(
            feature_name=feature_name.strip(),
            new_feature_name=new_feature_name.strip(),
        )

        self._config = {
            "input_path": input_path,
            "output_path": output_path,
            "feature_name": feature_name.strip(),
            "new_feature_name": new_feature_name.strip(),
        }
        return self

    @beartype
    def preprocess(self, **kwargs: Any) -> RenameFeatureResult:
        """Rename a configured column and write the updated CSV file.

        Args:
            **kwargs (Any): Optional configuration keys identical to `fit(...)`.
                If provided, they override any existing fitted configuration.

        Returns:
            RenameFeatureResult: Typed summary of rename outputs.
        """

        if kwargs:
            self.fit(**kwargs)
        if self._config is None:
            raise InputValidationError(
                "No configuration provided. Call `fit(...)` first or pass kwargs to `preprocess(...)`."
            )

        input_path = _as_path(self._config.get("input_path"), key="input_path")
        output_path = _as_path(self._config.get("output_path"), key="output_path")
        feature_name = str(self._config.get("feature_name"))
        new_feature_name = str(self._config.get("new_feature_name"))

        data = pd.read_csv(input_path)
        _validate_rename_is_possible(
            available_columns=list(data.columns),
            feature_name=feature_name,
            new_feature_name=new_feature_name,
        )

        updated = data.rename(columns={feature_name: new_feature_name})
        output_path.parent.mkdir(parents=True, exist_ok=True)
        updated.to_csv(output_path, index=False)

        return RenameFeatureResult(
            output_path=output_path.resolve(),
            row_count=len(updated),
            column_count=int(updated.shape[1]),
            renamed_from=feature_name,
            renamed_to=new_feature_name,
        )


@beartype
def run_rename_feature(*, technique: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Run one rename-feature technique for the Go CLI bridge.

    !!! warning

        This `run_*` function is primarily for the Go CLI bridge and should not be
        treated as the Python library API. In Python scripts or notebooks, import
        and call `RenameFeature` directly from `ldt.data_preprocessing`.

    Args:
        technique (str): Rename-feature technique key from the catalogue.
        params (Mapping[str, Any]): Wrapper parameters forwarded to
            `RenameFeature.fit_preprocess(...)`:
            `input_path`, `output_path`, `feature_name`, and
            `new_feature_name`.

    Returns:
        dict[str, Any]: Serialised rename summary for the Go CLI bridge.
    """

    _, resolved = resolve_technique_with_defaults(
        section_key="rename_feature",
        technique_id=technique,
        provided_params=dict(params),
    )

    result = RenameFeature().fit_preprocess(
        input_path=Path(_as_required_string(resolved, "input_path")).expanduser(),
        output_path=Path(_as_required_string(resolved, "output_path")).expanduser(),
        feature_name=_as_required_string(resolved, "feature_name"),
        new_feature_name=_as_required_string(resolved, "new_feature_name"),
    )

    return {
        "output_path": str(result.output_path),
        "row_count": result.row_count,
        "column_count": result.column_count,
        "renamed_from": result.renamed_from,
        "renamed_to": result.renamed_to,
    }


@beartype
def _as_required_string(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"Missing required string parameter: {key}")
    return value.strip()


@beartype
def _as_path(value: Any, *, key: str) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value.strip():
        return Path(value).expanduser()
    raise InputValidationError(f"Missing required path parameter: {key}")


@beartype
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


@beartype
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


@beartype
def _validate_feature_names(*, feature_name: str, new_feature_name: str) -> None:
    if not feature_name:
        raise InputValidationError("Feature name to rename cannot be empty.")
    if not new_feature_name:
        raise InputValidationError("New feature name cannot be empty.")
    if feature_name == new_feature_name:
        raise InputValidationError(
            "Feature name and new feature name must be different."
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
