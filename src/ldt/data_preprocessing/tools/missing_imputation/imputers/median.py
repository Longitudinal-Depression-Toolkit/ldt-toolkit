from __future__ import annotations

from pathlib import Path
from typing import Any

from beartype import beartype

from ldt.utils.errors import InputValidationError
from ldt.utils.metadata import ComponentMetadata

from .common import (
    count_missing_values,
    load_csv,
    numeric_columns,
    validate_input_csv_path,
    validate_output_csv_path,
    write_csv,
)
from .imputer import MissingImputer
from .results import SimpleImputationResult


@beartype
class MedianImputer(MissingImputer):
    """Impute missing numeric values with each column median.

    Runtime parameters:
        - no extra parameters are required
        - numeric columns are imputed independently using their own median
        - non-numeric columns are left unchanged
        - columns containing only missing values are skipped and remain missing
    """

    metadata = ComponentMetadata(
        name="median_imputation",
        full_name="Median Imputation",
        abstract_description="Fill missing numeric values with each column median.",
    )

    def __init__(self) -> None:
        self._config: dict[str, Path] | None = None

    @beartype
    def fit(
        self, *, input_path: Path, output_path: Path, **kwargs: Any
    ) -> MedianImputer:
        """Validate and store median-imputation configuration.

        Args:
            input_path (Path): Input CSV path.
            output_path (Path): Output CSV path.
            **kwargs (Any): No additional parameters are supported.

        Returns:
            MedianImputer: The fitted imputer instance.
        """

        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise InputValidationError(
                f"Median imputation does not accept extra parameters: {unexpected}"
            )

        validate_input_csv_path(input_path)
        validate_output_csv_path(output_path, input_path=input_path)
        self._config = {
            "input_path": input_path,
            "output_path": output_path,
        }
        return self

    @beartype
    def impute(
        self, *, input_path: Path, output_path: Path, **kwargs: Any
    ) -> SimpleImputationResult:
        """Execute median imputation and write the imputed CSV output.

        Args:
            input_path (Path): Input CSV path.
            output_path (Path): Output CSV path.
            **kwargs (Any): Optional configuration keys identical to `fit(...)`.
                If provided, they override any existing fitted configuration.

        Returns:
            SimpleImputationResult: Typed summary of the imputation outputs and
            the columns imputed or skipped.
        """

        if kwargs:
            self.fit(input_path=input_path, output_path=output_path, **kwargs)
        if self._config is None:
            raise InputValidationError(
                "No median-imputation configuration provided. Call `fit(...)` before `impute(...)`."
            )

        data = load_csv(input_path)
        missing_before = count_missing_values(data)
        eligible_columns = numeric_columns(data)
        if missing_before > 0 and not eligible_columns:
            raise InputValidationError(
                "Median imputation requires at least one numeric column with missing values."
            )

        imputed = data.copy()
        imputed_columns: list[str] = []
        skipped_columns: list[str] = []

        for column in eligible_columns:
            series = imputed[column]
            if not series.isna().any():
                continue
            if series.dropna().empty:
                skipped_columns.append(column)
                continue
            fill_value = series.median(skipna=True)
            imputed[column] = series.fillna(fill_value)
            imputed_columns.append(column)

        resolved_output = write_csv(imputed, output_path=output_path)
        return SimpleImputationResult(
            output_path=resolved_output,
            row_count=len(imputed),
            column_count=int(imputed.shape[1]),
            missing_before=missing_before,
            missing_after=count_missing_values(imputed),
            strategy="median",
            eligible_columns=eligible_columns,
            imputed_columns=tuple(imputed_columns),
            skipped_columns=tuple(skipped_columns),
        )
