from __future__ import annotations

from pathlib import Path
from typing import Any

from beartype import beartype

from ldt.utils.errors import InputValidationError
from ldt.utils.metadata import ComponentMetadata

from .common import (
    count_missing_values,
    load_csv,
    validate_input_csv_path,
    validate_output_csv_path,
    write_csv,
)
from .imputer import MissingImputer
from .results import SimpleImputationResult


@beartype
class MostFrequentImputer(MissingImputer):
    """Impute missing values with each column's most frequent non-missing value.

    Runtime parameters:
        - no extra parameters are required
        - every column is considered eligible, including categorical columns
        - columns containing only missing values are skipped and remain missing
    """

    metadata = ComponentMetadata(
        name="most_frequent_imputation",
        full_name="Most Frequent Imputation",
        abstract_description="Fill missing values with each column's most frequent value.",
    )

    def __init__(self) -> None:
        self._config: dict[str, Path] | None = None

    @beartype
    def fit(
        self, *, input_path: Path, output_path: Path, **kwargs: Any
    ) -> MostFrequentImputer:
        """Validate and store most-frequent-imputation configuration.

        Args:
            input_path (Path): Input CSV path.
            output_path (Path): Output CSV path.
            **kwargs (Any): No additional parameters are supported.

        Returns:
            MostFrequentImputer: The fitted imputer instance.
        """

        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise InputValidationError(
                "Most-frequent imputation does not accept extra parameters: "
                f"{unexpected}"
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
        """Execute most-frequent imputation and write the imputed CSV output.

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
                "No most-frequent-imputation configuration provided. Call `fit(...)` before `impute(...)`."
            )

        data = load_csv(input_path)
        missing_before = count_missing_values(data)
        eligible_columns = tuple(data.columns.tolist())

        imputed = data.copy()
        imputed_columns: list[str] = []
        skipped_columns: list[str] = []

        for column in eligible_columns:
            series = imputed[column]
            if not series.isna().any():
                continue
            mode = series.mode(dropna=True)
            if mode.empty:
                skipped_columns.append(column)
                continue
            imputed[column] = series.fillna(mode.iloc[0])
            imputed_columns.append(column)

        resolved_output = write_csv(imputed, output_path=output_path)
        return SimpleImputationResult(
            output_path=resolved_output,
            row_count=len(imputed),
            column_count=int(imputed.shape[1]),
            missing_before=missing_before,
            missing_after=count_missing_values(imputed),
            strategy="most_frequent",
            eligible_columns=eligible_columns,
            imputed_columns=tuple(imputed_columns),
            skipped_columns=tuple(skipped_columns),
        )
