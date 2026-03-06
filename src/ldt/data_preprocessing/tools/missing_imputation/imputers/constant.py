from __future__ import annotations

from pathlib import Path
from typing import Any

from beartype import beartype

from ldt.utils.errors import InputValidationError
from ldt.utils.metadata import ComponentMetadata

from .common import (
    as_scalar_fill_value,
    count_missing_values,
    default_constant_fill_value,
    load_csv,
    validate_input_csv_path,
    validate_output_csv_path,
    write_csv,
)
from .imputer import MissingImputer
from .results import SimpleImputationResult


@beartype
class ConstantImputer(MissingImputer):
    """Impute missing values with one constant fill value.

    Runtime parameters:
        - `fill_value`: Optional scalar replacement applied to every missing
          cell. When omitted, numeric columns default to `0` and all other
          columns default to `"missing"`.
    """

    metadata = ComponentMetadata(
        name="constant_imputation",
        full_name="Constant Imputation",
        abstract_description="Fill missing values with a scalar constant.",
    )

    def __init__(self) -> None:
        self._config: dict[str, Any] | None = None

    @beartype
    def fit(
        self, *, input_path: Path, output_path: Path, **kwargs: Any
    ) -> ConstantImputer:
        """Validate and store constant-imputation configuration.

        Args:
            input_path (Path): Input CSV path.
            output_path (Path): Output CSV path.
            **kwargs (Any): Constant configuration keys:
                - `fill_value` (str | int | float | bool | None): Optional
                  scalar replacement value.

        Returns:
            ConstantImputer: The fitted imputer instance.
        """

        validate_input_csv_path(input_path)
        validate_output_csv_path(output_path, input_path=input_path)

        unknown_keys = set(kwargs) - {"fill_value"}
        if unknown_keys:
            unexpected = ", ".join(sorted(unknown_keys))
            raise InputValidationError(
                f"Constant imputation does not accept extra parameters: {unexpected}"
            )

        fill_value = kwargs.get("fill_value")
        if fill_value is not None:
            fill_value = as_scalar_fill_value(fill_value, field_name="fill_value")

        self._config = {
            "input_path": input_path,
            "output_path": output_path,
            "fill_value": fill_value,
        }
        return self

    @beartype
    def impute(
        self, *, input_path: Path, output_path: Path, **kwargs: Any
    ) -> SimpleImputationResult:
        """Execute constant imputation and write the imputed CSV output.

        Args:
            input_path (Path): Input CSV path.
            output_path (Path): Output CSV path.
            **kwargs (Any): Optional configuration keys identical to `fit(...)`.
                If provided, they override any existing fitted configuration.

        Returns:
            SimpleImputationResult: Typed summary of the imputation outputs and
            the columns imputed.
        """

        if kwargs:
            self.fit(input_path=input_path, output_path=output_path, **kwargs)
        if self._config is None:
            raise InputValidationError(
                "No constant-imputation configuration provided. Call `fit(...)` before `impute(...)`."
            )

        data = load_csv(input_path)
        missing_before = count_missing_values(data)
        eligible_columns = tuple(data.columns.tolist())
        fill_value = self._config["fill_value"]

        imputed = data.copy()
        imputed_columns = [
            column for column in eligible_columns if imputed[column].isna().any()
        ]

        if fill_value is None:
            for column in imputed_columns:
                imputed[column] = imputed[column].fillna(
                    default_constant_fill_value(imputed[column])
                )
        else:
            imputed = imputed.fillna(fill_value)

        resolved_output = write_csv(imputed, output_path=output_path)
        return SimpleImputationResult(
            output_path=resolved_output,
            row_count=len(imputed),
            column_count=int(imputed.shape[1]),
            missing_before=missing_before,
            missing_after=count_missing_values(imputed),
            strategy="constant",
            eligible_columns=eligible_columns,
            imputed_columns=tuple(imputed_columns),
            fill_value=fill_value,
        )
