from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import miceforest as mf
import pandas as pd
from beartype import beartype

from ldt.utils.errors import InputValidationError
from ldt.utils.metadata import ComponentMetadata

from .imputer import MissingImputer


@beartype
@dataclass(frozen=True)
class MICEImputationResult:
    """Structured result for MICE missing-data imputation."""

    output_path: Path
    row_count: int
    column_count: int
    missing_before: int
    missing_after: int
    iterations: int
    num_datasets: int
    dataset_index: int


@beartype
class MICEImputer(MissingImputer):
    """Impute missing values using `miceforest` MICE.

    Runtime parameters:
        - `iterations`: Number of MICE iterations.
        - `num_datasets`: Number of imputed datasets produced by the kernel.
        - `dataset_index`: Which imputed dataset to export.
        - `mean_match_candidates`: Predictive mean matching candidate count.
        - `random_state`: Optional random seed.
        - `cast_object_to_category`: Convert object/string columns to
          categorical before kernel fitting.
    """

    metadata = ComponentMetadata(
        name="mice_imputation",
        full_name="MICE Imputation",
        abstract_description="Impute missing values with MICE via miceforest.",
    )

    def __init__(self) -> None:
        self._config: dict[str, Any] | None = None

    @beartype
    def fit(self, *, input_path: Path, output_path: Path, **kwargs: Any) -> MICEImputer:
        """Validate and store MICE configuration.

        Args:
            input_path (Path): Input CSV path.
            output_path (Path): Output CSV path.
            **kwargs (Any): MICE configuration keys:
                - `iterations` (int): Number of MICE iterations.
                - `num_datasets` (int): Number of imputed datasets.
                - `dataset_index` (int): Index of dataset to export.
                - `mean_match_candidates` (int): PMM candidate count.
                - `random_state` (int | None): Optional random seed.
                - `cast_object_to_category` (bool): Cast text columns before
                  imputation.

        Returns:
            MICEImputer: The fitted imputer instance.
        """

        _validate_csv_path(input_path)
        _validate_output_csv_path(output_path, input_path=input_path)

        iterations = _as_int(kwargs.get("iterations", 5), field_name="iterations")
        num_datasets = _as_int(
            kwargs.get("num_datasets", 1),
            field_name="num_datasets",
        )
        dataset_index = _as_int(
            kwargs.get("dataset_index", 0),
            field_name="dataset_index",
        )
        mean_match_candidates = _as_int(
            kwargs.get("mean_match_candidates", 5),
            field_name="mean_match_candidates",
        )
        random_state = _as_optional_int(kwargs.get("random_state"))
        cast_object_to_category = _as_bool(
            kwargs.get("cast_object_to_category", True),
            field_name="cast_object_to_category",
        )

        if iterations < 1:
            raise InputValidationError("iterations must be >= 1.")
        if num_datasets < 1:
            raise InputValidationError("num_datasets must be >= 1.")
        if dataset_index < 0:
            raise InputValidationError("dataset_index must be >= 0.")
        if dataset_index >= num_datasets:
            raise InputValidationError("dataset_index must be < num_datasets.")
        if mean_match_candidates < 0:
            raise InputValidationError("mean_match_candidates must be >= 0.")

        self._config = {
            "input_path": input_path,
            "output_path": output_path,
            "iterations": iterations,
            "num_datasets": num_datasets,
            "dataset_index": dataset_index,
            "mean_match_candidates": mean_match_candidates,
            "random_state": random_state,
            "cast_object_to_category": cast_object_to_category,
        }
        return self

    @beartype
    def impute(
        self, *, input_path: Path, output_path: Path, **kwargs: Any
    ) -> MICEImputationResult:
        """Execute MICE and write the imputed CSV output.

        Args:
            input_path (Path): Input CSV path.
            output_path (Path): Output CSV path.
            **kwargs (Any): Optional configuration keys identical to `fit(...)`.
                If provided, they override any existing fitted configuration.

        Returns:
            MICEImputationResult: Typed summary of imputation outputs and
            missingness counts.
        """

        if kwargs:
            self.fit(input_path=input_path, output_path=output_path, **kwargs)
        if self._config is None:
            raise InputValidationError(
                "No MICE configuration provided. Call `fit(...)` before `impute(...)`."
            )

        iterations = int(self._config["iterations"])
        num_datasets = int(self._config["num_datasets"])
        dataset_index = int(self._config["dataset_index"])
        mean_match_candidates = int(self._config["mean_match_candidates"])
        random_state = self._config["random_state"]
        cast_object_to_category = bool(self._config["cast_object_to_category"])

        data = pd.read_csv(input_path)
        missing_before = int(data.isna().sum().sum())

        if cast_object_to_category:
            object_columns = data.select_dtypes(include=["object", "string"]).columns
            for column in object_columns:
                data[column] = data[column].astype("category")

        if missing_before == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(output_path, index=False)
            return MICEImputationResult(
                output_path=output_path.resolve(),
                row_count=len(data),
                column_count=int(data.shape[1]),
                missing_before=0,
                missing_after=0,
                iterations=iterations,
                num_datasets=num_datasets,
                dataset_index=dataset_index,
            )

        try:
            kernel = mf.ImputationKernel(
                data=data,
                num_datasets=num_datasets,
                mean_match_candidates=mean_match_candidates,
                save_all_iterations_data=True,
                random_state=random_state,
            )
            kernel.mice(iterations=iterations, verbose=False)
            imputed = kernel.complete_data(dataset=dataset_index, inplace=False)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise InputValidationError(f"MICE imputation failed: {exc}") from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        imputed.to_csv(output_path, index=False)

        return MICEImputationResult(
            output_path=output_path.resolve(),
            row_count=len(imputed),
            column_count=int(imputed.shape[1]),
            missing_before=missing_before,
            missing_after=int(imputed.isna().sum().sum()),
            iterations=iterations,
            num_datasets=num_datasets,
            dataset_index=dataset_index,
        )


@beartype
def _as_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise InputValidationError(f"`{field_name}` must be an integer value.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise InputValidationError(f"`{field_name}` must be an integer value.")
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except ValueError as exc:
            raise InputValidationError(
                f"`{field_name}` must be an integer value."
            ) from exc
    raise InputValidationError(f"`{field_name}` must be an integer value.")


@beartype
def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return _as_int(value, field_name="random_state")


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
