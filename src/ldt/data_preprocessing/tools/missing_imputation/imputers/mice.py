from __future__ import annotations

from pathlib import Path
from typing import Any

import miceforest as mf
from beartype import beartype

from ldt.utils.errors import InputValidationError
from ldt.utils.metadata import ComponentMetadata

from .common import (
    as_bool,
    as_int,
    as_optional_int,
    count_missing_values,
    load_csv,
    validate_input_csv_path,
    validate_output_csv_path,
    write_csv,
)
from .imputer import MissingImputer
from .results import MICEImputationResult


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

        validate_input_csv_path(input_path)
        validate_output_csv_path(output_path, input_path=input_path)

        iterations = as_int(kwargs.get("iterations", 5), field_name="iterations")
        num_datasets = as_int(
            kwargs.get("num_datasets", 1),
            field_name="num_datasets",
        )
        dataset_index = as_int(
            kwargs.get("dataset_index", 0),
            field_name="dataset_index",
        )
        mean_match_candidates = as_int(
            kwargs.get("mean_match_candidates", 5),
            field_name="mean_match_candidates",
        )
        random_state = as_optional_int(
            kwargs.get("random_state"),
            field_name="random_state",
        )
        cast_object_to_category = as_bool(
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

        data = load_csv(input_path)
        missing_before = count_missing_values(data)

        if cast_object_to_category:
            object_columns = data.select_dtypes(include=["object", "string"]).columns
            for column in object_columns:
                data[column] = data[column].astype("category")

        if missing_before == 0:
            resolved_output = write_csv(data, output_path=output_path)
            return MICEImputationResult(
                output_path=resolved_output,
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

        resolved_output = write_csv(imputed, output_path=output_path)
        return MICEImputationResult(
            output_path=resolved_output,
            row_count=len(imputed),
            column_count=int(imputed.shape[1]),
            missing_before=missing_before,
            missing_after=count_missing_values(imputed),
            iterations=iterations,
            num_datasets=num_datasets,
            dataset_index=dataset_index,
        )
