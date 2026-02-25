from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from beartype import beartype

from ldt.data_preprocessing.catalog import resolve_technique_with_defaults
from ldt.utils.errors import InputValidationError
from ldt.utils.metadata import ComponentMetadata, resolve_component_metadata
from ldt.utils.templates.tools.data_preprocessing import DataPreprocessingTool

from .discovery import discover_missing_imputers
from .imputers import MICEImputationResult, MissingImputer


@beartype
class MissingImputation(DataPreprocessingTool):
    """Dispatch missing-data imputation to concrete imputer implementations.

    This stage tool is deliberately thin and delegates method-specific logic to
    classes in `tools/missing_imputation/imputers`.

    Runtime parameters:
        - `input_path`: Input CSV path.
        - `output_path`: Output CSV path.
        - `technique` or `imputer`: Imputation key (currently
          `mice_imputation`).
        - any additional key-value pairs are forwarded to the selected imputer.

    Examples:
        ```python
        from ldt.data_preprocessing import MissingImputation

        tool = MissingImputation()
        result = tool.fit_preprocess(
            technique="mice_imputation",
            input_path="./data/raw.csv",
            output_path="./outputs/imputed.csv",
            iterations=5,
            num_datasets=1,
            dataset_index=0,
            mean_match_candidates=5,
            cast_object_to_category=True,
            random_state=42,
        )
        ```
    """

    metadata = ComponentMetadata(
        name="missing_imputation",
        full_name="Missing Imputation",
        abstract_description="Run missing-data imputation using registered imputers.",
    )

    def __init__(self) -> None:
        self._input_path: Path | None = None
        self._output_path: Path | None = None
        self._imputer_key: str | None = None
        self._imputer_kwargs: dict[str, Any] = {}
        self._imputer: MissingImputer | None = None

    @beartype
    def fit(self, **kwargs: Any) -> MissingImputation:
        """Resolve imputer, validate paths, and fit imputer configuration.

        Args:
            **kwargs (Any): Configuration keys:
                - `input_path` (str | Path): Input CSV path.
                - `output_path` (str | Path): Output CSV path.
                - `technique` (str | None): Imputer key.
                - `imputer` (str | None): Alias for `technique`.
                - any additional keys are forwarded unchanged to the selected
                  imputer `fit(...)`.

        Returns:
            MissingImputation: The fitted dispatcher instance.
        """

        input_path = _as_path(kwargs.get("input_path"), key="input_path")
        output_path = _as_path(kwargs.get("output_path"), key="output_path")
        _validate_csv_path(input_path)
        _validate_output_csv_path(output_path, input_path=input_path)

        technique = _normalise_key(
            _as_optional_string(kwargs.get("technique"))
            or _as_optional_string(kwargs.get("imputer"))
            or "mice_imputation"
        )

        imputers = discover_missing_imputers()
        imputer_cls = _resolve_imputer_class(imputers, technique)

        imputer_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"input_path", "output_path", "technique", "imputer"}
        }

        imputer = imputer_cls()
        imputer.fit(
            input_path=input_path,
            output_path=output_path,
            **imputer_kwargs,
        )

        self._input_path = input_path
        self._output_path = output_path
        self._imputer_key = technique
        self._imputer_kwargs = dict(imputer_kwargs)
        self._imputer = imputer
        return self

    @beartype
    def preprocess(self, **kwargs: Any) -> MICEImputationResult:
        """Run the configured imputer and return a typed imputation result.

        Args:
            **kwargs (Any): Optional configuration keys identical to `fit(...)`.
                If provided, they override any existing fitted configuration.

        Returns:
            MICEImputationResult: Typed imputation summary returned by the
            current imputer implementation.
        """

        if kwargs:
            self.fit(**kwargs)
        if (
            self._input_path is None
            or self._output_path is None
            or self._imputer is None
            or self._imputer_key is None
        ):
            raise InputValidationError(
                "No configuration provided. Call `fit(...)` first or pass kwargs to `preprocess(...)`."
            )

        result = self._imputer.impute(
            input_path=self._input_path,
            output_path=self._output_path,
            **self._imputer_kwargs,
        )

        if not isinstance(result, MICEImputationResult):
            raise InputValidationError(
                "Unsupported imputation result payload returned by imputer."
            )
        return result


@beartype
def run_missing_imputation(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one missing-imputation technique for the Go CLI bridge.

    !!! warning

        This `run_*` function is primarily for the Go CLI bridge and should not be
        treated as the Python library API. In Python scripts or notebooks, import
        and call `MissingImputation` directly from `ldt.data_preprocessing`.

    Args:
        technique (str): Imputation technique key from the catalogue.
        params (Mapping[str, Any]): Wrapper parameters forwarded to
            `MissingImputation.fit_preprocess(...)`, including paths, imputer
            selection keys, and imputer-specific configuration values.

    Returns:
        dict[str, Any]: Serialised imputation summary for the Go CLI bridge.
    """

    _, resolved = resolve_technique_with_defaults(
        section_key="missing_imputation",
        technique_id=technique,
        provided_params=dict(params),
    )

    payload = dict(resolved)
    payload["technique"] = technique

    result = MissingImputation().fit_preprocess(**payload)
    return _serialise_result(result=result, technique=technique)


@beartype
def _serialise_result(*, result: Any, technique: str) -> dict[str, Any]:
    if is_dataclass(result):
        payload = asdict(result)
    elif isinstance(result, dict):
        payload = dict(result)
    else:
        raise InputValidationError("Unsupported imputation result payload.")

    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)

    payload["imputer"] = _normalise_key(technique)
    return payload


@beartype
def _resolve_imputer_class(
    imputers: Mapping[str, type[MissingImputer]],
    imputer_key: str,
) -> type[MissingImputer]:
    target = _normalise_key(imputer_key)
    for key, imputer_cls in imputers.items():
        if _normalise_key(key) == target:
            return imputer_cls

    choices = ", ".join(
        sorted(
            resolve_component_metadata(imputer).name for imputer in imputers.values()
        )
    )
    raise InputValidationError(
        f"Unknown missing-data imputer `{imputer_key}`. Available: {choices}"
    )


@beartype
def _normalise_key(raw: str) -> str:
    return raw.strip().lower().replace("-", "_")


@beartype
def _as_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed if trimmed else None
    raise InputValidationError("Expected a string value.")


@beartype
def _as_path(value: Any, *, key: str) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value.strip():
        return Path(value).expanduser()
    raise InputValidationError(f"Missing required path parameter: {key}")


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
