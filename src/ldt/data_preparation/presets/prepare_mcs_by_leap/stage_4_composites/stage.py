from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from beartype import beartype

from ldt.utils.errors import InputValidationError

_STAGE_DIR = Path(__file__).resolve().parent
_COMPOSITES_CONFIG = _STAGE_DIR / "composites.yaml"

CompositeOperation = Literal["sum", "mean", "median", "min", "max", "coalesce"]


@beartype
@dataclass(frozen=True)
class CompositeFeatureSpec:
    """Configuration for one composite feature operation.

    Attributes:
        output (str): Output.
        operation (CompositeOperation): Operation.
        inputs (tuple[str, ...]): Inputs.
    """

    output: str
    operation: CompositeOperation
    inputs: tuple[str, ...]


@beartype
@dataclass(frozen=True)
class CompositeFeatureSummary:
    """Summary for stage-4 composite feature preparation.

    Attributes:
        requested (int): Requested.
        created (int): Created.
    """

    requested: int
    created: int


@beartype
def resolve_composite_features(wave: str) -> tuple[CompositeFeatureSpec, ...]:
    """Resolve configured composite feature operations for one wave.

    This helper is used by higher-level workflows and keeps input/output handling consistent.

    Args:
        wave (str): Wave identifier.

    Returns:
        tuple[CompositeFeatureSpec, ...]: Tuple of resolved values.
    """

    normalised = wave.strip().upper()
    raw = _load_yaml_config(_COMPOSITES_CONFIG)
    waves = raw.get("waves")
    if not isinstance(waves, dict):
        raise InputValidationError(
            "Stage 4 composites config must define `waves` mapping."
        )

    specs_raw = waves.get(normalised, [])
    if specs_raw in (None, "none"):
        return ()
    if not isinstance(specs_raw, list):
        raise InputValidationError(
            f"Invalid composites config for `{normalised}`. Expected list."
        )

    parsed: list[CompositeFeatureSpec] = []
    for index, item in enumerate(specs_raw):
        context = f"waves.{normalised}[{index}]"
        if not isinstance(item, dict):
            raise InputValidationError(
                f"Invalid `{context}` in composites config. Expected mapping."
            )

        output = item.get("output")
        if not isinstance(output, str) or not output.strip():
            raise InputValidationError(
                f"Invalid `{context}.output` in composites config."
            )

        operation = item.get("operation", "sum")
        supported_operations = {"sum", "mean", "median", "min", "max", "coalesce"}
        if operation not in supported_operations:
            raise InputValidationError(
                f"Unsupported `{context}.operation` `{operation}`. "
                "Supported operations: sum, mean, median, min, max, coalesce."
            )

        inputs = item.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise InputValidationError(
                f"Invalid `{context}.inputs` in composites config. Expected non-empty list."
            )

        parsed_inputs: list[str] = []
        for input_name in inputs:
            if not isinstance(input_name, str) or not input_name.strip():
                raise InputValidationError(
                    f"Invalid entry in `{context}.inputs` in composites config."
                )
            parsed_inputs.append(input_name.strip())

        parsed.append(
            CompositeFeatureSpec(
                output=output.strip(),
                operation=operation,
                inputs=tuple(parsed_inputs),
            )
        )
    return tuple(parsed)


@beartype
def apply_composites(
    *,
    wave: str,
    data: pd.DataFrame,
    source_data: pd.DataFrame,
) -> tuple[pd.DataFrame, CompositeFeatureSummary]:
    """Apply configured composites to stage-3 data using selected or source columns.

    Args:
        wave (str): Wave identifier.
        data (pd.DataFrame): Input dataset.
        source_data (pd.DataFrame): Source dataframe for this stage.

    Returns:
        tuple[pd.DataFrame, CompositeFeatureSummary]: Transformed dataset as a pandas DataFrame.
    """

    specs = resolve_composite_features(wave)
    if not specs:
        return data, CompositeFeatureSummary(requested=0, created=0)

    prepared = data.copy()
    created = 0
    for spec in specs:
        input_series = [
            _resolve_composite_input_series(
                data=prepared,
                source_data=source_data,
                input_name=input_name,
                output_name=spec.output,
            )
            for input_name in spec.inputs
        ]
        prepared[spec.output] = _run_composite_operation(
            input_series=tuple(input_series),
            operation=spec.operation,
            output_name=spec.output,
        )
        created += 1

    return prepared, CompositeFeatureSummary(requested=len(specs), created=created)


@beartype
def print_composite_summary(*, summary: CompositeFeatureSummary) -> None:
    """Print stage-4 summary.

    Args:
        summary (CompositeFeatureSummary): Summary object from the previous stage.
    """

    print(
        f"Stage 4 - Composite features: created={summary.created}/{summary.requested}"
    )


@beartype
def _resolve_composite_input_series(
    *,
    data: pd.DataFrame,
    source_data: pd.DataFrame,
    input_name: str,
    output_name: str,
) -> pd.Series:
    """Resolve one composite input from selected data then raw source data."""

    in_data = _resolve_column_name(data=data, source_name=input_name)
    if in_data is not None:
        return data[in_data]

    in_source = _resolve_column_name(data=source_data, source_name=input_name)
    if in_source is not None:
        return source_data[in_source]

    raise InputValidationError(
        f"Composite output `{output_name}` cannot resolve input `{input_name}`."
    )


@beartype
def _resolve_column_name(*, data: pd.DataFrame, source_name: str) -> str | None:
    """Resolve one column by exact match or unique suffix match."""

    if source_name in data.columns:
        return source_name

    suffix_matches = sorted(
        column for column in data.columns if column.endswith(f"__{source_name}")
    )
    if not suffix_matches:
        return None
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    raise InputValidationError(
        f"Composite input `{source_name}` is ambiguous. "
        f"Matches: {', '.join(suffix_matches)}."
    )


@beartype
def _run_composite_operation(
    *,
    input_series: tuple[pd.Series, ...],
    operation: CompositeOperation,
    output_name: str,
) -> pd.Series:
    """Execute one composite operation over aligned pandas series."""

    if operation == "coalesce":
        output = input_series[0]
        for series in input_series[1:]:
            output = output.combine_first(series)
        return output

    frame = pd.concat(input_series, axis=1)
    numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
    if operation == "sum":
        return numeric_frame.sum(axis=1, min_count=1)
    if operation == "mean":
        return numeric_frame.mean(axis=1)
    if operation == "median":
        return numeric_frame.median(axis=1)
    if operation == "min":
        return numeric_frame.min(axis=1)
    if operation == "max":
        return numeric_frame.max(axis=1)

    raise InputValidationError(
        f"Unsupported composite operation `{operation}` for output `{output_name}`."
    )


@beartype
@cache
def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load one YAML config file and validate mapping root."""

    if not config_path.exists() or not config_path.is_file():
        raise InputValidationError(f"Missing config file: {config_path.resolve()}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise InputValidationError(
            f"Config file `{config_path.name}` must contain a YAML mapping."
        )
    return loaded
