from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_preparation.catalog import (
    list_synthetic_techniques,
    resolve_technique_with_defaults,
)
from src.data_preparation.tools.synthetic_data_generation.generators.harmonisation_challenge import (
    HarmonisationChallenge,
)
from src.data_preparation.tools.synthetic_data_generation.generators.missing_data_scenarios import (
    MissingDataScenarios,
)
from src.data_preparation.tools.synthetic_data_generation.generators.trend_patterns import (
    SyntheticWaveDataset,
    TrajectoryFeatureSpec,
    TrajectoryPatternSpec,
)
from src.utils.errors import InputValidationError


def run_synthetic_generation(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Run a synthetic-data-generation technique and write CSV output.

    Supported techniques model different longitudinal patterns:
    `trend_patterns`, `event_shock_recovery`, `piecewise_changepoint`,
    `missing_data_scenarios`, and `harmonisation_challenge`.
    The function resolves defaults from the catalog, executes the selected
    generator, writes the resulting dataset to CSV, and returns output metadata.

    Args:
        technique (str): Synthetic technique key from the catalog.
        params (Mapping[str, Any]): Technique parameters and output settings.

    Returns:
        dict[str, Any]: Output summary including path, shape, and column names.

    Examples:
        ```python
        from ldt.data_preparation.tools.synthetic_data_generation.run import run_synthetic_generation
        result = run_synthetic_generation(
            technique="event_shock_recovery",
            params={
                "output_path": "./synthetic_waves.csv",
                "n_samples": 500,
                "n_waves": 6,
                "random_state": 42,
                "feature_cols": "depressive_score,anxiety_score",
                "shock_wave": 3,
                "shock_mean": 4.0,
                "recovery_rate": 0.8,
                "noise_sd": 1.0,
            },
        )
        ```
    """

    _, resolved = resolve_technique_with_defaults(
        section_key="synthetic_data_generation",
        technique_id=technique,
        provided_params=dict(params),
    )

    canonical = _normalise_key(technique)
    if canonical == "trend_patterns":
        data, output_path = _run_trend_patterns(resolved)
    elif canonical == "event_shock_recovery":
        data, output_path = _run_event_shock_recovery(resolved)
    elif canonical == "piecewise_changepoint":
        data, output_path = _run_piecewise_changepoint(resolved)
    elif canonical == "missing_data_scenarios":
        data, output_path = _run_missing_data_scenarios(resolved)
    elif canonical == "harmonisation_challenge":
        data, output_path = _run_harmonisation_challenge(resolved)
    else:
        raise InputValidationError(f"Unsupported synthetic technique: {technique}")

    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination, index=False)

    return {
        "technique": technique,
        "output_path": str(destination.resolve()),
        "row_count": int(len(data)),
        "column_count": int(data.shape[1]),
        "columns": [str(column) for column in data.columns],
    }


def _run_trend_patterns(params: Mapping[str, Any]) -> tuple[pd.DataFrame, str]:
    output_path = _as_required_string(params, "output_path")
    n_samples = _as_required_int(params, "n_samples", minimum=1)
    n_waves = _as_required_int(params, "n_waves", minimum=2)
    random_state = _as_optional_int(params, "random_state")
    feature_cols = _as_required_string_list(params, "feature_cols")
    spec_mode = _as_required_string(params, "spec_mode").strip().lower()

    dataset = SyntheticWaveDataset(
        n_samples=n_samples,
        n_waves=n_waves,
        random_state=random_state,
    )

    if spec_mode in {"default", "d"}:
        class_specs = dataset.default_class_specs()
    elif spec_mode in {"random", "r"}:
        class_specs = dataset.random_class_specs()
    elif spec_mode in {"custom", "c"}:
        class_specs = _parse_custom_class_specs(params.get("custom_class_specs_json"))
    else:
        raise InputValidationError("spec_mode must be one of: default, random, custom.")

    feature_specs = [
        TrajectoryFeatureSpec(name=feature, class_specs=class_specs)
        for feature in feature_cols
    ]

    final_dataset = SyntheticWaveDataset(
        n_samples=n_samples,
        n_waves=n_waves,
        random_state=random_state,
        feature_specs=feature_specs,
    )
    return final_dataset.generate(), output_path


def _run_event_shock_recovery(params: Mapping[str, Any]) -> tuple[pd.DataFrame, str]:
    output_path = _as_required_string(params, "output_path")
    n_samples = _as_required_int(params, "n_samples", minimum=1)
    n_waves = _as_required_int(params, "n_waves", minimum=2)
    random_state = _as_optional_int(params, "random_state")
    feature_cols = _as_required_string_list(params, "feature_cols")
    shock_wave = _as_required_int(params, "shock_wave")
    shock_mean = _as_required_float(params, "shock_mean")
    recovery_rate = _as_required_float(params, "recovery_rate")
    noise_sd = _as_required_float(params, "noise_sd", minimum=0.0)

    if shock_wave <= 1 or shock_wave >= n_waves:
        raise InputValidationError("Shock wave must be between wave 2 and n_waves-1.")

    rng = np.random.default_rng(random_state)
    times = np.arange(1, n_waves + 1, dtype=float)

    records: list[dict[str, float | int]] = []
    for subject_id in range(1, n_samples + 1):
        baseline_intercept = rng.normal(6.0, 1.2)
        baseline_slope = rng.normal(0.0, 0.3)
        subject_shock = rng.normal(shock_mean, 0.8)

        for wave_index, time in enumerate(times, start=1):
            baseline = baseline_intercept + baseline_slope * time
            if wave_index < shock_wave:
                shock_component = 0.0
            else:
                elapsed = wave_index - shock_wave
                shock_component = subject_shock * np.exp(-recovery_rate * elapsed)
            value = float(rng.normal(baseline + shock_component, noise_sd))

            record: dict[str, float | int] = {
                "subject_id": subject_id,
                "wave": wave_index,
            }
            for feature in feature_cols:
                feature_shift = rng.normal(0.0, 0.3)
                record[feature] = value + feature_shift
            records.append(record)

    return pd.DataFrame.from_records(records), output_path


def _run_piecewise_changepoint(params: Mapping[str, Any]) -> tuple[pd.DataFrame, str]:
    output_path = _as_required_string(params, "output_path")
    n_samples = _as_required_int(params, "n_samples", minimum=1)
    n_waves = _as_required_int(params, "n_waves", minimum=2)
    random_state = _as_optional_int(params, "random_state")
    feature_cols = _as_required_string_list(params, "feature_cols")
    changepoint_wave = _as_required_int(params, "changepoint_wave")
    noise_sd = _as_required_float(params, "noise_sd", minimum=0.0)
    pre_slope_sd = _as_required_float(params, "pre_slope_sd", minimum=0.0)
    post_slope_sd = _as_required_float(params, "post_slope_sd", minimum=0.0)

    if changepoint_wave <= 1 or changepoint_wave >= n_waves:
        raise InputValidationError(
            "Changepoint wave must be between wave 2 and n_waves-1."
        )

    rng = np.random.default_rng(random_state)
    times = np.arange(1, n_waves + 1, dtype=float)

    records: list[dict[str, float | int]] = []
    for subject_id in range(1, n_samples + 1):
        intercept = rng.normal(7.0, 1.5)
        pre_slope = rng.normal(0.0, pre_slope_sd)
        post_slope_delta = rng.normal(0.0, post_slope_sd)
        post_slope = pre_slope + post_slope_delta

        values_by_time: dict[int, float] = {}
        cp_time = float(changepoint_wave)
        level_at_cp = intercept + pre_slope * cp_time
        for wave_index, time in enumerate(times, start=1):
            if wave_index <= changepoint_wave:
                mean_value = intercept + pre_slope * time
            else:
                mean_value = level_at_cp + post_slope * (time - cp_time)
            values_by_time[wave_index] = float(rng.normal(mean_value, noise_sd))

        for wave_index, _time in enumerate(times, start=1):
            record: dict[str, float | int] = {
                "subject_id": subject_id,
                "wave": wave_index,
            }
            for feature in feature_cols:
                feature_shift = rng.normal(0.0, 0.25)
                record[feature] = values_by_time[wave_index] + feature_shift
            records.append(record)

    return pd.DataFrame.from_records(records), output_path


def _run_missing_data_scenarios(params: Mapping[str, Any]) -> tuple[pd.DataFrame, str]:
    output_path = _as_required_string(params, "output_path")
    n_samples = _as_required_int(params, "n_samples", minimum=1)
    n_waves = _as_required_int(params, "n_waves", minimum=2)
    random_state = _as_optional_int(params, "random_state")
    feature_cols = _as_required_string_list(params, "feature_cols")
    mechanism = _as_required_string(params, "mechanism").strip().lower()
    missing_rate = _as_required_float(params, "missing_rate", minimum=0.0, maximum=0.95)
    dropout_rate = _as_required_float(params, "dropout_rate", minimum=0.0, maximum=0.95)
    mar_strength = _as_required_float(params, "mar_strength", minimum=0.0)

    if mechanism not in {"mcar", "mar", "dropout", "mixed"}:
        raise InputValidationError(
            "mechanism must be one of: mcar, mar, dropout, mixed."
        )

    runner = MissingDataScenarios()
    rng = np.random.default_rng(random_state)

    data = runner._generate_complete_panel(
        n_samples=n_samples,
        n_waves=n_waves,
        feature_cols=feature_cols,
        rng=rng,
    )
    data = runner._apply_missingness(
        data=data,
        feature_cols=feature_cols,
        mechanism=mechanism,
        missing_rate=missing_rate,
        dropout_rate=dropout_rate,
        mar_strength=mar_strength,
        n_waves=n_waves,
        rng=rng,
    )
    data = data.drop(
        columns=["time", "class", "dropout_wave"],
        errors="ignore",
    )
    return data, output_path


def _run_harmonisation_challenge(params: Mapping[str, Any]) -> tuple[pd.DataFrame, str]:
    output_path = _as_required_string(params, "output_path")
    n_samples = _as_required_int(params, "n_samples", minimum=1)
    n_waves = _as_required_int(params, "n_waves", minimum=2)
    random_state = _as_optional_int(params, "random_state")
    feature_cols = _as_required_string_list(params, "feature_cols")
    noise_rate = _as_required_float(params, "noise_rate", minimum=0.0, maximum=1.0)
    missing_label_rate = _as_required_float(
        params,
        "missing_label_rate",
        minimum=0.0,
        maximum=0.5,
    )
    include_canonical_columns = _as_required_bool(params, "include_canonical_columns")

    runner = HarmonisationChallenge()
    rng = np.random.default_rng(random_state)

    data = runner._generate_base_panel(
        n_samples=n_samples,
        n_waves=n_waves,
        feature_cols=feature_cols,
        rng=rng,
    )
    data = runner._inject_harmonisation_noise(
        data=data,
        noise_rate=noise_rate,
        missing_label_rate=missing_label_rate,
        include_canonical_columns=include_canonical_columns,
        rng=rng,
    )
    data = data.drop(columns=["time", "class"], errors="ignore")
    return data, output_path


def _parse_custom_class_specs(raw: Any) -> list[TrajectoryPatternSpec]:
    if isinstance(raw, str):
        if raw.strip() == "":
            raise InputValidationError(
                "custom_class_specs_json is required when spec_mode=custom."
            )
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise InputValidationError(
                f"Invalid custom_class_specs_json: {exc}"
            ) from exc
    elif isinstance(raw, list):
        parsed = raw
    else:
        raise InputValidationError(
            "custom_class_specs_json must be a JSON string or list of objects."
        )

    if not isinstance(parsed, list) or not parsed:
        raise InputValidationError(
            "custom_class_specs_json must contain at least one class specification."
        )

    specs: list[TrajectoryPatternSpec] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            raise InputValidationError(
                "Each custom class spec must be an object with numeric fields."
            )
        name = _as_required_string(entry, "name")
        intercept_mean = _as_required_float(entry, "intercept_mean")
        slope_mean = _as_required_float(entry, "slope_mean")
        intercept_sd = _as_optional_float(entry, "intercept_sd", default=1.0)
        slope_sd = _as_optional_float(entry, "slope_sd", default=0.2)
        specs.append(
            TrajectoryPatternSpec(
                name=name,
                intercept_mean=intercept_mean,
                slope_mean=slope_mean,
                intercept_sd=intercept_sd,
                slope_sd=slope_sd,
            )
        )
    return specs


def _as_required_string(params: Mapping[str, Any], key: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"Missing required string parameter: {key}")
    return value.strip()


def _as_required_int(
    params: Mapping[str, Any],
    key: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    raw = params.get(key)
    if isinstance(raw, bool):
        raise InputValidationError(f"Parameter `{key}` must be an integer.")

    if isinstance(raw, int):
        value = raw
    elif isinstance(raw, float):
        if raw.is_integer():
            value = int(raw)
        else:
            raise InputValidationError(f"Parameter `{key}` must be an integer.")
    elif isinstance(raw, str) and raw.strip():
        try:
            value = int(raw.strip())
        except ValueError as exc:
            raise InputValidationError(
                f"Parameter `{key}` must be an integer."
            ) from exc
    else:
        raise InputValidationError(f"Missing required integer parameter: {key}")

    if minimum is not None and value < minimum:
        raise InputValidationError(f"Parameter `{key}` must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise InputValidationError(f"Parameter `{key}` must be <= {maximum}.")
    return value


def _as_optional_int(params: Mapping[str, Any], key: str) -> int | None:
    raw = params.get(key)
    if raw is None:
        return None
    if isinstance(raw, str) and raw.strip() == "":
        return None
    return _as_required_int(params, key)


def _as_required_float(
    params: Mapping[str, Any],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    raw = params.get(key)
    if isinstance(raw, bool):
        raise InputValidationError(f"Parameter `{key}` must be numeric.")

    if isinstance(raw, int | float):
        value = float(raw)
    elif isinstance(raw, str) and raw.strip():
        try:
            value = float(raw.strip())
        except ValueError as exc:
            raise InputValidationError(f"Parameter `{key}` must be numeric.") from exc
    else:
        raise InputValidationError(f"Missing required numeric parameter: {key}")

    if minimum is not None and value < minimum:
        raise InputValidationError(f"Parameter `{key}` must be >= {minimum}.")
    if maximum is not None and value > maximum:
        raise InputValidationError(f"Parameter `{key}` must be <= {maximum}.")
    return value


def _as_optional_float(
    params: Mapping[str, Any],
    key: str,
    *,
    default: float,
) -> float:
    raw = params.get(key)
    if raw is None:
        return default
    if isinstance(raw, str) and raw.strip() == "":
        return default
    return _as_required_float(params, key)


def _as_required_bool(params: Mapping[str, Any], key: str) -> bool:
    raw = params.get(key)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "y"}:
            return True
        if value in {"0", "false", "no", "n"}:
            return False
    raise InputValidationError(f"Parameter `{key}` must be a boolean.")


def _as_required_string_list(params: Mapping[str, Any], key: str) -> list[str]:
    raw = params.get(key)
    values: list[str] = []

    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, str):
                raise InputValidationError(
                    f"Parameter `{key}` entries must be strings."
                )
            candidate = item.strip()
            if candidate:
                values.append(candidate)
    elif isinstance(raw, str):
        values = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        raise InputValidationError(
            f"Parameter `{key}` must be a comma-separated string or list of strings."
        )

    if not values:
        raise InputValidationError(
            f"Parameter `{key}` must include at least one value."
        )
    if len(values) != len(set(values)):
        raise InputValidationError(f"Parameter `{key}` entries must be unique.")
    return values


def _normalise_key(raw: str) -> str:
    return raw.strip().lower().replace("-", "_")


def list_synthetic_generation_catalog() -> list[dict[str, Any]]:
    """Return synthetic generation catalog entries.

    Returns:
        list[dict[str, Any]]: Catalog rows describing available techniques.
    """

    return list_synthetic_techniques()
