from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, get_args, get_origin

import pandas as pd

from src.data_preprocessing.catalog import resolve_technique_with_defaults
from src.data_preprocessing.support.inputs import (
    as_bool,
    as_optional_int,
    as_optional_object,
    as_optional_string,
    as_optional_string_list_or_csv,
    as_required_string,
    ensure_columns,
    normalise_key,
    run_with_validation,
)
from src.utils.errors import InputValidationError

from .discovery import discover_trajectory_builders
from .trajectory import TrajectoryModel


def run_build_trajectories(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Build trajectory assignments from longitudinal data.

    This tool fits a trajectory builder (or reloads a saved one), then writes a
    subject-level assignment dataset with trajectory IDs and labels.

    Supported execution techniques:

    | Technique | What it does |
    | --- | --- |
    | `from_scratch` | Fits a selected builder on the input data. |
    | `from_saved_model` | Loads an already-fitted model and applies `transform`. |
    | `<builder-key>` | Convenience mode equivalent to `from_scratch` with `builder=<builder-key>`. |

    Available trajectory builders:

    | Builder key | Method summary |
    | --- | --- |
    | `dtw_kmeans` | Dynamic-Time-Warping k-means on shape similarity. |
    | `clusterMLD` | Hierarchical clustering on spline-based trajectory summaries. |
    | `GMM` | Growth Mixture Model (R `lcmm`) with class-specific random effects. |
    | `LCGA` | Latent Class Growth Analysis (R `lcmm`) without class-specific random-effects variance. |

    Args:
        technique (str): Execution technique key.
        params (Mapping[str, Any]): Builder selection, data-column mapping,
            model IO paths, and builder hyperparameters.

    Returns:
        dict[str, Any]: Output assignment path, builder metadata, and optional
            saved-model path.

    Examples:
        ```python
        from ldt.data_preprocessing.tools.build_trajectories.run import run_build_trajectories

        result = run_build_trajectories(
            technique="from_scratch",
            params={
                "input_path": "./data/longitudinal.csv",
                "output_path": "./outputs/trajectory_assignments.csv",
                "id_col": "subject_id",
                "time_col": "wave",
                "value_cols": "depression_score",
                "builder": "LCGA",
                "n_trajectories": 4,
                "builder_kwargs_json": {"max_iter": 200, "n_init": 20},
                "save_model_path": "./outputs/lcga_model.pkl",
            },
        )
        ```
    """

    return run_with_validation(
        lambda: _run_build_trajectories(technique=technique, params=params)
    )


def _run_build_trajectories(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    _, resolved = resolve_technique_with_defaults(
        section_key="build_trajectories",
        technique_id=technique,
        provided_params=dict(params),
    )

    mode = normalise_key(technique)
    run_mode = mode
    input_path = Path(as_required_string(resolved, "input_path")).expanduser()
    output_path = Path(as_required_string(resolved, "output_path")).expanduser()
    id_col = as_required_string(resolved, "id_col")
    time_col = as_required_string(resolved, "time_col")
    value_cols = as_optional_string_list_or_csv(resolved, "value_cols")
    class_col = as_optional_string(resolved, "class_col")

    data = pd.read_csv(input_path)
    ensure_columns(data, [id_col, time_col])
    if class_col:
        ensure_columns(data, [class_col])

    builder_label: str
    saved_model_path: str | None = None

    if mode == "from_saved_model":
        run_mode = "from_saved_model"
        model_path = Path(as_required_string(resolved, "input_model_path")).expanduser()
        model = TrajectoryModel.load(model_path)
        builder_label = type(model).__name__
    else:
        run_mode = "from_scratch"
        builders = discover_trajectory_builders()
        builder_key = as_optional_string(resolved, "builder")
        if not builder_key:
            if mode == "from_scratch":
                raise InputValidationError(
                    "Missing required parameter `builder` for technique `from_scratch`."
                )
            builder_key = technique
        builder_cls = _resolve_builder_class(builders, builder_key)
        raw_builder_kwargs = as_optional_object(resolved, "builder_kwargs_json")
        builder_kwargs = dict(raw_builder_kwargs)

        n_trajectories = as_optional_int(resolved, "n_trajectories")
        if n_trajectories is not None:
            builder_kwargs.setdefault("n_trajectories", n_trajectories)

        init_kwargs = _coerce_kwargs_for_signature(builder_cls.__init__, builder_kwargs)
        model = builder_cls(**init_kwargs)
        builder_label = builder_key
        model_save_path = as_optional_string(resolved, "save_model_path")
        if model_save_path:
            saved_model_path = str(Path(model_save_path).expanduser())

    resolved_value_cols = _resolve_value_columns(
        data=data,
        provided=value_cols,
        id_col=id_col,
        time_col=time_col,
        class_col=class_col,
        max_value_cols=getattr(model, "max_value_cols", None),
    )

    model.fit(data, id_col=id_col, time_col=time_col, value_cols=resolved_value_cols)
    assignments = model.transform(
        data,
        id_col=id_col,
        time_col=time_col,
        value_cols=resolved_value_cols,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output_path, index=False)

    if saved_model_path:
        model.save(saved_model_path)

    last_result = getattr(model, "_last_result", None)
    selected_n_trajectories = None
    if last_result is not None and hasattr(last_result, "n_trajectories"):
        selected_n_trajectories = int(last_result.n_trajectories)

    return {
        "mode": run_mode,
        "builder": builder_label,
        "output_path": str(output_path.resolve()),
        "row_count": int(len(assignments)),
        "column_count": int(assignments.shape[1]),
        "value_cols": list(resolved_value_cols),
        "saved_model_path": (
            str(Path(saved_model_path).resolve()) if saved_model_path else None
        ),
        "n_trajectories": selected_n_trajectories,
    }


def _resolve_builder_class(
    builders: dict[str, type[TrajectoryModel]],
    builder_key: str,
) -> type[TrajectoryModel]:
    target = normalise_key(builder_key)
    for key, builder_cls in builders.items():
        if normalise_key(key) == target:
            return builder_cls
    choices = ", ".join(sorted(builders.keys()))
    raise InputValidationError(f"Unknown builder `{builder_key}`. Available: {choices}")


def _coerce_kwargs_for_signature(
    signature_target: Any,
    raw_kwargs: dict[str, Any],
) -> dict[str, Any]:
    signature = inspect.signature(signature_target)
    coerced: dict[str, Any] = {}

    valid_parameters = {
        name: parameter
        for name, parameter in signature.parameters.items()
        if name != "self"
    }

    for key, value in raw_kwargs.items():
        parameter = valid_parameters.get(key)
        if parameter is None:
            raise InputValidationError(
                f"Unknown builder parameter `{key}` in builder_kwargs_json."
            )
        coerced[key] = _coerce_value(value, parameter.annotation)

    return coerced


def _coerce_value(value: Any, annotation: Any) -> Any:
    if annotation is inspect._empty:
        return value

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is None:
        if annotation is Any:
            return value
        if annotation is bool:
            return as_bool(value, field_name="builder kwargs")
        if annotation is int:
            if isinstance(value, bool):
                raise InputValidationError("Boolean is not a valid integer value.")
            if isinstance(value, int):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
            if isinstance(value, str) and value.strip():
                try:
                    return int(value.strip())
                except ValueError as exc:
                    raise InputValidationError("Expected an integer value.") from exc
            raise InputValidationError("Expected an integer value.")
        if annotation is float:
            if isinstance(value, bool):
                raise InputValidationError("Boolean is not a valid float value.")
            if isinstance(value, int | float):
                return float(value)
            if isinstance(value, str) and value.strip():
                try:
                    return float(value.strip())
                except ValueError as exc:
                    raise InputValidationError("Expected a float value.") from exc
            raise InputValidationError("Expected a float value.")
        if annotation is str:
            return str(value)
        return value

    if origin in {list, tuple, Sequence}:
        inner = args[0] if args else Any
        if isinstance(value, str):
            values = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, list):
            values = value
        elif isinstance(value, tuple):
            values = list(value)
        else:
            raise InputValidationError(
                "Builder sequence kwargs must be provided as an array or CSV string."
            )

        coerced_items = [_coerce_value(item, inner) for item in values]
        if origin is tuple:
            return tuple(coerced_items)
        return coerced_items

    if origin is dict:
        if not isinstance(value, dict):
            raise InputValidationError("Builder dict kwargs must be JSON objects.")
        return value

    if origin is set:
        if isinstance(value, list | tuple | set):
            inner = args[0] if args else Any
            return {_coerce_value(item, inner) for item in value}
        raise InputValidationError("Builder set kwargs must be JSON arrays.")

    if origin in {type(None), None}:
        return value

    if origin is getattr(__import__("types"), "UnionType", object):
        pass

    if origin is None and not args:
        return value

    if args:
        non_none = [item for item in args if item is not type(None)]
        if value is None:
            return None
        for candidate in non_none:
            try:
                return _coerce_value(value, candidate)
            except InputValidationError:
                continue
        return value

    return value


def _resolve_value_columns(
    *,
    data: pd.DataFrame,
    provided: list[str],
    id_col: str,
    time_col: str,
    class_col: str | None,
    max_value_cols: int | None,
) -> list[str]:
    if provided:
        value_cols = list(provided)
    else:
        excluded = {id_col, time_col}
        if class_col:
            excluded.add(class_col)
        value_cols = [
            col
            for col in data.columns
            if col not in excluded and pd.api.types.is_numeric_dtype(data[col])
        ]

    if not value_cols:
        raise InputValidationError(
            "No value columns available for trajectory building."
        )

    ensure_columns(data, value_cols)

    if max_value_cols == 1 and len(value_cols) > 1:
        return [value_cols[0]]
    if max_value_cols is not None and len(value_cols) > max_value_cols:
        raise InputValidationError(
            f"Selected value columns exceed builder limit ({max_value_cols})."
        )
    return value_cols
