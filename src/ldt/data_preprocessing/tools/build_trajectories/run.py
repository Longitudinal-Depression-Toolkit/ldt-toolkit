from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, get_args, get_origin

import pandas as pd
from beartype import beartype

from ldt.data_preprocessing.catalog import resolve_technique_with_defaults
from ldt.utils.errors import InputValidationError
from ldt.utils.metadata import ComponentMetadata
from ldt.utils.templates.tools.data_preprocessing import DataPreprocessingTool

from .discovery import discover_trajectory_builders
from .trajectory import TrajectoryModel


@beartype
class BuildTrajectories(DataPreprocessingTool):
    """Build trajectory assignments from longitudinal data.

    This tool fits one trajectory builder (or reloads a saved one), then writes
    subject-level assignments with trajectory IDs and names.

    Runtime parameters:
        - `mode`: `from_scratch` or `from_saved_model`.
        - `input_path`: Input long-format CSV.
        - `output_path`: Output CSV for trajectory assignments.
        - `id_col`: Subject identifier column.
        - `time_col`: Time or wave column.
        - `value_cols`: Optional longitudinal outcome columns. If omitted, they
          are inferred from non-ID/time columns.
        - `class_col`: Optional existing class column to exclude from value
          column inference.
        - `builder`: Builder key when running from scratch.
        - `builder_kwargs_json`: Builder constructor kwargs payload.
        - `n_trajectories`: Optional class-count override passed to builder
          kwargs.
        - `input_model_path`: Required in `from_saved_model` mode.
        - `save_model_path`: Optional model save path in `from_scratch` mode.

    Examples:
        ```python
        from ldt.data_preprocessing import BuildTrajectories

        tool = BuildTrajectories()
        result = tool.fit_preprocess(
            mode="from_scratch",
            input_path="./data/longitudinal.csv",
            output_path="./outputs/trajectory_assignments.csv",
            id_col="subject_id",
            time_col="wave",
            value_cols=["depression_score"],
            builder="lcga",
            n_trajectories=4,
            builder_kwargs_json={"max_iter": 200, "n_init": 20},
            save_model_path="./outputs/lcga_model.pkl",
        )
        ```
    """

    metadata = ComponentMetadata(
        name="build_trajectories",
        full_name="Build Trajectories",
        abstract_description="Build subject trajectory assignments from longitudinal data.",
    )

    def __init__(self) -> None:
        self._config: dict[str, Any] | None = None

    @beartype
    def fit(self, **kwargs: Any) -> BuildTrajectories:
        """Validate and store trajectory-building configuration.

        Args:
            **kwargs (Any): Configuration keys:
                - `mode` (str): `from_scratch` or `from_saved_model`.
                - `input_path` (str | Path): Input long-format CSV path.
                - `output_path` (str | Path): Assignment output CSV path.
                - `id_col` (str): Subject identifier column.
                - `time_col` (str): Time or wave column.
                - `value_cols` (str | list[str] | None): Outcome columns.
                - `class_col` (str | None): Optional existing class label column.
                - `builder` (str | None): Trajectory builder key.
                - `builder_kwargs_json` (dict[str, Any] | None): Builder
                  constructor kwargs.
                - `n_trajectories` (int | None): Optional override for builder
                  class count.
                - `input_model_path` (str | Path): Required in
                  `from_saved_model` mode.
                - `save_model_path` (str | Path | None): Optional fitted model
                  export path in `from_scratch` mode.

        Returns:
            BuildTrajectories: The fitted tool instance.
        """

        mode = _normalise_mode(kwargs.get("mode", "from_scratch"))

        input_path_raw = kwargs.get("input_path")
        output_path_raw = kwargs.get("output_path")
        id_col_raw = kwargs.get("id_col")
        time_col_raw = kwargs.get("time_col")

        if input_path_raw is None:
            raise InputValidationError("Missing required parameter: input_path")
        if output_path_raw is None:
            raise InputValidationError("Missing required parameter: output_path")
        if not isinstance(id_col_raw, str) or not id_col_raw.strip():
            raise InputValidationError("Missing required parameter: id_col")
        if not isinstance(time_col_raw, str) or not time_col_raw.strip():
            raise InputValidationError("Missing required parameter: time_col")

        input_path = Path(str(input_path_raw)).expanduser()
        output_path = Path(str(output_path_raw)).expanduser()
        _validate_csv_path(input_path, field_name="input_path")
        _validate_output_csv_path(output_path)

        data = pd.read_csv(input_path)
        id_col = id_col_raw.strip()
        time_col = time_col_raw.strip()
        _ensure_columns(data, [id_col, time_col])

        class_col = _as_optional_string(kwargs.get("class_col"))
        if class_col:
            _ensure_columns(data, [class_col])

        builders = discover_trajectory_builders()
        builder_label: str | None = None
        builder_kwargs: dict[str, Any] = {}
        model_path: Path | None = None
        save_model_path: Path | None = None

        if mode == "from_saved_model":
            model_path_raw = kwargs.get("input_model_path")
            if model_path_raw is None:
                raise InputValidationError(
                    "Missing required parameter `input_model_path` for `from_saved_model`."
                )
            model_path = Path(str(model_path_raw)).expanduser()
            if not model_path.exists() or not model_path.is_file():
                raise InputValidationError(
                    f"Saved model path does not exist: {model_path}"
                )
            builder_label = "from_saved_model"
        else:
            builder_raw = kwargs.get("builder")
            if not builder_raw and mode == "from_scratch":
                raise InputValidationError(
                    "Missing required parameter `builder` for mode `from_scratch`."
                )

            if builder_raw is None:
                builder_raw = mode
            builder_label = str(builder_raw)
            _resolve_builder_class(builders, builder_label)

            builder_kwargs = _as_optional_object(kwargs.get("builder_kwargs_json"))
            n_trajectories = _as_optional_int(kwargs.get("n_trajectories"))
            if n_trajectories is not None:
                builder_kwargs.setdefault("n_trajectories", n_trajectories)

            save_model_raw = _as_optional_string(kwargs.get("save_model_path"))
            if save_model_raw:
                save_model_path = Path(save_model_raw).expanduser()

        value_cols = _as_optional_string_list(kwargs.get("value_cols"))

        self._config = {
            "mode": mode,
            "input_path": input_path,
            "output_path": output_path,
            "id_col": id_col,
            "time_col": time_col,
            "value_cols": tuple(value_cols),
            "class_col": class_col,
            "builder": builder_label,
            "builder_kwargs_json": builder_kwargs,
            "input_model_path": model_path,
            "save_model_path": save_model_path,
        }
        return self

    @beartype
    def preprocess(self, **kwargs: Any) -> dict[str, Any]:
        """Execute trajectory building and write assignment output.

        Args:
            **kwargs (Any): Optional configuration keys identical to `fit(...)`.
                If provided, they override any existing fitted configuration.

        Returns:
            dict[str, Any]: Serialised run summary with builder metadata,
            output path, shape details, and selected class count when available.
        """

        if kwargs:
            self.fit(**kwargs)
        if self._config is None:
            raise InputValidationError(
                "No configuration provided. Call `fit(...)` first or pass kwargs to `preprocess(...)`."
            )

        mode = str(self._config.get("mode"))
        input_path = _as_path(self._config.get("input_path"), key="input_path")
        output_path = _as_path(self._config.get("output_path"), key="output_path")
        id_col = str(self._config.get("id_col"))
        time_col = str(self._config.get("time_col"))
        class_col = _as_optional_string(self._config.get("class_col"))
        value_cols = list(self._config.get("value_cols", ()))

        data = pd.read_csv(input_path)
        _ensure_columns(data, [id_col, time_col])
        if class_col:
            _ensure_columns(data, [class_col])

        builder_label: str
        saved_model_path: str | None = None

        if mode == "from_saved_model":
            run_mode = "from_saved_model"
            model_path = _as_path(
                self._config.get("input_model_path"),
                key="input_model_path",
            )
            model = TrajectoryModel.load(model_path)
            builder_label = type(model).__name__
        else:
            run_mode = "from_scratch"
            builders = discover_trajectory_builders()
            builder_key = _as_optional_string(self._config.get("builder"))
            if not builder_key:
                raise InputValidationError("Builder is required for from-scratch mode.")

            builder_cls = _resolve_builder_class(builders, builder_key)
            raw_builder_kwargs = _as_optional_object(
                self._config.get("builder_kwargs_json")
            )
            init_kwargs = _coerce_kwargs_for_signature(
                builder_cls.__init__, raw_builder_kwargs
            )
            model = builder_cls(**init_kwargs)
            builder_label = builder_key

            model_save_path = self._config.get("save_model_path")
            if isinstance(model_save_path, Path):
                saved_model_path = str(model_save_path)

        resolved_value_cols = _resolve_value_columns(
            data=data,
            provided=value_cols,
            id_col=id_col,
            time_col=time_col,
            class_col=class_col,
            max_value_cols=getattr(model, "max_value_cols", None),
        )

        model.fit(
            data, id_col=id_col, time_col=time_col, value_cols=resolved_value_cols
        )
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


@beartype
def run_build_trajectories(
    *, technique: str, params: Mapping[str, Any]
) -> dict[str, Any]:
    """Run one trajectory-building technique for the Go CLI bridge.

    !!! warning

        This `run_*` function is primarily for the Go CLI bridge and should not be
        treated as the Python library API. In Python scripts or notebooks, import
        and call `BuildTrajectories` directly from `ldt.data_preprocessing`.

    Args:
        technique (str): Builder mode/technique key from the catalogue.
        params (Mapping[str, Any]): Wrapper parameters forwarded to
            `BuildTrajectories.fit_preprocess(...)`, including path keys,
            column keys, mode-specific model keys, and optional builder kwargs.

    Returns:
        dict[str, Any]: Serialised trajectory-building summary for the Go CLI
        bridge.
    """

    _, resolved = resolve_technique_with_defaults(
        section_key="build_trajectories",
        technique_id=technique,
        provided_params=dict(params),
    )

    mode = _normalise_mode(technique)

    payload: dict[str, Any] = {
        "mode": mode,
        "input_path": Path(_as_required_string(resolved, "input_path")).expanduser(),
        "output_path": Path(_as_required_string(resolved, "output_path")).expanduser(),
        "id_col": _as_required_string(resolved, "id_col"),
        "time_col": _as_required_string(resolved, "time_col"),
        "value_cols": _as_optional_string_list(resolved.get("value_cols")),
        "class_col": _as_optional_string(resolved.get("class_col")),
    }

    if mode == "from_saved_model":
        payload["input_model_path"] = Path(
            _as_required_string(resolved, "input_model_path")
        ).expanduser()
    else:
        payload["builder"] = _as_optional_string(resolved.get("builder")) or technique
        payload["builder_kwargs_json"] = _as_optional_object(
            resolved.get("builder_kwargs_json")
        )
        payload["n_trajectories"] = _as_optional_int(resolved.get("n_trajectories"))
        save_model_path = _as_optional_string(resolved.get("save_model_path"))
        if save_model_path:
            payload["save_model_path"] = Path(save_model_path).expanduser()

    return BuildTrajectories().fit_preprocess(**payload)


@beartype
def _normalise_mode(raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise InputValidationError("Missing required trajectory mode.")
    token = raw.strip().lower().replace("-", "_")
    if token == "from_saved_model":
        return "from_saved_model"
    return "from_scratch"


@beartype
def _resolve_builder_class(
    builders: dict[str, type[TrajectoryModel]],
    builder_key: str,
) -> type[TrajectoryModel]:
    target = _normalise_key(builder_key)
    for key, builder_cls in builders.items():
        if _normalise_key(key) == target:
            return builder_cls
    choices = ", ".join(sorted(builders.keys()))
    raise InputValidationError(f"Unknown builder `{builder_key}`. Available: {choices}")


@beartype
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


@beartype
def _coerce_value(value: Any, annotation: Any) -> Any:
    if annotation is inspect._empty:
        return value

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is None:
        if annotation is Any:
            return value
        if annotation is bool:
            return _as_bool(value, field_name="builder kwargs")
        if annotation is int:
            return _as_int(value, field_name="builder kwargs")
        if annotation is float:
            return _as_float(value, field_name="builder kwargs")
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


@beartype
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

    _ensure_columns(data, value_cols)

    if max_value_cols == 1 and len(value_cols) > 1:
        return [value_cols[0]]
    if max_value_cols is not None and len(value_cols) > max_value_cols:
        raise InputValidationError(
            f"Selected value columns exceed builder limit ({max_value_cols})."
        )
    return value_cols


@beartype
def _normalise_key(raw: str) -> str:
    return raw.strip().lower().replace("-", "_")


@beartype
def _as_required_string(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"Missing required string parameter: {key}")
    return value.strip()


@beartype
def _as_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed if trimmed else None
    raise InputValidationError("Expected a string value.")


@beartype
def _as_optional_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        parsed: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise InputValidationError("List parameters must contain strings.")
            stripped = item.strip()
            if stripped:
                parsed.append(stripped)
        return parsed
    raise InputValidationError("Expected a CSV string or string list.")


@beartype
def _as_optional_object(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise InputValidationError("builder_kwargs_json must be a JSON object.")
    return dict(value)


@beartype
def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return _as_int(value, field_name="integer")


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
def _as_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise InputValidationError(f"`{field_name}` must be a float value.")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value.strip())
        except ValueError as exc:
            raise InputValidationError(
                f"`{field_name}` must be a float value."
            ) from exc
    raise InputValidationError(f"`{field_name}` must be a float value.")


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
def _as_path(value: Any, *, key: str) -> Path:
    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value.strip():
        return Path(value).expanduser()
    raise InputValidationError(f"Missing required path parameter: {key}")


@beartype
def _ensure_columns(data: pd.DataFrame, required: list[str]) -> None:
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise InputValidationError(
            f"Missing required columns: {', '.join(sorted(missing))}"
        )


@beartype
def _validate_csv_path(path: Path, *, field_name: str) -> None:
    if not path.exists():
        raise InputValidationError(f"{field_name} does not exist: {path}")
    if not path.is_file():
        raise InputValidationError(f"{field_name} is not a file: {path}")
    if path.suffix.lower() != ".csv":
        raise InputValidationError(f"{field_name} must point to a .csv file.")


@beartype
def _validate_output_csv_path(path: Path) -> None:
    if path.suffix.lower() != ".csv":
        raise InputValidationError("output_path must point to a .csv file.")
