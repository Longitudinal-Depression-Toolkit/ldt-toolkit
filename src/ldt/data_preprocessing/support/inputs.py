from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ldt.utils.errors import InputValidationError


def run_with_validation(callback: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Execute a runner and normalise low-level errors into validation errors.

    Args:
        callback (Callable[[], dict[str, Any]]): Callable used to execute the workflow step.

    Returns:
        dict[str, Any]: Dictionary containing tool results.
    """

    try:
        return callback()
    except InputValidationError:
        raise
    except (TypeError, ValueError, FileNotFoundError) as exc:
        raise InputValidationError(str(exc)) from exc


def normalise_key(raw: str) -> str:
    """Normalise a catalog key for case/slug tolerant comparisons.

    Args:
        raw (str): Raw value to parse.

    Returns:
        str: Parsed string value.
    """

    return raw.strip().lower().replace("-", "_")


def ensure_columns(data: pd.DataFrame, required: Sequence[str]) -> None:
    """Validate required columns exist in a dataframe.

    Args:
        data (pd.DataFrame): Input dataset.
        required (Sequence[str]): Required column names.
    """

    missing = [column for column in required if column not in data.columns]
    if missing:
        raise InputValidationError(
            f"Missing required columns: {', '.join(sorted(missing))}"
        )


def ensure_distinct_paths(
    input_path: Path, output_path: Path, *, field_name: str
) -> None:
    """Ensure output path is not the same as input path.

    Args:
        input_path (Path): Filesystem path used by the workflow.
        output_path (Path): Filesystem path used by the workflow.
        field_name (str): Parameter name.
    """

    if input_path.resolve() == output_path.resolve():
        raise InputValidationError(
            f"`{field_name}` must be different from the input path."
        )


def as_required_string(params: Mapping[str, Any], key: str) -> str:
    """Read a required non-empty string parameter.

    Args:
        params (Mapping[str, Any]): Parameter mapping provided by the caller.
        key (str): Parameter name.

    Returns:
        str: Parsed string value.
    """

    value = params.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"Missing required string parameter: {key}")
    return value.strip()


def as_optional_string(params: Mapping[str, Any], key: str) -> str | None:
    """Read an optional string parameter.

    Args:
        params (Mapping[str, Any]): Parameter mapping provided by the caller.
        key (str): Parameter name.

    Returns:
        str | None: Parsed string value.
    """

    value = params.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed if trimmed else None
    raise InputValidationError(f"`{key}` must be a string when provided.")


def as_required_string_list_or_csv(params: Mapping[str, Any], key: str) -> list[str]:
    """Read a required string list from either CSV text or list input.

    Args:
        params (Mapping[str, Any]): Parameter mapping provided by the caller.
        key (str): Parameter name.

    Returns:
        list[str]: List of parsed values.
    """

    parsed = _string_list_or_csv(params.get(key))
    if not parsed:
        raise InputValidationError(f"Missing required string-list parameter: {key}")
    return parsed


def as_optional_string_list_or_csv(params: Mapping[str, Any], key: str) -> list[str]:
    """Read an optional string list from either CSV text or list input.

    Args:
        params (Mapping[str, Any]): Parameter mapping provided by the caller.
        key (str): Parameter name.

    Returns:
        list[str]: List of parsed values.
    """

    return _string_list_or_csv(params.get(key))


def as_required_int(
    params: Mapping[str, Any],
    key: str,
    *,
    minimum: int | None = None,
) -> int:
    """Read a required integer parameter, optionally with minimum bound.

    Args:
        params (Mapping[str, Any]): Parameter mapping provided by the caller.
        key (str): Parameter name.
        minimum (int | None): Minimum.

    Returns:
        int: Parsed integer value.
    """

    parsed = _coerce_int(params.get(key))
    if minimum is not None and parsed < minimum:
        raise InputValidationError(f"`{key}` must be >= {minimum}.")
    return parsed


def as_optional_int(params: Mapping[str, Any], key: str) -> int | None:
    """Read an optional integer parameter.

    Args:
        params (Mapping[str, Any]): Parameter mapping provided by the caller.
        key (str): Parameter name.

    Returns:
        int | None: Parsed integer value.
    """

    value = params.get(key)
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return _coerce_int(value)


def as_optional_float(params: Mapping[str, Any], key: str) -> float | None:
    """Read an optional float parameter.

    Args:
        params (Mapping[str, Any]): Parameter mapping provided by the caller.
        key (str): Parameter name.

    Returns:
        float | None: Parsed floating-point value.
    """

    value = params.get(key)
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return _coerce_float(value)


def as_optional_object(params: Mapping[str, Any], key: str) -> dict[str, Any]:
    """Read an optional JSON-like object parameter.

    Args:
        params (Mapping[str, Any]): Parameter mapping provided by the caller.
        key (str): Parameter name.

    Returns:
        dict[str, Any]: Dictionary containing tool results.
    """

    value = params.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise InputValidationError(f"`{key}` must be a JSON object.")
    return dict(value)


def as_choice(value: str, *, choices: Sequence[str], field_name: str) -> str:
    """Validate a string parameter against an enum-like choice set.

    Args:
        value (str): Value to validate or coerce.
        choices (Sequence[str]): Allowed choices.
        field_name (str): Parameter name.

    Returns:
        str: Parsed string value.
    """

    candidate = value.strip().lower()
    if candidate not in choices:
        raise InputValidationError(
            f"`{field_name}` must be one of: {', '.join(choices)}."
        )
    return candidate


def as_bool(value: Any, *, field_name: str) -> bool:
    """Read a boolean parameter from bool/int/string-like values.

    Args:
        value (Any): Value to validate or coerce.
        field_name (str): Parameter name.

    Returns:
        bool: Parsed boolean value.
    """

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


def _string_list_or_csv(value: Any) -> list[str]:
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


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        raise InputValidationError("Boolean is not a valid integer value.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise InputValidationError("Expected an integer value.")
    if isinstance(value, str):
        token = value.strip()
        if token == "":
            raise InputValidationError("Expected an integer value.")
        try:
            return int(token)
        except ValueError as exc:
            raise InputValidationError("Expected an integer value.") from exc
    raise InputValidationError("Expected an integer value.")


def _coerce_float(value: Any) -> float:
    if isinstance(value, bool):
        raise InputValidationError("Boolean is not a valid float value.")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        token = value.strip()
        if token == "":
            raise InputValidationError("Expected a float value.")
        try:
            return float(token)
        except ValueError as exc:
            raise InputValidationError("Expected a float value.") from exc
    raise InputValidationError("Expected a float value.")
