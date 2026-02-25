from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from beartype import beartype

from ldt.utils.errors import InputValidationError


@beartype
def as_required_string(params: Mapping[str, Any], key: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"Missing required string parameter: {key}")
    return value.strip()


@beartype
def as_int(params: Mapping[str, Any], key: str, *, minimum: int | None = None) -> int:
    value = params.get(key)
    if isinstance(value, bool):
        raise InputValidationError(f"`{key}` must be an integer value.")

    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float) and value.is_integer():
        parsed = int(value)
    elif isinstance(value, str) and value.strip():
        try:
            parsed = int(value.strip())
        except ValueError as exc:
            raise InputValidationError(f"`{key}` must be an integer value.") from exc
    else:
        raise InputValidationError(f"`{key}` must be an integer value.")

    if minimum is not None and parsed < minimum:
        raise InputValidationError(f"`{key}` must be >= {minimum}.")
    return parsed


@beartype
def as_bool(params: Mapping[str, Any], key: str, *, default: bool) -> bool:
    value = params.get(key, default)
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
    raise InputValidationError(f"`{key}` must be a boolean value.")


@beartype
def as_choice(
    params: Mapping[str, Any],
    key: str,
    *,
    choices: tuple[str, ...],
) -> str:
    value = as_required_string(params, key).lower()
    if value not in choices:
        raise InputValidationError(f"`{key}` must be one of: {', '.join(choices)}.")
    return value
