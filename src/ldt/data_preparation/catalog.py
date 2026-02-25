from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from ldt.bridge.catalog import (
    expand_parameter_templates,
    load_catalog_file,
    section_dict_items,
)
from ldt.utils.errors import InputValidationError

_CATALOG_PATH = Path(__file__).resolve().parent / "config" / "go_cli_catalog.json"


def load_data_preparation_catalog() -> dict[str, Any]:
    """Load and normalise the data-preparation catalogue JSON.

    Returns:
        dict[str, Any]: Dictionary containing tool results.
    """

    payload = load_catalog_file(_CATALOG_PATH)
    expand_parameter_templates(payload)
    return payload


def list_synthetic_techniques() -> list[dict[str, Any]]:
    """Return synthetic-data-generation techniques with prompt schemas.

    Returns:
        list[dict[str, Any]]: List of parsed values.
    """

    catalog = load_data_preparation_catalog()
    return section_dict_items(
        catalog,
        section_key="synthetic_data_generation",
        list_key="techniques",
    )


def list_data_conversion_techniques() -> list[dict[str, Any]]:
    """Return data-conversion techniques with prompt schemas.

    Returns:
        list[dict[str, Any]]: List of parsed values.
    """

    catalog = load_data_preparation_catalog()
    return section_dict_items(
        catalog,
        section_key="data_conversion",
        list_key="techniques",
    )


def list_data_preparation_presets() -> list[dict[str, Any]]:
    """Return data-preparation reproducibility presets metadata.

    Returns:
        list[dict[str, Any]]: List of parsed values.
    """

    catalog = load_data_preparation_catalog()
    return section_dict_items(
        catalog,
        section_key="presets_reproducibility",
        list_key="items",
    )


def resolve_technique_with_defaults(
    *,
    section_key: str,
    technique_id: str,
    provided_params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve technique entry and merge provided params with configured defaults.

    Args:
        section_key (str): Technique or operation identifier.
        technique_id (str): Identifier for technique id.
        provided_params (dict[str, Any]): Parameter mapping used for this call.

    Returns:
        tuple[dict[str, Any], dict[str, Any]]: Tuple of resolved values.
    """

    technique = _find_technique(section_key=section_key, technique_id=technique_id)
    parameters = technique.get("parameters", [])
    if not isinstance(parameters, list):
        raise InputValidationError(
            f"Technique `{technique_id}` parameters must be a list after template expansion."
        )

    resolved = dict(provided_params)
    for parameter in parameters:
        if not isinstance(parameter, dict):
            continue
        if not _condition_matches(parameter=parameter, values=resolved):
            continue

        key = parameter.get("key")
        if not isinstance(key, str) or not key.strip():
            continue
        key = key.strip()

        current = resolved.get(key)
        if not _is_missing_value(current):
            continue

        if "default" in parameter:
            resolved[key] = parameter["default"]
            continue

        if bool(parameter.get("required", False)):
            raise InputValidationError(
                f"Missing required parameter `{key}` for technique `{technique_id}`."
            )

    return copy.deepcopy(technique), resolved


def _find_technique(*, section_key: str, technique_id: str) -> dict[str, Any]:
    catalog = load_data_preparation_catalog()
    section = catalog.get(section_key)
    if not isinstance(section, dict):
        raise InputValidationError(f"Unknown catalog section: {section_key}")

    techniques = section.get("techniques", [])
    if not isinstance(techniques, list):
        raise InputValidationError(f"`{section_key}.techniques` must be a list.")

    target = _normalise_key(technique_id)
    for technique in techniques:
        if not isinstance(technique, dict):
            continue
        raw_id = technique.get("id")
        if isinstance(raw_id, str) and _normalise_key(raw_id) == target:
            return technique

    raise InputValidationError(
        f"Unknown technique `{technique_id}` in section `{section_key}`."
    )


def _normalise_key(raw: str) -> str:
    return raw.strip().lower().replace("-", "_")


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, list):
        return len(value) == 0
    return False


def _condition_matches(*, parameter: dict[str, Any], values: dict[str, Any]) -> bool:
    when = parameter.get("when")
    if not isinstance(when, dict):
        return True

    field = when.get("field")
    if not isinstance(field, str) or not field.strip():
        return True

    expected = when.get("equals")
    actual = values.get(field)
    if isinstance(expected, str):
        if actual is None:
            return False
        return str(actual).strip().lower() == expected.strip().lower()
    return actual == expected
