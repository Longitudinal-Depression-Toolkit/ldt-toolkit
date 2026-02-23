from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from src.utils.catalog import (
    expand_parameter_templates,
    load_catalog_file,
    section_dict_items,
)
from src.utils.errors import InputValidationError

_CATALOG_PATH = Path(__file__).resolve().parent / "config" / "go_cli_catalog.json"


def load_data_preprocessing_catalog() -> dict[str, Any]:
    """Load and normalise the data-preprocessing catalog JSON.

    Returns:
        dict[str, Any]: Dictionary containing tool results.
    """

    payload = load_catalog_file(_CATALOG_PATH)
    expand_parameter_templates(payload)
    return payload


def list_remove_columns_techniques() -> list[dict[str, Any]]:
    """Return remove-columns techniques with prompt schemas.

    Returns:
        list[dict[str, Any]]: List of parsed values.
    """

    catalog = load_data_preprocessing_catalog()
    return section_dict_items(
        catalog,
        section_key="remove_columns",
        list_key="techniques",
    )


def list_build_trajectories_techniques() -> list[dict[str, Any]]:
    """Return build-trajectories techniques with prompt schemas.

    Returns:
        list[dict[str, Any]]: List of parsed values.
    """

    return list_tool_techniques("build_trajectories")


def list_combine_dataset_with_trajectories_techniques() -> list[dict[str, Any]]:
    """Return dataset-combination techniques with prompt schemas.

    Returns:
        list[dict[str, Any]]: List of parsed values.
    """

    return list_tool_techniques("combine_dataset_with_trajectories")


def list_clean_dataset_techniques() -> list[dict[str, Any]]:
    """Return clean-dataset techniques with prompt schemas.

    Returns:
        list[dict[str, Any]]: List of parsed values.
    """

    return list_tool_techniques("clean_dataset")


def list_missing_imputation_techniques() -> list[dict[str, Any]]:
    """Return missing-imputation techniques with prompt schemas.

    Returns:
        list[dict[str, Any]]: List of parsed values.
    """

    return list_tool_techniques("missing_imputation")


def list_harmonise_categories_techniques() -> list[dict[str, Any]]:
    """Return harmonise-categories techniques with prompt schemas.

    Returns:
        list[dict[str, Any]]: List of parsed values.
    """

    return list_tool_techniques("harmonise_categories")


def list_show_table_techniques() -> list[dict[str, Any]]:
    """Return show-table techniques with prompt schemas.

    Returns:
        list[dict[str, Any]]: List of parsed values.
    """

    return list_tool_techniques("show_table")


def list_aggregate_long_to_cross_sectional_techniques() -> list[dict[str, Any]]:
    """Return aggregate-long-to-cross-sectional techniques with prompt schemas.

    Returns:
        list[dict[str, Any]]: List of parsed values.
    """

    return list_tool_techniques("aggregate_long_to_cross_sectional")


def list_rename_feature_techniques() -> list[dict[str, Any]]:
    """Return rename-feature techniques with prompt schemas.

    Returns:
        list[dict[str, Any]]: List of parsed values.
    """

    return list_tool_techniques("rename_feature")


def list_pivot_long_to_wide_techniques() -> list[dict[str, Any]]:
    """Return long-to-wide pivot techniques with prompt schemas.

    Returns:
        list[dict[str, Any]]: List of parsed values.
    """

    return list_tool_techniques("pivot_long_to_wide")


def list_trajectories_viz_techniques() -> list[dict[str, Any]]:
    """Return trajectories-viz techniques with prompt schemas.

    Returns:
        list[dict[str, Any]]: List of parsed values.
    """

    return list_tool_techniques("trajectories_viz")


def list_tool_techniques(section_key: str) -> list[dict[str, Any]]:
    """Return techniques for one data-preprocessing catalog section.

    Args:
        section_key (str): Technique or operation identifier.

    Returns:
        list[dict[str, Any]]: List of parsed values.
    """

    catalog = load_data_preprocessing_catalog()
    return section_dict_items(
        catalog,
        section_key=section_key,
        list_key="techniques",
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
    catalog = load_data_preprocessing_catalog()
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
