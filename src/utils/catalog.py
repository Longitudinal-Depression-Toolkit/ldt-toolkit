from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from src.utils.errors import InputValidationError


def load_catalog_file(path: Path) -> dict[str, Any]:
    """Load a JSON catalog file as a dictionary.

    Args:
        path (Path): Filesystem path used by the workflow.

    Returns:
        dict[str, Any]: Dictionary containing tool results.
    """

    if not path.exists():
        raise InputValidationError(f"Catalog file not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InputValidationError(f"Invalid catalog JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise InputValidationError("Catalog root must be a JSON object.")
    return payload


def expand_parameter_templates(payload: dict[str, Any]) -> None:
    """Expand `@template` parameter references for all catalog sections.

    Args:
        payload (dict[str, Any]): Parameter mapping provided by the caller.
    """

    for section_value in payload.values():
        if not isinstance(section_value, dict):
            continue

        templates = section_value.get("parameter_templates", {})
        if not isinstance(templates, dict):
            templates = {}

        techniques = section_value.get("techniques", [])
        if not isinstance(techniques, list):
            continue

        for technique in techniques:
            if not isinstance(technique, dict):
                continue

            parameters = technique.get("parameters")
            if isinstance(parameters, str) and parameters.startswith("@"):
                template_key = parameters[1:]
                template_params = templates.get(template_key)
                if not isinstance(template_params, list):
                    raise InputValidationError(
                        f"Unknown parameter template `{parameters}` in catalog."
                    )
                technique["parameters"] = copy.deepcopy(template_params)
                continue

            if isinstance(parameters, list):
                technique["parameters"] = copy.deepcopy(parameters)


def section_dict_items(
    payload: dict[str, Any],
    *,
    section_key: str,
    list_key: str,
) -> list[dict[str, Any]]:
    """Extract a list of dict items from one catalog section key.

    Args:
        payload (dict[str, Any]): Parameter mapping provided by the caller.
        section_key (str): Technique or operation identifier.
        list_key (str): List key.

    Returns:
        list[dict[str, Any]]: List of parsed values.
    """

    section = payload.get(section_key, {})
    if not isinstance(section, dict):
        raise InputValidationError(f"`{section_key}` must be an object.")

    items = section.get(list_key, [])
    if not isinstance(items, list):
        raise InputValidationError(f"`{section_key}.{list_key}` must be a list.")
    return copy.deepcopy([item for item in items if isinstance(item, dict)])
