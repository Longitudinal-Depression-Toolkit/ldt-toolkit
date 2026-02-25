from __future__ import annotations

import importlib
import inspect
import pkgutil

from ldt.utils.metadata import resolve_component_metadata

from .imputers import MissingImputer

IMPUTERS_PACKAGE_NAME = f"{__package__}.imputers"


def discover_missing_imputers() -> dict[str, type[MissingImputer]]:
    """Discover missing-data imputer classes from the imputers package."""

    imputers: dict[str, type[MissingImputer]] = {}
    package = importlib.import_module(IMPUTERS_PACKAGE_NAME)
    for module_info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{package.__name__}.{module_info.name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, MissingImputer) and obj is not MissingImputer:
                label = resolve_component_metadata(obj).name
                imputers[label] = obj

    return dict(
        sorted(
            imputers.items(),
            key=lambda item: (
                resolve_component_metadata(item[1]).full_name.lower(),
                item[0].lower(),
            ),
        )
    )
