from __future__ import annotations

import importlib
import inspect
import pkgutil

from ldt.utils.metadata import resolve_component_metadata

from .estimators import (
    LongitudinalEstimatorTemplate,
    LongitudinalStrategyDefinition,
    list_longitudinal_strategies,
)

ESTIMATORS_PACKAGE_NAME = f"{__package__}.estimators"


def discover_longitudinal_estimators() -> dict[
    str, type[LongitudinalEstimatorTemplate]
]:
    """Discover longitudinal estimator templates.

    Returns:
        dict[str, type[LongitudinalEstimatorTemplate]]: Mapping of estimator keys to templates.
    """

    estimators: dict[str, type[LongitudinalEstimatorTemplate]] = {}
    package = importlib.import_module(ESTIMATORS_PACKAGE_NAME)
    for module_info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{package.__name__}.{module_info.name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, LongitudinalEstimatorTemplate)
                and obj is not LongitudinalEstimatorTemplate
            ):
                label = resolve_component_metadata(obj).name
                estimators[label] = obj
    return dict(
        sorted(
            estimators.items(),
            key=lambda item: (
                resolve_component_metadata(item[1]).full_name.lower(),
                item[0].lower(),
            ),
        )
    )


__all__ = [
    "LongitudinalEstimatorTemplate",
    "LongitudinalStrategyDefinition",
    "discover_longitudinal_estimators",
    "list_longitudinal_strategies",
]
