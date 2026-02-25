from __future__ import annotations

import importlib
import inspect
import pkgutil

from ldt.machine_learning.tools.templates import EstimatorTemplate
from ldt.utils.metadata import resolve_component_metadata

ESTIMATORS_PACKAGE_NAME = f"{__package__}.estimators"


def discover_standard_estimators() -> dict[str, type[EstimatorTemplate]]:
    """Discover cross-sectional estimator templates.

    Returns:
        dict[str, type[EstimatorTemplate]]: Mapping of estimator keys to templates.
    """

    estimators: dict[str, type[EstimatorTemplate]] = {}
    package = importlib.import_module(ESTIMATORS_PACKAGE_NAME)
    for module_info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{package.__name__}.{module_info.name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, EstimatorTemplate) and obj is not EstimatorTemplate:
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
