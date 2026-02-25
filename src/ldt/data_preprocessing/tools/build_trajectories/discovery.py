from __future__ import annotations

import importlib
import inspect
import pkgutil

from ldt.utils.metadata import resolve_component_metadata

from .trajectory import TrajectoryModel

TRAJECTORIES_PACKAGE_NAME = f"{__package__}.trajectories"


def discover_trajectory_builders() -> dict[str, type[TrajectoryModel]]:
    """Discover trajectory-builder classes from the trajectories package.

    Returns:
        dict[str, type[TrajectoryModel]]: Mapping of strategy keys to trajectory models.
    """
    builders: dict[str, type[TrajectoryModel]] = {}
    package = importlib.import_module(TRAJECTORIES_PACKAGE_NAME)
    for module_info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{package.__name__}.{module_info.name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, TrajectoryModel) and obj is not TrajectoryModel:
                label = resolve_component_metadata(obj).name
                builders[label] = obj
    return dict(
        sorted(
            builders.items(),
            key=lambda item: (
                resolve_component_metadata(item[1]).full_name.lower(),
                item[0].lower(),
            ),
        )
    )
