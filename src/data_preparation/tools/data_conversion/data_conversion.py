from __future__ import annotations

import importlib
import inspect
import pkgutil

from beartype import beartype

from src.utils.metadata import ComponentMetadata, resolve_component_metadata

conversion_converters_package_name = (
    "ldt.data_preparation.tools.data_conversion.converters"
)


@beartype
class Conversion:
    """Base interface for tabular data converters."""

    metadata = ComponentMetadata(
        name="base",
        full_name="Data Converter",
    )


@beartype
def discover_converters() -> dict[str, type[Conversion]]:
    """Discover converter classes from the converters package.

    Returns:
        dict[str, type[Conversion]]: Mapping of technique keys to converter classes.
    """

    converters: dict[str, type[Conversion]] = {}
    package = importlib.import_module(conversion_converters_package_name)
    for module_info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{package.__name__}.{module_info.name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Conversion) and obj is not Conversion:
                label = resolve_component_metadata(obj).name
                converters[label] = obj
    return dict(sorted(converters.items(), key=lambda item: item[0].lower()))
