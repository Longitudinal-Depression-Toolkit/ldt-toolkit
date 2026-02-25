from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComponentMetadata:
    """Typed metadata for discoverable library components.

    Attributes:
        name (str): Stable machine-readable key for lookup and registration.
        full_name (str): Display name used in catalogs.
        abstract_description (str): Short summary shown in catalogs.
        tutorial_goal (str | None): Optional tutorial goal text.
        tutorial_how_it_works (str | None): Optional tutorial explanation text.
    """

    name: str
    full_name: str
    abstract_description: str = "View Docstring/Documentation"
    tutorial_goal: str | None = None
    tutorial_how_it_works: str | None = None


def resolve_component_metadata(component: object) -> ComponentMetadata:
    """Resolve component metadata from a class or instance.

    Args:
        component (object): Component object to inspect for metadata.

    Returns:
        ComponentMetadata: Metadata attached to the input component.
    """
    raw = getattr(component, "metadata", None)
    if raw is None:
        raise AttributeError(
            f"Component {component!r} does not define required `metadata`."
        )
    if not isinstance(raw, ComponentMetadata):
        raise TypeError(
            f"Component {component!r} has invalid `metadata` type: "
            f"{type(raw).__name__}. Expected ComponentMetadata."
        )
    return raw
