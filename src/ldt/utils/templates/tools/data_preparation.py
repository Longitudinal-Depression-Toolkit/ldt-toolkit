from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from beartype import beartype

from ldt.utils.metadata import ComponentMetadata


@beartype
@dataclass(frozen=True)
class ToolParameterDefinition:
    """Describe one tool parameter for wrappers and user interfaces.

    This schema is intentionally lightweight. It is mainly used by orchestration
    layers (for example Go CLI prompts or future UI builders) to discover a
    tool's expected inputs without hard-coding prompt definitions in multiple
    places.
    """

    key: str
    type: str
    required: bool
    default: Any = None
    description: str = ""


@beartype
class DataPreparationTool:
    """Authoring template for every data-preparation tool in the Python API.

    This base class defines the contract that all data-preparation tools should
    follow. It is designed for library-first usage, so users can import tools
    directly from `ldt.data_preparation` and call methods in a predictable way.

    A compliant tool should:
        - expose a clear class name and a meaningful `metadata` object
        - accept `**kwargs` in `fit(...)` and `prepare(...)`
        - validate input early and fail with explicit error messages
        - return native Python objects (for example `pd.DataFrame`,
          result dataclasses, or typed payloads), not CLI-shaped dictionaries
        - keep CLI-specific translation logic in separate `run.py` wrappers

    Method contract:
        - `fit(**kwargs)`:
          optional pre-computation or configuration step. Use this to parse,
          validate, and store internal state.
        - `prepare(**kwargs)`:
          required execution step. This method performs the actual
          data-preparation behaviour.
        - `fit_prepare(**kwargs)`:
          convenience method that runs `fit(...)` followed by `prepare(...)`
          with the same payload.

    Guidance for new tools:
        - prefer deterministic behaviour when a random seed is supplied
        - keep side effects explicit (for example writing files only when
          output paths are provided)
        - document expected inputs and outputs with concrete examples
        - expose `params_definition()` when wrapper or UI discovery is useful

    Example:
        ```python
        from typing import Any
        import pandas as pd
        from beartype import beartype

        from ldt.data_preparation import DataPreparationTool
        from ldt.utils.metadata import ComponentMetadata

        @beartype
        class ExampleTool(DataPreparationTool):
            metadata = ComponentMetadata(
                name="example_tool",
                full_name="Example Tool",
            )

            def fit(self, **kwargs: Any) -> "ExampleTool":
                self._feature = str(kwargs.get("feature", "x"))
                return self

            def prepare(self, **kwargs: Any) -> pd.DataFrame:
                value = float(kwargs.get("value", 0.0))
                return pd.DataFrame({self._feature: [value]})
        ```
    """

    metadata = ComponentMetadata(
        name="base",
        full_name="Data Preparation Tool",
    )

    @classmethod
    @beartype
    def params_definition(cls) -> tuple[ToolParameterDefinition, ...]:
        """Return an optional parameter schema for wrappers or UI discovery.

        Override this when external interfaces need a declarative input schema.
        """

        return ()

    @beartype
    def fit(self, **kwargs: Any) -> DataPreparationTool:
        """Run an optional fitting or configuration step.

        Args:
            **kwargs (Any): Tool-specific configuration values supplied by the
                caller. There is no stage-wide fixed key set at this level.
                Every concrete tool must document its supported keys and value
                types in its own `fit(...)` docstring.

        Returns:
            DataPreparationTool: The fitted tool instance.
        """

        _ = kwargs
        return self

    @beartype
    def prepare(self, **kwargs: Any) -> Any:
        """Execute data-preparation logic.

        Args:
            **kwargs (Any): Tool-specific runtime values. There is no shared
                key schema at the base-class level. Every concrete tool must
                document expected keys in its own `prepare(...)` docstring.

        Returns:
            Any: Tool-specific output payload.

        Raises:
            NotImplementedError: Always raised by the base template.
        """

        _ = kwargs
        raise NotImplementedError

    @beartype
    def fit_prepare(self, **kwargs: Any) -> Any:
        """Run `fit(...)` then `prepare(...)` with one shared payload.

        Args:
            **kwargs (Any): Tool-specific values forwarded unchanged to both
                `fit(...)` and `prepare(...)`.

        Returns:
            Any: Tool-specific output payload returned by `prepare(...)`.
        """

        self.fit(**kwargs)
        return self.prepare(**kwargs)
