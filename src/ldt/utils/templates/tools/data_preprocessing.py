from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from beartype import beartype

from ldt.utils.metadata import ComponentMetadata


@beartype
@dataclass(frozen=True)
class ToolParameterDefinition:
    """Describe one tool parameter for wrappers and user interfaces.

    This schema stays deliberately small and is mainly used by orchestration
    layers, including Go CLI prompts and future UI forms.
    """

    key: str
    type: str
    required: bool
    default: Any = None
    description: str = ""


@beartype
class DataPreprocessingTool:
    """Authoring template for every data-preprocessing tool in the Python API.

    This base class defines the shared contract for all tools in
    `ldt.data_preprocessing`.

    A compliant tool should:
        - expose a clear class name and `metadata` object
        - accept `**kwargs` in `fit(...)` and `preprocess(...)`
        - validate input early and fail with explicit messages
        - return Python-native outputs, not CLI-shaped dictionaries
        - keep Go-CLI translation in separate `run.py` wrappers

    Method contract:
        - `fit(**kwargs)`
          optional preparation/configuration step
        - `preprocess(**kwargs)`
          required execution step
        - `fit_preprocess(**kwargs)`
          convenience method calling `fit(...)` then `preprocess(...)`

    Example:
        ```python
        from typing import Any

        import pandas as pd
        from beartype import beartype

        from ldt.data_preprocessing import DataPreprocessingTool
        from ldt.utils.metadata import ComponentMetadata


        @beartype
        class ExamplePreprocessingTool(DataPreprocessingTool):
            metadata = ComponentMetadata(
                name="example_preprocessing_tool",
                full_name="Example Preprocessing Tool",
            )

            def fit(self, **kwargs: Any) -> "ExamplePreprocessingTool":
                self._column = str(kwargs.get("column", "x"))
                return self

            def preprocess(self, **kwargs: Any) -> pd.DataFrame:
                value = float(kwargs.get("value", 0.0))
                return pd.DataFrame({self._column: [value]})
        ```
    """

    metadata = ComponentMetadata(
        name="base",
        full_name="Data Preprocessing Tool",
    )

    @classmethod
    @beartype
    def params_definition(cls) -> tuple[ToolParameterDefinition, ...]:
        """Return an optional parameter schema for wrappers or UI discovery."""

        return ()

    @beartype
    def fit(self, **kwargs: Any) -> DataPreprocessingTool:
        """Run an optional fitting or configuration step.

        Args:
            **kwargs (Any): Tool-specific configuration values supplied by the
                caller. There is no stage-wide fixed key set at this level.
                Every concrete preprocessing tool must document supported keys
                and value types in its own `fit(...)` docstring.

        Returns:
            DataPreprocessingTool: The fitted tool instance.
        """

        _ = kwargs
        return self

    @beartype
    def preprocess(self, **kwargs: Any) -> Any:
        """Execute preprocessing logic.

        Args:
            **kwargs (Any): Tool-specific runtime values. There is no shared
                key schema at the base-class level. Every concrete tool must
                document expected keys in its own `preprocess(...)` docstring.

        Returns:
            Any: Tool-specific output payload.

        Raises:
            NotImplementedError: Always raised by the base template.
        """

        _ = kwargs
        raise NotImplementedError

    @beartype
    def fit_preprocess(self, **kwargs: Any) -> Any:
        """Run `fit(...)` then `preprocess(...)` with one shared payload.

        Args:
            **kwargs (Any): Tool-specific values forwarded unchanged to both
                `fit(...)` and `preprocess(...)`.

        Returns:
            Any: Tool-specific output payload returned by `preprocess(...)`.
        """

        self.fit(**kwargs)
        return self.preprocess(**kwargs)
