from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar

from beartype import beartype

from ldt.utils.errors import InputValidationError
from ldt.utils.metadata import ComponentMetadata

SelfType = TypeVar("SelfType", bound="MachineLearningTool")


@beartype
@dataclass(frozen=True)
class ToolParameterDefinition:
    """Describe one machine-learning tool parameter for wrappers and user interfaces.

    This schema is intentionally lightweight and is mainly consumed by external
    orchestration layers such as the Go CLI.
    """

    key: str
    type: str
    required: bool
    default: Any = None
    description: str = ""


@beartype
class MachineLearningTool:
    """Authoring template for every machine-learning tool in the Python API.

    A compliant machine-learning tool should:
        - expose a clear class name and a meaningful `metadata` object
        - accept `**kwargs` in `fit(...)`, `predict(...)`, and optionally
          `predict_proba(...)`
        - validate input early and fail with explicit error messages
        - return Python-native payloads for library users
        - keep Go CLI translation and bridge formatting in separate `run.py`
          wrappers
    """

    metadata = ComponentMetadata(
        name="base",
        full_name="Machine Learning Tool",
    )

    @classmethod
    @beartype
    def params_definition(cls) -> tuple[ToolParameterDefinition, ...]:
        """Return an optional parameter schema for wrappers or UI discovery."""

        return ()

    @beartype
    def fit(self: SelfType, **kwargs: Any) -> SelfType:
        """Run an optional fitting or configuration step.

        Args:
            **kwargs (Any): Tool-specific configuration values supplied by the
                caller. There is no stage-wide fixed key set at this level.
                Every concrete tool must document supported keys and value types
                in its own `fit(...)` docstring.

        Returns:
            SelfType: The fitted tool instance.
        """

        _ = kwargs
        return self

    @beartype
    def predict(self, **kwargs: Any) -> Any:
        """Execute the primary machine-learning behaviour.

        Args:
            **kwargs (Any): Tool-specific runtime values.

        Returns:
            Any: Tool-specific output payload.

        Raises:
            NotImplementedError: Always raised by the base template.
        """

        _ = kwargs
        raise NotImplementedError

    @beartype
    def predict_proba(self, **kwargs: Any) -> Any:
        """Execute probability-oriented machine-learning behaviour.

        Args:
            **kwargs (Any): Tool-specific runtime values.

        Returns:
            Any: Tool-specific probability output payload.

        Raises:
            InputValidationError: Always raised by default. Concrete tools can
                override this when probability outputs are supported.
        """

        _ = kwargs
        raise InputValidationError(
            f"`{self.__class__.__name__}` does not provide `predict_proba(...)`."
        )

    @beartype
    def fit_predict(self, **kwargs: Any) -> Any:
        """Run `fit(...)` then `predict(...)` with one shared payload.

        Args:
            **kwargs (Any): Tool-specific values forwarded unchanged to both
                `fit(...)` and `predict(...)`.

        Returns:
            Any: Tool-specific output payload returned by `predict(...)`.
        """

        self.fit(**kwargs)
        return self.predict(**kwargs)

    @beartype
    def fit_predict_proba(self, **kwargs: Any) -> Any:
        """Run `fit(...)` then `predict_proba(...)` with one shared payload.

        Args:
            **kwargs (Any): Tool-specific values forwarded unchanged to both
                `fit(...)` and `predict_proba(...)`.

        Returns:
            Any: Tool-specific output payload returned by `predict_proba(...)`.
        """

        self.fit(**kwargs)
        return self.predict_proba(**kwargs)
