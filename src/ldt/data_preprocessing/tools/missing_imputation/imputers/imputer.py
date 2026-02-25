from __future__ import annotations

from pathlib import Path
from typing import Any

from beartype import beartype

from ldt.utils.metadata import ComponentMetadata


@beartype
class MissingImputer:
    """Base interface for concrete missing-data imputers.

    Concrete imputers should document all supported `**kwargs` keys in both
    `fit(...)` and `impute(...)`.
    """

    metadata = ComponentMetadata(
        name="base",
        full_name="Missing Imputer",
    )

    @beartype
    def fit(
        self, *, input_path: Path, output_path: Path, **kwargs: Any
    ) -> MissingImputer:
        """Validate and store imputer-specific configuration.

        Args:
            input_path (Path): Input dataset path.
            output_path (Path): Output dataset path.
            **kwargs (Any): Imputer-specific configuration values. No shared
                key schema is enforced by the base class.

        Returns:
            MissingImputer: The fitted imputer instance.
        """

        _ = input_path, output_path, kwargs
        return self

    @beartype
    def impute(self, *, input_path: Path, output_path: Path, **kwargs: Any) -> Any:
        """Execute imputation and return a typed result payload.

        Args:
            input_path (Path): Input dataset path.
            output_path (Path): Output dataset path.
            **kwargs (Any): Imputer-specific runtime values.

        Returns:
            Any: Imputer-specific result payload.
        """

        _ = input_path, output_path, kwargs
        raise NotImplementedError

    @beartype
    def fit_impute(self, *, input_path: Path, output_path: Path, **kwargs: Any) -> Any:
        """Run `fit(...)` then `impute(...)` using one shared payload.

        Args:
            input_path (Path): Input dataset path.
            output_path (Path): Output dataset path.
            **kwargs (Any): Imputer-specific values forwarded to both methods.

        Returns:
            Any: Imputer-specific result payload returned by `impute(...)`.
        """

        self.fit(input_path=input_path, output_path=output_path, **kwargs)
        return self.impute(input_path=input_path, output_path=output_path, **kwargs)
