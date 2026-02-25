from __future__ import annotations

from typing import Any

from beartype import beartype

from ldt.utils.errors import InputValidationError
from ldt.utils.metadata import ComponentMetadata
from ldt.utils.templates.tools.data_preprocessing import DataPreprocessingTool


@beartype
class PreprocessMCSByLEAP(DataPreprocessingTool):
    """Placeholder API class for the incoming LEAP preprocessing preset.

    This preset is reserved for a full LEAP-oriented preprocessing pipeline that
    will transform prepared longitudinal datasets into ML-ready datasets.
    """

    metadata = ComponentMetadata(
        name="preprocess_mcs_by_leap",
        full_name="Preprocess MCS by LEAP",
        abstract_description=(
            "Incoming preset for preprocessing prepared longitudinal datasets "
            "into ML-ready datasets."
        ),
    )

    @beartype
    def fit(self, **kwargs: Any) -> PreprocessMCSByLEAP:
        """Accept and store preset configuration once implemented.

        Args:
            **kwargs (Any): Placeholder for future preset configuration keys.
                The concrete key schema is not yet finalised.

        Returns:
            PreprocessMCSByLEAP: The preset instance.
        """

        _ = kwargs
        return self

    @beartype
    def preprocess(self, **kwargs: Any) -> dict[str, Any]:
        """Run the preset once implementation is available.

        Args:
            **kwargs (Any): Placeholder for future runtime keys. No keys are
                currently accepted because this preset is not implemented yet.

        Returns:
            dict[str, Any]: Reserved return type for the future preset payload.
        """

        _ = kwargs
        raise InputValidationError(
            "Preset `preprocess_mcs_by_leap` is incoming and not available yet."
        )
