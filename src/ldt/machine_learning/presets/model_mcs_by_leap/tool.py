from __future__ import annotations

from typing import Any

from beartype import beartype

from ldt.utils.errors import InputValidationError
from ldt.utils.metadata import ComponentMetadata
from ldt.utils.templates.tools.machine_learning import MachineLearningTool


@beartype
class ModelMCSByLEAP(MachineLearningTool):
    """Placeholder API class for the incoming LEAP machine-learning preset.

    This preset is reserved for a full LEAP-oriented machine-learning pipeline
    that will model prepared and preprocessed MCS datasets.
    """

    metadata = ComponentMetadata(
        name="model_mcs_by_leap",
        full_name="Model MCS by LEAP",
        abstract_description=(
            "Incoming preset for running machine-learning workflows on prepared "
            "and preprocessed MCS datasets."
        ),
    )

    @beartype
    def fit(self, **kwargs: Any) -> ModelMCSByLEAP:
        """Accept and store preset configuration once implemented.

        Args:
            **kwargs (Any): Placeholder for future preset configuration keys.
                The concrete key schema is not yet finalised.

        Returns:
            ModelMCSByLEAP: The preset instance.
        """

        _ = kwargs
        return self

    @beartype
    def predict(self, **kwargs: Any) -> dict[str, Any]:
        """Run the preset once implementation is available.

        Args:
            **kwargs (Any): Placeholder for future runtime keys. No keys are
                currently accepted because this preset is not implemented yet.

        Returns:
            dict[str, Any]: Reserved return type for the future preset payload.
        """

        _ = kwargs
        raise InputValidationError(
            "Preset `model_mcs_by_leap` is incoming and not available yet."
        )
