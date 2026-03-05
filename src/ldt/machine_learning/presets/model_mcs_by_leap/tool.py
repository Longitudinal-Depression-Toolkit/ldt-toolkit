from __future__ import annotations

from typing import Any

from beartype import beartype

from ldt.utils.errors import InputValidationError
from ldt.utils.metadata import ComponentMetadata
from ldt.utils.templates.tools.machine_learning import MachineLearningTool


@beartype
class ModelMCSByLEAP(MachineLearningTool):
    """Python API placeholder for the incoming Model MCS by LEAP preset.

    Reserved for downstream modelling on prepared/preprocessed MCS artefacts.
    The preset is not implemented yet.

    Examples:
        ```python
        from ldt.machine_learning import ModelMCSByLEAP

        model = ModelMCSByLEAP().fit()
        # model.predict(...) currently raises InputValidationError
        ```
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
        """Accept placeholder kwargs and return `self` (no-op).

        Args:
            **kwargs: Reserved for future model preset configuration keys.

        Returns:
            ModelMCSByLEAP: The preset instance.
        """

        _ = kwargs
        return self

    @beartype
    def predict(self, **kwargs: Any) -> dict[str, Any]:
        """Raise `InputValidationError` until the preset is implemented.

        Args:
            **kwargs: Reserved for future runtime/prediction parameters.

        Returns:
            dict[str, Any]: Reserved output payload type for future versions.

        Raises:
            InputValidationError: Always, until the preset is implemented.
        """

        _ = kwargs
        raise InputValidationError(
            "Preset `model_mcs_by_leap` is incoming and not available yet."
        )
