from __future__ import annotations

from ldt.machine_learning.tools.standard_machine_learning.estimators.gradient_boosting import (
    GradientBoostingEstimatorTemplate,
)
from ldt.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class SepWavStackingDTGradientBoostingLongitudinalEstimatorTemplate(
    LongitudinalEstimatorTemplate
):
    """Template definition for `sepwav_stacking_dt__gradient_boosting` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="sepwav_stacking_dt__gradient_boosting",
        full_name="SepWav (Stacking + Decision Tree) + Gradient Boosting",
        abstract_description=(
            "[Longitudinal] Train one base estimator per wave and stack with Decision Tree. Base estimator: "
            "Gradient Boosting."
        ),
    )
    strategy_key = "sepwav_stacking_dt"
    base_estimator_key = "gradient_boosting"
    _builder = make_standard_primitive_builder(
        primitive_key="sepwav_stacking_dt",
        standard_template=GradientBoostingEstimatorTemplate,
    )
