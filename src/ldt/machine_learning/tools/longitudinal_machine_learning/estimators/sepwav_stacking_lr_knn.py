from __future__ import annotations

from ldt.machine_learning.tools.standard_machine_learning.estimators.knn import (
    KNNEstimatorTemplate,
)
from ldt.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class SepWavStackingLRKNNLongitudinalEstimatorTemplate(LongitudinalEstimatorTemplate):
    """Template definition for `sepwav_stacking_lr__knn` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="sepwav_stacking_lr__knn",
        full_name="SepWav (Stacking + Logistic Regression) + K-Nearest Neighbours",
        abstract_description=(
            "[Longitudinal] Train one base estimator per wave and stack with Logistic Regression. Base estimator: "
            "K-Nearest Neighbours."
        ),
    )
    strategy_key = "sepwav_stacking_lr"
    base_estimator_key = "knn"
    _builder = make_standard_primitive_builder(
        primitive_key="sepwav_stacking_lr",
        standard_template=KNNEstimatorTemplate,
    )
