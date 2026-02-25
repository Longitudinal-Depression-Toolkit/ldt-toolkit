from __future__ import annotations

from ldt.machine_learning.tools.standard_machine_learning.estimators.svm import (
    SVMEstimatorTemplate,
)
from ldt.utils.metadata import ComponentMetadata

from .base import LongitudinalEstimatorTemplate, make_standard_primitive_builder


class SepWavVotingSVMLongitudinalEstimatorTemplate(LongitudinalEstimatorTemplate):
    """Template definition for `sepwav_voting__svm` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="sepwav_voting__svm",
        full_name="SepWav (Voting) + Support Vector Machine",
        abstract_description=(
            "[Longitudinal] Train one base estimator per wave and combine by voting. Base estimator: "
            "Support Vector Machine."
        ),
    )
    strategy_key = "sepwav_voting"
    base_estimator_key = "svm"
    _builder = make_standard_primitive_builder(
        primitive_key="sepwav_voting",
        standard_template=SVMEstimatorTemplate,
    )
