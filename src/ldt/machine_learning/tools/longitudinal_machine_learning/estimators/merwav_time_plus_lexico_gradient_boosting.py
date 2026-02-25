from __future__ import annotations

from scikit_longitudinal.estimators.ensemble.lexicographical.lexico_gradient_boosting import (
    LexicoGradientBoostingClassifier,
)

from ldt.utils.metadata import ComponentMetadata

from .base import (
    LongitudinalEstimatorTemplate,
    make_merwav_time_plus_longitudinal_builder,
)


class MerWavTimePlusLexicoGradientBoostingLongitudinalEstimatorTemplate(
    LongitudinalEstimatorTemplate
):
    """Template definition for `merwav_time_plus__lexico_gradient_boosting` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="merwav_time_plus__lexico_gradient_boosting",
        full_name="MerWavTimePlus + Lexicographical Gradient Boosting",
        abstract_description=(
            "[Longitudinal] Preserve temporal dependency and train a "
            "longitudinal-data-aware estimator. Longitudinal algorithm: "
            "Lexicographical Gradient Boosting. Longitudinal gradient boosting with lexicographical split logic."
        ),
    )
    strategy_key = "merwav_time_plus"
    base_estimator_key = "lexico_gradient_boosting"
    _builder = make_merwav_time_plus_longitudinal_builder(
        estimator_cls=LexicoGradientBoostingClassifier
    )
