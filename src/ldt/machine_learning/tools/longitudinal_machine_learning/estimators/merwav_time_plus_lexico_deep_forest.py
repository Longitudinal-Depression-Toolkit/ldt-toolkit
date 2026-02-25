from __future__ import annotations

from scikit_longitudinal.estimators.ensemble.lexicographical.lexico_deep_forest import (
    LexicoDeepForestClassifier,
)

from ldt.utils.metadata import ComponentMetadata

from .base import (
    LongitudinalEstimatorTemplate,
    make_merwav_time_plus_longitudinal_builder,
)


class MerWavTimePlusLexicoDeepForestLongitudinalEstimatorTemplate(
    LongitudinalEstimatorTemplate
):
    """Template definition for `merwav_time_plus__lexico_deep_forest` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="merwav_time_plus__lexico_deep_forest",
        full_name="MerWavTimePlus + Lexicographical Deep Forest",
        abstract_description=(
            "[Longitudinal] Preserve temporal dependency and train a "
            "longitudinal-data-aware estimator. Longitudinal algorithm: "
            "Lexicographical Deep Forest. Longitudinal deep-forest ensemble with temporal-aware features."
        ),
    )
    strategy_key = "merwav_time_plus"
    base_estimator_key = "lexico_deep_forest"
    _builder = make_merwav_time_plus_longitudinal_builder(
        estimator_cls=LexicoDeepForestClassifier
    )
