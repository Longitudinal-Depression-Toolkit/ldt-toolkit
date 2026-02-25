from __future__ import annotations

from scikit_longitudinal.estimators.ensemble.lexicographical.lexico_random_forest import (
    LexicoRandomForestClassifier,
)

from ldt.utils.metadata import ComponentMetadata

from .base import (
    LongitudinalEstimatorTemplate,
    make_merwav_time_plus_longitudinal_builder,
)


class MerWavTimePlusLexicoRandomForestLongitudinalEstimatorTemplate(
    LongitudinalEstimatorTemplate
):
    """Template definition for `merwav_time_plus__lexico_random_forest` longitudinal estimator."""

    metadata = ComponentMetadata(
        name="merwav_time_plus__lexico_random_forest",
        full_name="MerWavTimePlus + Lexicographical Random Forest",
        abstract_description=(
            "[Longitudinal] Preserve temporal dependency and train a "
            "longitudinal-data-aware estimator. Longitudinal algorithm: "
            "Lexicographical Random Forest. Longitudinal random-forest ensemble with lexicographical splits."
        ),
    )
    strategy_key = "merwav_time_plus"
    base_estimator_key = "lexico_random_forest"
    _builder = make_merwav_time_plus_longitudinal_builder(
        estimator_cls=LexicoRandomForestClassifier
    )
