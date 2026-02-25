from .aggrfunc_mean_decision_tree import (
    AggrFuncMeanDecisionTreeLongitudinalEstimatorTemplate,
)
from .aggrfunc_mean_extra_trees import (
    AggrFuncMeanExtraTreesLongitudinalEstimatorTemplate,
)
from .aggrfunc_mean_gradient_boosting import (
    AggrFuncMeanGradientBoostingLongitudinalEstimatorTemplate,
)
from .aggrfunc_mean_knn import AggrFuncMeanKNNLongitudinalEstimatorTemplate
from .aggrfunc_mean_logistic_regression import (
    AggrFuncMeanLogisticRegressionLongitudinalEstimatorTemplate,
)
from .aggrfunc_mean_random_forest import (
    AggrFuncMeanRandomForestLongitudinalEstimatorTemplate,
)
from .aggrfunc_mean_svm import AggrFuncMeanSVMLongitudinalEstimatorTemplate
from .aggrfunc_median_decision_tree import (
    AggrFuncMedianDecisionTreeLongitudinalEstimatorTemplate,
)
from .aggrfunc_median_extra_trees import (
    AggrFuncMedianExtraTreesLongitudinalEstimatorTemplate,
)
from .aggrfunc_median_gradient_boosting import (
    AggrFuncMedianGradientBoostingLongitudinalEstimatorTemplate,
)
from .aggrfunc_median_knn import AggrFuncMedianKNNLongitudinalEstimatorTemplate
from .aggrfunc_median_logistic_regression import (
    AggrFuncMedianLogisticRegressionLongitudinalEstimatorTemplate,
)
from .aggrfunc_median_random_forest import (
    AggrFuncMedianRandomForestLongitudinalEstimatorTemplate,
)
from .aggrfunc_median_svm import AggrFuncMedianSVMLongitudinalEstimatorTemplate
from .base import (
    LongitudinalEstimatorTemplate,
    LongitudinalStrategyDefinition,
    list_longitudinal_strategies,
)
from .merwav_time_minus_decision_tree import (
    MerWavTimeMinusDecisionTreeLongitudinalEstimatorTemplate,
)
from .merwav_time_minus_extra_trees import (
    MerWavTimeMinusExtraTreesLongitudinalEstimatorTemplate,
)
from .merwav_time_minus_gradient_boosting import (
    MerWavTimeMinusGradientBoostingLongitudinalEstimatorTemplate,
)
from .merwav_time_minus_knn import MerWavTimeMinusKNNLongitudinalEstimatorTemplate
from .merwav_time_minus_logistic_regression import (
    MerWavTimeMinusLogisticRegressionLongitudinalEstimatorTemplate,
)
from .merwav_time_minus_random_forest import (
    MerWavTimeMinusRandomForestLongitudinalEstimatorTemplate,
)
from .merwav_time_minus_svm import MerWavTimeMinusSVMLongitudinalEstimatorTemplate
from .merwav_time_plus_lexico_decision_tree import (
    MerWavTimePlusLexicoDecisionTreeLongitudinalEstimatorTemplate,
)
from .merwav_time_plus_lexico_deep_forest import (
    MerWavTimePlusLexicoDeepForestLongitudinalEstimatorTemplate,
)
from .merwav_time_plus_lexico_gradient_boosting import (
    MerWavTimePlusLexicoGradientBoostingLongitudinalEstimatorTemplate,
)
from .merwav_time_plus_lexico_random_forest import (
    MerWavTimePlusLexicoRandomForestLongitudinalEstimatorTemplate,
)
from .merwav_time_plus_nested_trees import (
    MerWavTimePlusNestedTreesLongitudinalEstimatorTemplate,
)
from .sepwav_stacking_dt_decision_tree import (
    SepWavStackingDTDecisionTreeLongitudinalEstimatorTemplate,
)
from .sepwav_stacking_dt_extra_trees import (
    SepWavStackingDTExtraTreesLongitudinalEstimatorTemplate,
)
from .sepwav_stacking_dt_gradient_boosting import (
    SepWavStackingDTGradientBoostingLongitudinalEstimatorTemplate,
)
from .sepwav_stacking_dt_knn import SepWavStackingDTKNNLongitudinalEstimatorTemplate
from .sepwav_stacking_dt_logistic_regression import (
    SepWavStackingDTLogisticRegressionLongitudinalEstimatorTemplate,
)
from .sepwav_stacking_dt_random_forest import (
    SepWavStackingDTRandomForestLongitudinalEstimatorTemplate,
)
from .sepwav_stacking_dt_svm import SepWavStackingDTSVMLongitudinalEstimatorTemplate
from .sepwav_stacking_lr_decision_tree import (
    SepWavStackingLRDecisionTreeLongitudinalEstimatorTemplate,
)
from .sepwav_stacking_lr_extra_trees import (
    SepWavStackingLRExtraTreesLongitudinalEstimatorTemplate,
)
from .sepwav_stacking_lr_gradient_boosting import (
    SepWavStackingLRGradientBoostingLongitudinalEstimatorTemplate,
)
from .sepwav_stacking_lr_knn import SepWavStackingLRKNNLongitudinalEstimatorTemplate
from .sepwav_stacking_lr_logistic_regression import (
    SepWavStackingLRLogisticRegressionLongitudinalEstimatorTemplate,
)
from .sepwav_stacking_lr_random_forest import (
    SepWavStackingLRRandomForestLongitudinalEstimatorTemplate,
)
from .sepwav_stacking_lr_svm import SepWavStackingLRSVMLongitudinalEstimatorTemplate
from .sepwav_voting_decision_tree import (
    SepWavVotingDecisionTreeLongitudinalEstimatorTemplate,
)
from .sepwav_voting_extra_trees import (
    SepWavVotingExtraTreesLongitudinalEstimatorTemplate,
)
from .sepwav_voting_gradient_boosting import (
    SepWavVotingGradientBoostingLongitudinalEstimatorTemplate,
)
from .sepwav_voting_knn import SepWavVotingKNNLongitudinalEstimatorTemplate
from .sepwav_voting_logistic_regression import (
    SepWavVotingLogisticRegressionLongitudinalEstimatorTemplate,
)
from .sepwav_voting_random_forest import (
    SepWavVotingRandomForestLongitudinalEstimatorTemplate,
)
from .sepwav_voting_svm import SepWavVotingSVMLongitudinalEstimatorTemplate

__all__ = [
    "LongitudinalEstimatorTemplate",
    "LongitudinalStrategyDefinition",
    "list_longitudinal_strategies",
    "AggrFuncMeanDecisionTreeLongitudinalEstimatorTemplate",
    "AggrFuncMeanExtraTreesLongitudinalEstimatorTemplate",
    "AggrFuncMeanGradientBoostingLongitudinalEstimatorTemplate",
    "AggrFuncMeanKNNLongitudinalEstimatorTemplate",
    "AggrFuncMeanLogisticRegressionLongitudinalEstimatorTemplate",
    "AggrFuncMeanRandomForestLongitudinalEstimatorTemplate",
    "AggrFuncMeanSVMLongitudinalEstimatorTemplate",
    "AggrFuncMedianDecisionTreeLongitudinalEstimatorTemplate",
    "AggrFuncMedianExtraTreesLongitudinalEstimatorTemplate",
    "AggrFuncMedianGradientBoostingLongitudinalEstimatorTemplate",
    "AggrFuncMedianKNNLongitudinalEstimatorTemplate",
    "AggrFuncMedianLogisticRegressionLongitudinalEstimatorTemplate",
    "AggrFuncMedianRandomForestLongitudinalEstimatorTemplate",
    "AggrFuncMedianSVMLongitudinalEstimatorTemplate",
    "MerWavTimeMinusDecisionTreeLongitudinalEstimatorTemplate",
    "MerWavTimeMinusExtraTreesLongitudinalEstimatorTemplate",
    "MerWavTimeMinusGradientBoostingLongitudinalEstimatorTemplate",
    "MerWavTimeMinusKNNLongitudinalEstimatorTemplate",
    "MerWavTimeMinusLogisticRegressionLongitudinalEstimatorTemplate",
    "MerWavTimeMinusRandomForestLongitudinalEstimatorTemplate",
    "MerWavTimeMinusSVMLongitudinalEstimatorTemplate",
    "MerWavTimePlusLexicoDecisionTreeLongitudinalEstimatorTemplate",
    "MerWavTimePlusLexicoDeepForestLongitudinalEstimatorTemplate",
    "MerWavTimePlusLexicoGradientBoostingLongitudinalEstimatorTemplate",
    "MerWavTimePlusLexicoRandomForestLongitudinalEstimatorTemplate",
    "MerWavTimePlusNestedTreesLongitudinalEstimatorTemplate",
    "SepWavStackingDTDecisionTreeLongitudinalEstimatorTemplate",
    "SepWavStackingDTExtraTreesLongitudinalEstimatorTemplate",
    "SepWavStackingDTGradientBoostingLongitudinalEstimatorTemplate",
    "SepWavStackingDTKNNLongitudinalEstimatorTemplate",
    "SepWavStackingDTLogisticRegressionLongitudinalEstimatorTemplate",
    "SepWavStackingDTRandomForestLongitudinalEstimatorTemplate",
    "SepWavStackingDTSVMLongitudinalEstimatorTemplate",
    "SepWavStackingLRDecisionTreeLongitudinalEstimatorTemplate",
    "SepWavStackingLRExtraTreesLongitudinalEstimatorTemplate",
    "SepWavStackingLRGradientBoostingLongitudinalEstimatorTemplate",
    "SepWavStackingLRKNNLongitudinalEstimatorTemplate",
    "SepWavStackingLRLogisticRegressionLongitudinalEstimatorTemplate",
    "SepWavStackingLRRandomForestLongitudinalEstimatorTemplate",
    "SepWavStackingLRSVMLongitudinalEstimatorTemplate",
    "SepWavVotingDecisionTreeLongitudinalEstimatorTemplate",
    "SepWavVotingExtraTreesLongitudinalEstimatorTemplate",
    "SepWavVotingGradientBoostingLongitudinalEstimatorTemplate",
    "SepWavVotingKNNLongitudinalEstimatorTemplate",
    "SepWavVotingLogisticRegressionLongitudinalEstimatorTemplate",
    "SepWavVotingRandomForestLongitudinalEstimatorTemplate",
    "SepWavVotingSVMLongitudinalEstimatorTemplate",
]
