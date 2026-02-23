from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.machine_learning.catalog import resolve_technique_with_defaults
from src.machine_learning.support.inputs import (
    as_bool,
    as_optional_string,
    as_required_int,
    as_required_string,
    load_input_dataset,
    parse_metric_keys,
    parse_validation_split,
    resolve_feature_columns,
    resolve_target_column,
    run_with_validation,
)
from src.machine_learning.support.runtime_validation import (
    validate_cv_split_feasibility,
    validate_one_vs_rest_cv_folds,
)
from src.machine_learning.tools.longitudinal_machine_learning.discovery import (
    discover_longitudinal_estimators,
    list_longitudinal_strategies,
)
from src.machine_learning.tools.longitudinal_machine_learning.inputs import (
    LongitudinalFeatureInputPrompter,
)
from src.machine_learning.tools.longitudinal_machine_learning.target_encoding import (
    LongitudinalTargetEncoder,
)
from src.machine_learning.tools.metrics import list_supported_metrics
from src.machine_learning.tools.target_scenarios import TargetScenarioPlanner
from src.machine_learning.tools.templates import (
    ClassificationExperimentTemplate,
)
from src.utils.errors import InputValidationError
from src.utils.metadata import resolve_component_metadata


def run_longitudinal_machine_learning_tool(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Run longitudinal-machine-learning workflows for temporal tabular data.

    This tool trains or inspects estimators that use longitudinal structure via
    `feature_groups` and `non_longitudinal_features`.

    Available estimator keys:

    | Estimator key | Family | Base estimator |
    | --- | --- | --- |
    | `aggrfunc_mean__decision_tree` | AggrFunc (Mean) | Decision Tree |
    | `aggrfunc_mean__extra_trees` | AggrFunc (Mean) | Extra Trees |
    | `aggrfunc_mean__gradient_boosting` | AggrFunc (Mean) | Gradient Boosting |
    | `aggrfunc_mean__knn` | AggrFunc (Mean) | K-Nearest Neighbours |
    | `aggrfunc_mean__logistic_regression` | AggrFunc (Mean) | Logistic Regression |
    | `aggrfunc_mean__random_forest` | AggrFunc (Mean) | Random Forest |
    | `aggrfunc_mean__svm` | AggrFunc (Mean) | Support Vector Machine |
    | `aggrfunc_median__decision_tree` | AggrFunc (Median) | Decision Tree |
    | `aggrfunc_median__extra_trees` | AggrFunc (Median) | Extra Trees |
    | `aggrfunc_median__gradient_boosting` | AggrFunc (Median) | Gradient Boosting |
    | `aggrfunc_median__knn` | AggrFunc (Median) | K-Nearest Neighbours |
    | `aggrfunc_median__logistic_regression` | AggrFunc (Median) | Logistic Regression |
    | `aggrfunc_median__random_forest` | AggrFunc (Median) | Random Forest |
    | `aggrfunc_median__svm` | AggrFunc (Median) | Support Vector Machine |
    | `merwav_time_minus__decision_tree` | MerWavTimeMinus | Decision Tree |
    | `merwav_time_minus__extra_trees` | MerWavTimeMinus | Extra Trees |
    | `merwav_time_minus__gradient_boosting` | MerWavTimeMinus | Gradient Boosting |
    | `merwav_time_minus__knn` | MerWavTimeMinus | K-Nearest Neighbours |
    | `merwav_time_minus__logistic_regression` | MerWavTimeMinus | Logistic Regression |
    | `merwav_time_minus__random_forest` | MerWavTimeMinus | Random Forest |
    | `merwav_time_minus__svm` | MerWavTimeMinus | Support Vector Machine |
    | `sepwav_stacking_dt__decision_tree` | SepWav (Stacking + Decision Tree) | Decision Tree |
    | `sepwav_stacking_dt__extra_trees` | SepWav (Stacking + Decision Tree) | Extra Trees |
    | `sepwav_stacking_dt__gradient_boosting` | SepWav (Stacking + Decision Tree) | Gradient Boosting |
    | `sepwav_stacking_dt__knn` | SepWav (Stacking + Decision Tree) | K-Nearest Neighbours |
    | `sepwav_stacking_dt__logistic_regression` | SepWav (Stacking + Decision Tree) | Logistic Regression |
    | `sepwav_stacking_dt__random_forest` | SepWav (Stacking + Decision Tree) | Random Forest |
    | `sepwav_stacking_dt__svm` | SepWav (Stacking + Decision Tree) | Support Vector Machine |
    | `sepwav_stacking_lr__decision_tree` | SepWav (Stacking + Logistic Regression) | Decision Tree |
    | `sepwav_stacking_lr__extra_trees` | SepWav (Stacking + Logistic Regression) | Extra Trees |
    | `sepwav_stacking_lr__gradient_boosting` | SepWav (Stacking + Logistic Regression) | Gradient Boosting |
    | `sepwav_stacking_lr__knn` | SepWav (Stacking + Logistic Regression) | K-Nearest Neighbours |
    | `sepwav_stacking_lr__logistic_regression` | SepWav (Stacking + Logistic Regression) | Logistic Regression |
    | `sepwav_stacking_lr__random_forest` | SepWav (Stacking + Logistic Regression) | Random Forest |
    | `sepwav_stacking_lr__svm` | SepWav (Stacking + Logistic Regression) | Support Vector Machine |
    | `sepwav_voting__decision_tree` | SepWav (Voting) | Decision Tree |
    | `sepwav_voting__extra_trees` | SepWav (Voting) | Extra Trees |
    | `sepwav_voting__gradient_boosting` | SepWav (Voting) | Gradient Boosting |
    | `sepwav_voting__knn` | SepWav (Voting) | K-Nearest Neighbours |
    | `sepwav_voting__logistic_regression` | SepWav (Voting) | Logistic Regression |
    | `sepwav_voting__random_forest` | SepWav (Voting) | Random Forest |
    | `sepwav_voting__svm` | SepWav (Voting) | Support Vector Machine |
    | `merwav_time_plus__lexico_decision_tree` | MerWavTimePlus | Lexicographical Decision Tree |
    | `merwav_time_plus__lexico_random_forest` | MerWavTimePlus | Lexicographical Random Forest |
    | `merwav_time_plus__lexico_deep_forest` | MerWavTimePlus | Lexicographical Deep Forest |
    | `merwav_time_plus__lexico_gradient_boosting` | MerWavTimePlus | Lexicographical Gradient Boosting |
    | `merwav_time_plus__nested_trees` | MerWavTimePlus | Nested Trees |

    Total default longitudinal estimator roster: 47 estimators.

    Supported techniques:

    | Technique | What it does |
    | --- | --- |
    | `list_estimators` | Returns strategies and estimator roster metadata. |
    | `list_metrics` | Returns supported scoring metrics. |
    | `run_experiment` | Trains/evaluates one selected longitudinal estimator. |

    Target support:

    | Target case | Supported | Behaviour |
    | --- | --- | --- |
    | Binary classification | Yes | Runs one experiment on the provided binary labels. |
    | Multi-class classification | Yes (via one-vs-rest) | Builds one binary scenario per class and runs one experiment per scenario. |
    | Single-class target | No | Raises `InputValidationError`. |

    Args:
        technique (str): Workflow technique (`list_estimators`,
            `list_metrics`, or `run_experiment`).
        params (Mapping[str, Any]): Configuration for the selected technique,
            including longitudinal feature-group definitions for training mode.

    Returns:
        dict[str, Any]: Listing output for discovery techniques, or experiment
            metrics/artifact paths for `run_experiment`.

    Examples:
        ```python
        from ldt.machine_learning.tools.longitudinal_machine_learning.run import run_longitudinal_machine_learning_tool

        result = run_longitudinal_machine_learning_tool(
            technique="run_experiment",
            params={
                "input_path": "./data/millennium_longitudinal.csv",
                "target_column": "depression_status",
                "feature_columns": "mood_w1,mood_w2,mood_w3,sleep_w1,sleep_w2,sleep_w3,sex",
                "feature_groups": "[[0,1,2],[3,4,5]]",
                "non_longitudinal_features": "[6]",
                "estimator_key": "merwav_time_plus__lexico_random_forest",
                "metric_keys": "accuracy,f1_macro",
                "cv_folds": 5,
                "validation_split": "none",
                "random_seed": 42,
            },
        )
        ```
    """

    return run_with_validation(
        lambda: _run_longitudinal_machine_learning_tool(
            technique=technique, params=params
        )
    )


def _run_longitudinal_machine_learning_tool(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    _, resolved = resolve_technique_with_defaults(
        section_key="longitudinal_machine_learning",
        technique_id=technique,
        provided_params=dict(params),
    )

    mode = technique.strip().lower().replace("-", "_")
    if mode == "list_estimators":
        estimators = discover_longitudinal_estimators()
        strategies = list_longitudinal_strategies()
        return {
            "strategies": [
                {
                    "key": strategy.key,
                    "name": strategy.full_name,
                    "description": strategy.description,
                    "uses_standard_base_estimator": strategy.uses_standard_base_estimator,
                }
                for strategy in strategies
            ],
            "estimators": [
                {
                    "key": key,
                    "name": resolve_component_metadata(template).full_name,
                    "description": resolve_component_metadata(
                        template
                    ).abstract_description,
                    "strategy_key": template.strategy_key,
                    "base_estimator_key": template.base_estimator_key,
                }
                for key, template in estimators.items()
            ],
            "count": len(estimators),
        }

    if mode == "list_metrics":
        metrics = list_supported_metrics()
        return {
            "metrics": [
                {
                    "key": metric.key,
                    "source": metric.source,
                    "description": metric.description,
                }
                for metric in metrics
            ],
            "count": len(metrics),
        }

    if mode != "run_experiment":
        raise InputValidationError(
            f"Unsupported longitudinal-machine-learning technique: {technique}"
        )

    estimators = discover_longitudinal_estimators()
    if not estimators:
        raise InputValidationError("No longitudinal estimators were discovered.")

    estimator_key = as_optional_string(resolved, "estimator_key")
    if estimator_key is None:
        estimator_key = next(iter(estimators.keys()))
    if estimator_key not in estimators:
        available = ", ".join(estimators.keys())
        raise InputValidationError(
            f"Unknown estimator key `{estimator_key}`. Available: {available}"
        )
    estimator_template = estimators[estimator_key]
    estimator_name = resolve_component_metadata(estimator_template).full_name

    input_path = Path(as_required_string(resolved, "input_path")).expanduser()
    dataset = load_input_dataset(input_path=input_path)

    target_column = resolve_target_column(
        dataset=dataset,
        requested=as_required_string(resolved, "target_column"),
    )
    feature_columns = resolve_feature_columns(
        dataset=dataset,
        target_column=target_column,
        raw_feature_columns=as_required_string(resolved, "feature_columns"),
    )

    feature_vectors = LongitudinalFeatureInputPrompter.resolve_feature_vectors(
        feature_columns=feature_columns,
        feature_groups_raw=as_required_string(resolved, "feature_groups"),
        non_longitudinal_raw=as_required_string(resolved, "non_longitudinal_features"),
    )

    modelling_data = dataset[[*feature_columns, target_column]].dropna(
        subset=[target_column]
    )
    if modelling_data.empty:
        raise InputValidationError(
            "No rows available after dropping missing target values."
        )

    X = modelling_data[feature_columns].copy()
    y_raw = modelling_data[target_column].copy()

    metric_keys = parse_metric_keys(as_required_string(resolved, "metric_keys"))
    cv_folds = as_required_int(resolved, "cv_folds", minimum=2)
    validation_split = parse_validation_split(
        as_required_string(resolved, "validation_split"),
        cv_folds=cv_folds,
    )
    random_seed = as_required_int(resolved, "random_seed")
    silent_training_output = as_bool(
        resolved.get("silent_training_output", True),
        field_name="silent_training_output",
    )

    output_dir_raw = as_optional_string(resolved, "output_dir")
    output_dir = (
        Path(output_dir_raw).expanduser()
        if output_dir_raw is not None
        else Path(f"outputs/longitudinal_ml/{estimator_key}").expanduser()
    )

    unique_class_count = int(y_raw.nunique(dropna=True))
    if unique_class_count < 2:
        raise InputValidationError(
            "Classification requires at least two distinct target classes."
        )

    experiment = ClassificationExperimentTemplate()

    if unique_class_count > 2:
        scenarios = TargetScenarioPlanner.build_one_vs_rest_scenarios(y=y_raw)
        validate_one_vs_rest_cv_folds(
            cv_folds=cv_folds,
            validation_split=validation_split,
            scenarios=scenarios,
        )

        scenario_results: list[dict[str, Any]] = []
        for scenario in scenarios:
            estimator = estimator_template.build_estimator(
                random_seed=random_seed,
                feature_groups=feature_vectors.feature_groups,
                non_longitudinal_features=feature_vectors.non_longitudinal_features,
                feature_list_names=tuple(feature_columns),
            )
            scenario_output_dir = output_dir / scenario.scenario_key
            result = experiment.run(
                estimator=estimator,
                estimator_name=estimator_name,
                X=X,
                y=scenario.y_binary,
                metric_keys=metric_keys,
                output_dir=scenario_output_dir,
                cv_folds=cv_folds,
                random_seed=random_seed,
                validation_split=validation_split,
                silent_training_output=silent_training_output,
            )
            scenario_results.append(
                {
                    "scenario_key": scenario.scenario_key,
                    "scenario_label": scenario.scenario_label,
                    "output_dir": str(scenario_output_dir.resolve()),
                    "metric_key": result.metric_key,
                    "mean_score": result.mean_score,
                    "std_score": result.std_score,
                    "model_path": str(result.artifacts.model_path.resolve()),
                    "summary_path": str(result.artifacts.summary_path.resolve()),
                }
            )

        return {
            "mode": "one_vs_rest",
            "estimator_key": estimator_key,
            "estimator_name": estimator_name,
            "scenario_count": len(scenario_results),
            "scenarios": scenario_results,
            "output_root_dir": str(output_dir.resolve()),
            "dropped_rows": int(len(dataset) - len(modelling_data)),
        }

    target_encoding = LongitudinalTargetEncoder.encode_if_needed(y=y_raw)
    y = target_encoding.encoded_target

    validate_cv_split_feasibility(
        y=y,
        cv_folds=cv_folds,
        validation_split=validation_split,
    )

    estimator = estimator_template.build_estimator(
        random_seed=random_seed,
        feature_groups=feature_vectors.feature_groups,
        non_longitudinal_features=feature_vectors.non_longitudinal_features,
        feature_list_names=tuple(feature_columns),
    )
    result = experiment.run(
        estimator=estimator,
        estimator_name=estimator_name,
        X=X,
        y=y,
        metric_keys=metric_keys,
        output_dir=output_dir,
        cv_folds=cv_folds,
        random_seed=random_seed,
        validation_split=validation_split,
        silent_training_output=silent_training_output,
    )

    return {
        "mode": "multiclass_or_binary",
        "estimator_key": estimator_key,
        "estimator_name": estimator_name,
        "metric_key": result.metric_key,
        "mean_score": result.mean_score,
        "std_score": result.std_score,
        "output_dir": str(output_dir.resolve()),
        "model_path": str(result.artifacts.model_path.resolve()),
        "summary_path": str(result.artifacts.summary_path.resolve()),
        "report_path": str(result.artifacts.report_path.resolve()),
        "target_was_encoded": target_encoding.was_encoded,
        "target_label_mapping": [
            {"encoded": encoded, "original": label}
            for encoded, label in target_encoding.label_mapping
        ],
        "dropped_rows": int(len(dataset) - len(modelling_data)),
    }
