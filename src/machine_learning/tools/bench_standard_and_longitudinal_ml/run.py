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
    parse_excluded_estimators,
    parse_metric_keys,
    parse_validation_split,
    resolve_feature_columns,
    resolve_notebook_path,
    resolve_target_column,
    run_with_validation,
)
from src.machine_learning.support.runtime_validation import (
    validate_cv_split_feasibility,
    validate_one_vs_rest_cv_folds,
)
from src.machine_learning.tools.longitudinal_machine_learning.discovery import (
    discover_longitudinal_estimators,
)
from src.machine_learning.tools.longitudinal_machine_learning.inputs import (
    LongitudinalFeatureInputPrompter,
)
from src.machine_learning.tools.longitudinal_machine_learning.target_encoding import (
    LongitudinalTargetEncoder,
)
from src.machine_learning.tools.metrics import list_supported_metrics
from src.machine_learning.tools.standard_machine_learning.discovery import (
    discover_standard_estimators,
)
from src.machine_learning.tools.target_scenarios import TargetScenarioPlanner
from src.machine_learning.tools.templates import (
    BenchmarkEstimatorSpec,
    ClassificationBenchmarkTemplate,
    MixedBenchmarkNotebookConfig,
    MixedBenchmarkNotebookTemplate,
)
from src.utils.errors import InputValidationError
from src.utils.metadata import resolve_component_metadata

_BENCHMARK_NAME = "bench_standard_and_longitudinal_ml"


def run_bench_standard_and_longitudinal_ml_tool(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Run a mixed benchmark across standard and longitudinal estimators.

    This tool combines cross-sectional and longitudinal estimator families in
    one benchmark workflow. It supports listing capabilities and running the
    benchmark with shared data splits, shared metrics, and unified ranking
    artifacts across both estimator groups.

    Exact estimator keys benchmarked by `run_benchmark`
    (before applying `excluded_estimators`):

    Standard estimators:

    | Estimator key | Estimator name |
    | --- | --- |
    | `decision_tree` | Decision Tree |
    | `extra_trees` | Extra Trees |
    | `gradient_boosting` | Gradient Boosting |
    | `knn` | K-Nearest Neighbours |
    | `logistic_regression` | Logistic Regression |
    | `random_forest` | Random Forest |
    | `svm` | Support Vector Machine |

    Longitudinal estimators:

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

    Total default mixed benchmark roster: 54 estimators
    (7 standard + 47 longitudinal).

    Supported techniques:

    | Technique | What it does |
    | --- | --- |
    | `list_estimators` | Returns both estimator rosters (`standard_estimators`, `longitudinal_estimators`). |
    | `list_metrics` | Returns supported scoring metrics. |
    | `run_benchmark` | Trains/evaluates included estimators, ranks them, and writes artifacts. |

    Target support:

    | Target case | Supported | Behaviour |
    | --- | --- | --- |
    | Binary classification | Yes | Runs one mixed benchmark on the encoded binary target. |
    | Multi-class classification | Yes (via one-vs-rest) | Builds one binary scenario per class and runs one mixed benchmark per scenario. |
    | Single-class target | No | Raises `InputValidationError`. |

    In multi-class mode the workflow applies one-vs-rest scenario generation to
    ensure longitudinal estimators can be evaluated consistently alongside
    standard estimators.

    Args:
        technique (str): Benchmark mode to run (`list_estimators`,
            `list_metrics`, or `run_benchmark`).
        params (Mapping[str, Any]): Parameters for the selected benchmark mode.

    Returns:
        dict[str, Any]: Mixed benchmark scenario outputs and artifact paths for
            execution mode; catalog metadata for listing modes.

    Examples:
        ```python
        from ldt.machine_learning.tools.bench_standard_and_longitudinal_ml.run import run_bench_standard_and_longitudinal_ml_tool

        result = run_bench_standard_and_longitudinal_ml_tool(
            technique="run_benchmark",
            params={
                "input_path": "./dataset.csv",
                "target_column": "depression_status",
                "feature_columns": "mood_w1,mood_w2,mood_w3,sleep_w1,sleep_w2,sleep_w3,sex",
                "feature_groups": "[[0,1,2],[3,4,5]]",
                "non_longitudinal_features": "[6]",
                "excluded_estimators": "",
                "metric_keys": "accuracy,f1_macro",
                "cv_folds": 5,
                "validation_split": "none",
                "random_seed": 42,
            },
        )
        ```
    """

    return run_with_validation(
        lambda: _run_bench_standard_and_longitudinal_ml_tool(
            technique=technique,
            params=params,
        )
    )


def _run_bench_standard_and_longitudinal_ml_tool(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    _, resolved = resolve_technique_with_defaults(
        section_key="bench_standard_and_longitudinal_ml",
        technique_id=technique,
        provided_params=dict(params),
    )

    mode = technique.strip().lower().replace("-", "_")
    if mode == "list_estimators":
        standard_estimators = discover_standard_estimators()
        longitudinal_estimators = discover_longitudinal_estimators()
        return {
            "standard_estimators": [
                {
                    "key": key,
                    "name": resolve_component_metadata(template).full_name,
                    "description": resolve_component_metadata(
                        template
                    ).abstract_description,
                    "source": "standard_ml",
                }
                for key, template in standard_estimators.items()
            ],
            "longitudinal_estimators": [
                {
                    "key": key,
                    "name": resolve_component_metadata(template).full_name,
                    "description": resolve_component_metadata(
                        template
                    ).abstract_description,
                    "source": "longitudinal_ml",
                }
                for key, template in longitudinal_estimators.items()
            ],
            "count": len(standard_estimators) + len(longitudinal_estimators),
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

    if mode != "run_benchmark":
        raise InputValidationError(
            f"Unsupported mixed-benchmark technique: {technique}"
        )

    standard_estimators = discover_standard_estimators()
    longitudinal_estimators = discover_longitudinal_estimators()
    if not standard_estimators and not longitudinal_estimators:
        raise InputValidationError(
            "No standard or longitudinal estimators were discovered."
        )

    excluded = parse_excluded_estimators(
        as_required_string(resolved, "excluded_estimators")
    )
    included_standard_estimators = {
        key: template
        for key, template in standard_estimators.items()
        if key not in excluded
    }
    included_longitudinal_estimators = {
        key: template
        for key, template in longitudinal_estimators.items()
        if key not in excluded
    }
    if not included_standard_estimators and not included_longitudinal_estimators:
        raise InputValidationError(
            "All estimators were excluded. Include at least one estimator."
        )

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

    output_root_raw = as_optional_string(resolved, "output_root_dir")
    output_root_dir = (
        Path(output_root_raw).expanduser()
        if output_root_raw is not None
        else Path("outputs/benchmarks/standard_and_longitudinal_ml").expanduser()
    )

    generate_notebook = as_bool(
        resolved.get("generate_notebook", True),
        field_name="generate_notebook",
    )
    notebook_path = resolve_notebook_path(
        generate_notebook=generate_notebook,
        requested_path=as_optional_string(resolved, "notebook_output_path"),
        default_path=output_root_dir
        / "bench_standard_and_longitudinal_ml_reproducibility_notebook.ipynb",
    )
    include_pipeline_profiler_at_end = as_bool(
        resolved.get("include_pipeline_profiler_at_end", False),
        field_name="include_pipeline_profiler_at_end",
    )
    generate_pipeline_profiler_html = as_bool(
        resolved.get("generate_pipeline_profiler_html", False),
        field_name="generate_pipeline_profiler_html",
    )

    unique_class_count = int(y_raw.nunique(dropna=True))
    if unique_class_count < 2:
        raise InputValidationError(
            "Classification requires at least two distinct target classes."
        )

    scenarios = None
    if unique_class_count > 2:
        scenarios = TargetScenarioPlanner.build_one_vs_rest_scenarios(y=y_raw)
        validate_one_vs_rest_cv_folds(
            cv_folds=cv_folds,
            validation_split=validation_split,
            scenarios=scenarios,
        )
    else:
        encoded = LongitudinalTargetEncoder.encode_if_needed(y=y_raw)
        y_raw = encoded.encoded_target
        validate_cv_split_feasibility(
            y=y_raw,
            cv_folds=cv_folds,
            validation_split=validation_split,
        )

    def _build_specs() -> tuple[Any, ...]:
        estimator_specs: list[BenchmarkEstimatorSpec] = []
        for key, estimator_template in included_standard_estimators.items():
            metadata = resolve_component_metadata(estimator_template)
            estimator_specs.append(
                BenchmarkEstimatorSpec(
                    estimator_key=key,
                    estimator_name=f"{metadata.full_name} [standard-ml]",
                    estimator=estimator_template.build_estimator(
                        hyperparameters={},
                        random_seed=random_seed,
                    ),
                )
            )
        for key, estimator_template in included_longitudinal_estimators.items():
            metadata = resolve_component_metadata(estimator_template)
            estimator_specs.append(
                BenchmarkEstimatorSpec(
                    estimator_key=key,
                    estimator_name=f"{metadata.full_name} [longitudinal-ml]",
                    estimator=estimator_template.build_estimator(
                        random_seed=random_seed,
                        feature_groups=feature_vectors.feature_groups,
                        non_longitudinal_features=feature_vectors.non_longitudinal_features,
                        feature_list_names=tuple(feature_columns),
                    ),
                )
            )
        return tuple(estimator_specs)

    benchmark_runs: list[dict[str, Any]] = []
    if scenarios is not None:
        for scenario in scenarios:
            scenario_output_root = output_root_dir / scenario.scenario_key
            benchmark = ClassificationBenchmarkTemplate().run(
                benchmark_name=_BENCHMARK_NAME,
                estimator_specs=_build_specs(),
                X=X,
                y=scenario.y_binary,
                metric_keys=metric_keys,
                output_root_dir=scenario_output_root,
                cv_folds=cv_folds,
                random_seed=random_seed,
                validation_split=validation_split,
                silent_training_output=silent_training_output,
                generate_pipeline_profiler_html=generate_pipeline_profiler_html,
            )
            benchmark_runs.append(
                {
                    "scenario_key": scenario.scenario_key,
                    "scenario_label": scenario.scenario_label,
                    "benchmark_output_dir": str(
                        benchmark.artifacts.benchmark_output_dir.resolve()
                    ),
                    "ranking_csv": str(benchmark.artifacts.ranking_path.resolve()),
                    "summary_json": str(benchmark.artifacts.summary_path.resolve()),
                    "report_path": str(benchmark.artifacts.report_path.resolve()),
                    "ranked_estimators": [
                        {
                            "rank": idx + 1,
                            "estimator_key": item.estimator_key,
                            "estimator_name": item.estimator_name,
                            "mean_score": item.mean_score,
                            "std_score": item.std_score,
                        }
                        for idx, item in enumerate(benchmark.ranked_estimators)
                    ],
                }
            )
    else:
        benchmark = ClassificationBenchmarkTemplate().run(
            benchmark_name=_BENCHMARK_NAME,
            estimator_specs=_build_specs(),
            X=X,
            y=y_raw,
            metric_keys=metric_keys,
            output_root_dir=output_root_dir,
            cv_folds=cv_folds,
            random_seed=random_seed,
            validation_split=validation_split,
            silent_training_output=silent_training_output,
            generate_pipeline_profiler_html=generate_pipeline_profiler_html,
        )
        benchmark_runs.append(
            {
                "scenario_key": "default",
                "scenario_label": "default",
                "benchmark_output_dir": str(
                    benchmark.artifacts.benchmark_output_dir.resolve()
                ),
                "ranking_csv": str(benchmark.artifacts.ranking_path.resolve()),
                "summary_json": str(benchmark.artifacts.summary_path.resolve()),
                "report_path": str(benchmark.artifacts.report_path.resolve()),
                "ranked_estimators": [
                    {
                        "rank": idx + 1,
                        "estimator_key": item.estimator_key,
                        "estimator_name": item.estimator_name,
                        "mean_score": item.mean_score,
                        "std_score": item.std_score,
                    }
                    for idx, item in enumerate(benchmark.ranked_estimators)
                ],
            }
        )

    saved_notebook = None
    if notebook_path is not None:
        notebook_cfg = MixedBenchmarkNotebookConfig(
            notebook_path=notebook_path,
            dataset_path=input_path,
            output_root_dir=output_root_dir,
            benchmark_name=_BENCHMARK_NAME,
            excluded_estimators=tuple(sorted(excluded)),
            target_column=target_column,
            feature_columns=tuple(feature_columns),
            feature_groups=feature_vectors.feature_groups,
            non_longitudinal_features=feature_vectors.non_longitudinal_features,
            metric_keys=metric_keys,
            cv_folds=cv_folds,
            validation_split=validation_split,
            random_seed=random_seed,
            silent_training_output=silent_training_output,
            include_pipeline_profiler_at_end=include_pipeline_profiler_at_end,
        )
        saved_notebook = MixedBenchmarkNotebookTemplate().write_notebook(
            config=notebook_cfg
        )

    return {
        "benchmark_name": _BENCHMARK_NAME,
        "included_standard_estimators": len(included_standard_estimators),
        "included_longitudinal_estimators": len(included_longitudinal_estimators),
        "excluded_estimators": sorted(excluded),
        "runs": benchmark_runs,
        "scenario_count": len(benchmark_runs),
        "output_root_dir": str(output_root_dir.resolve()),
        "notebook_path": (str(saved_notebook.resolve()) if saved_notebook else None),
        "dropped_rows": int(len(dataset) - len(modelling_data)),
    }
