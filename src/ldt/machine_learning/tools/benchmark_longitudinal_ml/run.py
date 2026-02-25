from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from beartype import beartype

from ldt.machine_learning.catalog import resolve_technique_with_defaults
from ldt.machine_learning.support.inputs import (
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
from ldt.machine_learning.support.runtime_validation import (
    validate_cv_split_feasibility,
    validate_one_vs_rest_cv_folds,
)
from ldt.machine_learning.tools.longitudinal_machine_learning.discovery import (
    discover_longitudinal_estimators,
)
from ldt.machine_learning.tools.longitudinal_machine_learning.inputs import (
    LongitudinalFeatureInputPrompter,
)
from ldt.machine_learning.tools.longitudinal_machine_learning.target_encoding import (
    LongitudinalTargetEncoder,
)
from ldt.machine_learning.tools.metrics import list_supported_metrics
from ldt.machine_learning.tools.target_scenarios import TargetScenarioPlanner
from ldt.machine_learning.tools.templates import (
    BenchmarkEstimatorSpec,
    ClassificationBenchmarkTemplate,
    LongitudinalBenchmarkNotebookConfig,
    LongitudinalBenchmarkNotebookTemplate,
)
from ldt.utils.errors import InputValidationError
from ldt.utils.metadata import ComponentMetadata, resolve_component_metadata
from ldt.utils.templates.tools.machine_learning import MachineLearningTool

_BENCHMARK_NAME = "benchmark_longitudinal_ml"


@beartype
class BenchmarkLongitudinalML(MachineLearningTool):
    """Benchmark longitudinal machine-learning estimators on temporal features.

    Available estimator families:

    | Family key | What it does |
    | --- | --- |
    | `aggrfunc_*` | Aggregates repeated waves (mean/median) before classification. |
    | `merwav_time_minus_*` | Merges waves with relative-time encoding. |
    | `sepwav_stacking_*` | Fits per-wave learners and stacks them. |
    | `sepwav_voting_*` | Fits per-wave learners and combines by voting. |
    | `merwav_time_plus_*` | Uses native longitudinal estimators. |

    Available longitudinal estimator keys:

    | Estimator key | Transformation Strategy | Base estimator |
    | --- | --- | --- |
    | `aggrfunc_mean__decision_tree` | `aggrfunc_mean` | `decision_tree` |
    | `aggrfunc_mean__extra_trees` | `aggrfunc_mean` | `extra_trees` |
    | `aggrfunc_mean__gradient_boosting` | `aggrfunc_mean` | `gradient_boosting` |
    | `aggrfunc_mean__knn` | `aggrfunc_mean` | `knn` |
    | `aggrfunc_mean__logistic_regression` | `aggrfunc_mean` | `logistic_regression` |
    | `aggrfunc_mean__random_forest` | `aggrfunc_mean` | `random_forest` |
    | `aggrfunc_mean__svm` | `aggrfunc_mean` | `svm` |
    | `aggrfunc_median__decision_tree` | `aggrfunc_median` | `decision_tree` |
    | `aggrfunc_median__extra_trees` | `aggrfunc_median` | `extra_trees` |
    | `aggrfunc_median__gradient_boosting` | `aggrfunc_median` | `gradient_boosting` |
    | `aggrfunc_median__knn` | `aggrfunc_median` | `knn` |
    | `aggrfunc_median__logistic_regression` | `aggrfunc_median` | `logistic_regression` |
    | `aggrfunc_median__random_forest` | `aggrfunc_median` | `random_forest` |
    | `aggrfunc_median__svm` | `aggrfunc_median` | `svm` |
    | `merwav_time_minus__decision_tree` | `merwav_time_minus` | `decision_tree` |
    | `merwav_time_minus__extra_trees` | `merwav_time_minus` | `extra_trees` |
    | `merwav_time_minus__gradient_boosting` | `merwav_time_minus` | `gradient_boosting` |
    | `merwav_time_minus__knn` | `merwav_time_minus` | `knn` |
    | `merwav_time_minus__logistic_regression` | `merwav_time_minus` | `logistic_regression` |
    | `merwav_time_minus__random_forest` | `merwav_time_minus` | `random_forest` |
    | `merwav_time_minus__svm` | `merwav_time_minus` | `svm` |
    | `merwav_time_plus__lexico_decision_tree` | `merwav_time_plus` | `lexico_decision_tree` |
    | `merwav_time_plus__lexico_deep_forest` | `merwav_time_plus` | `lexico_deep_forest` |
    | `merwav_time_plus__lexico_gradient_boosting` | `merwav_time_plus` | `lexico_gradient_boosting` |
    | `merwav_time_plus__lexico_random_forest` | `merwav_time_plus` | `lexico_random_forest` |
    | `merwav_time_plus__nested_trees` | `merwav_time_plus` | `nested_trees` |
    | `sepwav_stacking_dt__decision_tree` | `sepwav_stacking_dt` | `decision_tree` |
    | `sepwav_stacking_dt__extra_trees` | `sepwav_stacking_dt` | `extra_trees` |
    | `sepwav_stacking_dt__gradient_boosting` | `sepwav_stacking_dt` | `gradient_boosting` |
    | `sepwav_stacking_dt__knn` | `sepwav_stacking_dt` | `knn` |
    | `sepwav_stacking_dt__logistic_regression` | `sepwav_stacking_dt` | `logistic_regression` |
    | `sepwav_stacking_dt__random_forest` | `sepwav_stacking_dt` | `random_forest` |
    | `sepwav_stacking_dt__svm` | `sepwav_stacking_dt` | `svm` |
    | `sepwav_stacking_lr__decision_tree` | `sepwav_stacking_lr` | `decision_tree` |
    | `sepwav_stacking_lr__extra_trees` | `sepwav_stacking_lr` | `extra_trees` |
    | `sepwav_stacking_lr__gradient_boosting` | `sepwav_stacking_lr` | `gradient_boosting` |
    | `sepwav_stacking_lr__knn` | `sepwav_stacking_lr` | `knn` |
    | `sepwav_stacking_lr__logistic_regression` | `sepwav_stacking_lr` | `logistic_regression` |
    | `sepwav_stacking_lr__random_forest` | `sepwav_stacking_lr` | `random_forest` |
    | `sepwav_stacking_lr__svm` | `sepwav_stacking_lr` | `svm` |
    | `sepwav_voting__decision_tree` | `sepwav_voting` | `decision_tree` |
    | `sepwav_voting__extra_trees` | `sepwav_voting` | `extra_trees` |
    | `sepwav_voting__gradient_boosting` | `sepwav_voting` | `gradient_boosting` |
    | `sepwav_voting__knn` | `sepwav_voting` | `knn` |
    | `sepwav_voting__logistic_regression` | `sepwav_voting` | `logistic_regression` |
    | `sepwav_voting__random_forest` | `sepwav_voting` | `random_forest` |
    | `sepwav_voting__svm` | `sepwav_voting` | `svm` |

    Abbreviations used in estimator keys:
        - `dt`: Decision Tree
        - `lr`: Logistic Regression
        - `svm`: Support Vector Machine
        - `knn`: K-Nearest Neighbours

    Supported techniques:

    | Technique | What it does |
    | --- | --- |
    | `list_estimators` | Returns longitudinal estimator roster metadata. |
    | `list_metrics` | Returns supported scoring metrics. |
    | `run_benchmark` | Trains/evaluates included estimators and ranks them. |

    Examples:
        ```python
        from ldt.machine_learning import BenchmarkLongitudinalML

        tool = BenchmarkLongitudinalML()
        result = tool.fit_predict(
            technique="run_benchmark",
            input_path="./longitudinal_dataset.csv",
            target_column="depression_status",
            feature_columns="mood_w1,mood_w2,mood_w3,sleep_w1,sleep_w2,sleep_w3,sex",
            feature_groups="[[0,1,2],[3,4,5]]",
            non_longitudinal_features="[6]",
            excluded_estimators="",
            metric_keys="accuracy,f1_macro",
            cv_folds=5,
            validation_split="none",
            random_seed=42,
        )
        ```

    !!! info "Third-party documentation"

        - [scikit-longitudinal documentation](https://scikit-longitudinal.readthedocs.io/latest/)
        - [scikit-learn documentation](https://scikit-learn.org/stable/)

    !!! info "Reference paper"

        - [JOSS paper: scikit-longitudinal](https://joss.theoj.org/papers/10.21105/joss.08481)
    """

    metadata = ComponentMetadata(
        name="benchmark_longitudinal_ml",
        full_name="Benchmark Longitudinal Machine Learning",
        abstract_description=(
            "Benchmark longitudinal classifiers and rank scenario-aware results."
        ),
    )

    def __init__(self) -> None:
        self._technique: str | None = None
        self._params: dict[str, Any] = {}

    @beartype
    def fit(self, **kwargs: Any) -> BenchmarkLongitudinalML:
        """Validate and store one execution payload.

        Args:
            **kwargs (Any): Configuration keys:
                - `technique` (str): `list_estimators`, `list_metrics`, or
                  `run_benchmark`.
                - `params` (Mapping[str, Any] | None): Optional parameter object.
                - for `run_benchmark`, expected keys include:
                  `input_path`, `target_column`, `feature_columns`,
                  `feature_groups`, `non_longitudinal_features`,
                  `excluded_estimators`, `metric_keys`, `cv_folds`,
                  `validation_split`, `random_seed`, `output_root_dir`,
                  `silent_training_output`, `generate_notebook`,
                  `notebook_output_path`,
                  `include_pipeline_profiler_at_end`, and
                  `generate_pipeline_profiler_html`.
                - for `list_estimators` and `list_metrics`, no extra keys are
                  required.
                - any additional keys are merged into `params` as direct
                  library-friendly shorthand.

        Returns:
            BenchmarkLongitudinalML: The fitted tool instance.
        """

        technique_raw = kwargs.get("technique", self._technique)
        if not isinstance(technique_raw, str) or not technique_raw.strip():
            raise InputValidationError("Missing required string parameter: technique")

        params_payload: dict[str, Any] = {}
        params_raw = kwargs.get("params")
        if params_raw is not None:
            if not isinstance(params_raw, Mapping):
                raise InputValidationError("`params` must be an object.")
            params_payload.update(dict(params_raw))
        params_payload.update(
            {
                key: value
                for key, value in kwargs.items()
                if key not in {"technique", "params"}
            }
        )

        self._technique = technique_raw.strip()
        self._params = params_payload
        return self

    @beartype
    def predict(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the configured benchmark workflow.

        Args:
            **kwargs (Any): Optional keys identical to `fit(...)`. If provided,
                they override any existing fitted configuration. Expected keys:
                `technique` and optional `params`, or direct shorthand keys for
                `run_benchmark`:
                `input_path`, `target_column`, `feature_columns`,
                `feature_groups`, `non_longitudinal_features`,
                `excluded_estimators`, `metric_keys`, `cv_folds`,
                `validation_split`, `random_seed`, `output_root_dir`,
                `silent_training_output`, `generate_notebook`,
                `notebook_output_path`,
                `include_pipeline_profiler_at_end`, and
                `generate_pipeline_profiler_html`.

        Returns:
            dict[str, Any]: Technique-specific benchmark result payload.
        """

        if kwargs:
            self.fit(**kwargs)
        if self._technique is None:
            raise InputValidationError(
                "No configuration provided. Call `fit(...)` first or pass kwargs to `predict(...)`."
            )

        return run_with_validation(
            lambda: _run_benchmark_longitudinal_ml_tool(
                technique=self._technique or "",
                params=self._params,
            )
        )

    @beartype
    def predict_proba(self, **kwargs: Any) -> dict[str, Any]:
        """Benchmark tool-level probability payloads are not exposed.

        Args:
            **kwargs (Any): Ignored. Present for template contract compliance.

        Returns:
            dict[str, Any]: Never returned.

        Raises:
            InputValidationError: Always raised because this benchmark tool
                returns ranking metadata rather than raw probability arrays.
        """

        _ = kwargs
        raise InputValidationError(
            "`BenchmarkLongitudinalML` does not expose tool-level `predict_proba(...)`."
        )


def run_benchmark_longitudinal_ml_tool(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one longitudinal benchmark technique for the Go CLI bridge.

    !!! warning

        This `run_*` function is primarily for the Go CLI bridge and should not be
        treated as the Python library API. In Python scripts or notebooks, import
        and call `BenchmarkLongitudinalML` directly from `ldt.machine_learning`.

    Args:
        technique (str): Workflow technique (`list_estimators`, `list_metrics`,
            or `run_benchmark`).
        params (Mapping[str, Any]): Technique parameters. For `run_benchmark`,
            expected keys include:
            `input_path`, `target_column`, `feature_columns`, `feature_groups`,
            `non_longitudinal_features`, `excluded_estimators`, `metric_keys`,
            `cv_folds`, `validation_split`, `random_seed`; optional keys:
            `output_root_dir`, `silent_training_output`, `generate_notebook`,
            `notebook_output_path`, `include_pipeline_profiler_at_end`, and
            `generate_pipeline_profiler_html`.

    Returns:
        dict[str, Any]: Technique-specific serialised result payload.
    """

    return BenchmarkLongitudinalML().fit_predict(
        technique=technique,
        params=params,
    )


def _run_benchmark_longitudinal_ml_tool(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    _, resolved = resolve_technique_with_defaults(
        section_key="benchmark_longitudinal_ml",
        technique_id=technique,
        provided_params=dict(params),
    )

    mode = technique.strip().lower().replace("-", "_")
    if mode == "list_estimators":
        estimators = discover_longitudinal_estimators()
        return {
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

    if mode != "run_benchmark":
        raise InputValidationError(
            f"Unsupported benchmark-longitudinal-ml technique: {technique}"
        )

    estimators = discover_longitudinal_estimators()
    if not estimators:
        raise InputValidationError("No longitudinal estimators were discovered.")

    excluded = parse_excluded_estimators(
        as_required_string(resolved, "excluded_estimators")
    )
    included_estimators = {
        key: template for key, template in estimators.items() if key not in excluded
    }
    if not included_estimators:
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
        else Path("outputs/benchmarks/longitudinal_ml").expanduser()
    )

    generate_notebook = as_bool(
        resolved.get("generate_notebook", True),
        field_name="generate_notebook",
    )
    notebook_path = resolve_notebook_path(
        generate_notebook=generate_notebook,
        requested_path=as_optional_string(resolved, "notebook_output_path"),
        default_path=output_root_dir
        / "benchmark_longitudinal_ml_reproducibility_notebook.ipynb",
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
        return tuple(
            BenchmarkEstimatorSpec(
                estimator_key=key,
                estimator_name=resolve_component_metadata(estimator_template).full_name,
                estimator=estimator_template.build_estimator(
                    random_seed=random_seed,
                    feature_groups=feature_vectors.feature_groups,
                    non_longitudinal_features=feature_vectors.non_longitudinal_features,
                    feature_list_names=tuple(feature_columns),
                ),
            )
            for key, estimator_template in included_estimators.items()
        )

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
        notebook_cfg = LongitudinalBenchmarkNotebookConfig(
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
        saved_notebook = LongitudinalBenchmarkNotebookTemplate().write_notebook(
            config=notebook_cfg
        )

    return {
        "benchmark_name": _BENCHMARK_NAME,
        "included_estimators": len(included_estimators),
        "excluded_estimators": sorted(excluded),
        "runs": benchmark_runs,
        "scenario_count": len(benchmark_runs),
        "output_root_dir": str(output_root_dir.resolve()),
        "notebook_path": (str(saved_notebook.resolve()) if saved_notebook else None),
        "dropped_rows": int(len(dataset) - len(modelling_data)),
    }
