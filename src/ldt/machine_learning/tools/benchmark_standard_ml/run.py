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
)
from ldt.machine_learning.tools.metrics import list_supported_metrics
from ldt.machine_learning.tools.standard_machine_learning.discovery import (
    discover_standard_estimators,
)
from ldt.machine_learning.tools.templates import (
    BenchmarkEstimatorSpec,
    ClassificationBenchmarkTemplate,
    StandardMLBenchmarkNotebookConfig,
    StandardMLBenchmarkNotebookTemplate,
)
from ldt.utils.errors import InputValidationError
from ldt.utils.metadata import ComponentMetadata, resolve_component_metadata
from ldt.utils.templates.tools.machine_learning import MachineLearningTool

_BENCHMARK_NAME = "benchmark_standard_ml"


@beartype
class BenchmarkStandardML(MachineLearningTool):
    """Benchmark standard machine-learning estimators on one dataset.

    Available estimator families:

    | Estimator key | What it does |
    | --- | --- |
    | `decision_tree` | Rule-based tree classifier. |
    | `extra_trees` | Highly randomised tree ensemble. |
    | `gradient_boosting` | Sequential boosted-tree classifier. |
    | `knn` | Nearest-neighbour classifier. |
    | `logistic_regression` | Linear probabilistic classifier. |
    | `random_forest` | Bagged tree ensemble. |
    | `svm` | Margin-based kernel classifier. |

    Supported techniques:

    | Technique | What it does |
    | --- | --- |
    | `list_estimators` | Returns benchmark estimator roster metadata. |
    | `list_metrics` | Returns supported scoring metrics. |
    | `run_benchmark` | Trains/evaluates included estimators and ranks them. |

    Examples:
        ```python
        from ldt.machine_learning import BenchmarkStandardML

        tool = BenchmarkStandardML()
        result = tool.fit_predict(
            technique="run_benchmark",
            input_path="./dataset.csv",
            target_column="depression_status",
            feature_columns="age,sex,sleep_score,anxiety_score",
            excluded_estimators="",
            metric_keys="accuracy,roc_auc",
            cv_folds=5,
            validation_split="none",
            random_seed=42,
        )
        ```
    """

    metadata = ComponentMetadata(
        name="benchmark_standard_ml",
        full_name="Benchmark Standard Machine Learning",
        abstract_description="Benchmark cross-sectional classifiers and rank results.",
    )

    def __init__(self) -> None:
        self._technique: str | None = None
        self._params: dict[str, Any] = {}

    @beartype
    def fit(self, **kwargs: Any) -> BenchmarkStandardML:
        """Validate and store one execution payload.

        Args:
            **kwargs (Any): Configuration keys:
                - `technique` (str): `list_estimators`, `list_metrics`, or
                  `run_benchmark`.
                - `params` (Mapping[str, Any] | None): Optional parameter object.
                - for `run_benchmark`, expected keys include:
                  `input_path`, `target_column`, `feature_columns`,
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
            BenchmarkStandardML: The fitted tool instance.
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
            lambda: _run_benchmark_standard_ml_tool(
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
            "`BenchmarkStandardML` does not expose tool-level `predict_proba(...)`."
        )


def run_benchmark_standard_ml_tool(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one standard benchmark technique for the Go CLI bridge.

    !!! warning

        This `run_*` function is primarily for the Go CLI bridge and should not be
        treated as the Python library API. In Python scripts or notebooks, import
        and call `BenchmarkStandardML` directly from `ldt.machine_learning`.

    Args:
        technique (str): Workflow technique (`list_estimators`, `list_metrics`,
            or `run_benchmark`).
        params (Mapping[str, Any]): Technique parameters. For `run_benchmark`,
            expected keys include:
            `input_path`, `target_column`, `feature_columns`,
            `excluded_estimators`, `metric_keys`, `cv_folds`,
            `validation_split`, `random_seed`; optional keys:
            `output_root_dir`, `silent_training_output`, `generate_notebook`,
            `notebook_output_path`, `include_pipeline_profiler_at_end`, and
            `generate_pipeline_profiler_html`.

    Returns:
        dict[str, Any]: Technique-specific serialised result payload.
    """

    return BenchmarkStandardML().fit_predict(
        technique=technique,
        params=params,
    )


def _run_benchmark_standard_ml_tool(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    _, resolved = resolve_technique_with_defaults(
        section_key="benchmark_standard_ml",
        technique_id=technique,
        provided_params=dict(params),
    )

    mode = technique.strip().lower().replace("-", "_")
    if mode == "list_estimators":
        estimators = discover_standard_estimators()
        return {
            "estimators": [
                {
                    "key": key,
                    "name": resolve_component_metadata(template).full_name,
                    "description": resolve_component_metadata(
                        template
                    ).abstract_description,
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
            f"Unsupported benchmark-standard-ml technique: {technique}"
        )

    estimators = discover_standard_estimators()
    if not estimators:
        raise InputValidationError("No cross-sectional estimators were found.")

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

    modelling_data = dataset[[*feature_columns, target_column]].dropna(
        subset=[target_column]
    )
    if modelling_data.empty:
        raise InputValidationError(
            "No rows available after dropping missing target values."
        )

    X = modelling_data[feature_columns].copy()
    y = modelling_data[target_column].copy()

    metric_keys = parse_metric_keys(as_required_string(resolved, "metric_keys"))
    cv_folds = as_required_int(resolved, "cv_folds", minimum=2)
    validation_split = parse_validation_split(
        as_required_string(resolved, "validation_split"),
        cv_folds=cv_folds,
    )
    validate_cv_split_feasibility(
        y=y,
        cv_folds=cv_folds,
        validation_split=validation_split,
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
        else Path("outputs/benchmarks/standard_ml").expanduser()
    )

    generate_notebook = as_bool(
        resolved.get("generate_notebook", True),
        field_name="generate_notebook",
    )
    notebook_path = resolve_notebook_path(
        generate_notebook=generate_notebook,
        requested_path=as_optional_string(resolved, "notebook_output_path"),
        default_path=output_root_dir
        / "benchmark_standard_ml_reproducibility_notebook.ipynb",
    )
    include_pipeline_profiler_at_end = as_bool(
        resolved.get("include_pipeline_profiler_at_end", False),
        field_name="include_pipeline_profiler_at_end",
    )
    generate_pipeline_profiler_html = as_bool(
        resolved.get("generate_pipeline_profiler_html", False),
        field_name="generate_pipeline_profiler_html",
    )

    estimator_specs = tuple(
        BenchmarkEstimatorSpec(
            estimator_key=key,
            estimator_name=resolve_component_metadata(template).full_name,
            estimator=template.build_estimator(
                hyperparameters={},
                random_seed=random_seed,
            ),
        )
        for key, template in included_estimators.items()
    )

    benchmark = ClassificationBenchmarkTemplate().run(
        benchmark_name=_BENCHMARK_NAME,
        estimator_specs=estimator_specs,
        X=X,
        y=y,
        metric_keys=metric_keys,
        output_root_dir=output_root_dir,
        cv_folds=cv_folds,
        random_seed=random_seed,
        validation_split=validation_split,
        silent_training_output=silent_training_output,
        generate_pipeline_profiler_html=generate_pipeline_profiler_html,
    )

    saved_notebook = None
    if notebook_path is not None:
        notebook_cfg = StandardMLBenchmarkNotebookConfig(
            notebook_path=notebook_path,
            dataset_path=input_path,
            output_root_dir=output_root_dir,
            benchmark_name=_BENCHMARK_NAME,
            excluded_estimators=tuple(sorted(excluded)),
            target_column=target_column,
            feature_columns=tuple(feature_columns),
            metric_keys=metric_keys,
            cv_folds=cv_folds,
            validation_split=validation_split,
            random_seed=random_seed,
            silent_training_output=silent_training_output,
            include_pipeline_profiler_at_end=include_pipeline_profiler_at_end,
        )
        saved_notebook = StandardMLBenchmarkNotebookTemplate().write_notebook(
            config=notebook_cfg
        )

    ranking = [
        {
            "rank": idx + 1,
            "estimator_key": item.estimator_key,
            "estimator_name": item.estimator_name,
            "mean_score": item.mean_score,
            "std_score": item.std_score,
        }
        for idx, item in enumerate(benchmark.ranked_estimators)
    ]

    return {
        "benchmark_name": _BENCHMARK_NAME,
        "metric_key": benchmark.metric_key,
        "metric_keys": list(benchmark.metric_keys),
        "included_estimators": len(included_estimators),
        "excluded_estimators": sorted(excluded),
        "ranked_estimators": ranking,
        "skipped_estimators": [
            {
                "estimator_key": skipped.estimator_key,
                "reason": skipped.reason,
            }
            for skipped in benchmark.skipped_estimators
        ],
        "benchmark_output_dir": str(benchmark.artifacts.benchmark_output_dir.resolve()),
        "ranking_csv": str(benchmark.artifacts.ranking_path.resolve()),
        "summary_json": str(benchmark.artifacts.summary_path.resolve()),
        "report_path": str(benchmark.artifacts.report_path.resolve()),
        "pipeline_profiler_json": str(
            benchmark.artifacts.pipeline_profiler.input_json_path.resolve()
        ),
        "pipeline_profiler_html": (
            str(benchmark.artifacts.pipeline_profiler.html_path.resolve())
            if benchmark.artifacts.pipeline_profiler.html_path is not None
            else None
        ),
        "pipeline_profiler_warning": benchmark.artifacts.pipeline_profiler.warning,
        "notebook_path": (str(saved_notebook.resolve()) if saved_notebook else None),
        "dropped_rows": int(len(dataset) - len(modelling_data)),
    }
