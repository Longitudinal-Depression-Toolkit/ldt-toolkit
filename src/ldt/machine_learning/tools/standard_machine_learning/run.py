from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from beartype import beartype

from ldt.machine_learning.catalog import resolve_technique_with_defaults
from ldt.machine_learning.support.inputs import (
    as_bool,
    as_optional_object,
    as_optional_string,
    as_required_int,
    as_required_string,
    load_input_dataset,
    parse_metric_keys,
    parse_validation_split,
    resolve_feature_columns,
    resolve_notebook_path,
    resolve_target_column,
    run_with_validation,
)
from ldt.machine_learning.support.runtime_validation import (
    public_estimator_module,
    resolve_estimator_init_kwargs,
    validate_cv_split_feasibility,
    validate_one_vs_rest_cv_folds,
)
from ldt.machine_learning.tools.metrics import list_supported_metrics
from ldt.machine_learning.tools.standard_machine_learning.discovery import (
    discover_standard_estimators,
)
from ldt.machine_learning.tools.target_scenarios import TargetScenarioPlanner
from ldt.machine_learning.tools.templates import (
    ClassificationExperimentNotebookTemplate,
    ClassificationExperimentTemplate,
    ClassificationNotebookConfig,
)
from ldt.utils.errors import InputValidationError
from ldt.utils.metadata import ComponentMetadata, resolve_component_metadata
from ldt.utils.templates.tools.machine_learning import MachineLearningTool


@beartype
class StandardMachineLearning(MachineLearningTool):
    """Run standard machine-learning workflows for tabular classification.

    Available estimators:

    | Estimator key | Family summary |
    | --- | --- |
    | `decision_tree` | Non-linear rule-based tree classifier. |
    | `extra_trees` | Highly randomised tree ensemble. |
    | `gradient_boosting` | Sequential boosted-tree ensemble. |
    | `knn` | Instance-based nearest-neighbour classifier. |
    | `logistic_regression` | Linear probabilistic classifier. |
    | `random_forest` | Bagged decision-tree ensemble. |
    | `svm` | Margin-based classifier with kernels. |

    Supported techniques:

    | Technique | What it does |
    | --- | --- |
    | `list_estimators` | Returns supported estimator keys and descriptions. |
    | `list_metrics` | Returns supported scoring metrics. |
    | `run_experiment` | Trains/evaluates one selected estimator and writes artefacts. |

    Examples:
        ```python
        from ldt.machine_learning import StandardMachineLearning

        tool = StandardMachineLearning()
        result = tool.fit_predict(
            technique="run_experiment",
            input_path="./data/millennium.csv",
            target_column="depression_status",
            feature_columns="sleep_score,anxiety_score,sex,income",
            estimator_key="random_forest",
            metric_keys="accuracy,f1_macro",
            cv_folds=5,
            validation_split="none",
            multiclass_mode="multiclass",
            random_seed=42,
        )
        ```
    """

    metadata = ComponentMetadata(
        name="standard_machine_learning",
        full_name="Standard Machine Learning",
        abstract_description=(
            "Run cross-sectional classification workflows and estimator discovery."
        ),
    )

    def __init__(self) -> None:
        self._technique: str | None = None
        self._params: dict[str, Any] = {}

    @beartype
    def fit(self, **kwargs: Any) -> StandardMachineLearning:
        """Validate and store one execution payload.

        Args:
            **kwargs (Any): Configuration keys:
                - `technique` (str): `list_estimators`, `list_metrics`, or
                  `run_experiment`.
                - `params` (Mapping[str, Any] | None): Optional parameter object.
                - for `run_experiment`, expected keys include:
                  `input_path`, `target_column`, `feature_columns`,
                  `estimator_key`, `metric_keys`, `cv_folds`,
                  `validation_split`, `random_seed`, `output_dir`,
                  `hyperparameters_json`, `multiclass_mode`,
                  `silent_training_output`, `generate_notebook`, and
                  `notebook_output_path`.
                - for `list_estimators` and `list_metrics`, no extra keys are
                  required.
                - any additional keys are merged into `params` as direct
                  library-friendly shorthand.

        Returns:
            StandardMachineLearning: The fitted tool instance.
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
        """Execute the configured workflow.

        Args:
            **kwargs (Any): Optional keys identical to `fit(...)`. If provided,
                they override any existing fitted configuration. Expected keys:
                `technique` and optional `params`, or direct shorthand keys for
                `run_experiment`:
                `input_path`, `target_column`, `feature_columns`,
                `estimator_key`, `metric_keys`, `cv_folds`,
                `validation_split`, `random_seed`, `output_dir`,
                `hyperparameters_json`, `multiclass_mode`,
                `silent_training_output`, `generate_notebook`,
                and `notebook_output_path`.

        Returns:
            dict[str, Any]: Technique-specific workflow result payload.
        """

        if kwargs:
            self.fit(**kwargs)
        if self._technique is None:
            raise InputValidationError(
                "No configuration provided. Call `fit(...)` first or pass kwargs to `predict(...)`."
            )

        return run_with_validation(
            lambda: _run_standard_machine_learning_tool(
                technique=self._technique or "",
                params=self._params,
            )
        )

    @beartype
    def predict_proba(self, **kwargs: Any) -> dict[str, Any]:
        """Standard tool-level probability payloads are not exposed.

        Args:
            **kwargs (Any): Ignored. Present for template contract compliance.

        Returns:
            dict[str, Any]: Never returned.

        Raises:
            InputValidationError: Always raised because this stage tool returns
                experiment/benchmark metadata rather than raw probability arrays.
        """

        _ = kwargs
        raise InputValidationError(
            "`StandardMachineLearning` does not expose tool-level `predict_proba(...)`. "
            "Use saved model artefacts for probability inference."
        )


def run_standard_machine_learning_tool(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one standard-machine-learning technique for the Go CLI bridge.

    !!! warning

        This `run_*` function is primarily for the Go CLI bridge and should not be
        treated as the Python library API. In Python scripts or notebooks, import
        and call `StandardMachineLearning` directly from `ldt.machine_learning`.

    Args:
        technique (str): Workflow technique (`list_estimators`, `list_metrics`,
            or `run_experiment`).
        params (Mapping[str, Any]): Technique parameters. For `run_experiment`,
            expected keys include:
            `input_path`, `target_column`, `feature_columns`, `estimator_key`,
            `metric_keys`, `cv_folds`, `validation_split`, `multiclass_mode`,
            `random_seed`; optional keys:
            `output_dir`, `hyperparameters_json`, `silent_training_output`,
            `generate_notebook`, and `notebook_output_path`.

    Returns:
        dict[str, Any]: Technique-specific serialised result payload.
    """

    return StandardMachineLearning().fit_predict(
        technique=technique,
        params=params,
    )


def _run_standard_machine_learning_tool(
    *,
    technique: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    _, resolved = resolve_technique_with_defaults(
        section_key="standard_machine_learning",
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

    if mode != "run_experiment":
        raise InputValidationError(
            f"Unsupported standard-machine-learning technique: {technique}"
        )

    estimators = discover_standard_estimators()
    if not estimators:
        raise InputValidationError("No cross-sectional estimators were found.")

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
    random_seed = as_required_int(resolved, "random_seed")
    silent_training_output = as_bool(
        resolved.get("silent_training_output", True),
        field_name="silent_training_output",
    )
    multiclass_mode = as_required_string(resolved, "multiclass_mode").strip().lower()
    if multiclass_mode not in {"multiclass", "one_vs_rest"}:
        raise InputValidationError(
            "multiclass_mode must be `multiclass` or `one_vs_rest`."
        )

    output_dir_raw = as_optional_string(resolved, "output_dir")
    output_dir = (
        Path(output_dir_raw).expanduser()
        if output_dir_raw is not None
        else Path(f"outputs/standard_ml/{estimator_key}").expanduser()
    )

    hyperparameter_overrides = as_optional_object(resolved, "hyperparameters_json")

    generate_notebook = as_bool(
        resolved.get("generate_notebook", True),
        field_name="generate_notebook",
    )
    notebook_path = resolve_notebook_path(
        generate_notebook=generate_notebook,
        requested_path=as_optional_string(resolved, "notebook_output_path"),
        default_path=output_dir / "experiment_reproducibility_notebook.ipynb",
    )

    unique_class_count = int(y.nunique(dropna=True))
    if unique_class_count < 2:
        raise InputValidationError(
            "Classification requires at least two distinct target classes."
        )

    experiment = ClassificationExperimentTemplate()

    if unique_class_count > 2 and multiclass_mode == "one_vs_rest":
        scenarios = TargetScenarioPlanner.build_one_vs_rest_scenarios(y=y)
        validate_one_vs_rest_cv_folds(
            cv_folds=cv_folds,
            validation_split=validation_split,
            scenarios=scenarios,
        )

        scenario_results: list[dict[str, Any]] = []
        for scenario in scenarios:
            estimator = estimator_template.build_estimator(
                hyperparameters=hyperparameter_overrides,
                random_seed=random_seed,
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
            "notebook_generated": False,
            "dropped_rows": int(len(dataset) - len(modelling_data)),
        }

    validate_cv_split_feasibility(
        y=y,
        cv_folds=cv_folds,
        validation_split=validation_split,
    )

    estimator = estimator_template.build_estimator(
        hyperparameters=hyperparameter_overrides,
        random_seed=random_seed,
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

    saved_notebook = None
    if notebook_path is not None:
        estimator_module = public_estimator_module(
            estimator_template=estimator_template
        )
        estimator_class_name = estimator_template.estimator_cls.__name__
        estimator_init_kwargs = resolve_estimator_init_kwargs(
            estimator_template=estimator_template,
            hyperparameter_overrides=hyperparameter_overrides,
            random_seed=random_seed,
        )
        config = ClassificationNotebookConfig(
            notebook_path=notebook_path,
            dataset_path=input_path,
            output_dir=output_dir,
            estimator_module=estimator_module,
            estimator_class_name=estimator_class_name,
            estimator_name=estimator_name,
            estimator_init_kwargs=estimator_init_kwargs,
            target_column=target_column,
            feature_columns=tuple(feature_columns),
            metric_keys=metric_keys,
            cv_folds=cv_folds,
            validation_split=validation_split,
            random_seed=random_seed,
            silent_training_output=silent_training_output,
        )
        saved_notebook = ClassificationExperimentNotebookTemplate().write_notebook(
            config=config
        )

    return {
        "mode": "multiclass",
        "estimator_key": estimator_key,
        "estimator_name": estimator_name,
        "metric_key": result.metric_key,
        "mean_score": result.mean_score,
        "std_score": result.std_score,
        "output_dir": str(output_dir.resolve()),
        "model_path": str(result.artifacts.model_path.resolve()),
        "summary_path": str(result.artifacts.summary_path.resolve()),
        "report_path": str(result.artifacts.report_path.resolve()),
        "notebook_path": (str(saved_notebook.resolve()) if saved_notebook else None),
        "dropped_rows": int(len(dataset) - len(modelling_data)),
    }
