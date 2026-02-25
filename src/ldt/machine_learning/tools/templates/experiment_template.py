from __future__ import annotations

import copy
import json
import pickle
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from beartype import beartype
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ldt.machine_learning.tools.metrics import (
    MetricCompatibilityError,
    MetricDefinition,
    resolve_metric_definition,
)
from ldt.utils.errors import InputValidationError


@beartype
@dataclass(frozen=True)
class ClassificationMetricSummary:
    """Per-metric cross-validation summary.

    Attributes:
        metric_key (str): Identifier for metric key.
        fold_scores (tuple[float, ...]): Fold scores.
        mean_score (float): Mean score.
        std_score (float): Std score.

    """

    metric_key: str
    fold_scores: tuple[float, ...]
    mean_score: float
    std_score: float


@beartype
@dataclass(frozen=True)
class ClassificationExperimentArtifacts:
    """Artefact paths produced by one classification experiment.

    Attributes:
        model_path (Path): Path for model path.
        summary_path (Path): Path for summary path.
        report_path (Path): Path for report path.

    """

    model_path: Path
    summary_path: Path
    report_path: Path


@beartype
@dataclass(frozen=True)
class ClassificationExperimentResult:
    """Structured output produced by the classification experiment template.

    Attributes:
        estimator_name (str): Name for estimator.
        metric_key (str): Identifier for metric key.
        metric_keys (tuple[str, ...]): Metric keys.
        random_seed (int): Random seed for reproducibility.
        cv_folds (int): Cv folds.
        validation_split (float): Validation split.
        split_strategy (str): Split strategy.
        silent_training_output (bool): Whether to silent training output.
        fold_scores (tuple[float, ...]): Fold scores.
        mean_score (float): Mean score.
        std_score (float): Std score.
        metric_summaries (dict[str, ClassificationMetricSummary]): Metric summaries.
        classification_report_text (str): Classification report text.
        classification_report_dict (dict[str, Any]): Classification report dict.
        fitted_estimator (BaseEstimator): Fitted estimator.
        artifacts (ClassificationExperimentArtifacts): Artefacts.

    """

    estimator_name: str
    metric_key: str
    metric_keys: tuple[str, ...]
    random_seed: int
    cv_folds: int
    validation_split: float
    split_strategy: str
    silent_training_output: bool
    fold_scores: tuple[float, ...]
    mean_score: float
    std_score: float
    metric_summaries: dict[str, ClassificationMetricSummary]
    classification_report_text: str
    classification_report_dict: dict[str, Any]
    fitted_estimator: BaseEstimator
    artifacts: ClassificationExperimentArtifacts


class EstimatorMetricCompatibilityError(InputValidationError):
    """Raised when an estimator cannot satisfy metric output requirements."""


@beartype
class ClassificationExperimentTemplate:
    """Reusable template for cross-validated classification experiments."""

    @beartype
    def run(
        self,
        *,
        estimator: BaseEstimator,
        estimator_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        metric_key: str | None = None,
        metric_keys: tuple[str, ...] | None = None,
        output_dir: Path,
        cv_folds: int = 10,
        random_seed: int | None = None,
        validation_split: float | None = None,
        silent_training_output: bool = False,
    ) -> ClassificationExperimentResult:
        """Execute a classification experiment.

        Args:
            estimator (BaseEstimator): Estimator.
            estimator_name (str): Name for estimator.
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target labels.
            metric_key (str | None): Identifier for metric key.
            metric_keys (tuple[str, ...] | None): Metric keys.
            output_dir (Path): Filesystem location for output dir.
            cv_folds (int): Cv folds.
            random_seed (int | None): Random seed for reproducibility.
            validation_split (float | None): Validation split.
            silent_training_output (bool): Boolean option that controls behaviour.

        Returns:
            ClassificationExperimentResult: Result object for this operation.
        """

        if cv_folds < 2:
            raise InputValidationError("Cross-validation folds must be >= 2.")
        if len(y) < cv_folds:
            raise InputValidationError(
                "Cross-validation folds cannot exceed the number of samples."
            )
        if y.nunique(dropna=True) < 2:
            raise InputValidationError(
                "Classification requires at least two target classes."
            )
        min_class_count = int(y.value_counts(dropna=False).min())

        seed = random_seed
        if seed is None:
            seed = int(np.random.randint(0, 2_147_483_647))

        (
            cv,
            resolved_validation_split,
            split_strategy,
        ) = self._build_splitter(
            y=y,
            cv_folds=cv_folds,
            min_class_count=min_class_count,
            random_seed=seed,
            validation_split=validation_split,
        )
        resolved_metrics = self._resolve_metrics(
            metric_key=metric_key,
            metric_keys=metric_keys,
        )
        resolved_metric_keys = tuple(metric.key for metric in resolved_metrics)
        primary_metric_key = resolved_metric_keys[0]
        model = self._ensure_model_pipeline(estimator=estimator, X=X)
        fold_scores_by_metric: dict[str, list[float]] = {
            metric.key: [] for metric in resolved_metrics
        }
        validation_y_true: list[np.ndarray] = []
        validation_y_pred: list[np.ndarray] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
            task_id = progress.add_task(
                "Running cross-validation folds...",
                total=cv_folds + 2,
            )
            try:
                for fold_index, (train_indices, test_indices) in enumerate(
                    cv.split(X, y),
                    start=1,
                ):
                    fold_model = self._clone_model(model=model)
                    X_train = X.iloc[train_indices]
                    y_train = y.iloc[train_indices]
                    X_test = X.iloc[test_indices]
                    y_test = y.iloc[test_indices]
                    with self._silence_training_output(enabled=silent_training_output):
                        fold_model.fit(X_train, y_train)
                    for metric in resolved_metrics:
                        try:
                            with self._silence_training_output(
                                enabled=silent_training_output
                            ):
                                fold_score = float(
                                    metric.scorer(fold_model, X_test, y_test)
                                )
                        except MetricCompatibilityError as exc:
                            raise EstimatorMetricCompatibilityError(
                                "Estimator/metric compatibility failed: "
                                f"`{estimator_name}` does not support "
                                f"`{metric.key}`. {exc}"
                            ) from exc
                        fold_scores_by_metric[metric.key].append(fold_score)
                    with self._silence_training_output(enabled=silent_training_output):
                        fold_predictions = np.asarray(fold_model.predict(X_test))
                    validation_y_true.append(np.asarray(y_test))
                    validation_y_pred.append(fold_predictions)
                    progress.update(
                        task_id,
                        advance=1,
                        description=(
                            "Running cross-validation folds... "
                            f"({fold_index}/{cv_folds})"
                        ),
                    )
            except ValueError as exc:
                raise InputValidationError(
                    "Cross-validation failed for the selected metric/estimator "
                    f"combination: {exc}"
                ) from exc

            metric_summaries = {
                metric.key: ClassificationMetricSummary(
                    metric_key=metric.key,
                    fold_scores=tuple(fold_scores_by_metric[metric.key]),
                    mean_score=float(np.nanmean(fold_scores_by_metric[metric.key])),
                    std_score=float(np.nanstd(fold_scores_by_metric[metric.key])),
                )
                for metric in resolved_metrics
            }
            primary_metric_summary = metric_summaries[primary_metric_key]
            if not validation_y_true or not validation_y_pred:
                raise InputValidationError(
                    "Cross-validation did not produce validation predictions."
                )
            y_true_evaluation = np.concatenate(validation_y_true)
            y_pred_evaluation = np.concatenate(validation_y_pred)
            report_text = classification_report(
                y_true_evaluation,
                y_pred_evaluation,
                zero_division=0,
            )
            report_dict = classification_report(
                y_true_evaluation,
                y_pred_evaluation,
                zero_division=0,
                output_dict=True,
            )
            progress.update(task_id, description="Training final model...")
            with self._silence_training_output(enabled=silent_training_output):
                fitted_estimator = self._clone_model(model=model).fit(X, y)
            progress.update(task_id, advance=1, description="Writing artefacts...")
            artifacts = self._write_artifacts(
                estimator_name=estimator_name,
                metric_key=primary_metric_key,
                metric_keys=resolved_metric_keys,
                output_dir=output_dir,
                cv_folds=cv_folds,
                random_seed=seed,
                validation_split=resolved_validation_split,
                split_strategy=split_strategy,
                silent_training_output=silent_training_output,
                fold_scores=primary_metric_summary.fold_scores,
                metric_summaries=metric_summaries,
                report_text=report_text,
                fitted_estimator=fitted_estimator,
                class_distribution=y.value_counts(dropna=False).to_dict(),
            )
            progress.update(task_id, advance=1, description="Done")

        return ClassificationExperimentResult(
            estimator_name=estimator_name,
            metric_key=primary_metric_key,
            metric_keys=resolved_metric_keys,
            random_seed=seed,
            cv_folds=cv_folds,
            validation_split=resolved_validation_split,
            split_strategy=split_strategy,
            silent_training_output=silent_training_output,
            fold_scores=primary_metric_summary.fold_scores,
            mean_score=primary_metric_summary.mean_score,
            std_score=primary_metric_summary.std_score,
            metric_summaries=metric_summaries,
            classification_report_text=report_text,
            classification_report_dict=report_dict,
            fitted_estimator=fitted_estimator,
            artifacts=artifacts,
        )

    @staticmethod
    @beartype
    def _resolve_metrics(
        *,
        metric_key: str | None,
        metric_keys: tuple[str, ...] | None,
    ) -> tuple[MetricDefinition, ...]:
        """Resolve and validate one or more metric definitions."""

        if metric_key is not None and metric_keys is not None:
            raise InputValidationError(
                "Provide either `metric_key` or `metric_keys`, not both."
            )

        raw_metric_keys: tuple[str, ...]
        if metric_keys is not None:
            raw_metric_keys = tuple(metric.strip() for metric in metric_keys)
        elif metric_key is not None:
            raw_metric_keys = tuple(metric.strip() for metric in metric_key.split(","))
        else:
            raise InputValidationError(
                "At least one metric is required (metric_key or metric_keys)."
            )

        filtered_metric_keys = tuple(metric for metric in raw_metric_keys if metric)
        if not filtered_metric_keys:
            raise InputValidationError("At least one metric key is required.")

        resolved = []
        seen_metric_keys: set[str] = set()
        for raw_key in filtered_metric_keys:
            try:
                metric_definition = resolve_metric_definition(raw_key)
            except KeyError as exc:
                raise InputValidationError(f"Unknown metric: {raw_key}") from exc
            if metric_definition.key in seen_metric_keys:
                continue
            seen_metric_keys.add(metric_definition.key)
            resolved.append(metric_definition)
        return tuple(resolved)

    @staticmethod
    @beartype
    def _clone_model(*, model: BaseEstimator) -> BaseEstimator:
        """Clone estimator, with deepcopy fallback for non-cloneable models."""

        try:
            return clone(model)
        except (RuntimeError, TypeError, ValueError):
            pass

        try:
            return copy.deepcopy(model)
        except Exception as exc:
            raise InputValidationError(
                "Unable to duplicate estimator for cross-validation. "
                "This estimator is not sklearn-cloneable and deepcopy also failed: "
                f"{exc}"
            ) from exc

    @staticmethod
    @beartype
    def _build_splitter(
        *,
        y: pd.Series,
        cv_folds: int,
        min_class_count: int,
        random_seed: int,
        validation_split: float | None,
    ) -> tuple[StratifiedKFold | StratifiedShuffleSplit, float, str]:
        """Build and validate the CV splitter used by the experiment run."""

        if validation_split is None:
            if cv_folds > min_class_count:
                raise InputValidationError(
                    "Cross-validation folds exceed the smallest class count "
                    f"({min_class_count})."
                )
            return (
                StratifiedKFold(
                    n_splits=cv_folds,
                    shuffle=True,
                    random_state=random_seed,
                ),
                1.0 / float(cv_folds),
                "stratified_kfold",
            )

        if not 0.0 < validation_split < 1.0:
            raise InputValidationError(
                "Validation split must be a float strictly between 0 and 1."
            )

        class_count = int(y.nunique(dropna=True))
        sample_count = len(y)
        validation_sample_count = int(np.ceil(validation_split * sample_count))
        train_sample_count = sample_count - validation_sample_count
        if validation_sample_count < class_count:
            raise InputValidationError(
                "Validation split is too small for stratification: validation "
                "samples per split must be at least the number of classes "
                f"({class_count})."
            )
        if train_sample_count < class_count:
            raise InputValidationError(
                "Validation split is too large for stratification: training "
                "samples per split must be at least the number of classes "
                f"({class_count})."
            )
        if min_class_count < 2:
            raise InputValidationError(
                "Validation split requires at least 2 samples in each class."
            )

        return (
            StratifiedShuffleSplit(
                n_splits=cv_folds,
                test_size=validation_split,
                random_state=random_seed,
            ),
            float(validation_split),
            "stratified_shuffle_split",
        )

    @staticmethod
    @contextmanager
    @beartype
    def _silence_training_output(
        *,
        enabled: bool,
    ) -> Any:
        """Optionally silence estimator stdout/stderr during training operations."""

        if not enabled:
            yield
            return

        with StringIO() as captured_stdout, StringIO() as captured_stderr:
            with redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
                yield

    @staticmethod
    @beartype
    def _ensure_model_pipeline(
        *, estimator: BaseEstimator, X: pd.DataFrame
    ) -> BaseEstimator:
        """Wrap plain estimators with a preprocessing pipeline.

        Args:
            estimator: User-selected estimator.
            X: Raw feature matrix.

        Returns:
            BaseEstimator: Estimator (possibly wrapped in a pipeline).
        """

        if isinstance(estimator, Pipeline):
            return estimator

        numeric_columns = list(X.select_dtypes(include=["number"]).columns)
        categorical_columns = [
            column for column in X.columns if column not in numeric_columns
        ]

        transformers: list[tuple[str, object, list[str]]] = []
        if numeric_columns:
            numeric_transformer = Pipeline(
                steps=[("imputer", SimpleImputer(strategy="median"))]
            )
            transformers.append(("numeric", numeric_transformer, numeric_columns))
        if categorical_columns:
            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                ]
            )
            transformers.append(
                ("categorical", categorical_transformer, categorical_columns)
            )

        if not transformers:
            raise InputValidationError(
                "No usable feature columns were found for modelling."
            )

        preprocessor = ColumnTransformer(transformers=transformers)
        return Pipeline(
            steps=[("preprocessor", preprocessor), ("estimator", estimator)]
        )

    @staticmethod
    @beartype
    def _write_artifacts(
        *,
        estimator_name: str,
        metric_key: str,
        metric_keys: tuple[str, ...],
        output_dir: Path,
        cv_folds: int,
        random_seed: int,
        validation_split: float,
        split_strategy: str,
        silent_training_output: bool,
        fold_scores: tuple[float, ...],
        metric_summaries: dict[str, ClassificationMetricSummary],
        report_text: str,
        fitted_estimator: BaseEstimator,
        class_distribution: dict[object, int],
    ) -> ClassificationExperimentArtifacts:
        """Persist experiment artefacts.

        Args:
            estimator_name: Estimator display name.
            metric_key: Primary metric key used for scoring.
            metric_keys: Ordered metric keys evaluated for scoring.
            output_dir: Artefact output directory.
            cv_folds: Number of CV folds.
            random_seed: Random seed used for experiment.
            fold_scores: Per-fold primary-metric scores.
            metric_summaries: Per-metric fold/mean/std summaries.
            report_text: Classification report text.
            fitted_estimator: Retrained estimator.
            class_distribution: Observed target-class distribution.

        Returns:
            ClassificationExperimentArtifacts: Saved artefact paths.
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_estimator = estimator_name.lower().replace(" ", "_")
        artifact_prefix = f"{safe_estimator}_{timestamp}"

        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"{artifact_prefix}_model.pkl"
        summary_path = output_dir / f"{artifact_prefix}_summary.json"
        report_path = output_dir / f"{artifact_prefix}_classification_report.txt"

        with model_path.open("wb") as file:
            pickle.dump(fitted_estimator, file)

        metric_summaries_payload = {
            key: {
                "fold_scores": [float(score) for score in summary.fold_scores],
                "mean_score": float(summary.mean_score),
                "std_score": float(summary.std_score),
            }
            for key, summary in metric_summaries.items()
        }
        summary_payload = {
            "estimator": estimator_name,
            "metric_of_interest": metric_key,
            "metrics_of_interest": list(metric_keys),
            "cross_validation_folds": cv_folds,
            "random_seed": random_seed,
            "validation_split": float(validation_split),
            "split_strategy": split_strategy,
            "silent_training_output": bool(silent_training_output),
            "fold_scores": [float(score) for score in fold_scores],
            "mean_score": float(np.nanmean(fold_scores)),
            "std_score": float(np.nanstd(fold_scores)),
            "metric_summaries": metric_summaries_payload,
            "class_distribution": {
                str(key): int(value) for key, value in class_distribution.items()
            },
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2))
        report_path.write_text(report_text)

        return ClassificationExperimentArtifacts(
            model_path=model_path,
            summary_path=summary_path,
            report_path=report_path,
        )
