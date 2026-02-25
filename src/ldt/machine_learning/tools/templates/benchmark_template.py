from __future__ import annotations

import json
import re
import warnings
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path

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
from sklearn.base import BaseEstimator

from ldt.machine_learning.tools.metrics import resolve_metric_definition
from ldt.utils.errors import InputValidationError

from .experiment_template import (
    ClassificationExperimentArtifacts,
    ClassificationExperimentTemplate,
    ClassificationMetricSummary,
    EstimatorMetricCompatibilityError,
)

_PIPELINE_PROFILER_MAX_NOTEBOOK_PIPELINES = 48


@beartype
@dataclass(frozen=True)
class BenchmarkEstimatorSpec:
    """Specification for one estimator participating in a benchmark run.

    Attributes:
        estimator_key (str): Identifier for estimator key.
        estimator_name (str): Name for estimator.
        estimator (BaseEstimator): Estimator.

    """

    estimator_key: str
    estimator_name: str
    estimator: BaseEstimator


@beartype
@dataclass(frozen=True)
class BenchmarkEstimatorResult:
    """Benchmark summary for one estimator.

    Attributes:
        estimator_key (str): Identifier for estimator key.
        estimator_name (str): Name for estimator.
        metric_key (str): Identifier for metric key.
        metric_keys (tuple[str, ...]): Metric keys.
        fold_scores (tuple[float, ...]): Fold scores.
        mean_score (float): Mean score.
        std_score (float): Std score.
        metric_summaries (dict[str, ClassificationMetricSummary]): Metric summaries.
        artifacts (ClassificationExperimentArtifacts): Artefacts.

    """

    estimator_key: str
    estimator_name: str
    metric_key: str
    metric_keys: tuple[str, ...]
    fold_scores: tuple[float, ...]
    mean_score: float
    std_score: float
    metric_summaries: dict[str, ClassificationMetricSummary]
    artifacts: ClassificationExperimentArtifacts


@beartype
@dataclass(frozen=True)
class BenchmarkSkippedEstimator:
    """Estimator excluded during benchmark execution.

    Attributes:
        estimator_key (str): Identifier for estimator key.
        estimator_name (str): Name for estimator.
        reason (str): Reason.

    """

    estimator_key: str
    estimator_name: str
    reason: str


@beartype
@dataclass(frozen=True)
class BenchmarkPipelineProfilerArtifacts:
    """PipelineProfiler compatibility artefacts.

    Attributes:
        input_json_path (Path): Path for input json path.
        html_path (Path | None): Path for html path.
        warning (str | None): Warning.

    """

    input_json_path: Path
    html_path: Path | None
    warning: str | None


@beartype
@dataclass(frozen=True)
class ClassificationBenchmarkArtifacts:
    """Files produced by one benchmark run.

    Attributes:
        benchmark_output_dir (Path): Directory for benchmark output dir.
        ranking_path (Path): Path for ranking path.
        summary_path (Path): Path for summary path.
        report_path (Path): Path for report path.
        pipeline_profiler (BenchmarkPipelineProfilerArtifacts): Pipeline profiler.

    """

    benchmark_output_dir: Path
    ranking_path: Path
    summary_path: Path
    report_path: Path
    pipeline_profiler: BenchmarkPipelineProfilerArtifacts


@beartype
@dataclass(frozen=True)
class ClassificationBenchmarkResult:
    """Structured output for one benchmark run.

    Attributes:
        benchmark_name (str): Name for benchmark.
        metric_key (str): Identifier for metric key.
        metric_keys (tuple[str, ...]): Metric keys.
        random_seed (int): Random seed for reproducibility.
        cv_folds (int): Cv folds.
        validation_split (float): Validation split.
        split_strategy (str): Split strategy.
        silent_training_output (bool): Whether to silent training output.
        ranked_estimators (tuple[BenchmarkEstimatorResult, ...]): Ranked estimators.
        skipped_estimators (tuple[BenchmarkSkippedEstimator, ...]): Skipped estimators.
        artifacts (ClassificationBenchmarkArtifacts): Artefacts.

    """

    benchmark_name: str
    metric_key: str
    metric_keys: tuple[str, ...]
    random_seed: int
    cv_folds: int
    validation_split: float
    split_strategy: str
    silent_training_output: bool
    ranked_estimators: tuple[BenchmarkEstimatorResult, ...]
    skipped_estimators: tuple[BenchmarkSkippedEstimator, ...]
    artifacts: ClassificationBenchmarkArtifacts


@beartype
class ClassificationBenchmarkTemplate:
    """Reusable template for estimator benchmarking under one protocol."""

    @staticmethod
    @beartype
    def resolve_pipeline_profiler_max_pipelines_limit(*, available: int) -> int:
        """Resolve effective notebook plotting cap for PipelineProfiler.

        This helper is used by higher-level workflows and keeps input/output handling consistent.

        Args:
            available (int): Number of available items.

        Returns:
            int: Parsed integer value.
        """

        if available < 1:
            raise InputValidationError(
                "PipelineProfiler payload is empty. Run a benchmark first."
            )
        return min(available, _PIPELINE_PROFILER_MAX_NOTEBOOK_PIPELINES)

    @staticmethod
    @beartype
    def resolve_pipeline_profiler_max_pipelines(
        *,
        requested: int,
        available: int,
    ) -> int:
        """Cap requested notebook pipeline count to supported bounds.

        This helper is used by higher-level workflows and keeps input/output handling consistent.

        Args:
            requested (int): Number of requested items.
            available (int): Number of available items.

        Returns:
            int: Parsed integer value.
        """

        if requested < 1:
            raise InputValidationError("Maximum pipelines to plot must be >= 1.")
        limit = ClassificationBenchmarkTemplate.resolve_pipeline_profiler_max_pipelines_limit(
            available=available
        )
        return min(requested, limit)

    @staticmethod
    @beartype
    def resolve_pipeline_profiler_payload_path(*, input_path: Path) -> Path:
        """Resolve a PipelineProfiler payload path from benchmark artefacts.

        This helper is used by higher-level workflows and keeps input/output handling consistent.

        Args:
            input_path (Path): Filesystem path used by the workflow.

        Returns:
            Path: Resolved filesystem path.
        """

        candidate = input_path.expanduser()
        if not candidate.exists() or not candidate.is_file():
            raise InputValidationError(f"Path does not exist: {candidate}")
        if candidate.suffix.lower() != ".json":
            raise InputValidationError("Input path must point to a JSON file.")

        raw_payload = json.loads(candidate.read_text())
        if isinstance(raw_payload, list):
            return candidate
        if not isinstance(raw_payload, dict):
            raise InputValidationError(
                "JSON payload must be a PipelineProfiler list or benchmark summary."
            )

        pipeline_section = raw_payload.get("pipeline_profiler")
        if not isinstance(pipeline_section, dict):
            raise InputValidationError(
                "Summary JSON is missing `pipeline_profiler` metadata."
            )
        input_json_raw = pipeline_section.get("input_json")
        if not isinstance(input_json_raw, str) or not input_json_raw.strip():
            raise InputValidationError(
                "Summary JSON is missing `pipeline_profiler.input_json`."
            )

        resolved = Path(input_json_raw).expanduser()
        if not resolved.is_absolute():
            resolved = (candidate.parent / resolved).resolve()
        if not resolved.exists() or not resolved.is_file():
            raise InputValidationError(
                f"Resolved PipelineProfiler input JSON does not exist: {resolved}"
            )
        return resolved

    @staticmethod
    @beartype
    def load_pipeline_profiler_payload(
        *, payload_path: Path
    ) -> list[dict[str, object]]:
        """Load and validate a PipelineProfiler payload from JSON.

        Args:
            payload_path (Path): Filesystem path used by the workflow.

        Returns:
            list[dict[str, object]]: List of parsed values.
        """

        candidate = payload_path.expanduser()
        if not candidate.exists() or not candidate.is_file():
            raise InputValidationError(f"Path does not exist: {candidate}")
        raw_payload = json.loads(candidate.read_text())
        if not isinstance(raw_payload, list):
            raise InputValidationError(
                "PipelineProfiler payload JSON must contain a list."
            )
        validated_payload: list[dict[str, object]] = []
        for item in raw_payload:
            if not isinstance(item, dict):
                raise InputValidationError(
                    "PipelineProfiler payload entries must be JSON objects."
                )
            validated_payload.append(item)
        if not validated_payload:
            raise InputValidationError(
                "PipelineProfiler payload is empty. Run a benchmark first."
            )
        return validated_payload

    @staticmethod
    @beartype
    def _resolve_metric_keys(
        *,
        metric_key: str | None,
        metric_keys: tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        """Resolve one or more metric keys into canonical unique keys."""

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

        resolved_metric_keys: list[str] = []
        seen_metric_keys: set[str] = set()
        for raw_metric_key in filtered_metric_keys:
            try:
                resolved_metric_key = resolve_metric_definition(raw_metric_key).key
            except KeyError as exc:
                raise InputValidationError(f"Unknown metric: {raw_metric_key}") from exc
            if resolved_metric_key in seen_metric_keys:
                continue
            seen_metric_keys.add(resolved_metric_key)
            resolved_metric_keys.append(resolved_metric_key)
        return tuple(resolved_metric_keys)

    @beartype
    def run(
        self,
        *,
        benchmark_name: str,
        estimator_specs: tuple[BenchmarkEstimatorSpec, ...],
        X: pd.DataFrame,
        y: pd.Series,
        metric_key: str | None = None,
        metric_keys: tuple[str, ...] | None = None,
        output_root_dir: Path,
        cv_folds: int = 10,
        random_seed: int | None = None,
        validation_split: float | None = None,
        silent_training_output: bool = False,
        generate_pipeline_profiler_html: bool = False,
    ) -> ClassificationBenchmarkResult:
        """Execute a benchmark run across multiple estimators.

        Args:
            benchmark_name (str): Name for benchmark.
            estimator_specs (tuple[BenchmarkEstimatorSpec, ...]): Estimator specs.
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target labels.
            metric_key (str | None): Identifier for metric key.
            metric_keys (tuple[str, ...] | None): Metric keys.
            output_root_dir (Path): Filesystem location for output root dir.
            cv_folds (int): Cv folds.
            random_seed (int | None): Random seed for reproducibility.
            validation_split (float | None): Validation split.
            silent_training_output (bool): Boolean option that controls behaviour.
            generate_pipeline_profiler_html (bool): Boolean option that controls behaviour.

        Returns:
            ClassificationBenchmarkResult: Result object for this operation.
        """

        if not estimator_specs:
            raise InputValidationError("Benchmark requires at least one estimator.")

        estimator_keys = [spec.estimator_key for spec in estimator_specs]
        if len(set(estimator_keys)) != len(estimator_keys):
            raise InputValidationError("Benchmark estimator keys must be unique.")

        resolved_metric_keys = self._resolve_metric_keys(
            metric_key=metric_key,
            metric_keys=metric_keys,
        )
        ranking_metric_key = resolved_metric_keys[0]

        seed = (
            random_seed
            if random_seed is not None
            else int(np.random.randint(0, 2_147_483_647))
        )
        resolved_validation_split = (
            float(validation_split)
            if validation_split is not None
            else (1.0 / float(cv_folds))
        )
        split_strategy = (
            "stratified_shuffle_split"
            if validation_split is not None
            else "stratified_kfold"
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_benchmark_name = self._slugify_label(benchmark_name)
        benchmark_output_dir = (
            output_root_dir.expanduser() / f"{safe_benchmark_name}_{timestamp}"
        )
        estimators_output_dir = benchmark_output_dir / "estimators"
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)

        experiment_template = ClassificationExperimentTemplate()
        benchmark_results: list[BenchmarkEstimatorResult] = []
        skipped_estimators: list[BenchmarkSkippedEstimator] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            transient=False,
        ) as progress:
            task_id = progress.add_task(
                "Running estimator benchmark...", total=len(estimator_specs) + 3
            )
            for index, estimator_spec in enumerate(estimator_specs, start=1):
                progress.update(
                    task_id,
                    description=(
                        "Running estimator benchmark... "
                        f"({index}/{len(estimator_specs)}) "
                        f"{estimator_spec.estimator_key}"
                    ),
                )
                try:
                    estimator_result = experiment_template.run(
                        estimator=estimator_spec.estimator,
                        estimator_name=estimator_spec.estimator_name,
                        X=X,
                        y=y,
                        metric_keys=resolved_metric_keys,
                        output_dir=estimators_output_dir / estimator_spec.estimator_key,
                        cv_folds=cv_folds,
                        random_seed=seed,
                        validation_split=validation_split,
                        silent_training_output=silent_training_output,
                    )
                except (
                    EstimatorMetricCompatibilityError,
                    InputValidationError,
                    Exception,
                ) as exc:
                    failure_reason = str(exc).strip() or (
                        f"{type(exc).__name__}: estimator execution failed."
                    )
                    skipped_estimators.append(
                        BenchmarkSkippedEstimator(
                            estimator_key=estimator_spec.estimator_key,
                            estimator_name=estimator_spec.estimator_name,
                            reason=failure_reason,
                        )
                    )
                    benchmark_results.append(
                        self._build_failed_estimator_result(
                            estimator_key=estimator_spec.estimator_key,
                            estimator_name=estimator_spec.estimator_name,
                            metric_key=ranking_metric_key,
                            metric_keys=resolved_metric_keys,
                            cv_folds=cv_folds,
                            output_dir=estimators_output_dir
                            / estimator_spec.estimator_key,
                            reason=failure_reason,
                        )
                    )
                    progress.update(task_id, advance=1)
                    continue
                benchmark_results.append(
                    BenchmarkEstimatorResult(
                        estimator_key=estimator_spec.estimator_key,
                        estimator_name=estimator_spec.estimator_name,
                        metric_key=estimator_result.metric_key,
                        metric_keys=estimator_result.metric_keys,
                        fold_scores=estimator_result.fold_scores,
                        mean_score=estimator_result.mean_score,
                        std_score=estimator_result.std_score,
                        metric_summaries=estimator_result.metric_summaries,
                        artifacts=estimator_result.artifacts,
                    )
                )
                progress.update(task_id, advance=1)

            ranked_estimators = tuple(
                sorted(
                    benchmark_results,
                    key=lambda estimator_result: estimator_result.mean_score,
                    reverse=True,
                )
            )
            ranking_path = benchmark_output_dir / "benchmark_ranking.csv"
            report_path = benchmark_output_dir / "benchmark_report.txt"
            summary_path = benchmark_output_dir / "benchmark_summary.json"

            progress.update(task_id, description="Writing benchmark ranking...")
            ranking_table = self._build_ranking_dataframe(
                ranked_estimators=ranked_estimators
            )
            ranking_table.to_csv(ranking_path, index=False)
            progress.update(task_id, advance=1)

            progress.update(task_id, description="Writing benchmark report...")
            report_path.write_text(
                self._build_report_text(
                    benchmark_name=benchmark_name,
                    metric_key=ranking_metric_key,
                    metric_keys=resolved_metric_keys,
                    cv_folds=cv_folds,
                    validation_split=resolved_validation_split,
                    split_strategy=split_strategy,
                    silent_training_output=silent_training_output,
                    random_seed=seed,
                    ranked_estimators=ranked_estimators,
                    skipped_estimators=tuple(skipped_estimators),
                )
            )
            progress.update(task_id, advance=1)

            progress.update(
                task_id, description="Exporting PipelineProfiler artefacts..."
            )
            pipeline_payload = self._build_pipeline_profiler_payload(
                benchmark_name=benchmark_name,
                metric_key=ranking_metric_key,
                ranked_estimators=ranked_estimators,
            )
            pipeline_input_json_path = (
                benchmark_output_dir / "pipelineprofiler_input.json"
            )
            pipeline_input_json_path.write_text(json.dumps(pipeline_payload, indent=2))
            pipeline_html_path: Path | None = None
            pipeline_warning: str | None = None
            if generate_pipeline_profiler_html:
                candidate_html_path = (
                    benchmark_output_dir / "pipelineprofiler_view.html"
                )
                try:
                    candidate_html_path.write_text(
                        self._build_pipeline_profiler_html_document(
                            pipelines=pipeline_payload
                        ),
                        encoding="utf-8",
                    )
                    pipeline_html_path = candidate_html_path
                except Exception as exc:
                    pipeline_warning = (
                        "Failed to generate standalone PipelineProfiler HTML: "
                        f"{type(exc).__name__}: {str(exc).strip()}. "
                        "Use the reproducibility notebook to render "
                        "`PipelineProfiler.plot_pipeline_matrix(...)` inline."
                    )
            else:
                pipeline_warning = (
                    "Standalone PipelineProfiler HTML was not generated "
                    "(default setting). Use the reproducibility notebook to "
                    "render `PipelineProfiler.plot_pipeline_matrix(...)` inline."
                )
            pipeline_artifacts = BenchmarkPipelineProfilerArtifacts(
                input_json_path=pipeline_input_json_path,
                html_path=pipeline_html_path,
                warning=pipeline_warning,
            )
            summary_payload = self._build_summary_payload(
                benchmark_name=benchmark_name,
                metric_key=ranking_metric_key,
                metric_keys=resolved_metric_keys,
                random_seed=seed,
                cv_folds=cv_folds,
                validation_split=resolved_validation_split,
                split_strategy=split_strategy,
                silent_training_output=silent_training_output,
                ranked_estimators=ranked_estimators,
                skipped_estimators=tuple(skipped_estimators),
                artifacts=ClassificationBenchmarkArtifacts(
                    benchmark_output_dir=benchmark_output_dir,
                    ranking_path=ranking_path,
                    summary_path=summary_path,
                    report_path=report_path,
                    pipeline_profiler=pipeline_artifacts,
                ),
            )
            summary_path.write_text(json.dumps(summary_payload, indent=2))
            progress.update(task_id, advance=1, description="Done")

        return ClassificationBenchmarkResult(
            benchmark_name=benchmark_name,
            metric_key=ranking_metric_key,
            metric_keys=resolved_metric_keys,
            random_seed=seed,
            cv_folds=cv_folds,
            validation_split=resolved_validation_split,
            split_strategy=split_strategy,
            silent_training_output=silent_training_output,
            ranked_estimators=ranked_estimators,
            skipped_estimators=tuple(skipped_estimators),
            artifacts=ClassificationBenchmarkArtifacts(
                benchmark_output_dir=benchmark_output_dir,
                ranking_path=ranking_path,
                summary_path=summary_path,
                report_path=report_path,
                pipeline_profiler=pipeline_artifacts,
            ),
        )

    @staticmethod
    @beartype
    def _build_pipeline_profiler_html_document(
        *, pipelines: list[dict[str, object]]
    ) -> str:
        """Build standalone PipelineProfiler HTML from a payload list."""

        # PipelineProfiler prints non-critical notebook-detection messages during
        # import outside Jupyter/Colab; suppress them for CLI runs.
        with (
            warnings.catch_warnings(),
            redirect_stdout(StringIO()),
            redirect_stderr(StringIO()),
        ):
            warnings.filterwarnings(
                "ignore",
                message=r"pkg_resources is deprecated as an API\..*",
                category=UserWarning,
                module=r"PipelineProfiler\._plot_pipeline_matrix",
            )
            from PipelineProfiler import get_pipeline_profiler_html

        return str(get_pipeline_profiler_html(pipelines))

    @staticmethod
    @beartype
    def _build_failed_estimator_result(
        *,
        estimator_key: str,
        estimator_name: str,
        metric_key: str,
        metric_keys: tuple[str, ...],
        cv_folds: int,
        output_dir: Path,
        reason: str,
    ) -> BenchmarkEstimatorResult:
        """Build fallback benchmark result for one failed estimator."""

        fallback_fold_scores = tuple(-1.0 for _ in range(cv_folds))
        fallback_metric_summaries = {
            current_metric_key: ClassificationMetricSummary(
                metric_key=current_metric_key,
                fold_scores=fallback_fold_scores,
                mean_score=-1.0,
                std_score=0.0,
            )
            for current_metric_key in metric_keys
        }
        artifacts = ClassificationBenchmarkTemplate._write_failed_estimator_artifacts(
            estimator_name=estimator_name,
            metric_key=metric_key,
            metric_keys=metric_keys,
            output_dir=output_dir,
            cv_folds=cv_folds,
            reason=reason,
        )
        return BenchmarkEstimatorResult(
            estimator_key=estimator_key,
            estimator_name=estimator_name,
            metric_key=metric_key,
            metric_keys=metric_keys,
            fold_scores=fallback_fold_scores,
            mean_score=-1.0,
            std_score=0.0,
            metric_summaries=fallback_metric_summaries,
            artifacts=artifacts,
        )

    @staticmethod
    @beartype
    def _write_failed_estimator_artifacts(
        *,
        estimator_name: str,
        metric_key: str,
        metric_keys: tuple[str, ...],
        output_dir: Path,
        cv_folds: int,
        reason: str,
    ) -> ClassificationExperimentArtifacts:
        """Persist placeholder artefacts for failed estimators."""

        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "failed_model.pkl"
        summary_path = output_dir / "failed_summary.json"
        report_path = output_dir / "failed_classification_report.txt"

        model_path.write_bytes(b"")
        summary_path.write_text(
            json.dumps(
                {
                    "status": "failed",
                    "estimator": estimator_name,
                    "metric_of_interest": metric_key,
                    "metrics_of_interest": list(metric_keys),
                    "cross_validation_folds": cv_folds,
                    "assigned_score": -1.0,
                    "failure_reason": reason,
                },
                indent=2,
            )
        )
        report_path.write_text(
            "Estimator failed during benchmark execution and was assigned "
            "fallback score -1.0.\n"
            f"Reason: {reason}\n"
        )
        return ClassificationExperimentArtifacts(
            model_path=model_path,
            summary_path=summary_path,
            report_path=report_path,
        )

    @staticmethod
    @beartype
    def _build_ranking_dataframe(
        *, ranked_estimators: tuple[BenchmarkEstimatorResult, ...]
    ) -> pd.DataFrame:
        """Build a tabular ranking dataframe from benchmark results."""

        rows = []
        for rank, estimator in enumerate(ranked_estimators, start=1):
            rows.append(
                {
                    "rank": rank,
                    "estimator_key": estimator.estimator_key,
                    "estimator_name": estimator.estimator_name,
                    "metric_key": estimator.metric_key,
                    "metric_keys": json.dumps(list(estimator.metric_keys)),
                    "mean_score": estimator.mean_score,
                    "std_score": estimator.std_score,
                    "fold_scores": json.dumps(
                        [float(score) for score in estimator.fold_scores]
                    ),
                    "metric_summaries": json.dumps(
                        ClassificationBenchmarkTemplate._serialise_metric_summaries(
                            metric_summaries=estimator.metric_summaries
                        )
                    ),
                    "model_artifact": str(estimator.artifacts.model_path.resolve()),
                    "summary_artifact": str(estimator.artifacts.summary_path.resolve()),
                    "report_artifact": str(estimator.artifacts.report_path.resolve()),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    @beartype
    def _build_report_text(
        *,
        benchmark_name: str,
        metric_key: str,
        metric_keys: tuple[str, ...],
        cv_folds: int,
        validation_split: float,
        split_strategy: str,
        silent_training_output: bool,
        random_seed: int,
        ranked_estimators: tuple[BenchmarkEstimatorResult, ...],
        skipped_estimators: tuple[BenchmarkSkippedEstimator, ...],
    ) -> str:
        """Build human-readable benchmark report text."""

        lines = [
            "Classification Benchmark Report",
            f"Benchmark: {benchmark_name}",
            f"Ranking metric: {metric_key}",
            f"Metrics evaluated: {', '.join(metric_keys)}",
            f"Cross-validation folds: {cv_folds}",
            f"Validation split: {validation_split:.4f}",
            f"Split strategy: {split_strategy}",
            f"Silent training output: {silent_training_output}",
            f"Random seed: {random_seed}",
            "",
            "Ranking",
            "-------",
        ]
        for rank, estimator in enumerate(ranked_estimators, start=1):
            ranking_fold_scores_text = ", ".join(
                f"{score:.4f}" for score in estimator.fold_scores
            )
            lines.append(
                f"{rank}. {estimator.estimator_name} [{estimator.estimator_key}] "
                f"| mean={estimator.mean_score:.4f} "
                f"| std={estimator.std_score:.4f}"
            )
            lines.append(f"   {metric_key} folds: {ranking_fold_scores_text}")
            for metric in metric_keys:
                if metric == metric_key:
                    continue
                metric_summary = estimator.metric_summaries[metric]
                metric_fold_scores_text = ", ".join(
                    f"{score:.4f}" for score in metric_summary.fold_scores
                )
                lines.append(
                    f"   {metric}: mean={metric_summary.mean_score:.4f} "
                    f"| std={metric_summary.std_score:.4f}"
                )
                lines.append(f"   {metric} folds: {metric_fold_scores_text}")
        if skipped_estimators:
            lines.extend(
                [
                    "",
                    "Skipped/failed estimators",
                    "-------------------------",
                ]
            )
            for skipped_estimator in skipped_estimators:
                lines.append(
                    f"- {skipped_estimator.estimator_name} "
                    f"[{skipped_estimator.estimator_key}]"
                )
                lines.append("  assigned_score: -1.0")
                lines.append(f"  reason: {skipped_estimator.reason}")
        lines.append("")
        return "\n".join(lines)

    @staticmethod
    @beartype
    def _build_pipeline_profiler_payload(
        *,
        benchmark_name: str,
        metric_key: str,
        ranked_estimators: tuple[BenchmarkEstimatorResult, ...],
    ) -> list[dict[str, object]]:
        """Build a minimal PipelineProfiler payload from benchmark rankings."""

        metric_keys = (
            tuple(ranked_estimators[0].metric_keys)
            if ranked_estimators
            else (metric_key,)
        )
        if metric_key not in metric_keys:
            metric_keys = (metric_key, *metric_keys)
        safe_problem_id = (
            f"ldt_{ClassificationBenchmarkTemplate._slugify_label(benchmark_name)}"
        )

        normalised_scores_by_metric: dict[str, np.ndarray] = {}
        for current_metric_key in metric_keys:
            raw_scores = np.asarray(
                [
                    estimator.metric_summaries[current_metric_key].mean_score
                    for estimator in ranked_estimators
                ],
                dtype=float,
            )
            normalised_scores_by_metric[current_metric_key] = (
                ClassificationBenchmarkTemplate._normalise_scores(raw_scores)
            )

        payload: list[dict[str, object]] = []
        for index, estimator in enumerate(ranked_estimators, start=1):
            pipeline_digest = f"{estimator.estimator_key}_{index}"
            pipeline_steps = ClassificationBenchmarkTemplate._build_d3m_like_steps(
                estimator_key=estimator.estimator_key,
                estimator_name=estimator.estimator_name,
            )
            score_entries: list[dict[str, object]] = []
            for current_metric_key in metric_keys:
                metric_summary = estimator.metric_summaries[current_metric_key]
                score_entries.append(
                    {
                        "metric": {
                            "metric": current_metric_key,
                            "name": current_metric_key,
                        },
                        "value": float(metric_summary.mean_score),
                        "normalized": float(
                            normalised_scores_by_metric[current_metric_key][index - 1]
                        ),
                    }
                )
            payload.append(
                {
                    "pipeline_id": pipeline_digest,
                    "pipeline_digest": pipeline_digest,
                    "digest": pipeline_digest,
                    "rank": index,
                    "inputs": [
                        {"name": "inputs.0"},
                    ],
                    "outputs": [
                        {"data": "steps.1.produce"},
                    ],
                    "steps": pipeline_steps,
                    "scores": score_entries,
                    "pipeline_source": {
                        "name": estimator.estimator_name,
                    },
                    "problem": {
                        "id": safe_problem_id,
                        "digest": safe_problem_id,
                    },
                }
            )
        return payload

    @staticmethod
    @beartype
    def _build_d3m_like_steps(
        *, estimator_key: str, estimator_name: str
    ) -> list[dict[str, object]]:
        """Build D3M-style pipeline steps for PipelineProfiler rendering."""

        (
            preprocessor_python_path,
            preprocessor_name,
            classifier_python_path_component,
            classifier_name,
        ) = ClassificationBenchmarkTemplate._resolve_pipeline_profiler_steps_components(
            estimator_key=estimator_key,
            estimator_name=estimator_name,
        )
        return [
            {
                "type": "PRIMITIVE",
                "primitive": {
                    "python_path": preprocessor_python_path,
                    "name": preprocessor_name,
                },
                "arguments": {
                    "inputs": {
                        "type": "CONTAINER",
                        "data": "inputs.0",
                    }
                },
                "outputs": [{"id": "produce"}],
                "hyperparams": {},
            },
            {
                "type": "PRIMITIVE",
                "primitive": {
                    "python_path": (
                        "d3m.primitives.classification."
                        f"{classifier_python_path_component}.LDT"
                    ),
                    "name": classifier_name,
                },
                "arguments": {
                    "inputs": {
                        "type": "CONTAINER",
                        "data": "steps.0.produce",
                    }
                },
                "outputs": [{"id": "produce"}],
                "hyperparams": {},
            },
        ]

    @staticmethod
    @beartype
    def _resolve_pipeline_profiler_steps_components(
        *,
        estimator_key: str,
        estimator_name: str,
    ) -> tuple[str, str, str, str]:
        """Resolve profiler preprocessor/classifier step metadata.

        For longitudinal estimators (`strategy__estimator`), the first step is
        emitted as a strategy-specific preprocessing primitive and the second
        step is the underlying estimator. For standard estimators, we keep the
        default LDT tabular preprocessor + estimator structure.
        """

        strategy_labels = {
            "merwav_time_minus": "MerWavTimeMinus",
            "aggrfunc_mean": "AggrFunc (Mean)",
            "aggrfunc_median": "AggrFunc (Median)",
            "sepwav_voting": "SepWav (Voting)",
            "sepwav_stacking_lr": "SepWav (Stacking + Logistic Regression)",
            "sepwav_stacking_dt": "SepWav (Stacking + Decision Tree)",
            "merwav_time_plus": "MerWavTimePlus",
        }
        if "__" in estimator_key:
            strategy_key, classifier_key_raw = estimator_key.split("__", maxsplit=1)
            strategy_label = strategy_labels.get(strategy_key)
            if strategy_label is not None:
                safe_strategy_key = ClassificationBenchmarkTemplate._pipeline_profiler_safe_primitive_component(
                    strategy_key
                )
                safe_classifier_key = ClassificationBenchmarkTemplate._pipeline_profiler_safe_primitive_component(
                    classifier_key_raw
                )
                classifier_label = estimator_name
                if " + " in estimator_name:
                    classifier_label = estimator_name.split(" + ", maxsplit=1)[1]
                return (
                    f"d3m.primitives.data_preprocessing.{safe_strategy_key}.LDT",
                    strategy_label,
                    safe_classifier_key,
                    classifier_label,
                )

        safe_estimator_key = (
            ClassificationBenchmarkTemplate._pipeline_profiler_safe_primitive_component(
                estimator_key
            )
        )
        return (
            "d3m.primitives.data_preprocessing.ldt_tabular_preprocessor.Common",
            "LDT Tabular Preprocessor",
            safe_estimator_key,
            estimator_name,
        )

    @staticmethod
    @beartype
    def _pipeline_profiler_safe_primitive_component(component: str) -> str:
        """Normalise primitive component for PipelineProfiler JS compatibility.

        PipelineProfiler's label helper splits primitive names by `_` and assumes
        no empty tokens. Longitudinal keys can contain `__`, which creates empty
        tokens and breaks front-end rendering. We collapse separators and remove
        unsupported characters for the synthetic D3M path only.
        """

        safe_component = re.sub(r"[^a-zA-Z0-9_]+", "_", component)
        safe_component = re.sub(r"_+", "_", safe_component).strip("_")
        return safe_component or "estimator"

    @staticmethod
    @beartype
    def _normalise_scores(scores: np.ndarray) -> np.ndarray:
        """Normalise scores to [0, 1] for PipelineProfiler colour mapping."""

        if scores.size == 0:
            return np.asarray([], dtype=float)
        if np.all(np.isnan(scores)):
            return np.zeros_like(scores, dtype=float)
        minimum = np.nanmin(scores)
        maximum = np.nanmax(scores)
        spread = maximum - minimum
        if spread <= 0:
            normalised = np.ones_like(scores, dtype=float)
            normalised[np.isnan(scores)] = 0.0
            return normalised
        normalised = (scores - minimum) / spread
        normalised[np.isnan(scores)] = 0.0
        return normalised

    @staticmethod
    @beartype
    def _build_summary_payload(
        *,
        benchmark_name: str,
        metric_key: str,
        metric_keys: tuple[str, ...],
        random_seed: int,
        cv_folds: int,
        validation_split: float,
        split_strategy: str,
        silent_training_output: bool,
        ranked_estimators: tuple[BenchmarkEstimatorResult, ...],
        skipped_estimators: tuple[BenchmarkSkippedEstimator, ...],
        artifacts: ClassificationBenchmarkArtifacts,
    ) -> dict[str, object]:
        """Build benchmark JSON summary payload."""

        return {
            "benchmark_name": benchmark_name,
            "metric_of_interest": metric_key,
            "metrics_of_interest": list(metric_keys),
            "cross_validation_folds": cv_folds,
            "validation_split": float(validation_split),
            "split_strategy": split_strategy,
            "silent_training_output": bool(silent_training_output),
            "random_seed": random_seed,
            "benchmark_output_directory": str(artifacts.benchmark_output_dir.resolve()),
            "ranking_artifact": str(artifacts.ranking_path.resolve()),
            "report_artifact": str(artifacts.report_path.resolve()),
            "pipeline_profiler": {
                "input_json": str(
                    artifacts.pipeline_profiler.input_json_path.resolve()
                ),
                "html_view": (
                    str(artifacts.pipeline_profiler.html_path.resolve())
                    if artifacts.pipeline_profiler.html_path is not None
                    else None
                ),
                "warning": artifacts.pipeline_profiler.warning,
            },
            "ranked_estimators": [
                {
                    "rank": rank,
                    "estimator_key": estimator.estimator_key,
                    "estimator_name": estimator.estimator_name,
                    "metric_key": estimator.metric_key,
                    "metric_keys": list(estimator.metric_keys),
                    "mean_score": estimator.mean_score,
                    "std_score": estimator.std_score,
                    "fold_scores": [float(score) for score in estimator.fold_scores],
                    "metric_summaries": ClassificationBenchmarkTemplate._serialise_metric_summaries(
                        metric_summaries=estimator.metric_summaries
                    ),
                    "model_artifact": str(estimator.artifacts.model_path.resolve()),
                    "summary_artifact": str(estimator.artifacts.summary_path.resolve()),
                    "report_artifact": str(estimator.artifacts.report_path.resolve()),
                }
                for rank, estimator in enumerate(ranked_estimators, start=1)
            ],
            "skipped_estimators": [
                {
                    "estimator_key": skipped_estimator.estimator_key,
                    "estimator_name": skipped_estimator.estimator_name,
                    "reason": skipped_estimator.reason,
                }
                for skipped_estimator in skipped_estimators
            ],
        }

    @staticmethod
    @beartype
    def _serialise_metric_summaries(
        *, metric_summaries: dict[str, ClassificationMetricSummary]
    ) -> dict[str, dict[str, object]]:
        """Convert metric-summary dataclasses into JSON-serialisable dicts."""

        return {
            metric_key: {
                "fold_scores": [float(score) for score in metric_summary.fold_scores],
                "mean_score": float(metric_summary.mean_score),
                "std_score": float(metric_summary.std_score),
            }
            for metric_key, metric_summary in metric_summaries.items()
        }

    @staticmethod
    @beartype
    def _slugify_label(label: str) -> str:
        """Slugify a label for filesystem-safe output-folder names."""

        cleaned = "".join(char if char.isalnum() else "_" for char in label.lower())
        while "__" in cleaned:
            cleaned = cleaned.replace("__", "_")
        slug = cleaned.strip("_")
        return slug or "benchmark"
