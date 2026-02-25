from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from beartype import beartype
from pandas.errors import PerformanceWarning
from rich import box
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from ldt.utils.errors import InputValidationError

from .stage_1_wave_paths import (
    available_waves as stage_1_available_waves,
)
from .stage_1_wave_paths import resolve_wave_dataset_config, validate_wave_raw_path
from .stage_2_subjects import (
    IdentifierSummary,
    MergeSummary,
    build_subject_level_wave_dataset,
    subject_key_config,
)
from .stage_3_features import (
    AmbiguousSourceResolution,
    FeaturePreparationSummary,
    LongitudinalFeatureMapping,
    SourceCoverageCandidate,
    append_unresolved_longitudinal_features,
    prepare_wave_features,
)
from .stage_4_composites import (
    CompositeFeatureSummary,
    apply_composites,
)
from .stage_5_output_format import OutputFormattingSummary, build_final_output_dataset

OutputFormat = Literal["long", "wide", "long_and_wide"]

_PRESET_DIR = Path(__file__).resolve().parent
_DEFAULTS_CONFIG = _PRESET_DIR / "defaults.yaml"
_CONSOLE = Console()
_AUTO_PARALLEL_WORKER_CAP = 2


@beartype
@dataclass(frozen=True)
class GeneralDefaults:
    """Default values used by the final interactive pipeline flow.

    Attributes:
        show_summary_logs (bool): Whether to show summary logs.
        run_parallel_when_possible (bool): Whether to run parallel when possible.
        default_long_output_path (Path): Path for default long output path.
        default_wide_output_path (Path): Path for default wide output path.
        default_wave_output_dir (Path): Filesystem location for default wave output dir.
        default_wide_suffix_prefix (str): Default wide suffix prefix.
    """

    show_summary_logs: bool
    run_parallel_when_possible: bool
    default_long_output_path: Path
    default_wide_output_path: Path
    default_wave_output_dir: Path
    default_wide_suffix_prefix: str


@beartype
@dataclass(frozen=True)
class WaveInputSpec:
    """User-selected raw directory for one wave.

    Attributes:
        wave (str): Wave.
        raw_dir (Path): Filesystem location for raw dir.
    """

    wave: str
    raw_dir: Path


@beartype
@dataclass(frozen=True)
class PrepareMCSByLEAPPipelineConfig:
    """Collected runtime configuration for the final 5-stage preset pipeline.

    Attributes:
        wave_inputs (tuple[WaveInputSpec, ...]): Wave inputs.
        output_format (OutputFormat): Output format.
        wide_suffix_prefix (str): Wide suffix prefix.
        show_summary_logs (bool): Whether to show summary logs.
        save_wave_outputs (bool): Whether to save wave outputs.
        wave_output_dir (Path | None): Filesystem location for wave output dir.
        save_final_output (bool): Whether to save final output.
        long_output_path (Path | None): Path for long output path.
        wide_output_path (Path | None): Path for wide output path.
        parallel (bool): Whether to parallel.
        max_workers (int | None): Max workers.
    """

    wave_inputs: tuple[WaveInputSpec, ...]
    output_format: OutputFormat
    wide_suffix_prefix: str
    show_summary_logs: bool
    save_wave_outputs: bool
    wave_output_dir: Path | None
    save_final_output: bool
    long_output_path: Path | None
    wide_output_path: Path | None
    parallel: bool
    max_workers: int | None = None


@beartype
@dataclass(frozen=True)
class WaveProcessingResult:
    """Per-wave pipeline output and summaries collected across stages 2-4.

    Attributes:
        wave (str): Wave.
        prepared_wave (pd.DataFrame): Prepared wave.
        identifier_summaries (tuple[IdentifierSummary, ...]): Identifier summaries.
        merge_summaries (tuple[MergeSummary, ...]): Merge summaries.
        feature_summary (FeaturePreparationSummary): Feature summary.
        ambiguous_resolutions (tuple[AmbiguousSourceResolution, ...]): Ambiguous resolutions.
        unresolved_after_retry (tuple[LongitudinalFeatureMapping, ...]): Unresolved after retry.
        composite_summary (CompositeFeatureSummary): Composite summary.
    """

    wave: str
    prepared_wave: pd.DataFrame
    identifier_summaries: tuple[IdentifierSummary, ...]
    merge_summaries: tuple[MergeSummary, ...]
    feature_summary: FeaturePreparationSummary
    ambiguous_resolutions: tuple[AmbiguousSourceResolution, ...]
    unresolved_after_retry: tuple[LongitudinalFeatureMapping, ...]
    composite_summary: CompositeFeatureSummary


@beartype
@dataclass(frozen=True)
class PipelineRunResult:
    """Final outputs and saved-path metadata returned after one pipeline run.

    Attributes:
        long_output (pd.DataFrame | None): Long output.
        wide_output (pd.DataFrame | None): Wide output.
        long_output_path (Path | None): Path for long output path.
        wide_output_path (Path | None): Path for wide output path.
        wave_output_paths (tuple[tuple[str, Path], ...]): Wave output paths.
    """

    long_output: pd.DataFrame | None
    wide_output: pd.DataFrame | None
    long_output_path: Path | None
    wide_output_path: Path | None
    wave_output_paths: tuple[tuple[str, Path], ...]


@beartype
def available_waves() -> tuple[str, ...]:
    """Return supported wave labels in numeric order.

    Returns:
        tuple[str, ...]: Supported wave labels ordered by wave number,
            for example `(\"W1\", \"W2\", \"W3\")`.
    """

    return stage_1_available_waves()


@beartype
def general_defaults() -> GeneralDefaults:
    """Load default values for this preset from root defaults.yaml.

    Returns:
        GeneralDefaults: Result object for this operation.
    """

    raw = _load_yaml_config(_DEFAULTS_CONFIG)
    general = raw.get("general")
    if not isinstance(general, dict):
        raise InputValidationError("Defaults config must define `general` mapping.")

    output = raw.get("output")
    if not isinstance(output, dict):
        raise InputValidationError("Defaults config must define `output` mapping.")

    default_long = output.get("default_long_output_csv")
    default_wide = output.get("default_wide_output_csv")
    default_wave_output_dir = output.get("default_wave_output_dir")
    default_suffix = output.get("default_wide_suffix_prefix", "_w")

    if not isinstance(default_long, str) or not default_long.strip():
        raise InputValidationError(
            "Invalid `output.default_long_output_csv` in defaults."
        )
    if not isinstance(default_wide, str) or not default_wide.strip():
        raise InputValidationError(
            "Invalid `output.default_wide_output_csv` in defaults."
        )
    if (
        not isinstance(default_wave_output_dir, str)
        or not default_wave_output_dir.strip()
    ):
        raise InputValidationError(
            "Invalid `output.default_wave_output_dir` in defaults."
        )
    if not isinstance(default_suffix, str) or not default_suffix.strip():
        raise InputValidationError(
            "Invalid `output.default_wide_suffix_prefix` in defaults."
        )

    return GeneralDefaults(
        show_summary_logs=bool(general.get("show_summary_logs", True)),
        run_parallel_when_possible=bool(
            general.get("run_parallel_when_possible", True)
        ),
        default_long_output_path=Path(default_long.strip()),
        default_wide_output_path=Path(default_wide.strip()),
        default_wave_output_dir=Path(default_wave_output_dir.strip()),
        default_wide_suffix_prefix=default_suffix.strip(),
    )


@beartype
def parse_wave_list(*, raw: str, available: tuple[str, ...]) -> tuple[str, ...]:
    """Parse and validate a comma-separated list of waves (`W1,W2` or `ALL`).

    Args:
        raw (str): Raw wave selection string from the caller.
        available (tuple[str, ...]): Allowed wave labels that can be selected.

    Returns:
        tuple[str, ...]: Deduplicated, validated wave labels in caller order.
    """

    tokens = [token.strip().upper() for token in raw.split(",") if token.strip()]
    if not tokens:
        raise ValueError("No wave labels provided.")

    if len(tokens) == 1 and tokens[0] == "ALL":
        return available

    available_set = set(available)
    invalid = [token for token in tokens if token not in available_set]
    if invalid:
        raise ValueError(f"Unsupported waves: {', '.join(invalid)}.")

    deduplicated: list[str] = []
    for token in tokens:
        if token not in deduplicated:
            deduplicated.append(token)
    return tuple(deduplicated)


@beartype
def run_prepare_mcs_by_leap_pipeline(
    *, config: PrepareMCSByLEAPPipelineConfig
) -> PipelineRunResult:
    """Run the final five-stage prepare-MCS-by-LEAP pipeline.

    Args:
        config (PrepareMCSByLEAPPipelineConfig): Config object used by this workflow.

    Returns:
        PipelineRunResult: Result object for this operation.
    """

    _validate_pipeline_config(config=config)

    keys = subject_key_config()
    ordered_wave_inputs = tuple(
        sorted(config.wave_inputs, key=lambda item: int(item.wave[1:]))
    )
    effective_save_wave_outputs = (
        config.save_wave_outputs and len(ordered_wave_inputs) > 1
    )

    long_output: pd.DataFrame | None = None
    wide_output: pd.DataFrame | None = None
    long_summary: OutputFormattingSummary | None = None
    wide_summary: OutputFormattingSummary | None = None
    written_long_path: Path | None = None
    written_wide_path: Path | None = None
    wave_results: tuple[WaveProcessingResult, ...]
    wave_output_paths: tuple[tuple[str, Path], ...]
    worker_count: int

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=_CONSOLE,
        transient=False,
    ) as progress:
        total_steps = _estimate_progress_steps(config=config)
        task_id = progress.add_task("Preparing MCS by LEAP...", total=total_steps)

        progress.update(task_id, description="Preparing waves (stages 1-4)...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PerformanceWarning)
            warnings.filterwarnings("ignore", category=UnicodeWarning)

            if config.parallel and len(ordered_wave_inputs) > 1:
                wave_results, worker_count, wave_output_paths = (
                    _run_parallel_wave_processing(
                        wave_inputs=ordered_wave_inputs,
                        protected_columns=keys.output_fixed_columns,
                        max_workers=config.max_workers,
                        save_wave_outputs=effective_save_wave_outputs,
                        wave_output_dir=config.wave_output_dir,
                    )
                )
            else:
                worker_count = 1
                sequential_results: list[WaveProcessingResult] = []
                sequential_wave_output_paths: list[tuple[str, Path]] = []
                for wave_input in ordered_wave_inputs:
                    wave_result = _prepare_single_wave(
                        wave_input=wave_input,
                        protected_columns=keys.output_fixed_columns,
                    )
                    sequential_results.append(wave_result)
                    if effective_save_wave_outputs:
                        if config.wave_output_dir is None:
                            raise InputValidationError(
                                "Internal error: missing wave output directory."
                            )
                        path = (
                            config.wave_output_dir / f"{wave_result.wave}_prepared.csv"
                        )
                        _write_csv_output(path=path, data=wave_result.prepared_wave)
                        sequential_wave_output_paths.append((wave_result.wave, path))
                wave_results = tuple(sequential_results)
                wave_output_paths = tuple(sequential_wave_output_paths)
        progress.update(task_id, advance=1)

        progress.update(task_id, description="Combining wave outputs...")
        merged = pd.concat(
            [result.prepared_wave for result in wave_results],
            ignore_index=True,
            sort=False,
        )
        progress.update(task_id, advance=1)

        if config.output_format in {"long", "long_and_wide"}:
            progress.update(task_id, description="Building final long output...")
            long_output, long_summary = build_final_output_dataset(
                data=merged,
                output_format="long",
                fixed_columns=keys.output_fixed_columns,
                subject_id_column=keys.output_fixed_columns[0],
                wave_column=keys.output_fixed_columns[1],
                wide_suffix_prefix=config.wide_suffix_prefix,
            )
            progress.update(task_id, advance=1)

        if config.output_format in {"wide", "long_and_wide"}:
            progress.update(task_id, description="Building final wide output...")
            wide_output, wide_summary = build_final_output_dataset(
                data=merged,
                output_format="wide",
                fixed_columns=keys.output_fixed_columns,
                subject_id_column=keys.output_fixed_columns[0],
                wave_column=keys.output_fixed_columns[1],
                wide_suffix_prefix=config.wide_suffix_prefix,
            )
            progress.update(task_id, advance=1)

        if config.save_final_output:
            progress.update(task_id, description="Writing final output files...")
            written_long_path, written_wide_path = _write_final_outputs_if_requested(
                config=config,
                long_output=long_output,
                wide_output=wide_output,
            )
            progress.update(task_id, advance=1)

        if config.show_summary_logs:
            progress.update(task_id, description="Rendering summary tables...")
            progress.update(task_id, advance=1)

        progress.update(task_id, description="Done", completed=total_steps)

    if config.show_summary_logs:
        _print_pipeline_summary_tables(
            wave_results=wave_results,
            parallel_used=config.parallel and len(ordered_wave_inputs) > 1,
            worker_count=worker_count,
            wave_output_paths=wave_output_paths,
            long_summary=long_summary,
            wide_summary=wide_summary,
            long_output_path=written_long_path,
            wide_output_path=written_wide_path,
        )

    return PipelineRunResult(
        long_output=long_output,
        wide_output=wide_output,
        long_output_path=written_long_path,
        wide_output_path=written_wide_path,
        wave_output_paths=wave_output_paths,
    )


@beartype
def _estimate_progress_steps(*, config: PrepareMCSByLEAPPipelineConfig) -> int:
    """Estimate major outer-layer progress milestones for one pipeline run."""

    steps = 2  # prepare waves + combine wave outputs
    if config.output_format in {"long", "long_and_wide"}:
        steps += 1
    if config.output_format in {"wide", "long_and_wide"}:
        steps += 1
    if config.save_final_output:
        steps += 1
    if config.show_summary_logs:
        steps += 1
    return steps


@beartype
def _validate_pipeline_config(*, config: PrepareMCSByLEAPPipelineConfig) -> None:
    """Validate high-level pipeline configuration before execution."""

    if not config.wave_inputs:
        raise InputValidationError("At least one wave input path is required.")

    if not config.wide_suffix_prefix.strip():
        raise InputValidationError("Wide suffix prefix cannot be empty.")

    save_wave_outputs = config.save_wave_outputs and len(config.wave_inputs) > 1
    if save_wave_outputs and config.wave_output_dir is None:
        raise InputValidationError(
            "`wave_output_dir` must be set when `save_wave_outputs` is true."
        )

    if not config.save_final_output:
        return

    if (
        config.output_format in {"long", "long_and_wide"}
        and config.long_output_path is None
    ):
        raise InputValidationError(
            "`long_output_path` must be set when long output saving is enabled."
        )
    if (
        config.output_format in {"wide", "long_and_wide"}
        and config.wide_output_path is None
    ):
        raise InputValidationError(
            "`wide_output_path` must be set when wide output saving is enabled."
        )


@beartype
def _prepare_single_wave(
    *,
    wave_input: WaveInputSpec,
    protected_columns: tuple[str, ...],
) -> WaveProcessingResult:
    """Run stages 1-4 for one wave and return all artefacts."""

    wave = wave_input.wave.strip().upper()
    raw_dir = wave_input.raw_dir

    wave_config = resolve_wave_dataset_config(wave)
    validate_wave_raw_path(wave=wave, raw_dir=raw_dir)

    subject_result = build_subject_level_wave_dataset(
        wave=wave,
        raw_dir=raw_dir,
        wave_config=wave_config,
    )

    feature_result = prepare_wave_features(
        wave=wave,
        data=subject_result.data,
        protected_columns=protected_columns,
    )

    with_composites, composite_summary = apply_composites(
        wave=wave,
        data=feature_result.data,
        source_data=subject_result.data,
    )

    prepared_wave, unresolved_after_retry = append_unresolved_longitudinal_features(
        data=with_composites,
        source_data=subject_result.data,
        unresolved=feature_result.unresolved_longitudinal,
    )

    return WaveProcessingResult(
        wave=wave,
        prepared_wave=prepared_wave.reset_index(drop=True),
        identifier_summaries=subject_result.identifier_summaries,
        merge_summaries=subject_result.merge_summaries,
        feature_summary=feature_result.summary,
        ambiguous_resolutions=feature_result.ambiguous_resolutions,
        unresolved_after_retry=unresolved_after_retry,
        composite_summary=composite_summary,
    )


@beartype
def _run_parallel_wave_processing(
    *,
    wave_inputs: tuple[WaveInputSpec, ...],
    protected_columns: tuple[str, ...],
    max_workers: int | None,
    save_wave_outputs: bool,
    wave_output_dir: Path | None,
) -> tuple[tuple[WaveProcessingResult, ...], int, tuple[tuple[str, Path], ...]]:
    """Run wave preparation in parallel and collect ordered results."""

    worker_count = _resolve_parallel_worker_count(
        requested=max_workers,
        wave_count=len(wave_inputs),
    )

    if save_wave_outputs:
        if wave_output_dir is None:
            raise InputValidationError("Internal error: missing wave output directory.")
        wave_output_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    results: list[WaveProcessingResult] = []
    saved_wave_output_paths: list[tuple[str, Path]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_wave = {
            executor.submit(
                _prepare_single_wave,
                wave_input=wave_input,
                protected_columns=protected_columns,
            ): wave_input.wave
            for wave_input in wave_inputs
        }

        for future in as_completed(future_to_wave):
            wave = str(future_to_wave[future]).upper()
            try:
                result = future.result()
                results.append(result)
                if save_wave_outputs:
                    if wave_output_dir is None:
                        raise InputValidationError(
                            "Internal error: missing wave output directory."
                        )
                    path = wave_output_dir / f"{result.wave}_prepared.csv"
                    _write_csv_output(path=path, data=result.prepared_wave)
                    saved_wave_output_paths.append((result.wave, path))
            except Exception as exc:  # pragma: no cover - safety branch
                failures.append(f"{wave}: {exc}")

    if failures:
        raise InputValidationError(
            "Wave preparation failed for one or more waves: " + " | ".join(failures)
        )

    ordered_results = tuple(sorted(results, key=lambda item: int(item.wave[1:])))
    ordered_paths = tuple(
        sorted(saved_wave_output_paths, key=lambda item: int(item[0][1:]))
    )
    return ordered_results, worker_count, ordered_paths


@beartype
def _resolve_parallel_worker_count(*, requested: int | None, wave_count: int) -> int:
    """Resolve effective worker count for parallel wave processing."""

    if wave_count <= 1:
        return 1

    if requested is not None:
        return max(1, min(requested, wave_count))

    cpu_count = os.cpu_count() or 1
    # Auto mode is intentionally conservative to avoid OOM on large multi-wave runs.
    return max(1, min(cpu_count, wave_count, _AUTO_PARALLEL_WORKER_CAP))


@beartype
def _write_final_outputs_if_requested(
    *,
    config: PrepareMCSByLEAPPipelineConfig,
    long_output: pd.DataFrame | None,
    wide_output: pd.DataFrame | None,
) -> tuple[Path | None, Path | None]:
    """Write final long/wide outputs as configured by the user."""

    if not config.save_final_output:
        return None, None

    written_long_path: Path | None = None
    written_wide_path: Path | None = None

    if long_output is not None:
        if config.long_output_path is None:
            raise InputValidationError("Internal error: long output path is missing.")
        _write_csv_output(path=config.long_output_path, data=long_output)
        written_long_path = config.long_output_path

    if wide_output is not None:
        if config.wide_output_path is None:
            raise InputValidationError("Internal error: wide output path is missing.")
        _write_csv_output(path=config.wide_output_path, data=wide_output)
        written_wide_path = config.wide_output_path

    return written_long_path, written_wide_path


@beartype
def _write_csv_output(*, path: Path, data: pd.DataFrame) -> None:
    """Write one CSV output and prevent silent overwrite."""

    if path.exists():
        raise InputValidationError(
            f"Output path already exists and will not be overwritten: {path.resolve()}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)


@beartype
def _print_pipeline_summary_tables(
    *,
    wave_results: tuple[WaveProcessingResult, ...],
    parallel_used: bool,
    worker_count: int,
    wave_output_paths: tuple[tuple[str, Path], ...],
    long_summary: OutputFormattingSummary | None,
    wide_summary: OutputFormattingSummary | None,
    long_output_path: Path | None,
    wide_output_path: Path | None,
) -> None:
    """Render end-of-run summary tables with Rich for better readability."""

    _CONSOLE.rule("Prepare MCS by LEAP - Run Summary")

    run_table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold cyan")
    run_table.add_column("Setting")
    run_table.add_column("Value", overflow="fold")
    run_table.add_row("Waves processed", str(len(wave_results)))
    run_table.add_row("Parallel execution", "Yes" if parallel_used else "No")
    run_table.add_row("Workers", str(worker_count))
    run_table.add_row(
        "Wave-by-wave outputs saved", "Yes" if wave_output_paths else "No"
    )
    if wave_output_paths:
        run_table.add_row(
            "Wave output directory",
            str(wave_output_paths[0][1].parent.resolve()),
        )
    _CONSOLE.print(run_table)

    wave_output_lookup = {wave: path for wave, path in wave_output_paths}
    for wave_result in wave_results:
        _CONSOLE.rule(f"Wave {wave_result.wave}")
        _CONSOLE.print(
            _build_identifier_table(
                identifier_summaries=wave_result.identifier_summaries,
            )
        )
        _CONSOLE.print(
            _build_merge_table(
                merge_summaries=wave_result.merge_summaries,
            )
        )
        _CONSOLE.print(
            _build_feature_summary_table(
                feature_summary=wave_result.feature_summary,
                unresolved_after_retry=wave_result.unresolved_after_retry,
            )
        )

        ambiguous = wave_result.ambiguous_resolutions
        if ambiguous:
            _CONSOLE.print(_build_ambiguous_resolution_table(resolutions=ambiguous))

        _CONSOLE.print(
            _build_composite_summary_table(summary=wave_result.composite_summary)
        )

        if wave_result.wave in wave_output_lookup:
            _CONSOLE.print(
                f"[green]Wave output saved:[/green] {wave_output_lookup[wave_result.wave].resolve()}"
            )

    _CONSOLE.rule("Final Outputs")
    final_table = Table(
        box=box.SIMPLE_HEAD, show_header=True, header_style="bold green"
    )
    final_table.add_column("Output")
    final_table.add_column("Rows", justify="right")
    final_table.add_column("Columns", justify="right")
    final_table.add_column("Longitudinal features", justify="right")
    final_table.add_column("Saved path", overflow="fold")

    if long_summary is not None:
        final_table.add_row(
            "Long",
            f"{long_summary.subject_rows:,}",
            f"{long_summary.columns:,}",
            f"{long_summary.longitudinal_features:,}",
            (
                str(long_output_path.resolve())
                if long_output_path is not None
                else "(not saved)"
            ),
        )
    if wide_summary is not None:
        final_table.add_row(
            "Wide",
            f"{wide_summary.subject_rows:,}",
            f"{wide_summary.columns:,}",
            f"{wide_summary.longitudinal_features:,}",
            (
                str(wide_output_path.resolve())
                if wide_output_path is not None
                else "(not saved)"
            ),
        )

    _CONSOLE.print(final_table)


@beartype
def _build_identifier_table(
    *,
    identifier_summaries: tuple[IdentifierSummary, ...],
) -> Table:
    """Build stage-2 identifier summary table."""

    table = Table(
        title="Stage 2 - Identifier Checks",
        box=box.MINIMAL_DOUBLE_HEAD,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Dataset", overflow="fold")
    table.add_column("Role")
    table.add_column("Key scope")
    table.add_column("Rows (with keys/total)", justify="right")
    table.add_column("Duplicates", justify="right")

    for summary in identifier_summaries:
        table.add_row(
            summary.alias,
            summary.role,
            summary.key_scope,
            f"{summary.rows_with_keys:,}/{summary.rows_total:,}",
            f"{summary.duplicate_key_rows:,}",
        )
    return table


@beartype
def _build_merge_table(*, merge_summaries: tuple[MergeSummary, ...]) -> Table:
    """Build stage-2 merge coverage summary table."""

    table = Table(
        title="Stage 2 - Merge Coverage",
        box=box.MINIMAL_DOUBLE_HEAD,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Dataset", overflow="fold")
    table.add_column("Key scope")
    table.add_column("Matched / Anchor", justify="right")

    for summary in merge_summaries:
        table.add_row(
            summary.alias,
            summary.key_scope,
            f"{summary.matched_rows:,}/{summary.anchor_rows:,}",
        )
    return table


@beartype
def _build_feature_summary_table(
    *,
    feature_summary: FeaturePreparationSummary,
    unresolved_after_retry: tuple[LongitudinalFeatureMapping, ...],
) -> Table:
    """Build stage-3 compact feature summary table."""

    summary = feature_summary
    table = Table(
        title="Stage 3 - Feature Preparation",
        box=box.MINIMAL_DOUBLE_HEAD,
        show_header=True,
        header_style="bold yellow",
    )
    table.add_column("Longitudinal resolved", justify="right")
    table.add_column("Longitudinal unresolved (initial)", justify="right")
    table.add_column("Longitudinal unresolved (after retry)", justify="right")
    table.add_column("Non-longitudinal selected", justify="right")
    table.add_column("Non-longitudinal unresolved", justify="right")
    table.add_column("Ambiguous resolutions", justify="right")
    table.add_row(
        f"{summary.resolved_longitudinal:,}",
        f"{summary.unresolved_longitudinal:,}",
        f"{len(unresolved_after_retry):,}",
        f"{summary.selected_non_longitudinal:,}",
        f"{summary.unresolved_non_longitudinal:,}",
        f"{summary.ambiguous_source_resolutions:,}",
    )
    return table


@beartype
def _build_ambiguous_resolution_table(
    *,
    resolutions: tuple[AmbiguousSourceResolution, ...],
) -> Table:
    """Build stage-3 ambiguous resolution table with top-5 coverage details."""

    table = Table(
        title="Stage 3 - Ambiguous Source Resolution (highest coverage selected)",
        box=box.MINIMAL_DOUBLE_HEAD,
        show_header=True,
        header_style="bold yellow",
    )
    table.add_column("Group")
    table.add_column("Target")
    table.add_column("Source")
    table.add_column("Selected column", overflow="fold")
    table.add_column("Matches", justify="right")
    table.add_column("Top 5 coverage", overflow="fold")

    for resolution in resolutions:
        top_coverage = _render_top_coverage_candidates(
            candidates=resolution.top_coverage_candidates,
        )
        table.add_row(
            resolution.feature_group,
            resolution.target_name,
            resolution.source_name,
            resolution.selected_column,
            f"{resolution.total_matches:,}",
            top_coverage,
        )
    return table


@beartype
def _render_top_coverage_candidates(
    *,
    candidates: tuple[SourceCoverageCandidate, ...],
) -> str:
    """Render top coverage candidates for compact table output."""

    items = [
        f"{candidate.column}: {candidate.non_null_rows:,}"
        for candidate in candidates[:5]
    ]
    return " | ".join(items)


@beartype
def _build_composite_summary_table(*, summary: CompositeFeatureSummary) -> Table:
    """Build stage-4 composite summary table."""

    table = Table(
        title="Stage 4 - Composite Features",
        box=box.MINIMAL_DOUBLE_HEAD,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Requested", justify="right")
    table.add_column("Created", justify="right")
    table.add_row(f"{summary.requested:,}", f"{summary.created:,}")
    return table


@beartype
@cache
def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load one YAML config file and validate mapping root."""

    if not config_path.exists() or not config_path.is_file():
        raise InputValidationError(f"Missing config file: {config_path.resolve()}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise InputValidationError(
            f"Config file `{config_path.name}` must contain a YAML mapping."
        )
    return loaded
