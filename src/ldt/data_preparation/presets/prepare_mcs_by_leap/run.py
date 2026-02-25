from __future__ import annotations

import io
from collections.abc import Mapping
from typing import Any

from rich.console import Console

from ldt.utils.errors import InputValidationError

from . import pipeline as preset_pipeline
from .tool import PrepareMCSByLEAP


def prepare_mcs_by_leap_profile() -> dict[str, Any]:
    """CLI wrapper: return defaults and supported waves.

    Returns:
        dict[str, Any]: Profile payload with `available_waves` and defaults
        consumed by the Go CLI prompt flow.
    """

    return PrepareMCSByLEAP.profile()


def run_prepare_mcs_by_leap(*, params: Mapping[str, Any]) -> dict[str, Any]:
    """Execute Prepare-MCS-by-LEAP for the Go CLI bridge.

    !!! warning

        This `run_*` function is primarily for the Go CLI bridge and should not be
        treated as the Python library API. In Python scripts or notebooks, use
        `PrepareMCSByLEAP.fit(...)` + `prepare(...)` (or `fit_prepare(...)`).

    Args:
        params (Mapping[str, Any]): Preset runtime keys accepted by
            `PrepareMCSByLEAP.build_config(...)`, including:
            - `waves`, `wave_inputs`
            - `output_format`, `wide_suffix_prefix`
            - `show_summary_logs`
            - `save_wave_outputs`, `wave_output_dir`
            - `save_final_output`, `long_output_path`, `wide_output_path`
            - `parallel`, `max_workers`

    Returns:
        dict[str, Any]: Serialised pipeline run summary for the Go CLI bridge.
    """

    try:
        tool = PrepareMCSByLEAP()
        config = tool.build_config(**dict(params))
        result = _run_pipeline_silently(config=config)

        long_rows = (
            int(result.long_output.shape[0]) if result.long_output is not None else 0
        )
        long_columns = (
            int(result.long_output.shape[1]) if result.long_output is not None else 0
        )
        wide_rows = (
            int(result.wide_output.shape[0]) if result.wide_output is not None else 0
        )
        wide_columns = (
            int(result.wide_output.shape[1]) if result.wide_output is not None else 0
        )

        return {
            "preset": "prepare_MCS_by_LEAP",
            "waves": [spec.wave for spec in config.wave_inputs],
            "output_format": config.output_format,
            "save_wave_outputs": config.save_wave_outputs,
            "save_final_output": config.save_final_output,
            "parallel": config.parallel,
            "max_workers": config.max_workers,
            "long_output_path": (
                str(result.long_output_path.resolve())
                if result.long_output_path
                else None
            ),
            "wide_output_path": (
                str(result.wide_output_path.resolve())
                if result.wide_output_path
                else None
            ),
            "wave_output_paths": [
                {"wave": wave, "path": str(path.resolve())}
                for wave, path in result.wave_output_paths
            ],
            "long_shape": {"rows": long_rows, "columns": long_columns},
            "wide_shape": {"rows": wide_rows, "columns": wide_columns},
        }
    except (ValueError, TypeError) as exc:
        raise InputValidationError(str(exc)) from exc


def _run_pipeline_silently(
    *,
    config: preset_pipeline.PrepareMCSByLEAPPipelineConfig,
) -> preset_pipeline.PipelineRunResult:
    """Run pipeline while suppressing rich progress/table output for JSON bridge."""

    muted_stream = io.StringIO()
    muted_console = Console(file=muted_stream, force_terminal=False, color_system=None)

    original_console = preset_pipeline._CONSOLE
    preset_pipeline._CONSOLE = muted_console
    try:
        return preset_pipeline.run_prepare_mcs_by_leap_pipeline(config=config)
    finally:
        preset_pipeline._CONSOLE = original_console
