from __future__ import annotations

import io
from collections.abc import Mapping
from typing import Any

from beartype import beartype
from rich.console import Console

from ldt.utils.errors import InputValidationError

from . import pipeline as preset_pipeline
from .tool import PreprocessMCSByLEAP


def preprocess_mcs_by_leap_profile() -> dict[str, Any]:
    """CLI wrapper: return defaults and stage config paths."""

    return PreprocessMCSByLEAP.profile()


def run_preprocess_mcs_by_leap(*, params: Mapping[str, Any]) -> dict[str, Any]:
    """Execute preprocess MCS by LEAP for the Go CLI bridge."""

    try:
        tool = PreprocessMCSByLEAP()
        config = tool.build_config(**dict(params))
        result = _run_pipeline_silently(config=config)

        return {
            "preset": "preprocess_mcs_by_leap",
            "input_path": str(config.input_path.resolve()),
            "output_path": (
                str(result.output_path.resolve()) if result.output_path else None
            ),
            "audit_output_dir": (
                str(result.audit_output_dir.resolve())
                if result.audit_output_dir is not None
                else None
            ),
            "save_final_output": config.save_final_output,
            "save_audit_tables": config.save_audit_tables,
            "shape": {
                "rows": int(result.output_data.shape[0]),
                "columns": int(result.output_data.shape[1]),
            },
            "stage_0_summary": result.stage_0_summary,
            "stage_1_summary": result.stage_1_summary,
            "stage_2_summary": result.stage_2_summary,
            "stage_3_summary": result.stage_3_summary,
            "stage_4_summary": result.stage_4_summary,
            "stage_5_summary": result.stage_5_summary,
            "audit_files": [
                {"name": name, "path": str(path.resolve())}
                for name, path in result.audit_paths
            ],
        }
    except (TypeError, ValueError) as exc:
        raise InputValidationError(str(exc)) from exc


@beartype
def _run_pipeline_silently(
    *,
    config: preset_pipeline.PreprocessMCSByLEAPPipelineConfig,
) -> preset_pipeline.PipelineRunResult:
    """Run pipeline while suppressing rich output for JSON bridge."""

    muted_stream = io.StringIO()
    muted_console = Console(file=muted_stream, force_terminal=False, color_system=None)

    original_console = preset_pipeline._CONSOLE
    preset_pipeline._CONSOLE = muted_console
    try:
        return preset_pipeline.run_preprocess_mcs_by_leap_pipeline(config=config)
    finally:
        preset_pipeline._CONSOLE = original_console
