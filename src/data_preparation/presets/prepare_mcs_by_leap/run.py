from __future__ import annotations

import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rich.console import Console

from src.utils.errors import InputValidationError

from . import pipeline as preset_pipeline


def prepare_mcs_by_leap_profile() -> dict[str, Any]:
    """Return prompt defaults and supported waves for Go preset UX.

    Returns:
        dict[str, Any]: Dictionary containing tool results.
    """

    defaults = preset_pipeline.general_defaults()
    waves = preset_pipeline.available_waves()
    return {
        "available_waves": list(waves),
        "defaults": {
            "show_summary_logs": bool(defaults.show_summary_logs),
            "run_parallel_when_possible": bool(defaults.run_parallel_when_possible),
            "default_long_output_path": str(defaults.default_long_output_path),
            "default_wide_output_path": str(defaults.default_wide_output_path),
            "default_wave_output_dir": str(defaults.default_wave_output_dir),
            "default_wide_suffix_prefix": str(defaults.default_wide_suffix_prefix),
        },
    }


def run_prepare_mcs_by_leap(*, params: Mapping[str, Any]) -> dict[str, Any]:
    """Run the five-stage prepare-MCS-by-LEAP preset pipeline.

    Args:
        params (Mapping[str, Any]): Parameter mapping provided by the caller.

    Returns:
        dict[str, Any]: Dictionary containing tool results.
    """

    try:
        profile = prepare_mcs_by_leap_profile()
        available = tuple(_as_required_string_list(profile, "available_waves"))
        defaults = _as_object(profile, "defaults")

        waves = _parse_waves(params=params, available=available)
        wave_inputs = _parse_wave_inputs(params=params, waves=waves)

        output_format = _as_output_format(params.get("output_format", "long_and_wide"))
        wide_suffix_prefix = _as_optional_string(
            params.get("wide_suffix_prefix"),
            fallback=_as_required_string(defaults, "default_wide_suffix_prefix"),
        )
        show_summary_logs = _as_bool(
            params,
            "show_summary_logs",
            fallback=bool(defaults.get("show_summary_logs", True)),
        )

        save_wave_outputs_default = len(waves) > 1
        save_wave_outputs = (
            _as_bool(
                params,
                "save_wave_outputs",
                fallback=save_wave_outputs_default,
            )
            and len(waves) > 1
        )

        save_final_output = _as_bool(params, "save_final_output", fallback=True)

        parallel_default = (
            bool(defaults.get("run_parallel_when_possible", True)) and len(waves) > 1
        )
        parallel = (
            _as_bool(params, "parallel", fallback=parallel_default) and len(waves) > 1
        )

        max_workers = (
            _as_optional_positive_int(params.get("max_workers")) if parallel else None
        )

        wave_output_dir: Path | None = None
        if save_wave_outputs:
            wave_output_dir = Path(
                _as_optional_string(
                    params.get("wave_output_dir"),
                    fallback=_as_required_string(defaults, "default_wave_output_dir"),
                )
            ).expanduser()

        long_output_path: Path | None = None
        wide_output_path: Path | None = None
        if save_final_output and output_format in {"long", "long_and_wide"}:
            long_output_path = Path(
                _as_optional_string(
                    params.get("long_output_path"),
                    fallback=_as_required_string(defaults, "default_long_output_path"),
                )
            ).expanduser()
        if save_final_output and output_format in {"wide", "long_and_wide"}:
            wide_output_path = Path(
                _as_optional_string(
                    params.get("wide_output_path"),
                    fallback=_as_required_string(defaults, "default_wide_output_path"),
                )
            ).expanduser()

        config = preset_pipeline.PrepareMCSByLEAPPipelineConfig(
            wave_inputs=tuple(wave_inputs),
            output_format=output_format,
            wide_suffix_prefix=wide_suffix_prefix,
            show_summary_logs=show_summary_logs,
            save_wave_outputs=save_wave_outputs,
            wave_output_dir=wave_output_dir,
            save_final_output=save_final_output,
            long_output_path=long_output_path,
            wide_output_path=wide_output_path,
            parallel=parallel,
            max_workers=max_workers,
        )

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
            "waves": list(waves),
            "output_format": output_format,
            "save_wave_outputs": save_wave_outputs,
            "save_final_output": save_final_output,
            "parallel": parallel,
            "max_workers": max_workers,
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
    """Run pipeline while suppressing rich progress/table output for bridge JSON mode."""

    muted_stream = io.StringIO()
    muted_console = Console(file=muted_stream, force_terminal=False, color_system=None)

    original_console = preset_pipeline._CONSOLE
    preset_pipeline._CONSOLE = muted_console
    try:
        return preset_pipeline.run_prepare_mcs_by_leap_pipeline(config=config)
    finally:
        preset_pipeline._CONSOLE = original_console


def _parse_waves(
    *, params: Mapping[str, Any], available: tuple[str, ...]
) -> tuple[str, ...]:
    raw = params.get("waves", "ALL")
    if isinstance(raw, list):
        token = ",".join(str(entry).strip() for entry in raw if str(entry).strip())
    else:
        token = str(raw).strip()
    if not token:
        token = "ALL"
    return preset_pipeline.parse_wave_list(raw=token, available=available)


def _parse_wave_inputs(
    *,
    params: Mapping[str, Any],
    waves: tuple[str, ...],
) -> list[preset_pipeline.WaveInputSpec]:
    raw_inputs = _as_object(params, "wave_inputs")
    parsed: list[preset_pipeline.WaveInputSpec] = []

    for wave in waves:
        raw_path = None
        for candidate in (wave, wave.lower(), wave.upper()):
            value = raw_inputs.get(candidate)
            if isinstance(value, str) and value.strip():
                raw_path = value.strip()
                break
        if raw_path is None:
            raise InputValidationError(f"Missing raw wave directory for `{wave}`.")

        parsed.append(
            preset_pipeline.WaveInputSpec(
                wave=wave,
                raw_dir=Path(raw_path).expanduser(),
            )
        )

    return parsed


def _as_object(params: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = params.get(key, {})
    if not isinstance(value, dict):
        raise InputValidationError(f"`{key}` must be an object.")
    return value


def _as_required_string(params: Mapping[str, Any], key: str) -> str:
    value = params.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"Missing required string parameter: {key}")
    return value.strip()


def _as_required_string_list(params: Mapping[str, Any], key: str) -> list[str]:
    value = params.get(key)
    if not isinstance(value, list) or not value:
        raise InputValidationError(f"Missing required string-list parameter: {key}")

    parsed: list[str] = []
    for entry in value:
        if not isinstance(entry, str):
            raise InputValidationError(f"Parameter `{key}` entries must be strings.")
        candidate = entry.strip()
        if candidate:
            parsed.append(candidate)
    if not parsed:
        raise InputValidationError(f"Parameter `{key}` must include at least one item.")
    return parsed


def _as_optional_string(value: Any, *, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if fallback.strip():
        return fallback.strip()
    raise InputValidationError("Expected a non-empty string value.")


def _as_bool(params: Mapping[str, Any], key: str, *, fallback: bool) -> bool:
    value = params.get(key)
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    if isinstance(value, int | float):
        return bool(value)
    raise InputValidationError(f"`{key}` must be a boolean.")


def _as_optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"", "auto", "none"}:
            return None
        value = token

    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise InputValidationError(
            "`max_workers` must be `auto` or a positive integer."
        ) from exc

    if parsed <= 0:
        raise InputValidationError("`max_workers` must be > 0.")
    return parsed


def _as_output_format(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return "long_and_wide"

    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in {"long", "wide", "long_and_wide"}:
        raise InputValidationError(
            "`output_format` must be one of: long, wide, long_and_wide."
        )
    return normalized
