from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from beartype import beartype

from ldt.utils.errors import InputValidationError
from ldt.utils.metadata import ComponentMetadata
from ldt.utils.templates.tools.data_preparation import DataPreparationTool

from . import pipeline as preset_pipeline


@beartype
class PrepareMCSByLEAP(DataPreparationTool):
    """Python API for the Prepare MCS by LEAP preset.

    Transforms selected raw MCS wave directories into canonical longitudinal
    outputs (`long`, `wide`, or `long_and_wide`) used by
    `PreprocessMCSByLEAP`.

    Pipeline overview:
    1. validate wave manifests
    2. construct subject-level wave tables
    3. map configured features to canonical roots
    4. build configured composite features
    5. format/save final long and/or wide artefacts

    Examples:
        ```python
        from ldt.data_preparation import PrepareMCSByLEAP

        result = PrepareMCSByLEAP().fit_prepare(
            waves="ALL",
            wave_inputs={
                "W1": "/path/to/W1/stata",
                "W2": "/path/to/W2/stata",
                "W3": "/path/to/W3/stata",
                "W4": "/path/to/W4/stata",
                "W5": "/path/to/W5/stata",
                "W6": "/path/to/W6/stata",
                "W7": "/path/to/W7/stata",
            },
            output_format="wide",
            save_final_output=True,
        )
        print(result.wide_output_path)
        ```
    """

    metadata = ComponentMetadata(
        name="prepare_mcs_by_leap",
        full_name="Prepare MCS by LEAP",
        abstract_description=(
            "Run a full preparation of raw Stata datasets into per-wave or merged "
            "(wide and long) longitudinal outputs, with features of interest for "
            "LEAP depression youth trajectory analysis."
        ),
    )

    def __init__(self) -> None:
        self._fitted_config: preset_pipeline.PrepareMCSByLEAPPipelineConfig | None = (
            None
        )

    @staticmethod
    @beartype
    def profile() -> dict[str, Any]:
        """Return supported waves and default values."""

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

    @beartype
    def fit(self, **kwargs: Any) -> PrepareMCSByLEAP:
        """Store a validated pipeline config for a later `prepare(...)` call.

        Args:
            **kwargs: Same configuration keys accepted by `build_config(...)`.
                Expected keys:

                - `waves`: wave selection.
                - `wave_inputs`: raw path mapping per selected wave.
                - `output_format`: `long`, `wide`, or `long_and_wide`.
                - `wide_suffix_prefix`: wide suffix prefix.
                - `show_summary_logs`: summary logging toggle.
                - `save_wave_outputs`: per-wave output save toggle.
                - `wave_output_dir`: output directory for per-wave CSVs.
                - `save_final_output`: final merged output save toggle.
                - `long_output_path`: final long CSV path.
                - `wide_output_path`: final wide CSV path.
                - `parallel`: multi-wave parallel execution toggle.
                - `max_workers`: optional worker cap for parallel execution.

        Returns:
            PrepareMCSByLEAP: The fitted preset instance.
        """

        self._fitted_config = self.build_config(**kwargs)
        return self

    @beartype
    def prepare(self, **kwargs: Any) -> preset_pipeline.PipelineRunResult:
        """Execute the preparation pipeline.

        Config resolution order:
        1. `config` passed in `kwargs`
        2. other `kwargs` converted via `build_config(...)`
        3. config previously stored by `fit(...)`

        Args:
            **kwargs: Optional `config` plus any config override keys accepted
                by `build_config(...)`.
                `config` should be a `PrepareMCSByLEAPPipelineConfig` instance.

        Returns:
            preset_pipeline.PipelineRunResult: Full pipeline run payload.

        Raises:
            InputValidationError: If no valid config is provided.
        """

        config = kwargs.get("config")
        if config is None and kwargs:
            config = self.build_config(**kwargs)
        if config is None:
            config = self._fitted_config
        if config is None:
            raise InputValidationError(
                "No pipeline configuration provided. Pass `config` or kwargs to "
                "`prepare(...)`, or call `fit(...)` first."
            )
        if not isinstance(config, preset_pipeline.PrepareMCSByLEAPPipelineConfig):
            raise InputValidationError(
                "`config` must be a PrepareMCSByLEAPPipelineConfig instance."
            )
        return preset_pipeline.run_prepare_mcs_by_leap_pipeline(config=config)

    @beartype
    def build_config(
        self,
        **kwargs: Any,
    ) -> preset_pipeline.PrepareMCSByLEAPPipelineConfig:
        """Build a validated `PrepareMCSByLEAPPipelineConfig` from kwargs.

        Args:
            **kwargs: Runtime overrides. Expected keys:
                
                - `waves`: selects waves (`ALL` or `W1,W2,...`).
                - `wave_inputs`: maps each selected wave to a raw directory.
                - `output_format`: chooses `long`, `wide`, or `long_and_wide`.
                - `wide_suffix_prefix`: prefix before wide wave suffix token.
                - `save_wave_outputs`: enables per-wave output persistence.
                - `wave_output_dir`: target directory for per-wave outputs.
                - `save_final_output`: enables final merged output write.
                - `long_output_path`: destination for final long output.
                - `wide_output_path`: destination for final wide output.
                - `show_summary_logs`: toggles terminal summary logs.
                - `parallel`: enables multi-wave parallel execution.
                - `max_workers`: optional worker cap when parallel is enabled.

        Returns:
            preset_pipeline.PrepareMCSByLEAPPipelineConfig: Validated config.
        """

        profile = self.profile()
        available = tuple(_as_required_string_list(profile, "available_waves"))
        defaults = _as_object(profile.get("defaults"), "defaults")

        waves = _parse_waves(raw=kwargs.get("waves", "ALL"), available=available)
        wave_inputs = _parse_wave_inputs(raw=kwargs.get("wave_inputs"), waves=waves)

        output_format = _as_output_format(kwargs.get("output_format", "long_and_wide"))
        wide_suffix_prefix = _as_optional_string(
            kwargs.get("wide_suffix_prefix"),
            fallback=_as_required_string(defaults, "default_wide_suffix_prefix"),
        )
        show_summary_logs = _as_bool(
            kwargs.get("show_summary_logs"),
            fallback=bool(defaults.get("show_summary_logs", True)),
        )

        save_wave_outputs_default = len(waves) > 1
        save_wave_outputs = (
            _as_bool(
                kwargs.get("save_wave_outputs"),
                fallback=save_wave_outputs_default,
            )
            and len(waves) > 1
        )

        save_final_output = _as_bool(kwargs.get("save_final_output"), fallback=True)

        parallel_default = (
            bool(defaults.get("run_parallel_when_possible", True)) and len(waves) > 1
        )
        parallel = (
            _as_bool(kwargs.get("parallel"), fallback=parallel_default)
            and len(waves) > 1
        )
        max_workers = _as_optional_positive_int(kwargs.get("max_workers"))
        if not parallel:
            max_workers = None

        wave_output_dir: Path | None = None
        if save_wave_outputs:
            wave_output_dir = Path(
                _as_optional_string(
                    kwargs.get("wave_output_dir"),
                    fallback=_as_required_string(defaults, "default_wave_output_dir"),
                )
            ).expanduser()

        long_output_path: Path | None = None
        wide_output_path: Path | None = None
        if save_final_output and output_format in {"long", "long_and_wide"}:
            long_output_path = Path(
                _as_optional_string(
                    kwargs.get("long_output_path"),
                    fallback=_as_required_string(defaults, "default_long_output_path"),
                )
            ).expanduser()
        if save_final_output and output_format in {"wide", "long_and_wide"}:
            wide_output_path = Path(
                _as_optional_string(
                    kwargs.get("wide_output_path"),
                    fallback=_as_required_string(defaults, "default_wide_output_path"),
                )
            ).expanduser()

        return preset_pipeline.PrepareMCSByLEAPPipelineConfig(
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


@beartype
def _parse_waves(*, raw: Any, available: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(raw, Sequence) and not isinstance(raw, str):
        token = ",".join(str(entry).strip() for entry in raw if str(entry).strip())
    else:
        token = str(raw).strip()
    if not token:
        token = "ALL"
    return preset_pipeline.parse_wave_list(raw=token, available=available)


@beartype
def _parse_wave_inputs(
    *,
    raw: Any,
    waves: tuple[str, ...],
) -> list[preset_pipeline.WaveInputSpec]:
    if isinstance(raw, Mapping):
        payload = dict(raw)
    elif isinstance(raw, Sequence):
        payload = {}
        for item in raw:
            if isinstance(item, preset_pipeline.WaveInputSpec):
                payload[item.wave] = str(item.raw_dir)
            elif isinstance(item, Mapping):
                wave = item.get("wave")
                path = item.get("raw_dir")
                if isinstance(wave, str) and isinstance(path, str | Path):
                    payload[wave] = str(path)
    else:
        raise InputValidationError(
            "`wave_inputs` must be a mapping {wave: raw_dir} or sequence of "
            "WaveInputSpec-compatible items."
        )

    parsed: list[preset_pipeline.WaveInputSpec] = []
    for wave in waves:
        raw_path = None
        for candidate in (wave, wave.lower(), wave.upper()):
            value = payload.get(candidate)
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


@beartype
def _as_object(value: Any, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InputValidationError(f"`{key}` must be an object.")
    return value


@beartype
def _as_required_string(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"Missing required string parameter: {key}")
    return value.strip()


@beartype
def _as_required_string_list(values: Mapping[str, Any], key: str) -> list[str]:
    value = values.get(key)
    if not isinstance(value, list) or not value:
        raise InputValidationError(f"Missing required string-list parameter: {key}")
    parsed = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if not parsed:
        raise InputValidationError(f"Parameter `{key}` must include at least one item.")
    return parsed


@beartype
def _as_optional_string(value: Any, *, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if fallback.strip():
        return fallback.strip()
    raise InputValidationError("Expected a non-empty string value.")


@beartype
def _as_bool(value: Any, *, fallback: bool) -> bool:
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
    raise InputValidationError("Expected a boolean-compatible value.")


@beartype
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


@beartype
def _as_output_format(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return "long_and_wide"

    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in {"long", "wide", "long_and_wide"}:
        raise InputValidationError(
            "`output_format` must be one of: long, wide, long_and_wide."
        )
    return normalized
