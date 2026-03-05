from __future__ import annotations

from pathlib import Path
from typing import Any

from beartype import beartype

from ldt.utils.errors import InputValidationError
from ldt.utils.metadata import ComponentMetadata
from ldt.utils.templates.tools.data_preprocessing import DataPreprocessingTool

from . import pipeline as preset_pipeline


@beartype
class PreprocessMCSByLEAP(DataPreprocessingTool):
    """Python API for the Preprocess MCS by LEAP preset.

    Consumes prepared MCS wide data and runs the 6-stage policy pipeline:
    target rows, structural cleanup, sentinel handling, leakage policy,
    finalisation diagnostics, and encoding policy.

    Produces an ML-ready wide dataset plus optional stage audit tables.

    Examples:
        ```python
        from ldt.data_preprocessing import PreprocessMCSByLEAP

        result = PreprocessMCSByLEAP().fit_preprocess(
            input_path="data/processed/MCS/mcs_longitudinal_wide.csv",
            save_final_output=True,
            save_audit_tables=True,
        )
        print(result.output_path)
        print(result.audit_output_dir)
        ```
    """

    metadata = ComponentMetadata(
        name="preprocess_mcs_by_leap",
        full_name="Preprocess MCS by LEAP",
        abstract_description=(
            "Run a staged, config-driven preprocessing pipeline for the prepared "
            "MCS wide dataset, with transparent audit tables for each stage."
        ),
    )

    def __init__(self) -> None:
        self._fitted_config: preset_pipeline.PreprocessMCSByLEAPPipelineConfig | None = (
            None
        )

    @staticmethod
    @beartype
    def profile() -> dict[str, Any]:
        """Return defaults and stage-config paths for this preset."""

        defaults = preset_pipeline.general_defaults()
        return {
            "preset": "preprocess_mcs_by_leap",
            "defaults": {
                "show_summary_logs": bool(defaults.show_summary_logs),
                "save_audit_tables": bool(defaults.save_audit_tables),
                "default_output_path": str(defaults.default_output_path),
                "default_audit_output_dir": str(defaults.default_audit_output_dir),
                "default_stage_0_config_path": str(defaults.default_stage_0_config_path),
                "default_stage_1_config_path": str(defaults.default_stage_1_config_path),
                "default_stage_2_config_path": str(defaults.default_stage_2_config_path),
                "default_stage_3_config_path": str(defaults.default_stage_3_config_path),
                "default_stage_4_config_path": str(defaults.default_stage_4_config_path),
                "default_stage_5_config_path": str(defaults.default_stage_5_config_path),
            },
        }

    @beartype
    def fit(self, **kwargs: Any) -> PreprocessMCSByLEAP:
        """Store a validated pipeline config for a later `preprocess(...)` call.

        Args:
            **kwargs: Same configuration keys accepted by `build_config(...)`.
                Expected keys:

                - `input_path`: required prepared wide input.
                - `output_path`: destination for final preprocessed CSV.
                - `save_final_output`: final output write toggle.
                - `save_audit_tables`: audit table write toggle.
                - `audit_output_dir`: destination directory for audit outputs.
                - `show_summary_logs`: summary logging toggle.
                - `stage_0_config_path`: stage 0 policy override path.
                - `stage_1_config_path`: stage 1 policy override path.
                - `stage_2_config_path`: stage 2 policy override path.
                - `stage_3_config_path`: stage 3 policy override path.
                - `stage_4_config_path`: stage 4 policy override path.
                - `stage_5_config_path`: stage 5 policy override path.

        Returns:
            PreprocessMCSByLEAP: The fitted preset instance.
        """

        self._fitted_config = self.build_config(**kwargs)
        return self

    @beartype
    def preprocess(self, **kwargs: Any) -> preset_pipeline.PipelineRunResult:
        """Execute the preprocessing pipeline.

        Config resolution order:
        1. `config` passed in `kwargs`
        2. other `kwargs` converted via `build_config(...)`
        3. config previously stored by `fit(...)`

        Args:
            **kwargs: Optional `config` plus any config override keys accepted
                by `build_config(...)`.
                `config` should be a `PreprocessMCSByLEAPPipelineConfig`
                instance.

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
                "`preprocess(...)`, or call `fit(...)` first."
            )
        if not isinstance(config, preset_pipeline.PreprocessMCSByLEAPPipelineConfig):
            raise InputValidationError(
                "`config` must be a PreprocessMCSByLEAPPipelineConfig instance."
            )

        return preset_pipeline.run_preprocess_mcs_by_leap_pipeline(config=config)

    @beartype
    def build_config(
        self,
        **kwargs: Any,
    ) -> preset_pipeline.PreprocessMCSByLEAPPipelineConfig:
        """Build a validated `PreprocessMCSByLEAPPipelineConfig` from kwargs.

        Args:
            **kwargs: Runtime overrides. Expected keys:
                
                - `input_path`: required path to prepared wide CSV input.
                - `output_path`: destination for final preprocessed CSV.
                - `save_final_output`: enables final output write.
                - `save_audit_tables`: enables stage audit table persistence.
                - `audit_output_dir`: target directory for audit outputs.
                - `show_summary_logs`: toggles console summary logs.
                - `stage_0_config_path`: override path for stage 0 policy.
                - `stage_1_config_path`: override path for stage 1 policy.
                - `stage_2_config_path`: override path for stage 2 policy.
                - `stage_3_config_path`: override path for stage 3 policy.
                - `stage_4_config_path`: override path for stage 4 policy.
                - `stage_5_config_path`: override path for stage 5 policy.

        Returns:
            preset_pipeline.PreprocessMCSByLEAPPipelineConfig: Validated config.
        """

        defaults = preset_pipeline.general_defaults()

        input_path_raw = kwargs.get("input_path")
        if input_path_raw is None:
            raise InputValidationError("Missing required parameter: input_path")
        input_path = Path(str(input_path_raw)).expanduser()

        save_final_output = bool(kwargs.get("save_final_output", True))
        output_path_raw = kwargs.get("output_path")
        output_path = (
            Path(str(output_path_raw)).expanduser()
            if output_path_raw not in (None, "")
            else defaults.default_output_path
        )

        save_audit_tables = bool(
            kwargs.get("save_audit_tables", defaults.save_audit_tables)
        )
        audit_output_dir_raw = kwargs.get("audit_output_dir")
        audit_output_dir = (
            Path(str(audit_output_dir_raw)).expanduser()
            if audit_output_dir_raw not in (None, "")
            else defaults.default_audit_output_dir
        )

        show_summary_logs = bool(
            kwargs.get("show_summary_logs", defaults.show_summary_logs)
        )

        stage_0_cfg = _resolve_stage_config_path(
            raw=kwargs.get("stage_0_config_path"),
            default=defaults.default_stage_0_config_path,
        )
        stage_1_cfg = _resolve_stage_config_path(
            raw=kwargs.get("stage_1_config_path"),
            default=defaults.default_stage_1_config_path,
        )
        stage_2_cfg = _resolve_stage_config_path(
            raw=kwargs.get("stage_2_config_path"),
            default=defaults.default_stage_2_config_path,
        )
        stage_3_cfg = _resolve_stage_config_path(
            raw=kwargs.get("stage_3_config_path"),
            default=defaults.default_stage_3_config_path,
        )
        stage_4_cfg = _resolve_stage_config_path(
            raw=kwargs.get("stage_4_config_path"),
            default=defaults.default_stage_4_config_path,
        )
        stage_5_cfg = _resolve_stage_config_path(
            raw=kwargs.get("stage_5_config_path"),
            default=defaults.default_stage_5_config_path,
        )

        return preset_pipeline.PreprocessMCSByLEAPPipelineConfig(
            input_path=input_path,
            output_path=output_path,
            save_final_output=save_final_output,
            save_audit_tables=save_audit_tables,
            audit_output_dir=audit_output_dir,
            show_summary_logs=show_summary_logs,
            stage_0_config_path=stage_0_cfg,
            stage_1_config_path=stage_1_cfg,
            stage_2_config_path=stage_2_cfg,
            stage_3_config_path=stage_3_cfg,
            stage_4_config_path=stage_4_cfg,
            stage_5_config_path=stage_5_cfg,
        )


@beartype
def _resolve_stage_config_path(*, raw: Any, default: Path) -> Path:
    if raw in (None, ""):
        return default
    return Path(str(raw)).expanduser()
