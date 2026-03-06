from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

import pandas as pd
import yaml
from beartype import beartype
from rich import box
from rich.console import Console
from rich.table import Table

from ldt.utils.errors import InputValidationError

from .stage_0_target_rows import apply_stage_0, resolve_stage_0_config
from .stage_1_structural import apply_stage_1, resolve_stage_1_config
from .stage_2_sentinels import apply_stage_2, resolve_stage_2_config
from .stage_3_modelling_policy import apply_stage_3, resolve_stage_3_config
from .stage_4_finalisation import apply_stage_4, resolve_stage_4_config
from .stage_5_encoding_policy import apply_stage_5, resolve_stage_5_config

_PRESET_DIR = Path(__file__).resolve().parent
_DEFAULTS_CONFIG = _PRESET_DIR / "defaults.yaml"
_CONSOLE = Console()


@beartype
@dataclass(frozen=True)
class GeneralDefaults:
    """Default values for the preprocessing preset."""

    show_summary_logs: bool
    save_audit_tables: bool
    default_output_path: Path
    default_audit_output_dir: Path
    default_stage_0_config_path: Path
    default_stage_1_config_path: Path
    default_stage_2_config_path: Path
    default_stage_3_config_path: Path
    default_stage_4_config_path: Path
    default_stage_5_config_path: Path


@beartype
@dataclass(frozen=True)
class PreprocessMCSByLEAPPipelineConfig:
    """Resolved runtime config for preprocess MCS by LEAP preset."""

    input_path: Path
    output_path: Path | None
    save_final_output: bool
    save_audit_tables: bool
    audit_output_dir: Path | None
    show_summary_logs: bool
    stage_0_config_path: Path
    stage_1_config_path: Path
    stage_2_config_path: Path
    stage_3_config_path: Path
    stage_4_config_path: Path
    stage_5_config_path: Path


@beartype
@dataclass(frozen=True)
class PipelineRunResult:
    """Final output payload for one preprocessing pipeline execution."""

    output_data: pd.DataFrame
    output_path: Path | None
    audit_output_dir: Path | None
    audit_paths: tuple[tuple[str, Path], ...]
    stage_0_summary: dict[str, object]
    stage_1_summary: dict[str, object]
    stage_2_summary: dict[str, object]
    stage_3_summary: dict[str, object]
    stage_4_summary: dict[str, object]
    stage_5_summary: dict[str, object]


@beartype
def general_defaults() -> GeneralDefaults:
    """Load default values from `defaults.yaml`."""

    raw = _load_yaml(_DEFAULTS_CONFIG)
    general = _as_mapping(raw.get("general"), "general")
    output = _as_mapping(raw.get("output"), "output")
    stage_configs = _as_mapping(raw.get("stage_configs"), "stage_configs")

    default_output_path = _as_path(
        output.get("default_output_csv"), "output.default_output_csv"
    )
    default_audit_output_dir = _as_path(
        output.get("default_audit_output_dir"),
        "output.default_audit_output_dir",
    )

    stage_0_rel = _as_path(
        stage_configs.get("stage_0_target_rows"),
        "stage_configs.stage_0_target_rows",
    )
    stage_1_rel = _as_path(
        stage_configs.get("stage_1_structural"),
        "stage_configs.stage_1_structural",
    )
    stage_2_rel = _as_path(
        stage_configs.get("stage_2_sentinels"),
        "stage_configs.stage_2_sentinels",
    )
    stage_3_rel = _as_path(
        stage_configs.get("stage_3_modelling_policy"),
        "stage_configs.stage_3_modelling_policy",
    )
    stage_4_rel = _as_path(
        stage_configs.get("stage_4_finalisation"),
        "stage_configs.stage_4_finalisation",
    )
    stage_5_rel = _as_path(
        stage_configs.get("stage_5_encoding_policy"),
        "stage_configs.stage_5_encoding_policy",
    )

    return GeneralDefaults(
        show_summary_logs=bool(general.get("show_summary_logs", True)),
        save_audit_tables=bool(general.get("save_audit_tables", True)),
        default_output_path=default_output_path,
        default_audit_output_dir=default_audit_output_dir,
        default_stage_0_config_path=_PRESET_DIR / stage_0_rel,
        default_stage_1_config_path=_PRESET_DIR / stage_1_rel,
        default_stage_2_config_path=_PRESET_DIR / stage_2_rel,
        default_stage_3_config_path=_PRESET_DIR / stage_3_rel,
        default_stage_4_config_path=_PRESET_DIR / stage_4_rel,
        default_stage_5_config_path=_PRESET_DIR / stage_5_rel,
    )


@beartype
def run_preprocess_mcs_by_leap_pipeline(
    *,
    config: PreprocessMCSByLEAPPipelineConfig,
) -> PipelineRunResult:
    """Execute the 6-stage preprocess MCS by LEAP pipeline."""

    _validate_runtime_paths(config=config)

    data = pd.read_csv(config.input_path)
    if data.empty:
        raise InputValidationError(
            f"Input dataset is empty: {config.input_path.resolve()}"
        )

    if config.show_summary_logs:
        _print_run_header(config=config, data=data)

    stage_2_cfg = resolve_stage_2_config(config.stage_2_config_path)
    stage_0_cfg = resolve_stage_0_config(config.stage_0_config_path)
    stage_0_result = apply_stage_0(
        data=data,
        config=stage_0_cfg,
        sentinel_codes=stage_2_cfg.sentinel_codes,
    )
    if config.show_summary_logs:
        _print_stage_summary(stage_name="Stage 0 - Target Rows", metrics=stage_0_result.summary)

    stage_1_cfg = resolve_stage_1_config(config.stage_1_config_path)
    stage_1_result = apply_stage_1(
        data=stage_0_result.data,
        config=stage_1_cfg,
        sentinel_codes=stage_2_cfg.sentinel_codes,
    )
    if config.show_summary_logs:
        _print_stage_summary(stage_name="Stage 1 - Structural", metrics=stage_1_result.summary)

    stage_2_result = apply_stage_2(data=stage_1_result.data, config=stage_2_cfg)
    if config.show_summary_logs:
        _print_stage_summary(stage_name="Stage 2 - Sentinels", metrics=stage_2_result.summary)

    stage_3_cfg = resolve_stage_3_config(config.stage_3_config_path)
    stage_3_result = apply_stage_3(data=stage_2_result.data, config=stage_3_cfg)
    if config.show_summary_logs:
        _print_stage_summary(stage_name="Stage 3 - Modelling Policy", metrics=stage_3_result.summary)

    stage_4_cfg = resolve_stage_4_config(config.stage_4_config_path)
    stage_4_result = apply_stage_4(data=stage_3_result.data, config=stage_4_cfg)
    if config.show_summary_logs:
        _print_stage_summary(stage_name="Stage 4 - Finalisation", metrics=stage_4_result.summary)

    stage_5_cfg = resolve_stage_5_config(config.stage_5_config_path)
    stage_5_result = apply_stage_5(data=stage_4_result.data, config=stage_5_cfg)
    if config.show_summary_logs:
        _print_stage_summary(stage_name="Stage 5 - Encoding Policy", metrics=stage_5_result.summary)

    final_data = stage_5_result.data
    output_path: Path | None = None
    if config.save_final_output:
        if config.output_path is None:
            raise InputValidationError(
                "`save_final_output=True` requires a non-empty output path."
            )
        output_path = config.output_path.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_data.to_csv(output_path, index=False)

    audit_paths: list[tuple[str, Path]] = []
    if config.save_audit_tables:
        if config.audit_output_dir is None:
            raise InputValidationError(
                "`save_audit_tables=True` requires `audit_output_dir`."
            )

        audit_output_dir = config.audit_output_dir.expanduser()
        audit_output_dir.mkdir(parents=True, exist_ok=True)
        audit_paths.extend(
            _save_stage_tables(
                audit_output_dir=audit_output_dir,
                stage_tables=stage_0_result.tables,
            )
        )
        audit_paths.extend(
            _save_stage_tables(
                audit_output_dir=audit_output_dir,
                stage_tables=stage_1_result.tables,
            )
        )
        audit_paths.extend(
            _save_stage_tables(
                audit_output_dir=audit_output_dir,
                stage_tables=stage_2_result.tables,
            )
        )
        audit_paths.extend(
            _save_stage_tables(
                audit_output_dir=audit_output_dir,
                stage_tables=stage_3_result.tables,
            )
        )
        audit_paths.extend(
            _save_stage_tables(
                audit_output_dir=audit_output_dir,
                stage_tables=stage_4_result.tables,
            )
        )
        audit_paths.extend(
            _save_stage_tables(
                audit_output_dir=audit_output_dir,
                stage_tables=stage_5_result.tables,
            )
        )

        summary_payload = {
            "input_path": str(config.input_path.resolve()),
            "output_path": str(output_path.resolve()) if output_path is not None else None,
            "shape": {
                "rows": int(final_data.shape[0]),
                "columns": int(final_data.shape[1]),
            },
            "stage_0_summary": asdict(stage_0_result.summary),
            "stage_1_summary": asdict(stage_1_result.summary),
            "stage_2_summary": asdict(stage_2_result.summary),
            "stage_3_summary": asdict(stage_3_result.summary),
            "stage_4_summary": asdict(stage_4_result.summary),
            "stage_5_summary": asdict(stage_5_result.summary),
        }

        summary_path = audit_output_dir / "pipeline_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary_payload, handle, indent=2)
        audit_paths.append(("pipeline_summary", summary_path))
    else:
        audit_output_dir = None

    if config.show_summary_logs:
        _print_run_footer(
            final_data=final_data,
            output_path=output_path,
            audit_output_dir=audit_output_dir,
            audit_paths=audit_paths,
        )

    return PipelineRunResult(
        output_data=final_data,
        output_path=output_path.resolve() if output_path is not None else None,
        audit_output_dir=(audit_output_dir.resolve() if audit_output_dir is not None else None),
        audit_paths=tuple((name, path.resolve()) for name, path in audit_paths),
        stage_0_summary=asdict(stage_0_result.summary),
        stage_1_summary=asdict(stage_1_result.summary),
        stage_2_summary=asdict(stage_2_result.summary),
        stage_3_summary=asdict(stage_3_result.summary),
        stage_4_summary=asdict(stage_4_result.summary),
        stage_5_summary=asdict(stage_5_result.summary),
    )


@beartype
def _save_stage_tables(
    *,
    audit_output_dir: Path,
    stage_tables: dict[str, pd.DataFrame],
) -> list[tuple[str, Path]]:
    written: list[tuple[str, Path]] = []
    for key, table in stage_tables.items():
        path = audit_output_dir / f"{key}.csv"
        table.to_csv(path, index=False)
        written.append((key, path))
    return written


@beartype
def _print_run_header(*, config: PreprocessMCSByLEAPPipelineConfig, data: pd.DataFrame) -> None:
    table = Table(
        title="Preprocess MCS by LEAP - Run",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Input", str(config.input_path.resolve()))
    table.add_row("Rows", str(int(data.shape[0])))
    table.add_row("Columns", str(int(data.shape[1])))
    table.add_row("Save final output", str(config.save_final_output))
    table.add_row("Save audit tables", str(config.save_audit_tables))
    _CONSOLE.print(table)


@beartype
def _print_stage_summary(*, stage_name: str, metrics: object) -> None:
    table = Table(
        title=stage_name,
        box=box.SIMPLE,
        show_header=True,
        header_style="bold green",
    )
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in asdict(metrics).items():
        table.add_row(str(key), str(value))
    _CONSOLE.print(table)


@beartype
def _print_run_footer(
    *,
    final_data: pd.DataFrame,
    output_path: Path | None,
    audit_output_dir: Path | None,
    audit_paths: list[tuple[str, Path]],
) -> None:
    table = Table(
        title="Preprocess MCS by LEAP - Completed",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Final rows", str(int(final_data.shape[0])))
    table.add_row("Final columns", str(int(final_data.shape[1])))
    table.add_row("Output path", str(output_path.resolve()) if output_path else "(not saved)")
    table.add_row(
        "Audit directory",
        str(audit_output_dir.resolve()) if audit_output_dir is not None else "(not saved)",
    )
    table.add_row("Audit files", str(len(audit_paths)))
    _CONSOLE.print(table)


@beartype
def _validate_runtime_paths(*, config: PreprocessMCSByLEAPPipelineConfig) -> None:
    if not config.input_path.exists() or not config.input_path.is_file():
        raise InputValidationError(
            f"Input file does not exist: {config.input_path.expanduser()}"
        )

    for path in (
        config.stage_0_config_path,
        config.stage_1_config_path,
        config.stage_2_config_path,
        config.stage_3_config_path,
        config.stage_4_config_path,
        config.stage_5_config_path,
    ):
        if not path.exists() or not path.is_file():
            raise InputValidationError(f"Stage config file does not exist: {path}")


@beartype
def _as_mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise InputValidationError(f"`{context}` must be a mapping.")
    return value


@beartype
def _as_path(value: object, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"`{context}` must be a non-empty path string.")
    return Path(value.strip()).expanduser()


@cache
def _load_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        raise InputValidationError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise InputValidationError(f"Config root must be a mapping: {path}")
    return raw
