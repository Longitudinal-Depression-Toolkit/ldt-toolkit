from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pandas as pd
import yaml
from beartype import beartype

from ldt.utils.errors import InputValidationError

_STAGE_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _STAGE_DIR / "policy.yaml"


@beartype
@dataclass(frozen=True)
class Stage0Config:
    """Parsed configuration for stage 0 target-row filtering."""

    target_column: str
    require_target_non_missing: bool


@beartype
@dataclass(frozen=True)
class Stage0Summary:
    """Compact stage 0 summary metrics."""

    input_rows: int
    output_rows: int
    rows_dropped_missing_target: int
    target_missing_rate_before: float
    target_effective_missing_rate_before: float


@beartype
@dataclass(frozen=True)
class Stage0Result:
    """Output payload for stage 0."""

    data: pd.DataFrame
    summary: Stage0Summary
    tables: dict[str, pd.DataFrame]


@beartype
def resolve_stage_0_config(config_path: Path | None = None) -> Stage0Config:
    """Load and validate stage 0 YAML configuration."""

    path = (config_path or _DEFAULT_CONFIG).expanduser()
    raw = _load_yaml(path)

    policy = _as_mapping(raw.get("target_rows"), "target_rows")
    target_column = _as_string(
        policy.get("target_column"),
        "target_rows.target_column",
    )

    return Stage0Config(
        target_column=target_column,
        require_target_non_missing=bool(
            policy.get("require_target_non_missing", True)
        ),
    )


@beartype
def apply_stage_0(
    *,
    data: pd.DataFrame,
    config: Stage0Config,
    sentinel_codes: tuple[int, ...] = (),
) -> Stage0Result:
    """Run stage 0 target-row filtering."""

    if config.target_column not in data.columns:
        raise InputValidationError(
            f"Stage 0 target column not found: {config.target_column}"
        )

    input_rows = int(data.shape[0])
    target_series = data[config.target_column]
    target_missing_rate_before = float(target_series.isna().mean())

    numeric_target = pd.to_numeric(target_series, errors="coerce")
    sentinel_missing_mask = (
        numeric_target.isin(set(sentinel_codes)) if sentinel_codes else pd.Series(False, index=data.index)
    )
    target_effective_missing_mask = target_series.isna() | sentinel_missing_mask
    target_effective_missing_rate_before = float(target_effective_missing_mask.mean())

    if config.require_target_non_missing:
        keep_mask = ~target_effective_missing_mask
    else:
        keep_mask = pd.Series(True, index=data.index)

    updated = data.loc[keep_mask].copy()
    rows_dropped = int((~keep_mask).sum())

    summary = Stage0Summary(
        input_rows=input_rows,
        output_rows=int(updated.shape[0]),
        rows_dropped_missing_target=rows_dropped,
        target_missing_rate_before=target_missing_rate_before,
        target_effective_missing_rate_before=target_effective_missing_rate_before,
    )

    policy_table = pd.DataFrame(
        [
            {
                "target_column": config.target_column,
                "require_target_non_missing": (
                    "yes" if config.require_target_non_missing else "no"
                ),
                "target_missing_rate_before": target_missing_rate_before,
                "target_effective_missing_rate_before": target_effective_missing_rate_before,
                "rows_before": input_rows,
                "rows_after": int(updated.shape[0]),
                "rows_dropped_missing_target": rows_dropped,
            }
        ]
    )

    tables: dict[str, pd.DataFrame] = {
        "stage0_target_row_policy": policy_table,
    }
    return Stage0Result(data=updated, summary=summary, tables=tables)


@beartype
def _as_mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise InputValidationError(f"`{context}` must be a mapping.")
    return value


@beartype
def _as_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"`{context}` must be a non-empty string.")
    return value.strip()


@cache
def _load_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        raise InputValidationError(f"Stage-0 config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise InputValidationError("Stage-0 config root must be a mapping.")
    return raw
