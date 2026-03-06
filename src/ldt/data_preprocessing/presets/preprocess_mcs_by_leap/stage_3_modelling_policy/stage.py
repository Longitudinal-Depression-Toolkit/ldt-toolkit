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
class Stage3Config:
    """Parsed configuration for stage 3 modelling policy."""

    target_column: str
    leakage_remove_columns: tuple[str, ...]


@beartype
@dataclass(frozen=True)
class Stage3Summary:
    """Compact stage 3 summary metrics."""

    input_columns: int
    output_columns: int
    leakage_columns_removed: int


@beartype
@dataclass(frozen=True)
class Stage3Result:
    """Output payload for stage 3."""

    data: pd.DataFrame
    summary: Stage3Summary
    tables: dict[str, pd.DataFrame]


@beartype
def resolve_stage_3_config(config_path: Path | None = None) -> Stage3Config:
    """Load and validate stage 3 YAML configuration."""

    path = (config_path or _DEFAULT_CONFIG).expanduser()
    raw = _load_yaml(path)

    # No backward compatibility for historical low-cardinality config in stage 3.
    if "low_cardinality" in raw:
        raise InputValidationError(
            "Stage 3 no longer accepts `low_cardinality` settings. "
            "Encoding policy is now fully handled in Stage 5 (`stage_5_encoding_policy`)."
        )

    leakage = _as_mapping(raw.get("leakage"), "leakage")
    target_column = _as_string(leakage.get("target_column"), "leakage.target_column")
    leakage_remove_columns = _as_string_list(
        leakage.get("remove_columns", []),
        "leakage.remove_columns",
    )

    return Stage3Config(
        target_column=target_column,
        leakage_remove_columns=tuple(leakage_remove_columns),
    )


@beartype
def apply_stage_3(*, data: pd.DataFrame, config: Stage3Config) -> Stage3Result:
    """Run stage 3 leakage-only policy."""

    input_columns = int(data.shape[1])

    removal_rows: list[dict[str, str]] = []
    leakage_drop_columns: list[str] = []

    for column in config.leakage_remove_columns:
        exists = column in data.columns
        if exists and column != config.target_column:
            leakage_drop_columns.append(column)

        removal_rows.append(
            {
                "feature": column,
                "action": "remove" if exists else "remove_if_present",
                "reason": "target_history_not_used_as_predictor",
                "exists": "yes" if exists else "no",
            }
        )

    if config.target_column in data.columns:
        removal_rows.append(
            {
                "feature": config.target_column,
                "action": "keep_target",
                "reason": "final_target_variable",
                "exists": "yes",
            }
        )

    leakage_policy = (
        pd.DataFrame(removal_rows).sort_values(["action", "feature"]).reset_index(drop=True)
    )

    dropped_features = sorted(set(leakage_drop_columns))
    updated = data.drop(columns=dropped_features, errors="ignore")

    dropped_table = pd.DataFrame(
        [{"feature": feature, "reason": "stage3_leakage_policy"} for feature in dropped_features]
    )
    if dropped_table.empty:
        dropped_table = pd.DataFrame(columns=["feature", "reason"])

    summary = Stage3Summary(
        input_columns=input_columns,
        output_columns=int(updated.shape[1]),
        leakage_columns_removed=len(dropped_features),
    )

    tables: dict[str, pd.DataFrame] = {
        "stage3_leakage_policy": leakage_policy,
        "stage3_leakage_dropped_features": dropped_table,
    }

    return Stage3Result(data=updated, summary=summary, tables=tables)


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


@beartype
def _as_string_list(value: object, context: str) -> list[str]:
    if value in (None, "none"):
        return []
    if not isinstance(value, list):
        raise InputValidationError(f"`{context}` must be a list of strings.")

    out: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise InputValidationError(
                f"Invalid entry at `{context}[{index}]`. Expected non-empty string."
            )
        out.append(item.strip())
    return out


@cache
def _load_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        raise InputValidationError(f"Stage-3 config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise InputValidationError("Stage-3 config root must be a mapping.")
    return raw

