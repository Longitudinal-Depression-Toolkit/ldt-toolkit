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
class Stage4Config:
    """Parsed configuration for stage 4 finalisation."""

    target_column: str
    correlation_pruning_apply: bool
    correlation_method: str
    correlation_abs_threshold: float
    correlation_exclude_columns: tuple[str, ...]


@beartype
@dataclass(frozen=True)
class Stage4Summary:
    """Compact stage 4 summary metrics."""

    input_rows: int
    output_rows: int
    input_columns: int
    output_columns: int
    high_corr_pairs_detected: int
    high_corr_columns_dropped: int


@beartype
@dataclass(frozen=True)
class Stage4Result:
    """Output payload for stage 4."""

    data: pd.DataFrame
    summary: Stage4Summary
    tables: dict[str, pd.DataFrame]


@beartype
def resolve_stage_4_config(config_path: Path | None = None) -> Stage4Config:
    """Load and validate stage 4 YAML configuration."""

    path = (config_path or _DEFAULT_CONFIG).expanduser()
    raw = _load_yaml(path)

    target_column = _as_string(raw.get("target_column"), "target_column")

    corr = _as_mapping(raw.get("correlation_pruning"), "correlation_pruning")
    corr_method = str(corr.get("method", "spearman")).strip().lower()
    if corr_method != "spearman":
        raise InputValidationError(
            "Unsupported `correlation_pruning.method`. Supported: spearman."
        )

    abs_threshold = _as_float(
        corr.get("abs_threshold", 0.98),
        "correlation_pruning.abs_threshold",
    )
    if abs_threshold <= 0.0 or abs_threshold > 1.0:
        raise InputValidationError(
            "`correlation_pruning.abs_threshold` must be in (0, 1]."
        )

    exclude_columns = _as_string_list(
        corr.get("exclude_columns", []),
        "correlation_pruning.exclude_columns",
    )

    return Stage4Config(
        target_column=target_column,
        correlation_pruning_apply=bool(corr.get("apply", False)),
        correlation_method=corr_method,
        correlation_abs_threshold=float(abs_threshold),
        correlation_exclude_columns=tuple(exclude_columns),
    )


@beartype
def apply_stage_4(*, data: pd.DataFrame, config: Stage4Config) -> Stage4Result:
    """Run stage 4 finalisation policy."""

    input_rows, input_columns = int(data.shape[0]), int(data.shape[1])

    if config.target_column not in data.columns:
        raise InputValidationError(
            f"Stage 4 target column not found: {config.target_column}"
        )

    updated = data.copy()

    predictor_missingness = _build_predictor_missingness_table(
        data=updated,
        target_column=config.target_column,
    )
    missing_rate_map = _missing_rate_lookup(predictor_missingness)

    high_corr_pairs = _build_high_corr_pairs(
        data=updated,
        config=config,
        missing_rate_map=missing_rate_map,
    )
    corr_decisions, dropped_corr_features = _decide_corr_pruning(
        pairs=high_corr_pairs,
        apply=config.correlation_pruning_apply,
    )

    updated = updated.drop(
        columns=sorted(dropped_corr_features),
        errors="ignore",
    )

    dropped_corr_table = pd.DataFrame(
        [
            {
                "feature": feature,
                "reason": "stage4_high_correlation_pruning",
            }
            for feature in sorted(dropped_corr_features)
        ]
    )
    if dropped_corr_table.empty:
        dropped_corr_table = pd.DataFrame(columns=["feature", "reason"])

    summary = Stage4Summary(
        input_rows=input_rows,
        output_rows=int(updated.shape[0]),
        input_columns=input_columns,
        output_columns=int(updated.shape[1]),
        high_corr_pairs_detected=int(high_corr_pairs.shape[0]),
        high_corr_columns_dropped=int(len(dropped_corr_features)),
    )

    tables: dict[str, pd.DataFrame] = {
        "stage4_predictor_missingness_target_filtered": predictor_missingness,
        "stage4_high_correlation_pairs": corr_decisions,
        "stage4_high_correlation_dropped_features": dropped_corr_table,
    }

    return Stage4Result(data=updated, summary=summary, tables=tables)


@beartype
def _build_predictor_missingness_table(
    *,
    data: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n_rows = int(data.shape[0])

    for column in data.columns:
        if column == target_column:
            continue
        series = data[column]
        missing_count = int(series.isna().sum())
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "rows": n_rows,
                "missing_count": missing_count,
                "missing_rate": float(missing_count / n_rows) if n_rows else 0.0,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["column", "dtype", "rows", "missing_count", "missing_rate"]
        )

    return pd.DataFrame(rows).sort_values(
        ["missing_rate", "column"],
        ascending=[False, True],
    ).reset_index(drop=True)


@beartype
def _missing_rate_lookup(table: pd.DataFrame) -> dict[str, float]:
    if table.empty:
        return {}
    return {
        str(row["column"]): float(row["missing_rate"])
        for _, row in table.iterrows()
    }


@beartype
def _build_high_corr_pairs(
    *,
    data: pd.DataFrame,
    config: Stage4Config,
    missing_rate_map: dict[str, float],
) -> pd.DataFrame:
    exclude = set(config.correlation_exclude_columns) | {config.target_column}
    numeric_predictors = [
        column
        for column in data.columns
        if column not in exclude and pd.api.types.is_numeric_dtype(data[column])
    ]

    columns = [
        "feature_left",
        "feature_right",
        "correlation",
        "abs_correlation",
        "missing_rate_left",
        "missing_rate_right",
    ]
    if len(numeric_predictors) < 2:
        return pd.DataFrame(columns=columns)

    corr = data[numeric_predictors].corr(method=config.correlation_method)
    rows: list[dict[str, object]] = []
    for i, left in enumerate(corr.columns):
        for j in range(i + 1, len(corr.columns)):
            right = corr.columns[j]
            value = corr.iat[i, j]
            if pd.isna(value):
                continue
            abs_value = abs(float(value))
            if abs_value < config.correlation_abs_threshold:
                continue
            rows.append(
                {
                    "feature_left": left,
                    "feature_right": right,
                    "correlation": float(value),
                    "abs_correlation": abs_value,
                    "missing_rate_left": float(missing_rate_map.get(left, 0.0)),
                    "missing_rate_right": float(missing_rate_map.get(right, 0.0)),
                }
            )

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows).sort_values(
        ["abs_correlation", "feature_left", "feature_right"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


@beartype
def _decide_corr_pruning(
    *,
    pairs: pd.DataFrame,
    apply: bool,
) -> tuple[pd.DataFrame, set[str]]:
    columns = [
        "feature_left",
        "feature_right",
        "correlation",
        "abs_correlation",
        "missing_rate_left",
        "missing_rate_right",
        "keep_feature",
        "drop_feature",
        "tie_break_rule",
        "drop_applied",
    ]
    if pairs.empty:
        return pd.DataFrame(columns=columns), set()

    dropped: set[str] = set()
    rows: list[dict[str, object]] = []

    for _, rec in pairs.iterrows():
        left = str(rec["feature_left"])
        right = str(rec["feature_right"])
        left_missing = float(rec["missing_rate_left"])
        right_missing = float(rec["missing_rate_right"])

        keep_feature, drop_feature, rule = _select_keep_drop(
            left=left,
            right=right,
            left_missing=left_missing,
            right_missing=right_missing,
        )

        drop_applied = "no"
        if apply:
            if left in dropped or right in dropped:
                keep_feature = ""
                drop_feature = ""
                rule = "pair_contains_already_dropped_feature"
            else:
                dropped.add(drop_feature)
                drop_applied = "yes"
        else:
            rule = f"dry_run_{rule}"

        rows.append(
            {
                "feature_left": left,
                "feature_right": right,
                "correlation": float(rec["correlation"]),
                "abs_correlation": float(rec["abs_correlation"]),
                "missing_rate_left": left_missing,
                "missing_rate_right": right_missing,
                "keep_feature": keep_feature,
                "drop_feature": drop_feature,
                "tie_break_rule": rule,
                "drop_applied": drop_applied,
            }
        )

    decided = pd.DataFrame(rows, columns=columns)
    return decided, dropped


@beartype
def _select_keep_drop(
    *,
    left: str,
    right: str,
    left_missing: float,
    right_missing: float,
) -> tuple[str, str, str]:
    if left_missing < right_missing:
        return left, right, "drop_higher_missing_rate"
    if right_missing < left_missing:
        return right, left, "drop_higher_missing_rate"

    keep_feature = left if left < right else right
    drop_feature = right if keep_feature == left else left
    return keep_feature, drop_feature, "drop_lexicographically_larger_on_tie"


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


@beartype
def _as_float(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputValidationError(f"`{context}` must be a float.")
    as_float = float(value)
    if pd.isna(as_float):
        raise InputValidationError(f"`{context}` must be a valid float.")
    return as_float


@cache
def _load_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        raise InputValidationError(f"Stage-4 config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise InputValidationError("Stage-4 config root must be a mapping.")
    return raw
