from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pandas as pd
import yaml
from beartype import beartype

from ldt.utils.errors import InputValidationError

_STAGE_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _STAGE_DIR / "rules.yaml"


@beartype
@dataclass(frozen=True)
class Stage1Config:
    """Parsed configuration for stage 1 structural cleanup."""

    baseline_only_roots: tuple[str, ...]
    wave_start: int
    wave_end: int
    require_all_effectively_missing: bool
    keep_identifier_columns: tuple[str, ...]
    remove_identifier_columns: tuple[str, ...]
    target_column: str
    remove_target_history_columns: tuple[str, ...]
    partial_missing_threshold: float
    drop_partial_missing_over_threshold: bool
    drop_all_effectively_missing: bool
    drop_single_unique_valid: bool


@beartype
@dataclass(frozen=True)
class Stage1Summary:
    """Compact stage 1 summary metrics."""

    input_rows: int
    input_columns: int
    dropped_columns: int
    output_columns: int
    w1_only_over_time_dropped: int
    id_columns_dropped: int
    target_history_dropped: int
    all_effectively_missing_dropped: int
    single_unique_dropped: int
    partial_missing_over_threshold_dropped: int
    all_effectively_missing_detected: int
    single_unique_detected: int
    partial_missing_over_threshold_detected: int


@beartype
@dataclass(frozen=True)
class Stage1Result:
    """Output payload for stage 1."""

    data: pd.DataFrame
    summary: Stage1Summary
    tables: dict[str, pd.DataFrame]


@beartype
def resolve_stage_1_config(
    config_path: Path | None = None,
) -> Stage1Config:
    """Load and validate stage 1 YAML configuration."""

    path = (config_path or _DEFAULT_CONFIG).expanduser()
    raw = _load_yaml(path)

    baseline_policy = _as_mapping(
        raw.get("baseline_only_roots_over_time"),
        "baseline_only_roots_over_time",
    )
    roots = _as_string_list(
        baseline_policy.get("roots"), "baseline_only_roots_over_time.roots"
    )
    wave_start = _as_int(
        baseline_policy.get("wave_start", 2),
        "baseline_only_roots_over_time.wave_start",
    )
    wave_end = _as_int(
        baseline_policy.get("wave_end", 7),
        "baseline_only_roots_over_time.wave_end",
    )
    if wave_start < 1 or wave_end < wave_start:
        raise InputValidationError(
            "Invalid `baseline_only_roots_over_time` wave range. Expected 1 <= wave_start <= wave_end."
        )

    ids = _as_mapping(raw.get("id_policy"), "id_policy")
    keep_ids = _as_string_list(
        ids.get("keep_identifier_columns", []),
        "id_policy.keep_identifier_columns",
    )
    remove_ids = _as_string_list(
        ids.get("remove_identifier_columns", []),
        "id_policy.remove_identifier_columns",
    )

    target = _as_mapping(raw.get("target_policy"), "target_policy")
    target_column = _as_string(target.get("target_column"), "target_policy.target_column")
    remove_target_history = _as_string_list(
        target.get("remove_history_columns", []),
        "target_policy.remove_history_columns",
    )

    miss = _as_mapping(raw.get("missingness"), "missingness")
    threshold = _as_float(
        miss.get("partial_missing_threshold", 0.60),
        "missingness.partial_missing_threshold",
    )
    if threshold <= 0.0 or threshold >= 1.0:
        raise InputValidationError(
            "`missingness.partial_missing_threshold` must be in the open interval (0, 1)."
        )

    return Stage1Config(
        baseline_only_roots=tuple(roots),
        wave_start=wave_start,
        wave_end=wave_end,
        require_all_effectively_missing=bool(
            baseline_policy.get("require_all_effectively_missing", True)
        ),
        keep_identifier_columns=tuple(keep_ids),
        remove_identifier_columns=tuple(remove_ids),
        target_column=target_column,
        remove_target_history_columns=tuple(remove_target_history),
        partial_missing_threshold=threshold,
        drop_partial_missing_over_threshold=bool(
            miss.get("drop_partial_missing_over_threshold", False)
        ),
        drop_all_effectively_missing=bool(
            miss.get("drop_all_effectively_missing", True)
        ),
        drop_single_unique_valid=bool(miss.get("drop_single_unique_valid", True)),
    )


@beartype
def apply_stage_1(
    *,
    data: pd.DataFrame,
    config: Stage1Config,
    sentinel_codes: tuple[int, ...],
) -> Stage1Result:
    """Run stage 1 structural cleanup and produce audit tables."""

    n_rows, n_cols = int(data.shape[0]), int(data.shape[1])

    profile = _build_column_profile(
        data=data,
        sentinel_codes=set(sentinel_codes),
    )
    all_missing = (
        profile.loc[profile["effective_missing_rate"] == 1.0]
        .sort_values("column")
        .reset_index(drop=True)
    )
    single_unique = (
        profile.loc[
            (profile["valid_count"] > 0)
            & (profile["nunique_valid"] <= 1)
            & (profile["effective_missing_rate"] < 1.0)
        ]
        .sort_values(["nunique_valid", "column"])
        .reset_index(drop=True)
    )
    partial_missing = (
        profile.loc[
            (profile["effective_missing_rate"] > config.partial_missing_threshold)
            & (profile["effective_missing_rate"] < 1.0)
        ]
        .sort_values("effective_missing_rate", ascending=False)
        .reset_index(drop=True)
    )

    w1_table, w1_drop_columns = _build_w1_only_over_time_table(
        data=data,
        roots=config.baseline_only_roots,
        wave_start=config.wave_start,
        wave_end=config.wave_end,
        sentinel_codes=set(sentinel_codes),
        require_all_effectively_missing=config.require_all_effectively_missing,
    )

    id_policy_table = _build_id_policy_table(data=data, config=config)
    target_policy_table = _build_target_policy_table(data=data, config=config)

    drop_rows: list[dict[str, str]] = []

    for column in sorted(w1_drop_columns):
        drop_rows.append(
            {
                "feature": column,
                "reason": "stage1_w1_only_over_time_all_missing",
            }
        )

    for column in config.remove_identifier_columns:
        if column in data.columns:
            drop_rows.append(
                {
                    "feature": column,
                    "reason": "user_policy_secondary_identifier",
                }
            )

    for column in config.remove_target_history_columns:
        if column in data.columns and column != config.target_column:
            drop_rows.append(
                {
                    "feature": column,
                    "reason": "user_policy_drop_target_history",
                }
            )

    if config.drop_all_effectively_missing:
        for column in all_missing["column"].tolist():
            if column == config.target_column:
                continue
            drop_rows.append(
                {
                    "feature": str(column),
                    "reason": "structural_all_effectively_missing",
                }
            )

    if config.drop_single_unique_valid:
        for column in single_unique["column"].tolist():
            if column == config.target_column:
                continue
            drop_rows.append(
                {
                    "feature": str(column),
                    "reason": "structural_single_unique_valid",
                }
            )

    if config.drop_partial_missing_over_threshold:
        for column in partial_missing["column"].tolist():
            if column == config.target_column:
                continue
            drop_rows.append(
                {
                    "feature": str(column),
                    "reason": "structural_partial_missing_over_threshold",
                }
            )

    obvious_remove = _dedupe_drop_rows(rows=drop_rows)

    protected = set(config.keep_identifier_columns) | {config.target_column}
    drop_columns = [
        column
        for column in obvious_remove["feature"].tolist()
        if column in data.columns and column not in protected
    ]

    updated = data.drop(columns=drop_columns, errors="ignore")

    summary = Stage1Summary(
        input_rows=n_rows,
        input_columns=n_cols,
        dropped_columns=len(drop_columns),
        output_columns=int(updated.shape[1]),
        w1_only_over_time_dropped=int(
            obvious_remove[obvious_remove["reason"] == "stage1_w1_only_over_time_all_missing"].shape[0]
        ),
        id_columns_dropped=int(
            obvious_remove[obvious_remove["reason"] == "user_policy_secondary_identifier"].shape[0]
        ),
        target_history_dropped=int(
            obvious_remove[obvious_remove["reason"] == "user_policy_drop_target_history"].shape[0]
        ),
        all_effectively_missing_dropped=int(
            obvious_remove[obvious_remove["reason"] == "structural_all_effectively_missing"].shape[0]
        ),
        single_unique_dropped=int(
            obvious_remove[obvious_remove["reason"] == "structural_single_unique_valid"].shape[0]
        ),
        partial_missing_over_threshold_dropped=int(
            obvious_remove[
                obvious_remove["reason"] == "structural_partial_missing_over_threshold"
            ].shape[0]
        ),
        all_effectively_missing_detected=int(all_missing.shape[0]),
        single_unique_detected=int(single_unique.shape[0]),
        partial_missing_over_threshold_detected=int(partial_missing.shape[0]),
    )

    tables: dict[str, pd.DataFrame] = {
        "stage1_column_profile": profile,
        "stage1_w1_only_over_time_check": w1_table,
        "stage1_id_policy": id_policy_table,
        "stage1_target_policy": target_policy_table,
        "stage1_unusable_all_effectively_missing": all_missing,
        "stage1_unusable_single_unique_valid": single_unique,
        "stage1_partial_missing_over_threshold": partial_missing,
        "stage1_obvious_remove_features": obvious_remove,
    }

    return Stage1Result(data=updated, summary=summary, tables=tables)


@beartype
def _build_column_profile(
    *,
    data: pd.DataFrame,
    sentinel_codes: set[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    n_rows = len(data)

    for column in data.columns:
        series = data[column]
        missing_mask = _effective_missing_mask(series=series, sentinel_codes=sentinel_codes)
        valid = series[~missing_mask]
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "rows": n_rows,
                "na_count": int(series.isna().sum()),
                "effective_missing_count": int(missing_mask.sum()),
                "effective_missing_rate": float(missing_mask.mean()),
                "valid_count": int((~missing_mask).sum()),
                "nunique_valid": int(valid.nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows).sort_values("column").reset_index(drop=True)


@beartype
def _build_w1_only_over_time_table(
    *,
    data: pd.DataFrame,
    roots: tuple[str, ...],
    wave_start: int,
    wave_end: int,
    sentinel_codes: set[int],
    require_all_effectively_missing: bool,
) -> tuple[pd.DataFrame, set[str]]:
    rows: list[dict[str, object]] = []
    to_drop: set[str] = set()

    for root in roots:
        over_time_cols = [
            f"{root}_w{wave}"
            for wave in range(wave_start, wave_end + 1)
            if f"{root}_w{wave}" in data.columns
        ]

        all_missing_cols: list[str] = []
        non_missing_cols: list[str] = []

        for column in over_time_cols:
            missing_mask = _effective_missing_mask(
                series=data[column], sentinel_codes=sentinel_codes
            )
            valid_count = int((~missing_mask).sum())
            if valid_count == 0:
                all_missing_cols.append(column)
            else:
                non_missing_cols.append(column)

        drop_candidate = False
        if over_time_cols:
            if require_all_effectively_missing:
                drop_candidate = len(non_missing_cols) == 0
            else:
                drop_candidate = len(all_missing_cols) > 0

        if drop_candidate:
            to_drop.update(all_missing_cols if require_all_effectively_missing else over_time_cols)

        rows.append(
            {
                "root_feature": root,
                "over_time_columns_present": ",".join(over_time_cols),
                "over_time_all_missing_columns": ",".join(all_missing_cols),
                "over_time_with_signal_columns": ",".join(non_missing_cols),
                "drop_over_time_columns": ",".join(
                    all_missing_cols if require_all_effectively_missing else over_time_cols
                )
                if drop_candidate
                else "",
                "drop_over_time?": "yes" if drop_candidate else "no",
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "root_feature",
                "over_time_columns_present",
                "over_time_all_missing_columns",
                "over_time_with_signal_columns",
                "drop_over_time_columns",
                "drop_over_time?",
            ]
        )
    return frame.sort_values("root_feature").reset_index(drop=True), to_drop


@beartype
def _build_id_policy_table(*, data: pd.DataFrame, config: Stage1Config) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    for feature in config.keep_identifier_columns:
        rows.append(
            {
                "feature": feature,
                "action": "keep",
                "reason": "primary_unique_identifier",
                "exists": "yes" if feature in data.columns else "no",
            }
        )

    for feature in config.remove_identifier_columns:
        rows.append(
            {
                "feature": feature,
                "action": "remove",
                "reason": "secondary_identifier_not_used_for_modelling",
                "exists": "yes" if feature in data.columns else "no",
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["feature", "action", "reason", "exists"])
    return frame.sort_values(["action", "feature"]).reset_index(drop=True)


@beartype
def _build_target_policy_table(*, data: pd.DataFrame, config: Stage1Config) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for feature in config.remove_target_history_columns:
        rows.append(
            {
                "feature": feature,
                "action": "remove",
                "reason": "target_history_not_used_as_predictor",
                "exists": "yes" if feature in data.columns else "no",
            }
        )

    rows.append(
        {
            "feature": config.target_column,
            "action": "keep_target",
            "reason": "final_target_variable",
            "exists": "yes" if config.target_column in data.columns else "no",
        }
    )

    frame = pd.DataFrame(rows)
    return frame.sort_values(["action", "feature"]).reset_index(drop=True)


@beartype
def _dedupe_drop_rows(*, rows: list[dict[str, str]]) -> pd.DataFrame:
    ordered: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        feature = row.get("feature", "")
        if not feature or feature in seen:
            continue
        seen.add(feature)
        ordered.append(row)

    frame = pd.DataFrame(ordered)
    if frame.empty:
        return pd.DataFrame(columns=["feature", "reason"])
    return frame.sort_values(["reason", "feature"]).reset_index(drop=True)


@beartype
def _effective_missing_mask(*, series: pd.Series, sentinel_codes: set[int]) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return series.isna() | numeric.isin(sentinel_codes)


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

    parsed: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise InputValidationError(
                f"Invalid entry at `{context}[{index}]`. Expected non-empty string."
            )
        parsed.append(item.strip())
    return parsed


@beartype
def _as_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputValidationError(f"`{context}` must be an integer.")
    return int(value)


@beartype
def _as_float(value: object, context: str) -> float:
    if isinstance(value, bool):
        raise InputValidationError(f"`{context}` must be numeric.")
    if isinstance(value, (int, float)):
        return float(value)
    raise InputValidationError(f"`{context}` must be numeric.")


@beartype
def _as_int_list(value: object, context: str) -> list[int]:
    if value in (None, "none"):
        return []
    if not isinstance(value, list):
        raise InputValidationError(f"`{context}` must be a list of integers.")

    out: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise InputValidationError(
                f"Invalid entry at `{context}[{index}]`. Expected integer."
            )
        out.append(int(item))
    return out


@cache
def _load_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        raise InputValidationError(f"Stage-1 config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise InputValidationError("Stage-1 config root must be a mapping.")
    return raw
