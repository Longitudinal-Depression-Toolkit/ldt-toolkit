from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pandas as pd
import yaml
from beartype import beartype

from ldt.utils.errors import InputValidationError

_STAGE_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _STAGE_DIR / "sentinels.yaml"


@beartype
@dataclass(frozen=True)
class Stage2Config:
    """Parsed configuration for stage 2 sentinel handling."""

    sentinel_codes: tuple[int, ...]
    apply_nan_replacement: bool
    fallback_to_observed_when_no_label_match: bool
    root_overrides: dict[str, tuple[int, ...]]
    dictionary_validation_enabled: bool
    missing_label_terms: tuple[str, ...]


@beartype
@dataclass(frozen=True)
class Stage2Summary:
    """Compact stage 2 summary metrics."""

    columns_evaluated: int
    columns_with_final_sentinel_map: int
    columns_with_replacements: int
    total_cells_replaced: int
    columns_with_observed_overcapture: int
    columns_with_observed_undercapture: int


@beartype
@dataclass(frozen=True)
class Stage2Result:
    """Output payload for stage 2."""

    data: pd.DataFrame
    summary: Stage2Summary
    tables: dict[str, pd.DataFrame]


@beartype
def resolve_stage_2_config(
    config_path: Path | None = None,
) -> Stage2Config:
    """Load and validate stage 2 YAML configuration."""

    path = (config_path or _DEFAULT_CONFIG).expanduser()
    raw = _load_yaml(path)

    sent = _as_mapping(raw.get("sentinel_detection"), "sentinel_detection")
    configured_sentinels = sent.get("sentinel_codes", [])
    current_sentinels = _as_int_list(
        configured_sentinels,
        "sentinel_detection.sentinel_codes",
    )

    root_overrides_raw = raw.get("root_overrides", {})
    if not isinstance(root_overrides_raw, dict):
        raise InputValidationError("`root_overrides` must be a mapping of root->int list.")

    root_overrides: dict[str, tuple[int, ...]] = {}
    for key, value in root_overrides_raw.items():
        if not isinstance(key, str) or not key.strip():
            raise InputValidationError("`root_overrides` keys must be non-empty strings.")
        root_overrides[key.strip()] = tuple(
            _as_int_list(value, f"root_overrides.{key}")
        )

    dv = _as_mapping(raw.get("dictionary_validation"), "dictionary_validation")

    missing_label_terms = _as_string_list(
        dv.get("missing_label_terms", []),
        "dictionary_validation.missing_label_terms",
    )

    return Stage2Config(
        sentinel_codes=tuple(current_sentinels),
        apply_nan_replacement=bool(sent.get("apply_nan_replacement", True)),
        fallback_to_observed_when_no_label_match=bool(
            sent.get("fallback_to_observed_when_no_label_match", True)
        ),
        root_overrides=root_overrides,
        dictionary_validation_enabled=bool(dv.get("enabled", True)),
        missing_label_terms=tuple(missing_label_terms),
    )


@beartype
def apply_stage_2(*, data: pd.DataFrame, config: Stage2Config) -> Stage2Result:
    """Run stage 2 sentinel validation and NaN replacement."""

    parsed_columns = _collect_parsed_columns(data)

    if config.dictionary_validation_enabled:
        validation = _validate_columns_with_label_terms(
            data=data,
            parsed_columns=parsed_columns,
            config=config,
        )
    else:
        validation = _validation_when_disabled(
            data=data,
            parsed_columns=parsed_columns,
            current_sentinels=set(config.sentinel_codes),
        )

    final_map = _build_final_sentinel_map(validation=validation, config=config)
    compact = _build_compact_sentinel_map(final_map=final_map)

    updated, replacement = _apply_sentinel_nan_replacement(
        data=data,
        final_map=final_map,
        apply_nan_replacement=config.apply_nan_replacement,
    )

    overcapture = validation.loc[
        validation["observed_overcapture_codes"].fillna("").astype(str).str.len() > 0
    ].copy()
    undercapture = validation.loc[
        validation["observed_undercapture_codes"].fillna("").astype(str).str.len() > 0
    ].copy()
    status_counts = (
        validation["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="count")
    )

    summary = Stage2Summary(
        columns_evaluated=int(validation.shape[0]),
        columns_with_final_sentinel_map=int(final_map.shape[0]),
        columns_with_replacements=int((replacement["replaced_count"] > 0).sum())
        if not replacement.empty
        else 0,
        total_cells_replaced=int(replacement["replaced_count"].sum())
        if not replacement.empty
        else 0,
        columns_with_observed_overcapture=int(overcapture.shape[0]),
        columns_with_observed_undercapture=int(undercapture.shape[0]),
    )

    tables: dict[str, pd.DataFrame] = {
        "stage2_column_sentinel_validation": validation,
        "stage2_status_counts": status_counts,
        "stage2_observed_overcapture_columns": overcapture,
        "stage2_observed_undercapture_columns": undercapture,
        "stage2_sentinel_final_map": final_map,
        "stage2_sentinel_final_map_compact": compact,
        "stage2_sentinel_replacement_summary": replacement,
    }

    return Stage2Result(data=updated, summary=summary, tables=tables)


@beartype
def _collect_parsed_columns(data: pd.DataFrame) -> list[tuple[str, str, int, bool]]:
    parsed: list[tuple[str, str, int, bool]] = []
    for column in data.columns:
        parsed_column = _parse_wide_column(column)
        if parsed_column is None:
            continue
        base, wave, is_non_long = parsed_column
        parsed.append((column, base, wave, is_non_long))
    return parsed


@beartype
def _validate_columns_with_label_terms(
    *,
    data: pd.DataFrame,
    parsed_columns: list[tuple[str, str, int, bool]],
    config: Stage2Config,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    current_sentinels = set(config.sentinel_codes)

    for column, base, wave, is_non_long in parsed_columns:
        series = data[column]
        observed_codes = _to_int_codes(series)
        observed_current = sorted(observed_codes & current_sentinels)
        dict_missing = _extract_missing_like_codes_from_series(
            series=series,
            missing_terms=config.missing_label_terms,
        )
        observed_dict_missing = sorted(observed_codes & set(dict_missing))
        observed_overcapture = sorted(
            code for code in observed_current if code not in set(dict_missing)
        )
        observed_undercapture = sorted(
            code for code in observed_dict_missing if code not in current_sentinels
        )

        rows.append(
            {
                "column": column,
                "wave": wave,
                "base": base,
                "is_non_long": is_non_long,
                "status": (
                    "label_terms_in_data" if dict_missing else "no_label_term_match"
                ),
                "scope": "",
                "source_variable": "",
                "dta_file": "",
                "dict_missing_codes": _codes_to_string(dict_missing),
                "observed_current_sentinel_codes": _codes_to_string(observed_current),
                "observed_dict_missing_codes": _codes_to_string(observed_dict_missing),
                "observed_overcapture_codes": _codes_to_string(observed_overcapture),
                "observed_undercapture_codes": _codes_to_string(observed_undercapture),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "column",
                "wave",
                "base",
                "is_non_long",
                "status",
                "scope",
                "source_variable",
                "dta_file",
                "dict_missing_codes",
                "observed_current_sentinel_codes",
                "observed_dict_missing_codes",
                "observed_overcapture_codes",
                "observed_undercapture_codes",
            ]
        )
    return frame.sort_values(["wave", "base", "column"]).reset_index(drop=True)


@beartype
def _validation_when_disabled(
    *,
    data: pd.DataFrame,
    parsed_columns: list[tuple[str, str, int, bool]],
    current_sentinels: set[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column, base, wave, is_non_long in parsed_columns:
        observed_codes = _to_int_codes(data[column])
        rows.append(
            {
                "column": column,
                "wave": wave,
                "base": base,
                "is_non_long": is_non_long,
                "status": "dictionary_validation_disabled",
                "scope": "",
                "source_variable": "",
                "dta_file": "",
                "dict_missing_codes": "",
                "observed_current_sentinel_codes": _codes_to_string(
                    sorted(observed_codes & current_sentinels)
                ),
                "observed_dict_missing_codes": "",
                "observed_overcapture_codes": "",
                "observed_undercapture_codes": "",
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "column",
                "wave",
                "base",
                "is_non_long",
                "status",
                "scope",
                "source_variable",
                "dta_file",
                "dict_missing_codes",
                "observed_current_sentinel_codes",
                "observed_dict_missing_codes",
                "observed_overcapture_codes",
                "observed_undercapture_codes",
            ]
        )
    return frame.sort_values(["wave", "base", "column"]).reset_index(drop=True)


@beartype
def _build_final_sentinel_map(*, validation: pd.DataFrame, config: Stage2Config) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for _, rec in validation.iterrows():
        dict_codes = _parse_codes(rec.get("dict_missing_codes", ""))
        observed_current = _parse_codes(rec.get("observed_current_sentinel_codes", ""))

        root = str(rec.get("base", "")).strip()
        root_override = list(config.root_overrides.get(root, ()))

        selected_codes: list[int] = []
        source = ""

        if root_override:
            selected_codes = sorted(set(root_override))
            source = "root_override"
        elif dict_codes:
            selected_codes = sorted(set(dict_codes))
            source = "label_terms"
        elif config.fallback_to_observed_when_no_label_match and observed_current:
            selected_codes = sorted(set(observed_current))
            source = "observed_current"

        if not selected_codes:
            continue

        rows.append(
            {
                "root_feature": root,
                "column": str(rec.get("column", "")).strip(),
                "wave": int(rec.get("wave")),
                "sentinel_codes_to_nan": _codes_to_string(selected_codes),
                "add_as_nan": "yes",
                "source": source,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "root_feature",
                "column",
                "wave",
                "sentinel_codes_to_nan",
                "add_as_nan",
                "source",
            ]
        )
    return frame.sort_values(["root_feature", "wave", "column"]).reset_index(drop=True)


@beartype
def _build_compact_sentinel_map(*, final_map: pd.DataFrame) -> pd.DataFrame:
    if final_map.empty:
        return pd.DataFrame(
            columns=["root_feature", "waves", "sentinel_codes_to_nan", "add_as_nan"]
        )

    rows: list[dict[str, object]] = []
    for root, group in final_map.groupby("root_feature"):
        waves = sorted(set(int(v) for v in group["wave"].tolist()))
        codes: set[int] = set()
        for token in group["sentinel_codes_to_nan"].tolist():
            codes.update(_parse_codes(token))

        rows.append(
            {
                "root_feature": root,
                "waves": ",".join(str(v) for v in waves),
                "sentinel_codes_to_nan": _codes_to_string(sorted(codes)),
                "add_as_nan": "yes",
            }
        )

    return pd.DataFrame(rows).sort_values("root_feature").reset_index(drop=True)


@beartype
def _apply_sentinel_nan_replacement(
    *,
    data: pd.DataFrame,
    final_map: pd.DataFrame,
    apply_nan_replacement: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    updated = data.copy()
    rows: list[dict[str, object]] = []

    if final_map.empty:
        return updated, pd.DataFrame(
            columns=[
                "column",
                "root_feature",
                "wave",
                "sentinel_codes_to_nan",
                "replaced_count",
                "replacement_applied",
            ]
        )

    for _, rec in final_map.iterrows():
        column = str(rec["column"])
        root = str(rec["root_feature"])
        wave = int(rec["wave"])
        codes = set(_parse_codes(rec["sentinel_codes_to_nan"]))

        replaced_count = 0
        if apply_nan_replacement and column in updated.columns and codes:
            numeric = pd.to_numeric(updated[column], errors="coerce")
            mask = numeric.isin(codes)
            replaced_count = int(mask.sum())
            if replaced_count > 0:
                updated[column] = updated[column].mask(mask)

        rows.append(
            {
                "column": column,
                "root_feature": root,
                "wave": wave,
                "sentinel_codes_to_nan": _codes_to_string(sorted(codes)),
                "replaced_count": replaced_count,
                "replacement_applied": "yes" if apply_nan_replacement else "no",
            }
        )

    replacement = pd.DataFrame(rows).sort_values(
        ["root_feature", "wave", "column"]
    ).reset_index(drop=True)
    return updated, replacement


@beartype
def _parse_wide_column(column: str) -> tuple[str, int, bool] | None:
    if column.startswith("non_long__w"):
        match = re.match(r"^non_long__w([1-7])__(.+)$", column)
        if not match:
            return None
        return match.group(2), int(match.group(1)), True

    match = re.match(r"^(.+)_w([1-7])$", column)
    if not match:
        return None
    return match.group(1), int(match.group(2)), False


def _to_int_codes(series: pd.Series) -> set[int]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    out: set[int] = set()
    for value in numeric.unique():
        as_float = float(value)
        if as_float.is_integer():
            out.add(int(as_float))
    return out


@beartype
def _extract_missing_like_codes_from_series(
    *,
    series: pd.Series,
    missing_terms: tuple[str, ...],
) -> list[int]:
    if not missing_terms:
        return []

    extracted: set[int] = set()
    for raw in series.dropna().astype(str).unique():
        label = raw.strip()
        if not label:
            continue
        if not _is_missing_like_label(label=label, missing_terms=missing_terms):
            continue

        direct_code = _coerce_label_code(label)
        if direct_code is not None:
            extracted.add(direct_code)
            continue

        numeric_token = re.search(r"-?\d+", label)
        if numeric_token is None:
            continue
        parsed = _coerce_label_code(numeric_token.group(0))
        if parsed is not None:
            extracted.add(parsed)

    return sorted(extracted)


@beartype
def _coerce_label_code(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(as_float):
        return None
    if not as_float.is_integer():
        return None
    return int(as_float)


@beartype
def _is_missing_like_label(*, label: str, missing_terms: tuple[str, ...]) -> bool:
    low = label.strip().lower()
    return any(term in low for term in missing_terms)


@beartype
def _codes_to_string(codes: list[int]) -> str:
    if not codes:
        return ""
    return ",".join(str(code) for code in sorted(set(codes)))


@beartype
def _parse_codes(value: object) -> list[int]:
    if pd.isna(value):
        return []
    token = str(value).strip()
    if not token:
        return []

    out: list[int] = []
    for part in token.split(","):
        piece = part.strip()
        if not piece:
            continue
        out.append(int(float(piece)))
    return sorted(set(out))


@beartype
def _as_mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise InputValidationError(f"`{context}` must be a mapping.")
    return value


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
        out.append(item.strip().lower())
    return out


@cache
def _load_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        raise InputValidationError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise InputValidationError(f"Config root must be a mapping: {path}")
    return raw
