from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pandas as pd
import yaml
from beartype import beartype

from ldt.utils.errors import InputValidationError

_STAGE_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _STAGE_DIR / "policy.yaml"

_NON_LONG_RE = re.compile(r"^non_long__w(?P<wave>\d+)__(?P<root>.+)$")
_LONG_RE = re.compile(r"^(?P<root>.+)_w(?P<wave>\d+)$")
_ALLOWED_ENCODINGS = {
    "binary_to_01",
    "ordinal",
    "nominal_one_hot",
    "continuous_numeric_keep",
}


@beartype
@dataclass(frozen=True)
class RootEncodingPolicy:
    """One root-level encoding policy."""

    root_feature: str
    encoding: str
    binary_value_map: dict[int, int]
    ordinal_mode: str
    ordinal_ordered_values: tuple[int, ...]
    nominal_one_hot_values: tuple[int, ...]
    non_longitudinal: bool


@beartype
@dataclass(frozen=True)
class Stage5Config:
    """Parsed configuration for stage 5 encoding policy."""

    target_column: str
    keep_columns: tuple[str, ...]
    fail_on_unconfigured_predictor: bool
    drop_constant_predictors: bool
    nominal_drop_source_columns: bool
    nominal_keep_missing_as_nan: bool
    nominal_swap_non_longitudinal_name: bool
    nominal_infer_single_wave_as_non_longitudinal: bool
    nominal_output_name_template_longitudinal: str
    nominal_output_name_template_non_longitudinal: str
    root_policies: dict[str, RootEncodingPolicy]


@beartype
@dataclass(frozen=True)
class Stage5Summary:
    """Compact stage 5 summary metrics."""

    input_columns: int
    output_columns: int
    configured_root_policies: int
    predictor_features_evaluated: int
    unconfigured_predictor_features: int
    binary_features_transformed: int
    ordinal_features_transformed: int
    continuous_features_kept: int
    nominal_source_features_expanded: int
    nominal_created_columns: int
    nominal_single_wave_reclassified_columns: int
    source_columns_dropped: int
    constant_predictor_features_detected: int
    constant_predictor_features_dropped: int


@beartype
@dataclass(frozen=True)
class Stage5Result:
    """Output payload for stage 5."""

    data: pd.DataFrame
    summary: Stage5Summary
    tables: dict[str, pd.DataFrame]


@beartype
@dataclass(frozen=True)
class _ParsedFeature:
    column: str
    root_feature: str
    wave: int | None
    group_kind: str


@beartype
def resolve_stage_5_config(config_path: Path | None = None) -> Stage5Config:
    """Load and validate stage 5 YAML configuration."""

    path = (config_path or _DEFAULT_CONFIG).expanduser()
    raw = _load_yaml(path)

    target_column = _as_string(raw.get("target_column"), "target_column")
    keep_columns = _as_string_list(
        raw.get("identifier_columns_keep", []),
        "identifier_columns_keep",
    )

    strict_policy = _as_mapping(raw.get("strict_policy"), "strict_policy")
    fail_on_unconfigured_predictor = bool(
        strict_policy.get("fail_on_unconfigured_predictor", True)
    )
    post_cleanup_cfg = _as_mapping(
        raw.get("post_encoding_cleanup", {}),
        "post_encoding_cleanup",
    )
    drop_constant_predictors = bool(
        post_cleanup_cfg.get("drop_constant_predictors", True)
    )

    nominal_cfg = _as_mapping(raw.get("nominal_one_hot"), "nominal_one_hot")
    nominal_drop_source_columns = bool(
        nominal_cfg.get("drop_source_columns", True)
    )
    nominal_keep_missing_as_nan = bool(
        nominal_cfg.get("keep_missing_as_nan", True)
    )
    nominal_swap_non_longitudinal_name = bool(
        nominal_cfg.get("swap_non_longitudinal_name", True)
    )
    nominal_infer_single_wave_as_non_longitudinal = bool(
        nominal_cfg.get("infer_single_wave_as_non_longitudinal", True)
    )
    nominal_output_name_template_longitudinal = _as_string(
        nominal_cfg.get(
            "output_name_template_longitudinal",
            "{root}__v{code}_w{wave}",
        ),
        "nominal_one_hot.output_name_template_longitudinal",
    )
    nominal_output_name_template_non_longitudinal = _as_string(
        nominal_cfg.get(
            "output_name_template_non_longitudinal",
            "non_long__w{wave}__{root}__v{code}",
        ),
        "nominal_one_hot.output_name_template_non_longitudinal",
    )
    for token in ("{root}", "{code}", "{wave}"):
        if token not in nominal_output_name_template_longitudinal:
            raise InputValidationError(
                "Invalid `nominal_one_hot.output_name_template_longitudinal`. "
                "Expected tokens: {root}, {code}, {wave}."
            )
        if token not in nominal_output_name_template_non_longitudinal:
            raise InputValidationError(
                "Invalid `nominal_one_hot.output_name_template_non_longitudinal`. "
                "Expected tokens: {root}, {code}, {wave}."
            )

    root_policies_raw = _as_mapping(raw.get("root_policies"), "root_policies")
    root_policies: dict[str, RootEncodingPolicy] = {}
    for root_feature, item in root_policies_raw.items():
        if not isinstance(root_feature, str) or not root_feature.strip():
            raise InputValidationError(
                "`root_policies` keys must be non-empty root feature strings."
            )
        root = root_feature.strip()
        policy_raw = _as_mapping(item, f"root_policies.{root}")
        encoding = _as_string(policy_raw.get("encoding"), f"root_policies.{root}.encoding")
        if encoding not in _ALLOWED_ENCODINGS:
            raise InputValidationError(
                f"Unsupported encoding for `{root}`: {encoding}. "
                f"Allowed: {sorted(_ALLOWED_ENCODINGS)}"
            )

        binary_value_map: dict[int, int] = {}
        ordinal_mode = "identity"
        ordinal_ordered_values: tuple[int, ...] = ()
        nominal_one_hot_values: tuple[int, ...] = ()
        non_longitudinal = bool(policy_raw.get("non_longitudinal", False))

        if encoding == "binary_to_01":
            binary_value_map = _as_int_to_binary_map(
                policy_raw.get("value_map"),
                f"root_policies.{root}.value_map",
            )
            if not binary_value_map:
                raise InputValidationError(
                    f"`root_policies.{root}.value_map` must define at least one mapping."
                )

        if encoding == "ordinal":
            ordinal_mode = _as_string(
                policy_raw.get("mode", "identity"),
                f"root_policies.{root}.mode",
            ).lower()
            if ordinal_mode not in {"identity", "rank"}:
                raise InputValidationError(
                    f"`root_policies.{root}.mode` must be `identity` or `rank`."
                )
            if ordinal_mode == "rank":
                ordered = _as_int_list(
                    policy_raw.get("ordered_values", []),
                    f"root_policies.{root}.ordered_values",
                )
                if not ordered:
                    raise InputValidationError(
                        f"`root_policies.{root}.ordered_values` cannot be empty "
                        "when mode=`rank`."
                    )
                ordinal_ordered_values = tuple(ordered)

        if encoding == "nominal_one_hot":
            one_hot_values = _as_int_list(
                policy_raw.get("one_hot_values", []),
                f"root_policies.{root}.one_hot_values",
            )
            if not one_hot_values:
                raise InputValidationError(
                    f"`root_policies.{root}.one_hot_values` cannot be empty."
                )
            nominal_one_hot_values = tuple(one_hot_values)
        elif non_longitudinal:
            raise InputValidationError(
                f"`root_policies.{root}.non_longitudinal` is only valid when "
                "encoding is `nominal_one_hot`."
            )

        root_policies[root] = RootEncodingPolicy(
            root_feature=root,
            encoding=encoding,
            binary_value_map=binary_value_map,
            ordinal_mode=ordinal_mode,
            ordinal_ordered_values=ordinal_ordered_values,
            nominal_one_hot_values=nominal_one_hot_values,
            non_longitudinal=non_longitudinal,
        )

    return Stage5Config(
        target_column=target_column,
        keep_columns=tuple(keep_columns),
        fail_on_unconfigured_predictor=fail_on_unconfigured_predictor,
        drop_constant_predictors=drop_constant_predictors,
        nominal_drop_source_columns=nominal_drop_source_columns,
        nominal_keep_missing_as_nan=nominal_keep_missing_as_nan,
        nominal_swap_non_longitudinal_name=nominal_swap_non_longitudinal_name,
        nominal_infer_single_wave_as_non_longitudinal=nominal_infer_single_wave_as_non_longitudinal,
        nominal_output_name_template_longitudinal=nominal_output_name_template_longitudinal,
        nominal_output_name_template_non_longitudinal=nominal_output_name_template_non_longitudinal,
        root_policies=root_policies,
    )


@beartype
def apply_stage_5(*, data: pd.DataFrame, config: Stage5Config) -> Stage5Result:
    """Apply stage 5 root-level encoding policy."""

    if config.target_column not in data.columns:
        raise InputValidationError(
            f"Stage 5 target column not found: {config.target_column}"
        )

    input_columns = int(data.shape[1])
    updated = data.copy()

    predictors = _collect_predictor_features(data=updated, config=config)

    assignment_rows: list[dict[str, object]] = []
    unconfigured_rows: list[dict[str, object]] = []
    binary_rows: list[dict[str, object]] = []
    ordinal_rows: list[dict[str, object]] = []
    nominal_rows: list[dict[str, object]] = []
    continuous_rows: list[dict[str, object]] = []
    source_to_drop: set[str] = set()
    created_nominal_columns: set[str] = set()
    pending_nominal_columns: dict[str, pd.Series] = {}

    for parsed in predictors:
        policy = config.root_policies.get(parsed.root_feature)
        if policy is None:
            assignment_rows.append(
                {
                    "feature": parsed.column,
                    "root_feature": parsed.root_feature,
                    "wave": parsed.wave if parsed.wave is not None else "",
                    "group_kind": parsed.group_kind,
                    "encoding": "",
                    "status": "unconfigured_root",
                }
            )
            unconfigured_rows.append(
                {
                    "feature": parsed.column,
                    "root_feature": parsed.root_feature,
                    "wave": parsed.wave if parsed.wave is not None else "",
                    "group_kind": parsed.group_kind,
                }
            )
            continue

        assignment_rows.append(
            {
                "feature": parsed.column,
                "root_feature": parsed.root_feature,
                "wave": parsed.wave if parsed.wave is not None else "",
                "group_kind": parsed.group_kind,
                "encoding": policy.encoding,
                "status": "configured",
            }
        )

        if policy.encoding == "continuous_numeric_keep":
            updated[parsed.column] = pd.to_numeric(updated[parsed.column], errors="coerce")
            continuous_rows.append(
                {
                    "feature": parsed.column,
                    "root_feature": parsed.root_feature,
                    "wave": parsed.wave if parsed.wave is not None else "",
                    "action": "keep_numeric",
                }
            )
            continue

        if policy.encoding == "binary_to_01":
            source = pd.to_numeric(updated[parsed.column], errors="coerce")
            mapped = source.map(policy.binary_value_map)
            unmatched_mask = source.notna() & mapped.isna()
            updated[parsed.column] = mapped.astype("float64")
            binary_rows.append(
                {
                    "feature": parsed.column,
                    "root_feature": parsed.root_feature,
                    "wave": parsed.wave if parsed.wave is not None else "",
                    "value_map": _map_to_string(policy.binary_value_map),
                    "non_missing_before": int(source.notna().sum()),
                    "unmatched_non_missing": int(unmatched_mask.sum()),
                }
            )
            continue

        if policy.encoding == "ordinal":
            source = pd.to_numeric(updated[parsed.column], errors="coerce")
            if policy.ordinal_mode == "identity":
                updated[parsed.column] = source
                unmatched_count = 0
                mapping_used = "identity_keep_numeric"
            else:
                ordinal_map = {
                    int(value): index + 1
                    for index, value in enumerate(policy.ordinal_ordered_values)
                }
                mapped = source.map(ordinal_map)
                unmatched_mask = source.notna() & mapped.isna()
                updated[parsed.column] = mapped.astype("float64")
                unmatched_count = int(unmatched_mask.sum())
                mapping_used = _map_to_string(ordinal_map)

            ordinal_rows.append(
                {
                    "feature": parsed.column,
                    "root_feature": parsed.root_feature,
                    "wave": parsed.wave if parsed.wave is not None else "",
                    "mode": policy.ordinal_mode,
                    "ordered_values": _codes_to_string(policy.ordinal_ordered_values),
                    "mapping_used": mapping_used,
                    "non_missing_before": int(source.notna().sum()),
                    "unmatched_non_missing": unmatched_count,
                }
            )
            continue

        if policy.encoding == "nominal_one_hot":
            source = pd.to_numeric(updated[parsed.column], errors="coerce")
            use_non_long_template = (
                policy.non_longitudinal and config.nominal_swap_non_longitudinal_name
            )
            output_template = (
                config.nominal_output_name_template_non_longitudinal
                if use_non_long_template
                else config.nominal_output_name_template_longitudinal
            )
            for category in policy.nominal_one_hot_values:
                created_name = _render_one_hot_name(
                    template=output_template,
                    root_feature=parsed.root_feature,
                    wave=parsed.wave,
                    code=category,
                )
                if created_name in updated.columns or created_name in pending_nominal_columns:
                    raise InputValidationError(
                        f"Nominal one-hot output column already exists: {created_name}"
                    )

                encoded = (source == float(category)).astype("float64")
                if config.nominal_keep_missing_as_nan:
                    encoded[source.isna()] = math.nan

                pending_nominal_columns[created_name] = encoded
                created_nominal_columns.add(created_name)

                nominal_rows.append(
                    {
                        "source_feature": parsed.column,
                        "root_feature": parsed.root_feature,
                        "wave": parsed.wave if parsed.wave is not None else "",
                        "category_code": int(category),
                        "created_feature": created_name,
                        "output_template_used": output_template,
                        "positive_count": int((encoded == 1.0).sum()),
                        "missing_count": int(encoded.isna().sum()),
                    }
                )

            if config.nominal_drop_source_columns:
                source_to_drop.add(parsed.column)
            continue

        raise InputValidationError(
            f"Unhandled encoding for `{parsed.root_feature}`: {policy.encoding}"
        )

    if config.fail_on_unconfigured_predictor and unconfigured_rows:
        sample = ", ".join(
            sorted({str(rec["root_feature"]) for rec in unconfigured_rows})[:12]
        )
        raise InputValidationError(
            "Stage 5 found predictor roots without policy. "
            f"Add them under `root_policies`. Roots (sample): {sample}"
        )

    if pending_nominal_columns:
        nominal_frame = pd.DataFrame(pending_nominal_columns, index=updated.index)
        updated = pd.concat([updated, nominal_frame], axis=1)

    dropped_source_columns = sorted(
        column for column in source_to_drop if column in updated.columns
    )
    if dropped_source_columns:
        updated = updated.drop(columns=dropped_source_columns, errors="ignore")

    constant_profile = _build_constant_predictor_profile(data=updated, config=config)
    constant_detected_columns = sorted(
        constant_profile.loc[constant_profile["is_constant_non_na"] == "yes", "feature"].tolist()
    )
    dropped_constant_columns: list[str] = []
    if config.drop_constant_predictors:
        dropped_constant_columns = [
            column for column in constant_detected_columns if column in updated.columns
        ]
        if dropped_constant_columns:
            updated = updated.drop(columns=dropped_constant_columns, errors="ignore")

    updated, nominal_reclassified_rows = _reclassify_single_wave_nominal_columns(
        data=updated,
        nominal_rows=nominal_rows,
        config=config,
    )
    nominal_reclass_map = {
        str(row["old_feature"]): str(row["new_feature"])
        for row in nominal_reclassified_rows
    }
    for row in nominal_rows:
        original_name = str(row.get("created_feature", ""))
        row["created_feature_final"] = nominal_reclass_map.get(
            original_name, original_name
        )

    root_policy_table = _build_root_policy_table(config.root_policies)
    assignment_table = _ensure_table(
        assignment_rows,
        columns=["feature", "root_feature", "wave", "group_kind", "encoding", "status"],
        sort_by=["status", "root_feature", "feature"],
    )
    unconfigured_table = _ensure_table(
        unconfigured_rows,
        columns=["feature", "root_feature", "wave", "group_kind"],
        sort_by=["root_feature", "feature"],
    )
    binary_table = _ensure_table(
        binary_rows,
        columns=[
            "feature",
            "root_feature",
            "wave",
            "value_map",
            "non_missing_before",
            "unmatched_non_missing",
        ],
        sort_by=["root_feature", "feature"],
    )
    ordinal_table = _ensure_table(
        ordinal_rows,
        columns=[
            "feature",
            "root_feature",
            "wave",
            "mode",
            "ordered_values",
            "mapping_used",
            "non_missing_before",
            "unmatched_non_missing",
        ],
        sort_by=["root_feature", "feature"],
    )
    continuous_table = _ensure_table(
        continuous_rows,
        columns=["feature", "root_feature", "wave", "action"],
        sort_by=["root_feature", "feature"],
    )
    nominal_table = _ensure_table(
        nominal_rows,
        columns=[
            "source_feature",
            "root_feature",
            "wave",
            "category_code",
            "created_feature",
            "created_feature_final",
            "output_template_used",
            "positive_count",
            "missing_count",
        ],
        sort_by=["root_feature", "source_feature", "category_code"],
    )
    nominal_reclassified_table = _ensure_table(
        nominal_reclassified_rows,
        columns=[
            "root_feature",
            "category_code",
            "wave",
            "old_feature",
            "new_feature",
            "reason",
        ],
        sort_by=["root_feature", "category_code", "wave"],
    )
    dropped_sources_table = _ensure_table(
        [{"feature": column, "reason": "stage5_nominal_one_hot_source_dropped"} for column in dropped_source_columns],
        columns=["feature", "reason"],
        sort_by=["feature"],
    )
    dropped_constant_table = _ensure_table(
        [
            {"feature": column, "reason": "stage5_post_encoding_constant_predictor"}
            for column in dropped_constant_columns
        ],
        columns=["feature", "reason"],
        sort_by=["feature"],
    )

    encoding_summary_counts = _build_encoding_summary_counts(assignment_rows)

    summary = Stage5Summary(
        input_columns=input_columns,
        output_columns=int(updated.shape[1]),
        configured_root_policies=len(config.root_policies),
        predictor_features_evaluated=len(predictors),
        unconfigured_predictor_features=len(unconfigured_rows),
        binary_features_transformed=len(binary_rows),
        ordinal_features_transformed=len(ordinal_rows),
        continuous_features_kept=len(continuous_rows),
        nominal_source_features_expanded=len({row["source_feature"] for row in nominal_rows}),
        nominal_created_columns=len(created_nominal_columns),
        nominal_single_wave_reclassified_columns=len(nominal_reclassified_rows),
        source_columns_dropped=len(dropped_source_columns),
        constant_predictor_features_detected=len(constant_detected_columns),
        constant_predictor_features_dropped=len(dropped_constant_columns),
    )

    tables: dict[str, pd.DataFrame] = {
        "stage5_root_policy_table": root_policy_table,
        "stage5_feature_policy_assignment": assignment_table,
        "stage5_unconfigured_predictor_features": unconfigured_table,
        "stage5_binary_mapping_summary": binary_table,
        "stage5_ordinal_mapping_summary": ordinal_table,
        "stage5_continuous_keep_summary": continuous_table,
        "stage5_nominal_one_hot_created_columns": nominal_table,
        "stage5_nominal_single_wave_reclassified_columns": nominal_reclassified_table,
        "stage5_nominal_one_hot_dropped_sources": dropped_sources_table,
        "stage5_constant_predictor_profile": constant_profile,
        "stage5_constant_predictor_dropped_features": dropped_constant_table,
        "stage5_encoding_summary_counts": encoding_summary_counts,
    }

    return Stage5Result(data=updated, summary=summary, tables=tables)


@beartype
def _build_encoding_summary_counts(assignment_rows: list[dict[str, object]]) -> pd.DataFrame:
    configured = [
        str(row["encoding"])
        for row in assignment_rows
        if str(row.get("status", "")) == "configured"
    ]
    if not configured:
        return pd.DataFrame(columns=["encoding", "feature_count"])
    frame = pd.DataFrame({"encoding": configured})
    return (
        frame.groupby("encoding", as_index=False)
        .size()
        .rename(columns={"size": "feature_count"})
        .sort_values(["encoding"])
        .reset_index(drop=True)
    )


@beartype
def _reclassify_single_wave_nominal_columns(
    *,
    data: pd.DataFrame,
    nominal_rows: list[dict[str, object]],
    config: Stage5Config,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    if not config.nominal_infer_single_wave_as_non_longitudinal:
        return data, []

    families: dict[tuple[str, int], list[dict[str, object]]] = {}
    longitudinal_template = config.nominal_output_name_template_longitudinal

    for row in nominal_rows:
        if str(row.get("output_template_used", "")) != longitudinal_template:
            continue

        created_feature = str(row.get("created_feature", ""))
        if created_feature not in data.columns:
            continue

        root_feature = str(row.get("root_feature", ""))
        if not root_feature:
            continue

        category_raw = row.get("category_code")
        wave_raw = row.get("wave")
        if category_raw in (None, "") or wave_raw in (None, ""):
            continue

        try:
            category_code = int(category_raw)
            wave = int(wave_raw)
        except (TypeError, ValueError):
            continue

        families.setdefault((root_feature, category_code), []).append(
            {"created_feature": created_feature, "wave": wave}
        )

    rename_map: dict[str, str] = {}
    reclassified_rows: list[dict[str, object]] = []

    for (root_feature, category_code), items in sorted(families.items()):
        observed_waves = sorted({int(item["wave"]) for item in items})
        if len(observed_waves) != 1:
            continue

        wave = int(observed_waves[0])
        source_feature = str(items[0]["created_feature"])
        target_feature = _render_one_hot_name(
            template=config.nominal_output_name_template_non_longitudinal,
            root_feature=root_feature,
            wave=wave,
            code=category_code,
        )

        if target_feature == source_feature:
            continue
        if target_feature in rename_map.values():
            raise InputValidationError(
                f"Nominal single-wave reclass target collision: {target_feature}"
            )
        if target_feature in data.columns and target_feature not in rename_map:
            raise InputValidationError(
                f"Nominal single-wave reclass target already exists: {target_feature}"
            )

        rename_map[source_feature] = target_feature
        reclassified_rows.append(
            {
                "root_feature": root_feature,
                "category_code": int(category_code),
                "wave": wave,
                "old_feature": source_feature,
                "new_feature": target_feature,
                "reason": "single_wave_survivor_after_one_hot",
            }
        )

    if rename_map:
        data = data.rename(columns=rename_map)

    return data, reclassified_rows


@beartype
def _build_root_policy_table(root_policies: dict[str, RootEncodingPolicy]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for root_feature in sorted(root_policies):
        policy = root_policies[root_feature]
        rows.append(
            {
                "root_feature": root_feature,
                "encoding": policy.encoding,
                "binary_value_map": _map_to_string(policy.binary_value_map),
                "ordinal_mode": policy.ordinal_mode,
                "ordinal_ordered_values": _codes_to_string(policy.ordinal_ordered_values),
                "nominal_one_hot_values": _codes_to_string(policy.nominal_one_hot_values),
                "non_longitudinal": "yes" if policy.non_longitudinal else "no",
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "root_feature",
            "encoding",
            "non_longitudinal",
            "binary_value_map",
            "ordinal_mode",
            "ordinal_ordered_values",
            "nominal_one_hot_values",
        ],
    )


@beartype
def _collect_predictor_features(*, data: pd.DataFrame, config: Stage5Config) -> list[_ParsedFeature]:
    parsed: list[_ParsedFeature] = []
    protected = set(config.keep_columns) | {config.target_column}
    for column in data.columns:
        if column in protected:
            continue
        parsed.append(_parse_feature(column))
    return parsed


@beartype
def _parse_feature(column: str) -> _ParsedFeature:
    non_long_match = _NON_LONG_RE.match(column)
    if non_long_match is not None:
        return _ParsedFeature(
            column=column,
            root_feature=str(non_long_match.group("root")),
            wave=int(non_long_match.group("wave")),
            group_kind="non_long",
        )

    long_match = _LONG_RE.match(column)
    if long_match is not None:
        return _ParsedFeature(
            column=column,
            root_feature=str(long_match.group("root")),
            wave=int(long_match.group("wave")),
            group_kind="longitudinal",
        )

    return _ParsedFeature(
        column=column,
        root_feature=column,
        wave=None,
        group_kind="single_column",
    )


@beartype
def _render_one_hot_name(
    *,
    template: str,
    root_feature: str,
    wave: int | None,
    code: int,
) -> str:
    wave_token = str(int(wave)) if wave is not None else "na"
    code_token = str(int(code))
    return template.format(root=root_feature, code=code_token, wave=wave_token)


@beartype
def _codes_to_string(values: tuple[int, ...] | list[int]) -> str:
    if not values:
        return ""
    return ",".join(str(int(value)) for value in values)


@beartype
def _map_to_string(mapping: dict[int, int]) -> str:
    if not mapping:
        return ""
    chunks = [f"{int(key)}->{int(mapping[key])}" for key in sorted(mapping)]
    return ";".join(chunks)


@beartype
def _ensure_table(
    rows: list[dict[str, object]],
    *,
    columns: list[str],
    sort_by: list[str],
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    return frame[columns].sort_values(sort_by).reset_index(drop=True)


@beartype
def _build_constant_predictor_profile(*, data: pd.DataFrame, config: Stage5Config) -> pd.DataFrame:
    protected = set(config.keep_columns) | {config.target_column}
    rows: list[dict[str, object]] = []
    for column in data.columns:
        if column in protected:
            continue
        series = data[column]
        valid = series.dropna()
        nunique_non_na = int(valid.nunique(dropna=True))
        rows.append(
            {
                "feature": column,
                "dtype": str(series.dtype),
                "non_missing_count": int(valid.shape[0]),
                "missing_count": int(series.isna().sum()),
                "nunique_non_na": nunique_non_na,
                "is_constant_non_na": "yes" if nunique_non_na <= 1 else "no",
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature",
                "dtype",
                "non_missing_count",
                "missing_count",
                "nunique_non_na",
                "is_constant_non_na",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        ["is_constant_non_na", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)


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
    if value in (None, ""):
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
def _as_int_list(value: object, context: str) -> list[int]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise InputValidationError(f"`{context}` must be a list of integers.")

    out: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise InputValidationError(
                f"Invalid entry at `{context}[{index}]`. Expected integer."
            )
        as_int = int(item)
        if float(item) != float(as_int):
            raise InputValidationError(
                f"Invalid entry at `{context}[{index}]`. Expected integer-like value."
            )
        out.append(as_int)
    return out


@beartype
def _as_int_to_binary_map(value: object, context: str) -> dict[int, int]:
    if not isinstance(value, dict):
        raise InputValidationError(f"`{context}` must be a mapping of int->(0|1).")

    out: dict[int, int] = {}
    for key, raw_target in value.items():
        try:
            source = int(str(key).strip())
        except Exception as exc:  # pragma: no cover - defensive
            raise InputValidationError(
                f"Invalid key in `{context}`: {key}. Expected integer-like key."
            ) from exc

        if isinstance(raw_target, bool) or not isinstance(raw_target, (int, float)):
            raise InputValidationError(
                f"Invalid value in `{context}` for key `{key}`. Expected 0 or 1."
            )
        mapped = int(raw_target)
        if float(raw_target) != float(mapped) or mapped not in {0, 1}:
            raise InputValidationError(
                f"Invalid value in `{context}` for key `{key}`. Expected 0 or 1."
            )
        out[source] = mapped
    return out


@cache
def _load_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        raise InputValidationError(f"Stage-5 config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise InputValidationError("Stage-5 config root must be a mapping.")
    return raw
