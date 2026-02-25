from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from beartype import beartype

from ldt.utils.errors import InputValidationError

from ..stage_1_wave_paths import DatasetSpec, WaveDatasetConfig

_STAGE_DIR = Path(__file__).resolve().parent
_SUBJECT_KEYS_CONFIG = _STAGE_DIR / "subject_keys.yaml"


@beartype
@dataclass(frozen=True)
class SubjectKeyConfig:
    """Configured key columns used to build one unique subject per wave.

    Attributes:
        key_columns (tuple[str, ...]): Column names for key columns.
        parent_key_columns (tuple[str, ...]): Column names for parent key columns.
        link_key_columns (tuple[str, ...]): Column names for link key columns.
        output_fixed_columns (tuple[str, ...]): Column names for output fixed columns.
    """

    key_columns: tuple[str, ...]
    parent_key_columns: tuple[str, ...]
    link_key_columns: tuple[str, ...]
    output_fixed_columns: tuple[str, ...]


@beartype
@dataclass(frozen=True)
class PreparedDataset:
    """Prepared dataset with standardised keys and prefixed non-key columns.

    Attributes:
        spec (DatasetSpec): Spec.
        data (pd.DataFrame): Data.
    """

    spec: DatasetSpec
    data: pd.DataFrame


@beartype
@dataclass(frozen=True)
class IdentifierSummary:
    """Summary of identifier quality checks for one source dataset.

    Attributes:
        alias (str): Alias.
        role (str): Role.
        key_scope (str): Key scope.
        rows_total (int): Rows total.
        rows_with_keys (int): Rows with keys.
        duplicate_key_rows (int): Duplicate key rows.
    """

    alias: str
    role: str
    key_scope: str
    rows_total: int
    rows_with_keys: int
    duplicate_key_rows: int


@beartype
@dataclass(frozen=True)
class MergeSummary:
    """Summary of key coverage while merging one dataset into an anchor.

    Attributes:
        alias (str): Alias.
        key_scope (str): Key scope.
        matched_rows (int): Matched rows.
        anchor_rows (int): Anchor rows.
    """

    alias: str
    key_scope: str
    matched_rows: int
    anchor_rows: int


@beartype
@dataclass(frozen=True)
class SubjectConceptualisationResult:
    """Stage-2 output with subject-level wave data and diagnostic summaries.

    Attributes:
        data (pd.DataFrame): Data.
        identifier_summaries (tuple[IdentifierSummary, ...]): Identifier summaries.
        merge_summaries (tuple[MergeSummary, ...]): Merge summaries.
    """

    data: pd.DataFrame
    identifier_summaries: tuple[IdentifierSummary, ...]
    merge_summaries: tuple[MergeSummary, ...]


@beartype
@cache
def subject_key_config() -> SubjectKeyConfig:
    """Load key-column configuration used by the subject conceptualisation stage.

    Returns:
        SubjectKeyConfig: Result object for this operation.
    """

    raw = _load_yaml_config(_SUBJECT_KEYS_CONFIG)
    keys = raw.get("keys")
    if not isinstance(keys, dict):
        raise InputValidationError("Stage 2 config must define `keys` mapping.")

    key_columns = _parse_string_list(
        value=keys.get("key_columns"), context="keys.key_columns"
    )
    parent_key_columns = _parse_string_list(
        value=keys.get("parent_key_columns"),
        context="keys.parent_key_columns",
    )
    link_key_columns = _parse_string_list(
        value=keys.get("link_key_columns"),
        context="keys.link_key_columns",
    )
    output_fixed_columns = _parse_string_list(
        value=keys.get("output_fixed_columns"),
        context="keys.output_fixed_columns",
    )

    _validate_unique_columns(columns=key_columns, context="keys.key_columns")
    _validate_unique_columns(
        columns=parent_key_columns,
        context="keys.parent_key_columns",
    )
    _validate_unique_columns(columns=link_key_columns, context="keys.link_key_columns")
    _validate_unique_columns(
        columns=output_fixed_columns,
        context="keys.output_fixed_columns",
    )

    if len(key_columns) != 2:
        raise InputValidationError("`keys.key_columns` must contain exactly 2 values.")
    if len(parent_key_columns) != 2:
        raise InputValidationError(
            "`keys.parent_key_columns` must contain exactly 2 values."
        )
    if len(link_key_columns) != 3:
        raise InputValidationError(
            "`keys.link_key_columns` must contain exactly 3 values."
        )

    key_set = set(key_columns)
    parent_key_set = set(parent_key_columns)
    link_key_set = set(link_key_columns)
    if not key_set.issubset(link_key_set):
        raise InputValidationError("`keys.link_key_columns` must include child keys.")
    if not parent_key_set.issubset(link_key_set):
        raise InputValidationError("`keys.link_key_columns` must include parent keys.")

    required_fixed = ("CHID", "wave", *key_columns)
    missing_fixed = [
        column for column in required_fixed if column not in output_fixed_columns
    ]
    if missing_fixed:
        raise InputValidationError(
            "`keys.output_fixed_columns` is missing required columns: "
            + ", ".join(missing_fixed)
        )

    return SubjectKeyConfig(
        key_columns=key_columns,
        parent_key_columns=parent_key_columns,
        link_key_columns=link_key_columns,
        output_fixed_columns=output_fixed_columns,
    )


@beartype
def build_subject_level_wave_dataset(
    *,
    wave: str,
    raw_dir: Path,
    wave_config: WaveDatasetConfig,
) -> SubjectConceptualisationResult:
    """Build one child-level dataframe per wave using configured identifiers.

    Args:
        wave (str): Wave identifier.
        raw_dir (Path): Filesystem location for raw dir.
        wave_config (WaveDatasetConfig): Wave config.

    Returns:
        SubjectConceptualisationResult: Result object for this operation.
    """

    prepared, identifier_summaries = _load_and_validate_datasets(
        raw_dir=raw_dir,
        wave_config=wave_config,
    )
    final, merge_summaries = _assemble_wave_output(
        wave=wave,
        prepared_datasets=prepared,
    )
    return SubjectConceptualisationResult(
        data=final,
        identifier_summaries=tuple(identifier_summaries),
        merge_summaries=tuple(merge_summaries),
    )


@beartype
def print_subject_summaries(*, result: SubjectConceptualisationResult) -> None:
    """Print stage-2 diagnostic summaries.

    Args:
        result (SubjectConceptualisationResult): Result object used by this workflow.
    """

    print("Stage 2 - Subject conceptualisation checks:")
    for summary in result.identifier_summaries:
        print(
            f"- {summary.alias} [{summary.role}] ({summary.key_scope}): "
            f"rows={summary.rows_with_keys}/{summary.rows_total}, "
            f"duplicates={summary.duplicate_key_rows}"
        )

    print("Stage 2 - Merge coverage:")
    for summary in result.merge_summaries:
        print(
            f"- {summary.alias} ({summary.key_scope}): "
            f"matched={summary.matched_rows}/{summary.anchor_rows}"
        )


@beartype
def _load_and_validate_datasets(
    *,
    raw_dir: Path,
    wave_config: WaveDatasetConfig,
) -> tuple[list[PreparedDataset], list[IdentifierSummary]]:
    """Load all configured datasets for one wave and standardise identifiers."""

    prepared: list[PreparedDataset] = []
    summaries: list[IdentifierSummary] = []
    for spec in wave_config.datasets:
        loaded, summary = _load_single_dataset(raw_dir=raw_dir, spec=spec)
        prepared.append(loaded)
        summaries.append(summary)
    return prepared, summaries


@beartype
def _load_single_dataset(
    *,
    raw_dir: Path,
    spec: DatasetSpec,
) -> tuple[PreparedDataset, IdentifierSummary]:
    """Load, key-standardise, and prefix one configured source dataset."""

    source_path = raw_dir / spec.file_name
    if not source_path.exists() or not source_path.is_file():
        raise InputValidationError(
            f"Missing required dataset file: {source_path.resolve()}"
        )

    data = pd.read_stata(source_path, convert_categoricals=False)
    rows_total = len(data)

    normalised = _normalise_mcsid_column(data=data, file_name=spec.file_name)
    _validate_required_identifier_columns(data=normalised, spec=spec)
    standardised = _add_standardised_keys(data=normalised, spec=spec)

    role_keys = _role_merge_keys(role=spec.role)
    keyed = standardised.dropna(subset=list(role_keys)).copy()
    rows_with_keys = len(keyed)
    if rows_with_keys == 0:
        raise InputValidationError(
            f"Dataset `{spec.file_name}` has no rows after key filtering on "
            f"`{'+'.join(role_keys)}`."
        )

    duplicate_key_rows = int(keyed.duplicated(list(role_keys)).sum())
    if duplicate_key_rows > 0:
        raise InputValidationError(
            f"Dataset `{spec.file_name}` violates unique identifier pattern "
            f"`{'+'.join(role_keys)}` with {duplicate_key_rows} duplicate row(s)."
        )

    keyed = keyed.sort_values(list(role_keys)).drop_duplicates(list(role_keys))
    prefixed = _prefix_feature_columns(
        data=keyed,
        dataset_alias=spec.alias,
        key_columns=_link_key_columns(),
    )

    summary = IdentifierSummary(
        alias=spec.alias,
        role=spec.role,
        key_scope="+".join(role_keys),
        rows_total=rows_total,
        rows_with_keys=rows_with_keys,
        duplicate_key_rows=duplicate_key_rows,
    )
    return PreparedDataset(spec=spec, data=prefixed.reset_index(drop=True)), summary


@beartype
def _normalise_mcsid_column(*, data: pd.DataFrame, file_name: str) -> pd.DataFrame:
    """Normalise any case variant of MCSID to a canonical uppercase key column."""

    matching = [column for column in data.columns if column.upper() == "MCSID"]
    if not matching:
        raise InputValidationError(
            f"Dataset `{file_name}` does not contain required identifier `MCSID`."
        )

    canonical = matching[0]
    family_key = _family_key_column()
    renamed = data.rename(columns={canonical: family_key})
    extras = [column for column in matching[1:] if column != canonical]
    if extras:
        renamed = renamed.drop(columns=extras)

    renamed[family_key] = renamed[family_key].astype(str).str.strip()
    renamed = renamed[renamed[family_key] != ""]
    return renamed


@beartype
def _validate_required_identifier_columns(
    *, data: pd.DataFrame, spec: DatasetSpec
) -> None:
    """Validate required identifier columns declared for one dataset."""

    missing = [
        identifier
        for identifier in spec.required_identifiers
        if _resolve_column_name(data=data, preferred_name=identifier) is None
    ]
    if missing:
        raise InputValidationError(
            f"Dataset `{spec.file_name}` is missing required identifier column(s): "
            + ", ".join(sorted(missing))
        )


@beartype
def _add_standardised_keys(*, data: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Create standardised CNUM/PNUM keys based on dataset configuration."""

    prepared = data.copy()
    child_key = _child_key_column()
    parent_key = _parent_key_column()

    if spec.cnum_source is not None:
        cnum_source = _resolve_column_name(
            data=prepared, preferred_name=spec.cnum_source
        )
        if cnum_source is None:
            raise InputValidationError(
                f"Dataset `{spec.file_name}` missing CNUM source `{spec.cnum_source}`."
            )
        prepared[child_key] = pd.to_numeric(
            prepared[cnum_source], errors="coerce"
        ).astype("Int64")

    if spec.pnum_source is not None:
        pnum_source = _resolve_column_name(
            data=prepared, preferred_name=spec.pnum_source
        )
        if pnum_source is None:
            raise InputValidationError(
                f"Dataset `{spec.file_name}` missing PNUM source `{spec.pnum_source}`."
            )
        prepared[parent_key] = pd.to_numeric(
            prepared[pnum_source], errors="coerce"
        ).astype("Int64")

    if spec.filter_to_valid_child_cnum:
        if child_key not in prepared.columns:
            raise InputValidationError(
                f"Dataset `{spec.file_name}` requested child filtering but has no `{child_key}`."
            )
        prepared = prepared[prepared[child_key].notna() & (prepared[child_key] > 0)]

    return prepared


@beartype
def _resolve_column_name(*, data: pd.DataFrame, preferred_name: str) -> str | None:
    """Resolve one column name case-insensitively."""

    upper = preferred_name.upper()
    for column in data.columns:
        if column.upper() == upper:
            return column
    return None


@beartype
def _role_merge_keys(*, role: str) -> tuple[str, ...]:
    """Resolve merge keys for one dataset role."""

    if role == "family":
        return (_family_key_column(),)
    if role == "child":
        return _key_columns()
    if role == "parent":
        return _parent_key_columns()
    return _link_key_columns()


@beartype
def _prefix_feature_columns(
    *,
    data: pd.DataFrame,
    dataset_alias: str,
    key_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Prefix all non-key feature columns with dataset alias."""

    key_set = set(key_columns)
    renamed = {
        column: f"{dataset_alias}__{column}"
        for column in data.columns
        if column not in key_set
    }
    return data.rename(columns=renamed)


@beartype
def _assemble_wave_output(
    *,
    wave: str,
    prepared_datasets: list[PreparedDataset],
) -> tuple[pd.DataFrame, list[MergeSummary]]:
    """Assemble final child-level wave output from role-specific datasets."""

    role_groups = _group_prepared_by_role(prepared_datasets=prepared_datasets)
    child_datasets = role_groups["child"]
    family_datasets = role_groups["family"]
    parent_datasets = role_groups["parent"]
    link_datasets = role_groups["link"]

    child_anchor = _build_child_anchor(
        child_datasets=child_datasets,
        link_datasets=link_datasets,
    )
    child_block, child_summaries = _merge_tables_with_anchor(
        anchor=child_anchor,
        datasets=child_datasets,
        merge_keys=_key_columns(),
        merge_validate="one_to_one",
    )
    child_plus_family, family_summaries = _merge_tables_with_anchor(
        anchor=child_block,
        datasets=family_datasets,
        merge_keys=(_family_key_column(),),
        merge_validate="many_to_one",
    )

    parent_wide = _build_parent_wide_block(
        parent_datasets=parent_datasets,
        link_datasets=link_datasets,
    )
    final = child_plus_family.merge(
        parent_wide,
        on=list(_key_columns()),
        how="left",
        validate="one_to_one",
    )
    final = _add_output_identifiers(final, wave_label=wave)
    _validate_child_level_uniqueness(final)
    _validate_no_row_explosion(anchor=child_anchor, final=final)
    return final, child_summaries + family_summaries


@beartype
def _group_prepared_by_role(
    *,
    prepared_datasets: list[PreparedDataset],
) -> dict[str, list[PreparedDataset]]:
    """Group prepared datasets by role."""

    grouped: dict[str, list[PreparedDataset]] = {
        "family": [],
        "child": [],
        "parent": [],
        "link": [],
    }
    for prepared in prepared_datasets:
        grouped[prepared.spec.role].append(prepared)
    return grouped


@beartype
def _build_child_anchor(
    *,
    child_datasets: list[PreparedDataset],
    link_datasets: list[PreparedDataset],
) -> pd.DataFrame:
    """Build child anchor from union of child/link key coverage."""

    child_keys = _key_columns()
    frames = [dataset.data[list(child_keys)] for dataset in child_datasets]
    frames.extend(dataset.data[list(child_keys)] for dataset in link_datasets)
    if not frames:
        raise InputValidationError("No child/link datasets available to build anchor.")

    anchor = pd.concat(frames, ignore_index=True)
    anchor = anchor.dropna(subset=list(child_keys)).drop_duplicates(list(child_keys))
    anchor = anchor.sort_values(list(child_keys)).reset_index(drop=True)
    return anchor


@beartype
def _merge_tables_with_anchor(
    *,
    anchor: pd.DataFrame,
    datasets: list[PreparedDataset],
    merge_keys: tuple[str, ...],
    merge_validate: str,
) -> tuple[pd.DataFrame, list[MergeSummary]]:
    """Merge prepared datasets into an anchor table and collect coverage stats."""

    merged = anchor.copy()
    summaries: list[MergeSummary] = []
    anchor_keys = merged[list(merge_keys)].drop_duplicates(list(merge_keys))

    for dataset in datasets:
        table = dataset.data
        table_keys = table[list(merge_keys)].drop_duplicates(list(merge_keys))
        matched_rows = int(
            anchor_keys.merge(table_keys, on=list(merge_keys), how="inner").shape[0]
        )
        summaries.append(
            MergeSummary(
                alias=dataset.spec.alias,
                key_scope="+".join(merge_keys),
                matched_rows=matched_rows,
                anchor_rows=len(anchor_keys),
            )
        )
        merged = merged.merge(
            table,
            on=list(merge_keys),
            how="left",
            validate=merge_validate,
        )

    return merged, summaries


@beartype
def _build_parent_wide_block(
    *,
    parent_datasets: list[PreparedDataset],
    link_datasets: list[PreparedDataset],
) -> pd.DataFrame:
    """Build child-level parent-slot features from parent/link datasets."""

    parent_anchor = _build_union_key_anchor(
        datasets=parent_datasets,
        key_columns=_parent_key_columns(),
    )
    parent_block, _ = _merge_tables_with_anchor(
        anchor=parent_anchor,
        datasets=parent_datasets,
        merge_keys=_parent_key_columns(),
        merge_validate="one_to_one",
    )

    link_anchor = _build_union_key_anchor(
        datasets=link_datasets,
        key_columns=_link_key_columns(),
    )
    link_block, _ = _merge_tables_with_anchor(
        anchor=link_anchor,
        datasets=link_datasets,
        merge_keys=_link_key_columns(),
        merge_validate="one_to_one",
    )

    link_plus_parent = link_block.merge(
        parent_block,
        on=list(_parent_key_columns()),
        how="left",
        validate="many_to_one",
    )
    return _pivot_parent_slots(link_plus_parent)


@beartype
def _build_union_key_anchor(
    *,
    datasets: list[PreparedDataset],
    key_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Build a sorted unique key anchor from one or more datasets."""

    if not datasets:
        return pd.DataFrame(columns=list(key_columns))

    keys = pd.concat(
        [dataset.data[list(key_columns)] for dataset in datasets],
        ignore_index=True,
    )
    keys = keys.dropna(subset=list(key_columns)).drop_duplicates(list(key_columns))
    keys = keys.sort_values(list(key_columns)).reset_index(drop=True)
    return keys


@beartype
def _pivot_parent_slots(data: pd.DataFrame) -> pd.DataFrame:
    """Pivot parent/link records into p1/p2/... child-level columns."""

    family_key = _family_key_column()
    child_key = _child_key_column()
    parent_key = _parent_key_column()

    keyed = data.copy()
    keyed["PARENT_SLOT"] = keyed[parent_key].map(_format_parent_slot)
    keyed = keyed.dropna(subset=["PARENT_SLOT"])

    keyed = (
        keyed.sort_values([family_key, child_key, "PARENT_SLOT"])
        .groupby([family_key, child_key, "PARENT_SLOT"], dropna=False, as_index=False)
        .first()
    )

    value_columns = [
        column
        for column in keyed.columns
        if column not in {family_key, child_key, parent_key, "PARENT_SLOT"}
    ]
    if not value_columns:
        child_keys = _key_columns()
        return keyed[list(child_keys)].drop_duplicates(list(child_keys))

    wide = keyed.set_index([family_key, child_key, "PARENT_SLOT"])[
        value_columns
    ].unstack("PARENT_SLOT")
    wide.columns = [
        f"{slot}__{feature}" for feature, slot in wide.columns.to_flat_index()
    ]
    return wide.reset_index()


@beartype
def _format_parent_slot(value: object) -> str | None:
    """Convert numeric parent index to stable slot labels (p1, p2, ...)."""

    if pd.isna(value):
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    return f"p{numeric}"


@beartype
def _add_output_identifiers(data: pd.DataFrame, *, wave_label: str) -> pd.DataFrame:
    """Add CHID/wave and enforce configured fixed-column ordering."""

    family_key = _family_key_column()
    child_key = _child_key_column()
    output_fixed = _output_fixed_columns()

    output = data.copy()
    output["CHID"] = (
        output[family_key].astype(str) + "_" + output[child_key].astype(str)
    )
    output["wave"] = wave_label

    missing_fixed = [column for column in output_fixed if column not in output.columns]
    if missing_fixed:
        raise InputValidationError(
            "Configured output fixed columns are missing in final dataframe: "
            + ", ".join(missing_fixed)
        )

    remaining = [column for column in output.columns if column not in output_fixed]
    return output[list(output_fixed) + remaining]


@beartype
def _validate_child_level_uniqueness(data: pd.DataFrame) -> None:
    """Validate uniqueness on the configured child key columns."""

    duplicate_rows = int(data.duplicated(list(_key_columns())).sum())
    if duplicate_rows > 0:
        key_columns = ", ".join(_key_columns())
        raise InputValidationError(
            "Final output is not unique on "
            f"`({key_columns})` with {duplicate_rows} duplicate row(s)."
        )


@beartype
def _validate_no_row_explosion(*, anchor: pd.DataFrame, final: pd.DataFrame) -> None:
    """Ensure merging did not increase child rows unexpectedly."""

    if len(anchor) != len(final):
        raise InputValidationError(
            "Unexpected row expansion detected: "
            f"anchor_rows={len(anchor)}, final_rows={len(final)}."
        )


@beartype
def _key_columns() -> tuple[str, ...]:
    return subject_key_config().key_columns


@beartype
def _parent_key_columns() -> tuple[str, ...]:
    return subject_key_config().parent_key_columns


@beartype
def _link_key_columns() -> tuple[str, ...]:
    return subject_key_config().link_key_columns


@beartype
def _output_fixed_columns() -> tuple[str, ...]:
    return subject_key_config().output_fixed_columns


@beartype
def _family_key_column() -> str:
    return _key_columns()[0]


@beartype
def _child_key_column() -> str:
    return _key_columns()[1]


@beartype
def _parent_key_column() -> str:
    return _parent_key_columns()[1]


@beartype
@cache
def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load one YAML config file from disk and validate mapping root."""

    if not config_path.exists() or not config_path.is_file():
        raise InputValidationError(f"Missing config file: {config_path.resolve()}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise InputValidationError(
            f"Config file `{config_path.name}` must contain a YAML mapping."
        )
    return loaded


@beartype
def _parse_string_list(*, value: Any, context: str) -> tuple[str, ...]:
    """Parse one list of non-empty string values."""

    if not isinstance(value, list):
        raise InputValidationError(f"Expected list for `{context}`.")

    parsed: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise InputValidationError(f"Invalid string entry in `{context}`.")
        parsed.append(item.strip())
    return tuple(parsed)


@beartype
def _validate_unique_columns(*, columns: tuple[str, ...], context: str) -> None:
    """Validate no duplicate column names in configured tuples."""

    if len(columns) != len(set(columns)):
        raise InputValidationError(
            f"`{context}` must not contain duplicate column names."
        )
