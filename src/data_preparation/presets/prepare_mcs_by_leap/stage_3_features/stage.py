from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from beartype import beartype

from src.utils.errors import InputValidationError

_STAGE_DIR = Path(__file__).resolve().parent
_LONGITUDINAL_CONFIG = _STAGE_DIR / "longitudinal_features.yaml"
_NON_LONGITUDINAL_CONFIG = _STAGE_DIR / "non_longitudinal_features.yaml"

LONGITUDINAL_TEMP_PREFIX = "__tmp_long__"


@beartype
@dataclass(frozen=True)
class LongitudinalFeatureMapping:
    """Canonical longitudinal feature mapping for one wave.

    Attributes:
        canonical (str): Canonical.
        source (str): Source.
        scope (str | None): Scope.
    """

    canonical: str
    source: str
    scope: str | None = None


@beartype
@dataclass(frozen=True)
class NonLongitudinalFeatureSpec:
    """Non-longitudinal feature selection for one wave.

    Attributes:
        source (str): Source.
        output (str): Output.
        scope (str | None): Scope.
    """

    source: str
    output: str
    scope: str | None = None


@beartype
@dataclass(frozen=True)
class FeaturePreparationSummary:
    """Summary for stage-3 feature preparation.

    Attributes:
        resolved_longitudinal (int): Resolved longitudinal.
        unresolved_longitudinal (int): Unresolved longitudinal.
        selected_non_longitudinal (int): Selected non longitudinal.
        unresolved_non_longitudinal (int): Unresolved non longitudinal.
        ambiguous_source_resolutions (int): Ambiguous source resolutions.
    """

    resolved_longitudinal: int
    unresolved_longitudinal: int
    selected_non_longitudinal: int
    unresolved_non_longitudinal: int
    ambiguous_source_resolutions: int


@beartype
@dataclass(frozen=True)
class SourceCoverageCandidate:
    """One candidate column and its non-null row coverage.

    Attributes:
        column (str): Column.
        non_null_rows (int): Non null rows.
    """

    column: str
    non_null_rows: int


@beartype
@dataclass(frozen=True)
class AmbiguousSourceResolution:
    """Resolution details for one ambiguous source mapping.

    Attributes:
        source_name (str): Name for source.
        target_name (str): Name for target.
        feature_group (str): Feature group.
        selected_column (str): Column name for selected column.
        total_matches (int): Total matches.
        top_coverage_candidates (tuple[SourceCoverageCandidate, ...]): Top coverage candidates.
    """

    source_name: str
    target_name: str
    feature_group: str
    selected_column: str
    total_matches: int
    top_coverage_candidates: tuple[SourceCoverageCandidate, ...]


@beartype
@dataclass(frozen=True)
class FeaturePreparationResult:
    """Stage-3 output containing selected features and unresolved mappings.

    Attributes:
        data (pd.DataFrame): Data.
        unresolved_longitudinal (tuple[LongitudinalFeatureMapping, ...]): Unresolved longitudinal.
        summary (FeaturePreparationSummary): Summary.
        ambiguous_resolutions (tuple[AmbiguousSourceResolution, ...]): Ambiguous resolutions.
    """

    data: pd.DataFrame
    unresolved_longitudinal: tuple[LongitudinalFeatureMapping, ...]
    summary: FeaturePreparationSummary
    ambiguous_resolutions: tuple[AmbiguousSourceResolution, ...]


@beartype
def resolve_longitudinal_mappings(wave: str) -> tuple[LongitudinalFeatureMapping, ...]:
    """Resolve longitudinal canonical mappings for one wave.

    This helper is used by higher-level workflows and keeps input/output handling consistent.

    Args:
        wave (str): Wave identifier.

    Returns:
        tuple[LongitudinalFeatureMapping, ...]: Tuple of resolved values.
    """

    normalised = wave.strip().upper()
    raw = _load_yaml_config(_LONGITUDINAL_CONFIG)
    waves = raw.get("waves")
    if not isinstance(waves, dict):
        raise InputValidationError(
            "Stage 3 longitudinal config must define `waves` mapping."
        )

    selected = waves.get(normalised, [])
    items: Any = selected
    if isinstance(selected, dict):
        if "features" in selected:
            items = selected["features"]
        else:
            raise InputValidationError(
                f"Invalid longitudinal config for `{normalised}`. Expected `features` key."
            )

    if items in (None, "none"):
        return ()
    if not isinstance(items, list):
        raise InputValidationError(
            f"Invalid longitudinal feature entries for `{normalised}`. Expected list."
        )

    mappings: list[LongitudinalFeatureMapping] = []
    for index, item in enumerate(items):
        context = f"waves.{normalised}[{index}]"
        if isinstance(item, str):
            feature = item.strip()
            if not feature:
                raise InputValidationError(f"Invalid empty entry in `{context}`.")
            mappings.append(
                LongitudinalFeatureMapping(canonical=feature, source=feature)
            )
            continue

        if not isinstance(item, dict):
            raise InputValidationError(
                f"Invalid `{context}` in longitudinal config. Expected mapping."
            )

        canonical = item.get("canonical")
        source = item.get("source")
        scope = _parse_optional_scope(value=item.get("scope"), context=context)
        if not isinstance(canonical, str) or not canonical.strip():
            raise InputValidationError(
                f"Invalid `{context}.canonical` in longitudinal config."
            )
        if not isinstance(source, str) or not source.strip():
            raise InputValidationError(
                f"Invalid `{context}.source` in longitudinal config."
            )
        mappings.append(
            LongitudinalFeatureMapping(
                canonical=canonical.strip(),
                source=source.strip(),
                scope=scope,
            )
        )

    return tuple(mappings)


@beartype
def resolve_non_longitudinal_specs(wave: str) -> tuple[NonLongitudinalFeatureSpec, ...]:
    """Resolve non-longitudinal feature selection for one wave.

    This helper is used by higher-level workflows and keeps input/output handling consistent.

    Args:
        wave (str): Wave identifier.

    Returns:
        tuple[NonLongitudinalFeatureSpec, ...]: Tuple of resolved values.
    """

    normalised = wave.strip().upper()
    raw = _load_yaml_config(_NON_LONGITUDINAL_CONFIG)
    waves = raw.get("waves")
    if not isinstance(waves, dict):
        raise InputValidationError(
            "Stage 3 non-longitudinal config must define `waves` mapping."
        )

    selected = waves.get(normalised, [])
    if selected in (None, "none"):
        return ()
    if not isinstance(selected, list):
        raise InputValidationError(
            f"Invalid non-longitudinal entries for `{normalised}`. Expected list."
        )

    specs: list[NonLongitudinalFeatureSpec] = []
    for index, item in enumerate(selected):
        context = f"waves.{normalised}[{index}]"
        if isinstance(item, str):
            source = item.strip()
            if not source:
                raise InputValidationError(f"Invalid empty entry in `{context}`.")
            specs.append(NonLongitudinalFeatureSpec(source=source, output=source))
            continue

        if not isinstance(item, dict):
            raise InputValidationError(
                f"Invalid `{context}` in non-longitudinal config. Expected mapping."
            )

        source = item.get("source")
        output = item.get("output", source)
        scope = _parse_optional_scope(value=item.get("scope"), context=context)
        if not isinstance(source, str) or not source.strip():
            raise InputValidationError(
                f"Invalid `{context}.source` in non-longitudinal config."
            )
        if not isinstance(output, str) or not output.strip():
            raise InputValidationError(
                f"Invalid `{context}.output` in non-longitudinal config."
            )
        specs.append(
            NonLongitudinalFeatureSpec(
                source=source.strip(),
                output=output.strip(),
                scope=scope,
            )
        )
    return tuple(specs)


@beartype
def prepare_wave_features(
    *,
    wave: str,
    data: pd.DataFrame,
    protected_columns: tuple[str, ...],
) -> FeaturePreparationResult:
    """Select longitudinal and non-longitudinal features for one wave.

    Args:
        wave (str): Wave identifier.
        data (pd.DataFrame): Input dataset.
        protected_columns (tuple[str, ...]): Column names used by this workflow.

    Returns:
        FeaturePreparationResult: Result object for this operation.
    """

    mappings = resolve_longitudinal_mappings(wave)
    non_long_specs = resolve_non_longitudinal_specs(wave)

    protected = [column for column in protected_columns if column in data.columns]
    selected = data[protected].copy()

    unresolved_longitudinal: list[LongitudinalFeatureMapping] = []
    ambiguous_resolutions: list[AmbiguousSourceResolution] = []
    resolved_longitudinal = 0

    for mapping in mappings:
        resolved_column, resolution = _resolve_feature_source_column(
            data=data,
            source_name=mapping.source,
            scope=mapping.scope,
            feature_group="longitudinal",
            target_name=mapping.canonical,
        )
        if resolved_column is None:
            unresolved_longitudinal.append(mapping)
            continue
        if resolution is not None:
            ambiguous_resolutions.append(resolution)

        temp_column = build_longitudinal_temp_column_name(mapping=mapping)
        selected[temp_column] = data[resolved_column]
        resolved_longitudinal += 1

    unresolved_non_longitudinal = 0
    selected_non_longitudinal = 0
    for spec in non_long_specs:
        resolved_column, resolution = _resolve_feature_source_column(
            data=data,
            source_name=spec.source,
            scope=spec.scope,
            feature_group="non_longitudinal",
            target_name=spec.output,
        )
        if resolved_column is None:
            unresolved_non_longitudinal += 1
            continue
        if resolution is not None:
            ambiguous_resolutions.append(resolution)

        output_column = build_non_longitudinal_output_column_name(
            wave=wave,
            output_name=spec.output,
        )
        selected[output_column] = data[resolved_column]
        selected_non_longitudinal += 1

    summary = FeaturePreparationSummary(
        resolved_longitudinal=resolved_longitudinal,
        unresolved_longitudinal=len(unresolved_longitudinal),
        selected_non_longitudinal=selected_non_longitudinal,
        unresolved_non_longitudinal=unresolved_non_longitudinal,
        ambiguous_source_resolutions=len(ambiguous_resolutions),
    )
    return FeaturePreparationResult(
        data=selected,
        unresolved_longitudinal=tuple(unresolved_longitudinal),
        summary=summary,
        ambiguous_resolutions=tuple(ambiguous_resolutions),
    )


@beartype
def append_unresolved_longitudinal_features(
    *,
    data: pd.DataFrame,
    source_data: pd.DataFrame,
    unresolved: tuple[LongitudinalFeatureMapping, ...],
) -> tuple[pd.DataFrame, tuple[LongitudinalFeatureMapping, ...]]:
    """Retry unresolved longitudinal mappings after composite feature creation.

    Args:
        data (pd.DataFrame): Input dataset.
        source_data (pd.DataFrame): Source dataframe for this stage.
        unresolved (tuple[LongitudinalFeatureMapping, ...]): Unresolved.

    Returns:
        tuple[pd.DataFrame, tuple[LongitudinalFeatureMapping, ...]]: Transformed dataset as a pandas DataFrame.
    """

    if not unresolved:
        return data, ()

    working = data.copy()
    still_unresolved: list[LongitudinalFeatureMapping] = []

    for mapping in unresolved:
        temp_column = build_longitudinal_temp_column_name(mapping=mapping)
        if temp_column in working.columns:
            continue

        resolved_in_working, _ = _resolve_feature_source_column(
            data=working,
            source_name=mapping.source,
            scope=mapping.scope,
            feature_group="longitudinal_retry",
            target_name=mapping.canonical,
        )
        if resolved_in_working is not None:
            working[temp_column] = working[resolved_in_working]
            continue

        resolved_in_source, _ = _resolve_feature_source_column(
            data=source_data,
            source_name=mapping.source,
            scope=mapping.scope,
            feature_group="longitudinal_retry",
            target_name=mapping.canonical,
        )
        if resolved_in_source is not None:
            working[temp_column] = source_data[resolved_in_source]
            continue

        still_unresolved.append(mapping)

    return working, tuple(still_unresolved)


@beartype
def print_feature_summary(
    *,
    summary: FeaturePreparationSummary,
    unresolved_after_retry: tuple[LongitudinalFeatureMapping, ...],
    ambiguous_resolutions: tuple[AmbiguousSourceResolution, ...],
) -> None:
    """Print stage-3 summary and unresolved mappings.

    Args:
        summary (FeaturePreparationSummary): Summary object from the previous stage.
        unresolved_after_retry (tuple[LongitudinalFeatureMapping, ...]): Unresolved after retry.
        ambiguous_resolutions (tuple[AmbiguousSourceResolution, ...]): Ambiguous resolutions.
    """

    print(
        "Stage 3 - Feature preparation: "
        f"longitudinal_resolved={summary.resolved_longitudinal}, "
        f"longitudinal_unresolved_initial={summary.unresolved_longitudinal}, "
        f"non_longitudinal_selected={summary.selected_non_longitudinal}, "
        f"non_longitudinal_unresolved={summary.unresolved_non_longitudinal}, "
        f"ambiguous_source_resolutions={summary.ambiguous_source_resolutions}"
    )

    if ambiguous_resolutions:
        print("Stage 3 - Ambiguous source resolution (selected by highest coverage):")
        for resolution in ambiguous_resolutions:
            top_candidates = ", ".join(
                f"{candidate.column}={candidate.non_null_rows}"
                for candidate in resolution.top_coverage_candidates[:5]
            )
            print(
                f"- {resolution.feature_group} `{resolution.target_name}` "
                f"from source `{resolution.source_name}`: "
                f"selected `{resolution.selected_column}` "
                f"(matches={resolution.total_matches}; top_coverage={top_candidates})"
            )

    if unresolved_after_retry:
        print(
            "Stage 3 - Longitudinal mappings still unresolved after composites: "
            + ", ".join(
                f"{mapping.canonical}<-{mapping.source}"
                for mapping in unresolved_after_retry
            )
        )


@beartype
def build_longitudinal_temp_column_name(*, mapping: LongitudinalFeatureMapping) -> str:
    """Build temporary stage-3 column name using the source feature prefix.

    Args:
        mapping (LongitudinalFeatureMapping): Mapping.

    Returns:
        str: Parsed string value.
    """

    return f"{LONGITUDINAL_TEMP_PREFIX}{mapping.source}__{mapping.canonical}"


@beartype
def build_non_longitudinal_output_column_name(*, wave: str, output_name: str) -> str:
    """Build wave-specific output name for non-longitudinal selected features.

    Args:
        wave (str): Wave identifier.
        output_name (str): Name for output.

    Returns:
        str: Parsed string value.
    """

    return f"non_long__{wave.strip().lower()}__{output_name}"


@beartype
def _resolve_feature_source_column(
    *,
    data: pd.DataFrame,
    source_name: str,
    scope: str | None,
    feature_group: str,
    target_name: str,
) -> tuple[str | None, AmbiguousSourceResolution | None]:
    """Resolve one source column; break ambiguities by highest non-null coverage."""

    if source_name in data.columns:
        return source_name, None

    suffix_matches = sorted(
        column for column in data.columns if column.endswith(f"__{source_name}")
    )
    if not suffix_matches:
        return None, None

    candidates = suffix_matches
    if scope is not None:
        scoped = [
            column
            for column in suffix_matches
            if _column_matches_scope(column=column, scope=scope)
        ]
        if scoped:
            candidates = scoped

    if len(candidates) == 1:
        return candidates[0], None

    ranked = sorted(
        (
            SourceCoverageCandidate(
                column=column,
                non_null_rows=int(data[column].notna().sum()),
            )
            for column in candidates
        ),
        key=lambda candidate: (-candidate.non_null_rows, candidate.column),
    )
    selected_column = ranked[0].column
    resolution = AmbiguousSourceResolution(
        source_name=source_name,
        target_name=target_name,
        feature_group=feature_group,
        selected_column=selected_column,
        total_matches=len(candidates),
        top_coverage_candidates=tuple(ranked[:5]),
    )
    return selected_column, resolution


@beartype
def _parse_optional_scope(*, value: Any, context: str) -> str | None:
    """Parse optional scope token for feature-source disambiguation."""

    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"Invalid `{context}.scope` in config.")
    return _normalise_scope_token(value)


@beartype
def _normalise_scope_token(raw: str) -> str:
    """Normalise one scope token to a stable snake_case comparison key."""

    first_part = raw.split(",")[0].strip().lower()
    return first_part.replace("-", "_").replace(" ", "_")


@beartype
def _column_matches_scope(*, column: str, scope: str) -> bool:
    """Check if one prefixed column belongs to the requested dataset scope."""

    alias = _extract_dataset_alias(column)
    if alias is None:
        return False
    normalised_alias = alias.lower()
    return normalised_alias == scope or normalised_alias.endswith(f"_{scope}")


@beartype
def _extract_dataset_alias(column: str) -> str | None:
    """Extract dataset alias from one stage-2 prefixed feature column."""

    if "__" not in column:
        return None

    parts = column.split("__")
    if len(parts) < 2:
        return None

    first = parts[0].lower()
    if len(parts) >= 3 and first.startswith("p") and first[1:].isdigit():
        return parts[1]
    return parts[0]


@beartype
@cache
def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load one YAML config file and validate mapping root."""

    if not config_path.exists() or not config_path.is_file():
        raise InputValidationError(f"Missing config file: {config_path.resolve()}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise InputValidationError(
            f"Config file `{config_path.name}` must contain a YAML mapping."
        )
    return loaded
