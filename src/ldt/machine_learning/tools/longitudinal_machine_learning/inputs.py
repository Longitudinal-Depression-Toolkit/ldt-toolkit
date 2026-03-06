from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
from beartype import beartype
from scikit_longitudinal.data_preparation.longitudinal_dataset import (
    LongitudinalDataset,
)

from ldt.utils.errors import InputValidationError


@beartype
@dataclass(frozen=True)
class LongitudinalFeatureVectors:
    """Resolved longitudinal vectors passed to scikit-longitudinal models.

    Attributes:
        feature_groups (tuple[tuple[int, ...], ...]): Feature groups.
        non_longitudinal_features (tuple[int, ...]): Column names for non longitudinal features.

    """

    feature_groups: tuple[tuple[int, ...], ...]
    non_longitudinal_features: tuple[int, ...]


@beartype
class LongitudinalFeatureInputPrompter:
    """Resolve and validate longitudinal feature-vector definitions."""

    @staticmethod
    @beartype
    def print_wide_format_warning() -> None:
        """Print a wide-format reminder for callers that need it."""

        print(
            "Important: Longitudinal ML workflows require wide-format datasets.\n"
            "If your dataset is still long format, run data_preprocessing pivot-long-to-wide."
        )

    @staticmethod
    @beartype
    def resolve_feature_vectors(
        *,
        feature_columns: list[str],
        feature_groups_raw: Any,
        non_longitudinal_raw: Any,
        feature_groups_mode: str | None = None,
        feature_groups_preset: str | None = None,
        feature_groups_suffix: str | None = None,
        non_longitudinal_mode: str | None = None,
    ) -> LongitudinalFeatureVectors:
        """Resolve longitudinal vectors from raw string values.

        This helper is used by higher-level workflows and keeps input/output handling consistent.

        Args:
            feature_columns (list[str]): Column names used by this workflow.
            feature_groups_raw (Any): Raw value describing feature groups.
            non_longitudinal_raw (Any): Raw value describing non-longitudinal features.
            feature_groups_mode (str | None): Required feature-group mode.
            feature_groups_preset (str | None): scikit-longitudinal preset name for preset mode.
            feature_groups_suffix (str | None): Suffix used for wave inference in suffix mode.
            non_longitudinal_mode (str | None): Required non-longitudinal mode.

        Returns:
            LongitudinalFeatureVectors: Result object for this operation.
        """

        feature_groups = LongitudinalFeatureInputPrompter.resolve_feature_groups(
            feature_columns=feature_columns,
            feature_groups_raw=feature_groups_raw,
            feature_groups_mode=feature_groups_mode,
            feature_groups_preset=feature_groups_preset,
            feature_groups_suffix=feature_groups_suffix,
        )
        non_longitudinal = (
            LongitudinalFeatureInputPrompter.resolve_non_longitudinal_features(
                feature_columns=feature_columns,
                non_longitudinal_raw=non_longitudinal_raw,
                feature_groups=feature_groups,
                non_longitudinal_mode=non_longitudinal_mode,
            )
        )
        return LongitudinalFeatureVectors(
            feature_groups=feature_groups,
            non_longitudinal_features=non_longitudinal,
        )

    @staticmethod
    @beartype
    def resolve_feature_groups(
        *,
        feature_columns: list[str],
        feature_groups_raw: Any,
        feature_groups_mode: str | None = None,
        feature_groups_preset: str | None = None,
        feature_groups_suffix: str | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        """Parse and validate feature groups from raw input.

        This helper is used by higher-level workflows and keeps input/output handling consistent.

        Args:
            feature_columns (list[str]): Column names used by this workflow.
            feature_groups_raw (Any): Raw value describing feature groups.
            feature_groups_mode (str | None): Required feature-group mode.
            feature_groups_preset (str | None): scikit-longitudinal preset name for preset mode.
            feature_groups_suffix (str | None): Suffix used for wave inference in suffix mode.

        Returns:
            tuple[tuple[int, ...], ...]: Tuple of resolved values.
        """

        mode = LongitudinalFeatureInputPrompter._normalise_mode(
            raw_mode=feature_groups_mode,
            allowed={"manual", "preset", "suffix"},
            field_name="feature_groups_mode",
        )
        if mode is None:
            raise InputValidationError(
                "`feature_groups_mode` is required. Use `manual`, `preset`, or `suffix`."
            )
        if mode == "manual":
            if not LongitudinalFeatureInputPrompter._has_value(feature_groups_raw):
                raise InputValidationError(
                    "`feature_groups` is required when `feature_groups_mode=manual`."
                )
            return LongitudinalFeatureInputPrompter._resolve_manual_feature_groups(
                feature_columns=feature_columns,
                raw_groups=feature_groups_raw,
            )
        if mode == "preset":
            if not LongitudinalFeatureInputPrompter._has_value(feature_groups_preset):
                raise InputValidationError(
                    "`feature_groups_preset` is required when `feature_groups_mode=preset`."
                )
            return LongitudinalFeatureInputPrompter._resolve_feature_groups_from_preset(
                feature_columns=feature_columns,
                preset_name=str(feature_groups_preset),
            )
        if not LongitudinalFeatureInputPrompter._has_value(feature_groups_suffix):
            raise InputValidationError(
                "`feature_groups_suffix` is required when `feature_groups_mode=suffix`."
            )
        return LongitudinalFeatureInputPrompter._resolve_feature_groups_from_suffix(
            feature_columns=feature_columns,
            suffix=str(feature_groups_suffix),
        )

    @staticmethod
    @beartype
    def resolve_non_longitudinal_features(
        *,
        feature_columns: list[str],
        non_longitudinal_raw: Any,
        feature_groups: tuple[tuple[int, ...], ...],
        non_longitudinal_mode: str | None = None,
    ) -> tuple[int, ...]:
        """Parse and validate non-longitudinal feature vector.

        This helper is used by higher-level workflows and keeps input/output handling consistent.

        Args:
            feature_columns (list[str]): Column names used by this workflow.
            non_longitudinal_raw (Any): Raw value describing non-longitudinal features.
            feature_groups (tuple[tuple[int, ...], ...]): Longitudinal feature-group indices.
            non_longitudinal_mode (str | None): Required non-longitudinal mode.

        Returns:
            tuple[int, ...]: Tuple of resolved values.
        """

        longitudinal_indices = {
            index
            for feature_group in feature_groups
            for index in feature_group
            if index != -1
        }

        mode = LongitudinalFeatureInputPrompter._normalise_mode(
            raw_mode=non_longitudinal_mode,
            allowed={"auto", "manual"},
            field_name="non_longitudinal_mode",
        )
        if mode is None:
            raise InputValidationError(
                "`non_longitudinal_mode` is required. Use `auto` or `manual`."
            )
        if mode == "auto":
            return tuple(
                index
                for index, _ in enumerate(feature_columns)
                if index not in longitudinal_indices
            )
        return LongitudinalFeatureInputPrompter._resolve_manual_non_longitudinal(
            feature_columns=feature_columns,
            raw_features=non_longitudinal_raw,
            longitudinal_indices=longitudinal_indices,
        )

    @staticmethod
    @beartype
    def _resolve_manual_feature_groups(
        *,
        feature_columns: list[str],
        raw_groups: Any,
    ) -> tuple[tuple[int, ...], ...]:
        groups = LongitudinalFeatureInputPrompter._coerce_group_collection(raw_groups)
        if not groups:
            raise InputValidationError("At least one feature group is required.")

        resolved_groups: list[tuple[int, ...]] = []
        seen_indices: set[int] = set()
        for group in groups:
            group_indices = LongitudinalFeatureInputPrompter._resolve_one_feature_group(
                feature_columns=feature_columns,
                group=group,
            )
            overlap = seen_indices.intersection(
                index for index in group_indices if index != -1
            )
            if overlap:
                overlap_labels = ", ".join(
                    feature_columns[index] for index in sorted(overlap)
                )
                raise InputValidationError(
                    "Feature groups cannot overlap. Duplicated features: "
                    f"{overlap_labels}"
                )
            seen_indices.update(index for index in group_indices if index != -1)
            resolved_groups.append(group_indices)
        return tuple(resolved_groups)

    @staticmethod
    @beartype
    def _resolve_feature_token(
        *,
        feature_columns: list[str],
        token: Any,
        allow_padding: bool = False,
    ) -> int:
        """Resolve one feature token (index or name) to column index."""

        if isinstance(token, int):
            if token == -1 and allow_padding:
                return -1
            index = token
            if not 0 <= index < len(feature_columns):
                raise InputValidationError(f"Feature index out of range: {token}")
            return index

        if isinstance(token, str):
            cleaned = token.strip()
            if cleaned == "":
                raise InputValidationError("Feature tokens cannot be empty.")
            if cleaned == "-1" and allow_padding:
                return -1
            if re.fullmatch(r"-?\d+", cleaned):
                index = int(cleaned)
                if index == -1 and allow_padding:
                    return -1
                if not 0 <= index < len(feature_columns):
                    raise InputValidationError(f"Feature index out of range: {token}")
                return index
            if cleaned in feature_columns:
                return feature_columns.index(cleaned)

        raise InputValidationError(
            f"Unknown feature token: `{token}`. Use exact feature names or "
            "zero-based indices."
        )

    @staticmethod
    @beartype
    def _resolve_one_feature_group(
        *,
        feature_columns: list[str],
        group: list[Any],
    ) -> tuple[int, ...]:
        if not group:
            raise InputValidationError("Feature groups cannot be empty.")

        resolved: list[int] = []
        seen: set[int] = set()
        actual_feature_count = 0
        for token in group:
            index = LongitudinalFeatureInputPrompter._resolve_feature_token(
                feature_columns=feature_columns,
                token=token,
                allow_padding=True,
            )
            if index == -1:
                resolved.append(index)
                continue
            actual_feature_count += 1
            if index in seen:
                raise InputValidationError(
                    f"Feature group contains duplicate entries: {group}"
                )
            seen.add(index)
            resolved.append(index)

        if actual_feature_count < 2:
            raise InputValidationError(
                "Each feature group must contain at least two real features."
            )
        return tuple(resolved)

    @staticmethod
    @beartype
    def _resolve_feature_groups_from_preset(
        *,
        feature_columns: list[str],
        preset_name: str,
    ) -> tuple[tuple[int, ...], ...]:
        cleaned_preset = preset_name.strip()
        if not cleaned_preset:
            raise InputValidationError("`feature_groups_preset` cannot be empty.")

        dataset = LongitudinalDataset(
            "in_memory.csv",
            data_frame=pd.DataFrame(columns=feature_columns),
        )
        try:
            dataset.setup_features_group(cleaned_preset)
        except ValueError as exc:
            raise InputValidationError(
                "Unknown or invalid scikit-longitudinal feature-group preset "
                f"`{cleaned_preset}`."
            ) from exc

        groups = dataset.feature_groups()
        return LongitudinalFeatureInputPrompter._normalise_automatic_feature_groups(
            feature_columns=feature_columns,
            raw_groups=groups,
        )

    @staticmethod
    @beartype
    def _resolve_feature_groups_from_suffix(
        *,
        feature_columns: list[str],
        suffix: str,
    ) -> tuple[tuple[int, ...], ...]:
        cleaned_suffix = suffix.strip()
        if not cleaned_suffix:
            raise InputValidationError("`feature_groups_suffix` cannot be empty.")

        wave_pattern = re.compile(rf"{re.escape(cleaned_suffix)}(\d+)$")
        grouped_columns: dict[str, dict[int, int]] = {}
        max_wave = 0
        for index, column in enumerate(feature_columns):
            match = wave_pattern.search(column)
            if match is None:
                continue
            wave = int(match.group(1))
            base_name = column[: match.start()]
            wave_map = grouped_columns.setdefault(base_name, {})
            if wave in wave_map:
                raise InputValidationError(
                    "Cannot infer feature groups because multiple columns share "
                    f"the same base name and wave: `{column}`."
                )
            wave_map[wave] = index
            max_wave = max(max_wave, wave)

        if not grouped_columns:
            raise InputValidationError(
                "No feature groups matched the requested suffix "
                f"`{cleaned_suffix}`."
            )

        groups: list[list[int]] = []
        for wave_map in grouped_columns.values():
            if len(wave_map) < 2:
                continue
            padded = [-1] * max_wave
            for wave, index in wave_map.items():
                padded[wave - 1] = index
            groups.append(padded)

        if not groups:
            raise InputValidationError(
                "No longitudinal feature groups could be inferred from suffix "
                f"`{cleaned_suffix}`. At least two waves per feature are required."
            )

        return tuple(tuple(group) for group in groups)

    @staticmethod
    @beartype
    def _resolve_manual_non_longitudinal(
        *,
        feature_columns: list[str],
        raw_features: Any,
        longitudinal_indices: set[int],
    ) -> tuple[int, ...]:
        if not LongitudinalFeatureInputPrompter._has_value(raw_features):
            return tuple()

        tokens = LongitudinalFeatureInputPrompter._coerce_token_list(raw_features)
        if not tokens:
            return tuple()

        resolved: list[int] = []
        seen: set[int] = set()
        for token in tokens:
            index = LongitudinalFeatureInputPrompter._resolve_feature_token(
                feature_columns=feature_columns,
                token=token,
            )
            if index in longitudinal_indices:
                raise InputValidationError(
                    f"Feature `{feature_columns[index]}` is already used in "
                    "`feature_groups` and cannot be non-longitudinal."
                )
            if index in seen:
                continue
            seen.add(index)
            resolved.append(index)
        return tuple(resolved)

    @staticmethod
    @beartype
    def _normalise_automatic_feature_groups(
        *,
        feature_columns: list[str],
        raw_groups: Any,
    ) -> tuple[tuple[int, ...], ...]:
        groups = LongitudinalFeatureInputPrompter._coerce_group_collection(raw_groups)
        if not groups:
            raise InputValidationError("At least one feature group is required.")

        resolved_groups: list[tuple[int, ...]] = []
        seen_indices: set[int] = set()
        for group in groups:
            resolved_group: list[int] = []
            actual_indices: list[int] = []
            for token in group:
                index = LongitudinalFeatureInputPrompter._resolve_feature_token(
                    feature_columns=feature_columns,
                    token=token,
                    allow_padding=True,
                )
                resolved_group.append(index)
                if index != -1:
                    actual_indices.append(index)
            if len(actual_indices) < 2:
                continue
            overlap = seen_indices.intersection(actual_indices)
            if overlap:
                overlap_labels = ", ".join(
                    feature_columns[index] for index in sorted(overlap)
                )
                raise InputValidationError(
                    "Feature groups cannot overlap. Duplicated features: "
                    f"{overlap_labels}"
                )
            seen_indices.update(actual_indices)
            resolved_groups.append(tuple(resolved_group))

        if not resolved_groups:
            raise InputValidationError("At least one feature group is required.")
        return tuple(resolved_groups)

    @staticmethod
    @beartype
    def _coerce_group_collection(raw_groups: Any) -> list[list[Any]]:
        if isinstance(raw_groups, str):
            parsed_literal = LongitudinalFeatureInputPrompter._maybe_parse_literal(
                raw_groups
            )
            if parsed_literal is not None:
                raw_groups = parsed_literal
            else:
                return [
                    [
                        token.strip()
                        for token in raw_group.split(",")
                        if token.strip()
                    ]
                    for raw_group in raw_groups.split(";")
                    if raw_group.strip()
                ]

        if isinstance(raw_groups, tuple):
            raw_groups = list(raw_groups)
        if not isinstance(raw_groups, list):
            raise InputValidationError(
                "Feature groups must be provided as a nested list or a string."
            )
        if not raw_groups:
            return []
        if all(not isinstance(item, (list, tuple)) for item in raw_groups):
            return [list(raw_groups)]

        groups: list[list[Any]] = []
        for group in raw_groups:
            if isinstance(group, tuple):
                group = list(group)
            if not isinstance(group, list):
                raise InputValidationError(
                    "Feature groups must be a nested list, e.g. "
                    "`[[0, 1, 2], [3, 4, 5]]`."
                )
            groups.append(list(group))
        return groups

    @staticmethod
    @beartype
    def _coerce_token_list(raw_features: Any) -> list[Any]:
        if isinstance(raw_features, str):
            cleaned = raw_features.strip()
            if cleaned.lower() in {"", "none", "no", "n"}:
                return []
            parsed_literal = LongitudinalFeatureInputPrompter._maybe_parse_literal(
                cleaned
            )
            if parsed_literal is not None:
                raw_features = parsed_literal
            else:
                return [
                    token.strip()
                    for token in cleaned.split(",")
                    if token.strip()
                ]

        if isinstance(raw_features, tuple):
            raw_features = list(raw_features)
        if isinstance(raw_features, list):
            return list(raw_features)
        return [raw_features]

    @staticmethod
    @beartype
    def _normalise_mode(
        *,
        raw_mode: str | None,
        allowed: set[str],
        field_name: str,
    ) -> str | None:
        if raw_mode is None:
            return None
        cleaned = raw_mode.strip().lower()
        if cleaned == "":
            return None
        if cleaned not in allowed:
            options = ", ".join(sorted(allowed))
            raise InputValidationError(
                f"`{field_name}` must be one of: {options}."
            )
        return cleaned

    @staticmethod
    @beartype
    def _maybe_parse_literal(raw_value: str) -> Any | None:
        stripped = raw_value.strip()
        if not stripped or stripped[0] not in {"[", "("}:
            return None
        try:
            return ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return None

    @staticmethod
    @beartype
    def _has_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip() != ""
        if isinstance(value, (list, tuple)):
            return len(value) > 0
        return True
