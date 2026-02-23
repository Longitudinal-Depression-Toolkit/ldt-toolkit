from __future__ import annotations

from dataclasses import dataclass

from beartype import beartype

from src.utils.errors import InputValidationError


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
        feature_groups_raw: str,
        non_longitudinal_raw: str,
    ) -> LongitudinalFeatureVectors:
        """Resolve longitudinal vectors from raw string values.

        This helper is used by higher-level workflows and keeps input/output handling consistent.

        Args:
            feature_columns (list[str]): Column names used by this workflow.
            feature_groups_raw (str): Raw text describing feature groups.
            non_longitudinal_raw (str): Raw text describing non-longitudinal features.

        Returns:
            LongitudinalFeatureVectors: Result object for this operation.
        """

        feature_groups = LongitudinalFeatureInputPrompter.resolve_feature_groups(
            feature_columns=feature_columns,
            feature_groups_raw=feature_groups_raw,
        )
        non_longitudinal = (
            LongitudinalFeatureInputPrompter.resolve_non_longitudinal_features(
                feature_columns=feature_columns,
                non_longitudinal_raw=non_longitudinal_raw,
                feature_groups=feature_groups,
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
        feature_groups_raw: str,
    ) -> tuple[tuple[int, ...], ...]:
        """Parse and validate feature groups from raw input.

        This helper is used by higher-level workflows and keeps input/output handling consistent.

        Args:
            feature_columns (list[str]): Column names used by this workflow.
            feature_groups_raw (str): Raw text describing feature groups.

        Returns:
            tuple[tuple[int, ...], ...]: Tuple of resolved values.
        """

        raw_groups = [
            raw_group.strip()
            for raw_group in feature_groups_raw.split(";")
            if raw_group.strip()
        ]
        if not raw_groups:
            raise InputValidationError("At least one feature group is required.")

        resolved_groups: list[tuple[int, ...]] = []
        seen_indices: set[int] = set()
        for raw_group in raw_groups:
            tokens = [token.strip() for token in raw_group.split(",") if token.strip()]
            if len(tokens) < 2:
                raise InputValidationError(
                    "Each feature group must contain at least two features."
                )
            group_indices = tuple(
                LongitudinalFeatureInputPrompter._resolve_feature_token(
                    feature_columns=feature_columns,
                    token=token,
                )
                for token in tokens
            )
            if len(set(group_indices)) != len(group_indices):
                raise InputValidationError(
                    f"Feature group contains duplicate entries: {raw_group}"
                )
            overlap = seen_indices.intersection(group_indices)
            if overlap:
                overlap_labels = ", ".join(
                    feature_columns[index] for index in sorted(overlap)
                )
                raise InputValidationError(
                    "Feature groups cannot overlap. Duplicated features: "
                    f"{overlap_labels}"
                )
            seen_indices.update(group_indices)
            resolved_groups.append(group_indices)

        return tuple(resolved_groups)

    @staticmethod
    @beartype
    def resolve_non_longitudinal_features(
        *,
        feature_columns: list[str],
        non_longitudinal_raw: str,
        feature_groups: tuple[tuple[int, ...], ...],
    ) -> tuple[int, ...]:
        """Parse and validate non-longitudinal feature vector.

        This helper is used by higher-level workflows and keeps input/output handling consistent.

        Args:
            feature_columns (list[str]): Column names used by this workflow.
            non_longitudinal_raw (str): Raw text describing non-longitudinal features.
            feature_groups (tuple[tuple[int, ...], ...]): Longitudinal feature-group indices.

        Returns:
            tuple[int, ...]: Tuple of resolved values.
        """

        normalised = non_longitudinal_raw.strip().lower()
        longitudinal_indices = {
            index for feature_group in feature_groups for index in feature_group
        }

        if normalised in {"", "none", "no", "n"}:
            return tuple()
        if normalised == "auto":
            return tuple(
                index
                for index, _ in enumerate(feature_columns)
                if index not in longitudinal_indices
            )

        tokens = [
            token.strip() for token in non_longitudinal_raw.split(",") if token.strip()
        ]
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
    def _resolve_feature_token(
        *,
        feature_columns: list[str],
        token: str,
    ) -> int:
        """Resolve one feature token (index or name) to column index."""

        if token.isdigit():
            index = int(token)
            if not 0 <= index < len(feature_columns):
                raise InputValidationError(f"Feature index out of range: {token}")
            return index

        if token in feature_columns:
            return feature_columns.index(token)

        raise InputValidationError(
            f"Unknown feature token: `{token}`. Use exact feature names or "
            "zero-based indices."
        )
