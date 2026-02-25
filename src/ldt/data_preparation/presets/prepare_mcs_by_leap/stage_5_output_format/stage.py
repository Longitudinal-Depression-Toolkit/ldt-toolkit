from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from beartype import beartype

from ldt.utils.errors import InputValidationError

from ..stage_3_features import LONGITUDINAL_TEMP_PREFIX

OutputFormat = Literal["long", "wide"]


@beartype
@dataclass(frozen=True)
class OutputFormattingSummary:
    """Summary for stage-5 longitudinal output formatting.

    Attributes:
        output_format (OutputFormat): Output format.
        subject_rows (int): Subject rows.
        columns (int): Column names for columns.
        longitudinal_features (int): Column names for longitudinal features.
    """

    output_format: OutputFormat
    subject_rows: int
    columns: int
    longitudinal_features: int


@beartype
def build_final_output_dataset(
    *,
    data: pd.DataFrame,
    output_format: OutputFormat,
    fixed_columns: tuple[str, ...],
    subject_id_column: str,
    wave_column: str,
    wide_suffix_prefix: str,
) -> tuple[pd.DataFrame, OutputFormattingSummary]:
    """Build final dataset in long or wide format.

    Args:
        data (pd.DataFrame): Input dataset.
        output_format (OutputFormat): Output format.
        fixed_columns (tuple[str, ...]): Column names used by this workflow.
        subject_id_column (str): Column name for subject id column.
        wave_column (str): Column name for wave column.
        wide_suffix_prefix (str): Wide suffix prefix.

    Returns:
        tuple[pd.DataFrame, OutputFormattingSummary]: Transformed dataset as a pandas DataFrame.
    """

    if output_format == "long":
        final, canonical_columns = _build_long_output(
            data=data,
            subject_id_column=subject_id_column,
            wave_column=wave_column,
        )
        summary = OutputFormattingSummary(
            output_format="long",
            subject_rows=len(final),
            columns=int(final.shape[1]),
            longitudinal_features=len(canonical_columns),
        )
        return final, summary

    final, canonical_columns = _build_wide_output(
        data=data,
        fixed_columns=fixed_columns,
        subject_id_column=subject_id_column,
        wave_column=wave_column,
        wide_suffix_prefix=wide_suffix_prefix,
    )
    summary = OutputFormattingSummary(
        output_format="wide",
        subject_rows=len(final),
        columns=int(final.shape[1]),
        longitudinal_features=len(canonical_columns),
    )
    return final, summary


@beartype
def print_output_summary(*, summary: OutputFormattingSummary) -> None:
    """Print stage-5 output formatting summary.

    Args:
        summary (OutputFormattingSummary): Summary object from the previous stage.
    """

    print(
        "Stage 5 - Output formatting: "
        f"format={summary.output_format}, "
        f"rows={summary.subject_rows}, "
        f"columns={summary.columns}, "
        f"longitudinal_features={summary.longitudinal_features}"
    )


@beartype
def _build_long_output(
    *,
    data: pd.DataFrame,
    subject_id_column: str,
    wave_column: str,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Build long-format output by collapsing temporary longitudinal columns."""

    collapsed, canonical_columns = _collapse_longitudinal_temp_columns(data=data)
    if subject_id_column not in collapsed.columns:
        raise InputValidationError(
            f"Cannot build long output. Missing subject column `{subject_id_column}`."
        )

    if wave_column in collapsed.columns:
        wave_rank = pd.to_numeric(
            collapsed[wave_column].astype(str).str.extract(r"(\d+)")[0],
            errors="coerce",
        )
        if wave_rank.notna().all():
            collapsed = collapsed.assign(_wave_rank=wave_rank).sort_values(
                [subject_id_column, "_wave_rank"]
            )
            collapsed = collapsed.drop(columns=["_wave_rank"])
        else:
            collapsed = collapsed.sort_values([subject_id_column, wave_column])
    else:
        collapsed = collapsed.sort_values([subject_id_column])

    collapsed = collapsed.reset_index(drop=True)
    return collapsed, canonical_columns


@beartype
def _build_wide_output(
    *,
    data: pd.DataFrame,
    fixed_columns: tuple[str, ...],
    subject_id_column: str,
    wave_column: str,
    wide_suffix_prefix: str,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Build wide-format output with one row per subject."""

    if not wide_suffix_prefix:
        raise InputValidationError("Wide suffix prefix cannot be empty.")

    collapsed, canonical_columns = _collapse_longitudinal_temp_columns(data=data)
    if subject_id_column not in collapsed.columns:
        raise InputValidationError(
            f"Cannot build wide output. Missing subject column `{subject_id_column}`."
        )
    if wave_column not in collapsed.columns:
        raise InputValidationError(
            f"Cannot build wide output. Missing wave column `{wave_column}`."
        )

    static_columns = [
        column
        for column in fixed_columns
        if column != wave_column and column in collapsed.columns
    ]
    if (
        subject_id_column not in static_columns
        and subject_id_column in collapsed.columns
    ):
        static_columns = [subject_id_column, *static_columns]

    base = collapsed.groupby(subject_id_column, dropna=False)[static_columns].first()

    used_columns = set(static_columns + [wave_column, *canonical_columns])
    non_longitudinal_columns = [
        column for column in collapsed.columns if column not in used_columns
    ]
    if non_longitudinal_columns:
        non_longitudinal = collapsed.groupby(subject_id_column, dropna=False)[
            non_longitudinal_columns
        ].first()
        base = base.join(non_longitudinal, how="left")

    for canonical in canonical_columns:
        pivoted = collapsed.pivot(
            index=subject_id_column,
            columns=wave_column,
            values=canonical,
        )
        pivoted = pivoted.rename(
            columns={
                wave_label: f"{canonical}{wide_suffix_prefix}{_wave_label_to_number(wave_label)}"
                for wave_label in pivoted.columns
            }
        )
        base = base.join(pivoted, how="left")

    result = base.reset_index(drop=True)
    return result, canonical_columns


@beartype
def _collapse_longitudinal_temp_columns(
    *,
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Collapse temporary longitudinal columns into canonical feature columns."""

    frame = data.copy()
    temp_columns = [
        column
        for column in frame.columns
        if column.startswith(LONGITUDINAL_TEMP_PREFIX)
    ]
    if not temp_columns:
        return frame, ()

    grouped: dict[str, list[str]] = {}
    for column in temp_columns:
        source, canonical = _parse_temp_longitudinal_column(column=column)
        _ = source
        grouped.setdefault(canonical, []).append(column)

    canonical_columns: list[str] = []
    for canonical, columns in grouped.items():
        merged = frame[columns[0]]
        for column in columns[1:]:
            merged = merged.combine_first(frame[column])

        if canonical in frame.columns:
            frame[canonical] = frame[canonical].combine_first(merged)
        else:
            frame[canonical] = merged
        canonical_columns.append(canonical)

    frame = frame.drop(columns=temp_columns)
    ordered = tuple(sorted(canonical_columns))
    return frame, ordered


@beartype
def _parse_temp_longitudinal_column(*, column: str) -> tuple[str, str]:
    """Parse one temporary longitudinal column name into source/canonical."""

    payload = column[len(LONGITUDINAL_TEMP_PREFIX) :]
    parts = payload.split("__", 1)
    if len(parts) != 2:
        raise InputValidationError(
            f"Invalid temporary longitudinal column format: `{column}`."
        )
    source, canonical = parts
    if not source or not canonical:
        raise InputValidationError(
            f"Invalid temporary longitudinal column format: `{column}`."
        )
    return source, canonical


@beartype
def _wave_label_to_number(wave_label: object) -> str:
    """Convert wave labels (for example W4) to numeric suffix token (4)."""

    as_text = str(wave_label).strip()
    digits = "".join(char for char in as_text if char.isdigit())
    return digits or as_text
