from __future__ import annotations

import numpy as np
import pandas as pd
from beartype import beartype

from ldt.utils.errors import InputValidationError
from ldt.utils.metadata import ComponentMetadata
from ldt.utils.templates.tools.data_preparation import DataPreparationTool


@beartype
class MissingDataScenarios(DataPreparationTool):
    """Synthetic data generator for missingness stress-test scenarios.

    This generator starts from a complete longitudinal panel and injects
    missingness patterns using one of four mechanisms:
    - `mcar`: missing completely at random.
    - `mar`: missing at random, probability tied to observed values/wave.
    - `dropout`: monotone missingness after participant dropout waves.
    - `mixed`: combination of the above.

    It is designed to benchmark imputation and robustness pipelines under
    controlled missing-data regimes.

    Examples:
        ```python
        from ldt.data_preparation import MissingDataScenarios

        generator = MissingDataScenarios()
        # The public tool runner orchestrates panel generation + missingness injection.
        ```
    """

    metadata = ComponentMetadata(
        name="missing-data-scenarios",
        full_name="Missing-data scenario benchmark",
        abstract_description=(
            "Generate long-format data with MCAR, MAR, dropout, or mixed "
            "missingness patterns."
        ),
    )

    @beartype
    def prepare(
        self,
        *,
        n_samples: int,
        n_waves: int,
        random_state: int | None,
        feature_cols: list[str],
        mechanism: str,
        missing_rate: float,
        dropout_rate: float,
        mar_strength: float,
    ) -> pd.DataFrame:
        """Generate long-format panel data with configurable missingness.

        Args:
            n_samples (int): Number of samples.
            n_waves (int): Number of waves.
            random_state (int | None): Optional random seed.
            feature_cols (list[str]): Feature columns to synthesise.
            mechanism (str): One of `mcar`, `mar`, `dropout`, `mixed`.
            missing_rate (float): Base missingness rate.
            dropout_rate (float): Subject-level dropout probability.
            mar_strength (float): Strength of MAR dependence on observed values.

        Returns:
            pd.DataFrame: Generated long-format dataset.
        """

        if not feature_cols:
            raise InputValidationError("At least one feature column must be provided.")

        mechanism_normalised = mechanism.strip().lower()
        if mechanism_normalised not in {"mcar", "mar", "dropout", "mixed"}:
            raise InputValidationError(
                "mechanism must be one of: mcar, mar, dropout, mixed."
            )

        self._validate_unit_interval(
            value=missing_rate,
            label="missing_rate",
            max_value=0.95,
        )
        self._validate_unit_interval(
            value=dropout_rate,
            label="dropout_rate",
            max_value=0.95,
        )
        if mar_strength < 0:
            raise InputValidationError("mar_strength must be non-negative.")

        rng = np.random.default_rng(random_state)
        data = self._generate_complete_panel(
            n_samples=n_samples,
            n_waves=n_waves,
            feature_cols=feature_cols,
            rng=rng,
        )
        data = self._apply_missingness(
            data=data,
            feature_cols=feature_cols,
            mechanism=mechanism_normalised,
            missing_rate=missing_rate,
            dropout_rate=dropout_rate,
            mar_strength=mar_strength,
            n_waves=n_waves,
            rng=rng,
        )
        return data.drop(columns=["time", "class", "dropout_wave"], errors="ignore")

    @staticmethod
    @beartype
    def _parse_feature_columns(raw: str) -> list[str]:
        parsed = [item.strip() for item in raw.split(",") if item.strip()]
        if not parsed:
            raise InputValidationError("At least one feature column must be provided.")
        if len(parsed) != len(set(parsed)):
            raise InputValidationError("Feature column names must be unique.")
        return parsed

    @staticmethod
    @beartype
    def _validate_unit_interval(*, value: float, label: str, max_value: float) -> None:
        if value < 0 or value > max_value:
            raise InputValidationError(
                f"{label} must be between 0 and {max_value} (inclusive)."
            )

    @beartype
    def _generate_complete_panel(
        self,
        *,
        n_samples: int,
        n_waves: int,
        feature_cols: list[str],
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        if n_samples <= 0:
            raise InputValidationError("Number of participants must be positive.")
        if n_waves < 2:
            raise InputValidationError("Number of waves must be at least 2.")

        sex_pool = ("female", "male")
        site_pool = ("Edinburgh", "Bristol", "Cardiff", "London")

        baseline_severity = rng.normal(0.0, 1.0, size=n_samples)
        slope = rng.normal(0.0, 0.35, size=n_samples)
        age_baseline = rng.integers(9, 15, size=n_samples)
        sex = rng.choice(sex_pool, size=n_samples, p=(0.52, 0.48))
        site = rng.choice(site_pool, size=n_samples)

        feature_shifts = {
            feature: float(idx) * 0.55 for idx, feature in enumerate(feature_cols)
        }
        records: list[dict[str, float | int | str]] = []

        for row_idx in range(n_samples):
            subject_id = row_idx + 1
            subject_base = baseline_severity[row_idx]
            subject_slope = slope[row_idx]
            for wave in range(1, n_waves + 1):
                time = float(wave)
                shared_signal = (
                    8.0
                    + 1.8 * subject_base
                    + subject_slope * (wave - 1)
                    + 0.22 * (wave - 1)
                )
                record: dict[str, float | int | str] = {
                    "subject_id": subject_id,
                    "wave": wave,
                    "time": time,
                    "age_baseline": int(age_baseline[row_idx]),
                    "sex": str(sex[row_idx]),
                    "site": str(site[row_idx]),
                }
                for feature in feature_cols:
                    value = (
                        shared_signal + feature_shifts[feature] + rng.normal(0.0, 0.9)
                    )
                    record[feature] = float(value)
                records.append(record)

        return pd.DataFrame.from_records(records)

    @beartype
    def _apply_missingness(
        self,
        *,
        data: pd.DataFrame,
        feature_cols: list[str],
        mechanism: str,
        missing_rate: float,
        dropout_rate: float,
        mar_strength: float,
        n_waves: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        target_cols = [col for col in feature_cols if col in data.columns]
        if not target_cols:
            raise InputValidationError("No requested feature columns exist in dataset.")

        result = data.copy()
        reference = data.copy()
        n_rows = len(result)

        if mechanism in {"mcar", "mixed"}:
            mask = rng.random((n_rows, len(target_cols))) < missing_rate
            for idx, col in enumerate(target_cols):
                result.loc[mask[:, idx], col] = np.nan

        if mechanism in {"mar", "mixed"}:
            mar_prob = self._mar_probability(
                data=reference,
                target_cols=target_cols,
                base_rate=missing_rate,
                mar_strength=mar_strength,
            )
            mask = rng.random((n_rows, len(target_cols))) < mar_prob.to_numpy()[:, None]
            for idx, col in enumerate(target_cols):
                result.loc[mask[:, idx], col] = np.nan

        dropout_wave_lookup: dict[int, int] = {}
        if mechanism in {"dropout", "mixed"}:
            subject_ids = np.sort(result["subject_id"].unique())
            dropout_subjects = subject_ids[rng.random(len(subject_ids)) < dropout_rate]
            for subject_id in dropout_subjects:
                dropout_wave = int(rng.integers(2, n_waves + 1))
                dropout_wave_lookup[int(subject_id)] = dropout_wave
                subject_mask = (result["subject_id"] == subject_id) & (
                    result["wave"] >= dropout_wave
                )
                result.loc[subject_mask, target_cols] = np.nan

        result["dropout_wave"] = (
            result["subject_id"].map(dropout_wave_lookup).fillna(0).astype(int)
        )
        result["missingness_mechanism"] = mechanism
        result["missing_feature_count"] = (
            result[target_cols].isna().sum(axis=1).astype(int)
        )
        return result

    @staticmethod
    @beartype
    def _mar_probability(
        *,
        data: pd.DataFrame,
        target_cols: list[str],
        base_rate: float,
        mar_strength: float,
    ) -> pd.Series:
        anchor_col = (
            "depressive_score" if "depressive_score" in data.columns else target_cols[0]
        )
        anchor = data[anchor_col].astype(float)
        anchor = anchor.fillna(float(anchor.median()))

        std = float(anchor.std(ddof=0))
        if std == 0.0:
            scaled_anchor = np.zeros(len(anchor), dtype=float)
        else:
            scaled_anchor = ((anchor - float(anchor.mean())) / std).to_numpy()

        waves = data["wave"].astype(float)
        wave_range = max(float(waves.max()) - float(waves.min()), 1.0)
        wave_component = ((waves - float(waves.min())) / wave_range).to_numpy()

        linear = -0.9 + mar_strength * scaled_anchor + 0.8 * wave_component
        sigmoid = 1.0 / (1.0 + np.exp(-linear))
        prob = np.clip(base_rate * (0.35 + sigmoid), 0.0, 0.98)
        return pd.Series(prob, index=data.index, dtype="float64")
