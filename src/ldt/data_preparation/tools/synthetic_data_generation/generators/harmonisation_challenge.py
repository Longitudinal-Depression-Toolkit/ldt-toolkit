from __future__ import annotations

import numpy as np
import pandas as pd
from beartype import beartype

from ldt.utils.errors import InputValidationError
from ldt.utils.metadata import ComponentMetadata
from ldt.utils.templates.tools.data_preparation import DataPreparationTool

_CATEGORY_VARIANTS: dict[str, dict[str, tuple[str, ...]]] = {
    "site_label": {
        "Edinburgh": ("Edinburgh", "edinburgh", "EDINBURGH", "Edinbrugh", "Edi nburgh"),
        "Bristol": ("Bristol", "bristol", "BRISTOL", "Bristol ", "Bris tol"),
        "Cardiff": ("Cardiff", "cardiff", "CARDIFF", "Cardif", "Card iff"),
        "London": ("London", "london", "LONDON", "Londn", "Lon don"),
    },
    "sex_label": {
        "Female": ("Female", "female", "F", "f", "woman", "Female "),
        "Male": ("Male", "male", "M", "m", "man", " Male"),
    },
    "treatment_group": {
        "Control": ("Control", "control", "CTRL", "ctl", "Control grp"),
        "Intervention": (
            "Intervention",
            "intervention",
            "INT",
            "interv",
            "Intervention grp",
        ),
    },
    "ethnicity_group": {
        "White": ("White", "white", "WHITE", "whte"),
        "Asian": ("Asian", "asian", "ASIAN", "Asn"),
        "Black": ("Black", "black", "BLACK", "Blk"),
        "Mixed": ("Mixed", "mixed", "MIXED", "mixd"),
    },
}


@beartype
class HarmonisationChallenge(DataPreparationTool):
    """Synthetic data generator for categorical harmonisation benchmarks.

    This generator creates a panel with clean canonical category labels
    (`site`, `sex`, `treatment`, `ethnicity`) and then injects realistic label
    noise such as typos, spacing variants, casing changes, abbreviations, and
    optional missing category labels.

    It is intended to test category harmonisation tools under controlled but
    messy real-world-like encoding conditions.

    Examples:
        ```python
        from ldt.data_preparation import HarmonisationChallenge

        generator = HarmonisationChallenge()
        # The public tool runner orchestrates panel generation + label corruption.
        ```
    """

    metadata = ComponentMetadata(
        name="harmonisation-challenge",
        full_name="Categorical harmonisation challenge",
        abstract_description=(
            "Generate messy category labels (typos, casing, spacing) to test "
            "`harmonise-categories` and `clean-dataset`."
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
        noise_rate: float,
        missing_label_rate: float,
        include_canonical_columns: bool,
    ) -> pd.DataFrame:
        """Generate a categorical harmonisation challenge dataset.

        Args:
            n_samples (int): Number of samples.
            n_waves (int): Number of waves.
            random_state (int | None): Optional random seed.
            feature_cols (list[str]): Numeric longitudinal feature columns.
            noise_rate (float): Fraction of labels receiving noisy variants.
            missing_label_rate (float): Fraction of labels replaced with missing values.
            include_canonical_columns (bool): Keep canonical side-by-side columns.

        Returns:
            pd.DataFrame: Generated long-format dataset with noisy categorical labels.
        """

        if not feature_cols:
            raise InputValidationError(
                "At least one numeric feature column is required."
            )
        self._validate_unit_interval(
            value=noise_rate,
            label="noise_rate",
            max_value=1.0,
        )
        self._validate_unit_interval(
            value=missing_label_rate,
            label="missing_label_rate",
            max_value=0.5,
        )

        rng = np.random.default_rng(random_state)
        data = self._generate_base_panel(
            n_samples=n_samples,
            n_waves=n_waves,
            feature_cols=feature_cols,
            rng=rng,
        )
        data = self._inject_harmonisation_noise(
            data=data,
            noise_rate=noise_rate,
            missing_label_rate=missing_label_rate,
            include_canonical_columns=include_canonical_columns,
            rng=rng,
        )
        return data.drop(columns=["time", "class"], errors="ignore")

    @staticmethod
    @beartype
    def _parse_feature_columns(raw: str) -> list[str]:
        parsed = [item.strip() for item in raw.split(",") if item.strip()]
        if not parsed:
            raise InputValidationError(
                "At least one numeric feature column is required."
            )
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
    def _generate_base_panel(
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

        subject_ids = np.arange(1, n_samples + 1)
        site_canonical = rng.choice(
            ("Edinburgh", "Bristol", "Cardiff", "London"),
            size=n_samples,
            p=(0.28, 0.24, 0.2, 0.28),
        )
        sex_canonical = rng.choice(("Female", "Male"), size=n_samples, p=(0.53, 0.47))
        treatment_canonical = rng.choice(
            ("Control", "Intervention"),
            size=n_samples,
            p=(0.52, 0.48),
        )
        ethnicity_canonical = rng.choice(
            ("White", "Asian", "Black", "Mixed"),
            size=n_samples,
            p=(0.63, 0.17, 0.12, 0.08),
        )
        baseline_severity = rng.normal(0.0, 1.0, size=n_samples)
        slope = rng.normal(0.0, 0.4, size=n_samples)

        records: list[dict[str, float | int | str]] = []
        for idx, subject_id in enumerate(subject_ids):
            base = baseline_severity[idx]
            trend = slope[idx]
            treatment_bonus = (
                0.45 if treatment_canonical[idx] == "Intervention" else 0.0
            )

            for wave in range(1, n_waves + 1):
                time = float(wave)
                signal = (
                    7.2 + 1.7 * base + trend * (wave - 1) - treatment_bonus * (wave - 1)
                )
                record: dict[str, float | int | str] = {
                    "subject_id": int(subject_id),
                    "wave": wave,
                    "time": time,
                    "site_label": str(site_canonical[idx]),
                    "sex_label": str(sex_canonical[idx]),
                    "treatment_group": str(treatment_canonical[idx]),
                    "ethnicity_group": str(ethnicity_canonical[idx]),
                }
                for feature_idx, feature in enumerate(feature_cols):
                    shift = 0.65 * feature_idx
                    value = signal + shift + rng.normal(0.0, 0.95)
                    record[feature] = float(value)
                records.append(record)

        return pd.DataFrame.from_records(records)

    @beartype
    def _inject_harmonisation_noise(
        self,
        *,
        data: pd.DataFrame,
        noise_rate: float,
        missing_label_rate: float,
        include_canonical_columns: bool,
        rng: np.random.Generator,
    ) -> pd.DataFrame:
        result = data.copy()
        categorical_columns = list(_CATEGORY_VARIANTS.keys())

        if include_canonical_columns:
            for column in categorical_columns:
                result[f"{column}_canonical"] = result[column].copy()

        for column in categorical_columns:
            variants = _CATEGORY_VARIANTS[column]
            noisy_values: list[str | None] = []
            for canonical in result[column].astype(str):
                if rng.random() < missing_label_rate:
                    noisy_values.append(None)
                    continue
                if rng.random() < noise_rate:
                    options = variants.get(canonical, (canonical,))
                    noisy_values.append(str(rng.choice(options)))
                    continue
                noisy_values.append(canonical)
            result[column] = pd.Series(noisy_values, dtype="object")

        result["label_noise_rate"] = float(noise_rate)
        result["missing_label_rate"] = float(missing_label_rate)
        return result
