from __future__ import annotations

import numpy as np
import pandas as pd
from beartype import beartype

from ldt.utils.errors import InputValidationError
from ldt.utils.metadata import ComponentMetadata
from ldt.utils.templates.tools.data_preparation import DataPreparationTool


@beartype
class EventShockRecovery(DataPreparationTool):
    """Synthetic data generator for shock-and-recovery trajectories.

    This generator creates long-format trajectories where each participant
    follows a baseline trend, experiences an acute shock at `shock_wave`,
    and then gradually recovers using an exponential decay curve.

    The generated pattern is useful when you want to benchmark methods that
    should detect abrupt deterioration followed by partial recovery over time.

    Examples:
        ```python
        from ldt.data_preparation import EventShockRecovery

        generator = EventShockRecovery()
        data = generator.prepare(
            n_samples=500,
            n_waves=6,
            random_state=42,
            feature_cols=["depressive_score", "anxiety_score"],
            shock_wave=3,
            shock_mean=4.0,
            recovery_rate=0.8,
            noise_sd=1.0,
        )
        ```
    """

    metadata = ComponentMetadata(
        name="event_shock_recovery",
        full_name="Event-shock + recovery trajectories",
        abstract_description=(
            "Generate trajectories with an event shock at a given wave "
            "followed by recovery."
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
        shock_wave: int,
        shock_mean: float,
        recovery_rate: float,
        noise_sd: float,
    ) -> pd.DataFrame:
        """Generate trajectories with a shock event followed by recovery.

        Each subject receives:
        1. A baseline intercept and slope.
        2. A subject-specific shock amplitude at `shock_wave`.
        3. Exponential recovery after the shock.

        This yields trajectories that are smooth before the event, sharply
        displaced at the event, and then partially or fully recover depending
        on `recovery_rate`.

        Args:
            n_samples (int): Number of samples.
            n_waves (int): Number of waves.
            random_state (int | None): Optional random seed for reproducibility.
            feature_cols (list[str]): Feature columns to simulate at each wave.
            shock_wave (int): Wave index where the event shock starts.
            shock_mean (float): Mean magnitude of the event shock.
            recovery_rate (float): Exponential recovery speed after the event.
            noise_sd (float): Observation noise standard deviation.

        Returns:
            pd.DataFrame: Long-format dataset with `subject_id`, `wave`, and
            one column per requested feature.

        Examples:
            ```python
            from ldt.data_preparation import EventShockRecovery

            generator = EventShockRecovery()
            data = generator.prepare(
                n_samples=200,
                n_waves=5,
                random_state=0,
                feature_cols=["depressive_score"],
                shock_wave=3,
                shock_mean=3.5,
                recovery_rate=0.9,
                noise_sd=0.8,
            )
            ```
        """
        if not feature_cols:
            raise InputValidationError("At least one feature column must be provided.")
        if shock_wave <= 1 or shock_wave >= n_waves:
            raise InputValidationError(
                "Shock wave must be between wave 2 and n_waves-1."
            )

        rng = np.random.default_rng(random_state)
        times = np.arange(1, n_waves + 1, dtype=float)

        records: list[dict[str, float | int]] = []
        for subject_id in range(1, n_samples + 1):
            baseline_intercept = rng.normal(6.0, 1.2)
            baseline_slope = rng.normal(0.0, 0.3)
            subject_shock = rng.normal(shock_mean, 0.8)

            for wave_index, time in enumerate(times, start=1):
                baseline = baseline_intercept + baseline_slope * time
                if wave_index < shock_wave:
                    shock_component = 0.0
                else:
                    elapsed = wave_index - shock_wave
                    shock_component = subject_shock * np.exp(-recovery_rate * elapsed)
                value = float(rng.normal(baseline + shock_component, noise_sd))

                record: dict[str, float | int] = {
                    "subject_id": subject_id,
                    "wave": wave_index,
                }
                for feature in feature_cols:
                    feature_shift = rng.normal(0.0, 0.3)
                    record[feature] = value + feature_shift
                records.append(record)

        return pd.DataFrame.from_records(records)
