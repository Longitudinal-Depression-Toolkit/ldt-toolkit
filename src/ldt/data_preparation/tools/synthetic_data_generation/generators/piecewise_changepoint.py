from __future__ import annotations

import numpy as np
import pandas as pd
from beartype import beartype

from ldt.utils.errors import InputValidationError
from ldt.utils.metadata import ComponentMetadata
from ldt.utils.templates.tools.data_preparation import DataPreparationTool


@beartype
class PiecewiseChangepoint(DataPreparationTool):
    """Synthetic data generator for piecewise changepoint trajectories.

    This generator creates trajectories with one structural break. Before the
    changepoint, observations evolve with a pre-change slope. After the
    changepoint, the slope shifts to a new regime while preserving continuity.

    This pattern is useful for benchmarking methods that should capture
    behavioural phase changes across time.

    Examples:
        ```python
        from ldt.data_preparation import PiecewiseChangepoint

        generator = PiecewiseChangepoint()
        data = generator.prepare(
            n_samples=500,
            n_waves=6,
            random_state=42,
            feature_cols=["depressive_score"],
            changepoint_wave=3,
            noise_sd=1.0,
            pre_slope_sd=0.5,
            post_slope_sd=0.5,
        )
        ```
    """

    metadata = ComponentMetadata(
        name="piecewise_changepoint",
        full_name="Piecewise changepoint trajectories",
        abstract_description=(
            "Generate trajectories with a changepoint: pre-change slope "
            "then post-change slope."
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
        changepoint_wave: int,
        noise_sd: float,
        pre_slope_sd: float,
        post_slope_sd: float,
    ) -> pd.DataFrame:
        """Generate trajectories with a piecewise slope changepoint.

        For each subject, the method samples an intercept and slope before the
        changepoint, then applies a post-changepoint slope adjustment.
        The value at the changepoint is held continuous so only trajectory
        direction/rate changes after the break.

        Args:
            n_samples (int): Number of samples.
            n_waves (int): Number of waves.
            random_state (int | None): Optional random seed for reproducibility.
            feature_cols (list[str]): Feature columns to simulate at each wave.
            changepoint_wave (int): Wave where slope regime switches.
            noise_sd (float): Observation noise standard deviation.
            pre_slope_sd (float): Standard deviation of pre-change slopes.
            post_slope_sd (float): Standard deviation of slope-shift offsets.

        Returns:
            pd.DataFrame: Long-format dataset with `subject_id`, `wave`, and
            one column per requested feature.

        Examples:
            ```python
            from ldt.data_preparation import PiecewiseChangepoint

            generator = PiecewiseChangepoint()
            data = generator.prepare(
                n_samples=300,
                n_waves=7,
                random_state=7,
                feature_cols=["depressive_score", "sleep_score"],
                changepoint_wave=4,
                noise_sd=0.9,
                pre_slope_sd=0.4,
                post_slope_sd=0.6,
            )
            ```
        """
        if not feature_cols:
            raise InputValidationError("At least one feature column must be provided.")
        if changepoint_wave <= 1 or changepoint_wave >= n_waves:
            raise InputValidationError(
                "Changepoint must be between wave 2 and n_waves-1."
            )

        rng = np.random.default_rng(random_state)
        times = np.arange(1, n_waves + 1, dtype=float)

        records: list[dict[str, float | int]] = []
        for subject_id in range(1, n_samples + 1):
            intercept = rng.normal(7.0, 1.5)
            pre_slope = rng.normal(0.0, pre_slope_sd)
            post_slope_delta = rng.normal(0.0, post_slope_sd)
            post_slope = pre_slope + post_slope_delta

            values_by_time: dict[int, float] = {}
            cp_time = float(changepoint_wave)
            level_at_cp = intercept + pre_slope * cp_time
            for wave_index, time in enumerate(times, start=1):
                if wave_index <= changepoint_wave:
                    mean_value = intercept + pre_slope * time
                else:
                    mean_value = level_at_cp + post_slope * (time - cp_time)
                values_by_time[wave_index] = float(rng.normal(mean_value, noise_sd))

            for wave_index, _time in enumerate(times, start=1):
                record: dict[str, float | int] = {
                    "subject_id": subject_id,
                    "wave": wave_index,
                }
                for feature in feature_cols:
                    feature_shift = rng.normal(0.0, 0.25)
                    record[feature] = values_by_time[wave_index] + feature_shift
                records.append(record)

        return pd.DataFrame.from_records(records)
