from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from beartype import beartype


@beartype
@dataclass(frozen=True)
class TrajectoryPatternSpec:
    """Trajectory-level pattern definition used during synthesis.

    The intercept and slope means define the expected class trajectory shape,
    while the standard deviations control participant-level heterogeneity
    around that class-level pattern.

    Attributes:
        name (str): Class label for the trajectory pattern.
        intercept_mean (float): Mean baseline value at the first wave.
        slope_mean (float): Mean per-wave change.
        intercept_sd (float): Spread of baseline values across participants.
        slope_sd (float): Spread of slopes across participants.
    """

    name: str
    intercept_mean: float
    slope_mean: float
    intercept_sd: float = 1.0
    slope_sd: float = 0.2


@beartype
@dataclass(frozen=True)
class TrajectoryFeatureSpec:
    """Feature-level specification for trend-pattern generation.

    Attributes:
        name (str): Output feature name in the generated dataset.
        class_specs (list[TrajectoryPatternSpec]): Pattern specs for each class.
        noise_sd (float | None): Optional feature-specific noise override.
    """

    name: str
    class_specs: list[TrajectoryPatternSpec]
    noise_sd: float | None = None


@beartype
@dataclass
class SyntheticWaveDataset:
    """Synthetic longitudinal dataset generator with configurable trend classes.

    This generator creates long-format participant trajectories where each
    participant is sampled into a trajectory class (for example persistently
    high, decreasing, increasing, persistently low) and then receives feature
    values derived from class-specific intercept/slope distributions.

    Attributes:
        n_samples (int): Number of samples.
        n_waves (int): Number of waves.
        wave_times (Iterable[float] | None): Optional explicit time vector.
        class_proportions (dict[str, float] | None): Optional class prevalence.
        feature_specs (Iterable[TrajectoryFeatureSpec] | None): Feature definitions.
        noise_sd (float): Default observation noise for features.
        random_state (int | None): Optional random seed.

    Examples:
        ```python
        from ldt.data_preparation.tools.synthetic_data_generation.generators.trend_patterns import SyntheticWaveDataset

        generator = SyntheticWaveDataset(n_samples=500, n_waves=4, random_state=42)
        data = generator.generate()
        ```
    """

    n_samples: int
    n_waves: int = 4
    wave_times: Iterable[float] | None = None
    class_proportions: dict[str, float] | None = None
    feature_specs: Iterable[TrajectoryFeatureSpec] | None = None
    noise_sd: float = 1.0
    random_state: int | None = None

    def __post_init__(self) -> None:
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.n_waves < 2:
            raise ValueError("n_waves must be at least 2")
        if self.wave_times is None:
            object.__setattr__(self, "wave_times", np.arange(self.n_waves))
        wave_times = np.array(list(self.wave_times), dtype=float)
        if len(wave_times) != self.n_waves:
            raise ValueError("wave_times must match n_waves length")
        object.__setattr__(self, "wave_times", wave_times)

    def default_class_specs(self) -> list[TrajectoryPatternSpec]:
        """Return canonical class patterns used by default.

        The default set produces four typical longitudinal shapes:
        persistently_decreasing, decreasing, increasing, and
        persistently_increasing.

        Returns:
            list[TrajectoryPatternSpec]: Class pattern specifications.
        """
        return [
            TrajectoryPatternSpec(
                "persistently_decreasing", intercept_mean=11.0, slope_mean=0.0
            ),
            TrajectoryPatternSpec("decreasing", intercept_mean=9.0, slope_mean=-1.5),
            TrajectoryPatternSpec("increasing", intercept_mean=5.0, slope_mean=1.2),
            TrajectoryPatternSpec(
                "persistently_increasing", intercept_mean=3.0, slope_mean=0.0
            ),
        ]

    def random_class_specs(self) -> list[TrajectoryPatternSpec]:
        """Sample randomised class patterns for simulation variety.

        Intercepts and slopes are sampled from bounded random ranges while
        preserving the same semantic class labels as the default mode.

        Returns:
            list[TrajectoryPatternSpec]: Randomised class pattern specifications.
        """
        rng = np.random.default_rng(self.random_state)
        names = [
            "persistently_decreasing",
            "decreasing",
            "increasing",
            "persistently_increasing",
        ]
        slopes = [0.0, -rng.uniform(0.5, 2.0), rng.uniform(0.5, 2.0), 0.0]
        intercepts = sorted(rng.uniform(2.0, 12.0, size=4), reverse=True)
        return [
            TrajectoryPatternSpec(
                name=names[idx],
                intercept_mean=float(intercepts[idx]),
                slope_mean=float(slopes[idx]),
            )
            for idx in range(4)
        ]

    def default_feature_specs(self) -> list[TrajectoryFeatureSpec]:
        """Return the default feature specification set.

        Returns:
            list[TrajectoryFeatureSpec]: Default feature specifications.
        """
        return [
            TrajectoryFeatureSpec(
                name="depressive_score",
                class_specs=self.default_class_specs(),
            )
        ]

    def default_class_proportions(self) -> dict[str, float]:
        """Return default class prevalence used for sampling assignments.

        Returns:
            dict[str, float]: Mapping from class label to sampling proportion.
        """
        return {
            "persistently_decreasing": 0.75,
            "decreasing": 0.09,
            "increasing": 0.11,
            "persistently_increasing": 0.05,
        }

    def generate(self) -> pd.DataFrame:
        """Generate a long-format synthetic longitudinal dataset.

        Participants are first assigned to classes using the configured class
        proportions. For each feature, class-specific intercepts/slopes are
        sampled, then wave-level values are generated with Gaussian noise.

        Returns:
            pd.DataFrame: Long-format dataset with `subject_id`, `wave`, and
            one column per configured feature.

        Examples:
            ```python
            from ldt.data_preparation.tools.synthetic_data_generation.generators.trend_patterns import SyntheticWaveDataset

            generator = SyntheticWaveDataset(n_samples=300, n_waves=5, random_state=7)
            data = generator.generate()
            ```
        """
        rng = np.random.default_rng(self.random_state)
        feature_specs = list(self.feature_specs or self.default_feature_specs())
        if not feature_specs:
            raise ValueError("feature_specs must include at least one feature.")
        feature_lookup = {
            feature.name: {spec.name: spec for spec in feature.class_specs}
            for feature in feature_specs
        }
        proportions = self.class_proportions or self.default_class_proportions()
        total_prop = sum(proportions.values())
        if not np.isclose(total_prop, 1.0):
            proportions = {
                key: value / total_prop for key, value in proportions.items()
            }

        class_names = list(proportions.keys())
        for feature_name, specs in feature_lookup.items():
            missing = set(class_names) - set(specs.keys())
            if missing:
                raise ValueError(
                    "Feature specs for "
                    f"{feature_name} missing classes: {sorted(missing)}"
                )
        class_probs = np.array([proportions[name] for name in class_names])
        assignments = rng.choice(class_names, size=self.n_samples, p=class_probs)

        records: list[dict[str, float | int | str]] = []
        for subject_id, class_name in enumerate(assignments, start=1):
            feature_params: dict[str, tuple[float, float, float]] = {}
            for feature in feature_specs:
                spec = feature_lookup[feature.name][class_name]
                intercept = rng.normal(spec.intercept_mean, spec.intercept_sd)
                slope = rng.normal(spec.slope_mean, spec.slope_sd)
                noise_sd = (
                    self.noise_sd if feature.noise_sd is None else feature.noise_sd
                )
                feature_params[feature.name] = (intercept, slope, noise_sd)
            for wave_index, time in enumerate(self.wave_times, start=1):
                record: dict[str, float | int] = {
                    "subject_id": subject_id,
                    "wave": wave_index,
                }
                for feature_name, (
                    intercept,
                    slope,
                    noise_sd,
                ) in feature_params.items():
                    mean_value = intercept + slope * time
                    value = rng.normal(mean_value, noise_sd)
                    record[feature_name] = float(value)
                records.append(record)

        return pd.DataFrame.from_records(records)
