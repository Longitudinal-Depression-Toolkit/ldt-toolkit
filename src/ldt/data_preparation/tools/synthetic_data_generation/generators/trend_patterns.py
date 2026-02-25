from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from beartype import beartype

from ldt.utils.metadata import ComponentMetadata
from ldt.utils.templates.tools.data_preparation import DataPreparationTool


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
class TrendPatterns(DataPreparationTool):
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
    """

    metadata = ComponentMetadata(
        name="trend_patterns",
        full_name="Trend patterns trajectories",
        abstract_description=(
            "Generate persistently decreasing/decreasing/increasing/"
            "persistently increasing longitudinal patterns."
        ),
    )

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

    def prepare(
        self,
        *,
        n_samples: int | None = None,
        n_waves: int | None = None,
        wave_times: Iterable[float] | None = None,
        class_proportions: dict[str, float] | None = None,
        feature_specs: Iterable[TrajectoryFeatureSpec] | None = None,
        noise_sd: float | None = None,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """Generate a long-format synthetic longitudinal dataset.

        Participants are first assigned to classes using the configured class
        proportions. For each feature, class-specific intercepts/slopes are
        sampled, then wave-level values are generated with Gaussian noise.

        Args:
            n_samples (int | None): Optional override for sample count.
            n_waves (int | None): Optional override for number of waves.
            wave_times (Iterable[float] | None): Optional explicit time axis.
            class_proportions (dict[str, float] | None): Optional class
                proportions keyed by class name.
            feature_specs (Iterable[TrajectoryFeatureSpec] | None): Optional
                feature and class-shape specifications.
            noise_sd (float | None): Optional default per-observation noise.
            random_state (int | None): Optional random seed.

        Returns:
            pd.DataFrame: Long-format dataset with `subject_id`, `wave`, and
            one column per configured feature.

        Examples:
            ```python
            from ldt.data_preparation import TrendPatterns

            generator = TrendPatterns(n_samples=300, n_waves=5, random_state=7)
            data = generator.prepare()
            ```
        """
        resolved_n_samples = self.n_samples if n_samples is None else n_samples
        resolved_n_waves = self.n_waves if n_waves is None else n_waves
        resolved_random_state = (
            self.random_state if random_state is None else random_state
        )
        resolved_noise_sd = self.noise_sd if noise_sd is None else noise_sd
        resolved_wave_times_input = (
            self.wave_times if wave_times is None else wave_times
        )

        if resolved_n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if resolved_n_waves < 2:
            raise ValueError("n_waves must be at least 2")

        resolved_wave_times = np.array(list(resolved_wave_times_input), dtype=float)
        if len(resolved_wave_times) != resolved_n_waves:
            raise ValueError("wave_times must match n_waves length")

        rng = np.random.default_rng(resolved_random_state)
        resolved_feature_specs = list(
            feature_specs if feature_specs is not None else self.feature_specs or []
        )
        if not resolved_feature_specs:
            resolved_feature_specs = self.default_feature_specs()
        if not resolved_feature_specs:
            raise ValueError("feature_specs must include at least one feature.")
        feature_lookup = {
            feature.name: {spec.name: spec for spec in feature.class_specs}
            for feature in resolved_feature_specs
        }
        proportions = (
            class_proportions
            if class_proportions is not None
            else self.class_proportions or self.default_class_proportions()
        )
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
        assignments = rng.choice(class_names, size=resolved_n_samples, p=class_probs)

        records: list[dict[str, float | int | str]] = []
        for subject_id, class_name in enumerate(assignments, start=1):
            feature_params: dict[str, tuple[float, float, float]] = {}
            for feature in resolved_feature_specs:
                spec = feature_lookup[feature.name][class_name]
                intercept = rng.normal(spec.intercept_mean, spec.intercept_sd)
                slope = rng.normal(spec.slope_mean, spec.slope_sd)
                noise_sd = (
                    resolved_noise_sd if feature.noise_sd is None else feature.noise_sd
                )
                feature_params[feature.name] = (intercept, slope, noise_sd)
            for wave_index, time in enumerate(resolved_wave_times, start=1):
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


SyntheticWaveDataset = TrendPatterns
