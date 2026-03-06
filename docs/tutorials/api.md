# API Tutorial

This is the advanced usage path for users who want to run toolkit operations in JupyterLab or their own Python scripts.

## A) Import the library and stages

API references: [`data_preparation`](../api/data-preparation.md), [`data_preprocessing`](../api/data-preprocessing.md), [`machine_learning` (Standard)](../api/standard-machine-learning.md), [`machine_learning` (Longitudinal)](../api/longitudinal-machine-learning.md), [`TrendPatterns`](../api/data-preparation.md), [`BuildTrajectories`](../api/data-preprocessing.md), [`StandardMachineLearning`](../api/standard-machine-learning.md)

```python
from ldt import data_preparation, data_preprocessing, machine_learning

# or import tools directly
from ldt.data_preparation import TrendPatterns
from ldt.data_preprocessing import BuildTrajectories
from ldt.machine_learning import LongitudinalMachineLearning, StandardMachineLearning
```

## B) Getting Started Example 1 - Generate Synthetic Data (Multi-Technique)

API references: [`TrendPatterns`](../api/data-preparation.md), [`EventShockRecovery`](../api/data-preparation.md), [`MissingDataScenarios`](../api/data-preparation.md)

```python
from pathlib import Path

from ldt.data_preparation import EventShockRecovery, MissingDataScenarios, TrendPatterns

out = Path("/path/to/your/project/data")
out.mkdir(parents=True, exist_ok=True)

TrendPatterns(n_samples=400, n_waves=5, random_state=7).prepare().to_csv(
    out / "synthetic_trend_patterns.csv", index=False
)

EventShockRecovery().prepare(
    n_samples=400,
    n_waves=5,
    random_state=7,
    feature_cols=["depressive_score"],
    shock_wave=3,
    shock_mean=3.5,
    recovery_rate=0.9,
    noise_sd=0.8,
).to_csv(out / "synthetic_event_shock.csv", index=False)

MissingDataScenarios().prepare(
    n_samples=400,
    n_waves=5,
    random_state=7,
    feature_cols=["depressive_score", "sleep_score"],
    mechanism="mixed",
    missing_rate=0.20,
    dropout_rate=0.10,
    mar_strength=1.0,
).to_csv(out / "synthetic_with_missing.csv", index=False)
```

## C) Getting Started Example 2 - Build Trajectories + ShowTable

API references: [`BuildTrajectories`](../api/data-preprocessing.md), [`ShowTable`](../api/data-preprocessing.md)

```python
from pathlib import Path

from ldt.data_preprocessing import BuildTrajectories, ShowTable

input_long = Path("/path/to/your/project/data/synthetic_trend_patterns.csv")
out = Path("/path/to/your/project/outputs")
out.mkdir(parents=True, exist_ok=True)

BuildTrajectories().fit_preprocess(
    mode="from_scratch",
    input_path=input_long,
    output_path=out / "trajectories_dtw_kmeans.csv",
    id_col="subject_id",
    time_col="wave",
    value_cols=["depressive_score"],
    builder="dtw_kmeans",
    n_trajectories=4,
)
BuildTrajectories().fit_preprocess(
    mode="from_scratch",
    input_path=input_long,
    output_path=out / "trajectories_clusterMLD.csv",
    id_col="subject_id",
    time_col="wave",
    value_cols=["depressive_score"],
    builder="clusterMLD",
    n_trajectories=4,
)

ShowTable().fit_preprocess(
    input_path=out / "trajectories_clusterMLD.csv",
    output_html=out / "trajectories_clusterMLD_report.html",
    open_browser=False,
)
```

## D) Getting Started Example 3 - End-to-End: Synthetic Data to Standard ML

API references: [`MissingDataScenarios`](../api/data-preparation.md), [`CleanDataset`](../api/data-preprocessing.md), [`MissingImputation`](../api/data-preprocessing.md), [`BuildTrajectories`](../api/data-preprocessing.md), [`AggregateLongToCrossSectional`](../api/data-preprocessing.md), [`CombineDatasetWithTrajectories`](../api/data-preprocessing.md), [`StandardMachineLearning`](../api/standard-machine-learning.md)

```python
from pathlib import Path

from ldt.data_preparation import MissingDataScenarios
from ldt.data_preprocessing import (
    AggregateLongToCrossSectional,
    BuildTrajectories,
    CleanDataset,
    CombineDatasetWithTrajectories,
    MissingImputation,
)
from ldt.machine_learning import StandardMachineLearning

root = Path("/path/to/your/project")
raw_long = root / "data/synthetic_long_with_missing.csv"
clean_long = root / "outputs/long_clean.csv"
imputed_long = root / "outputs/long_imputed.csv"
trajectories = root / "outputs/trajectories.csv"
cross_sectional = root / "outputs/cross_sectional.csv"
model_ready = root / "outputs/model_ready.csv"

long_df = MissingDataScenarios().prepare(
    n_samples=1200,
    n_waves=6,
    random_state=42,
    feature_cols=["depressive_score", "sleep_score", "anxiety_score"],
    mechanism="mixed",
    missing_rate=0.20,
    dropout_rate=0.15,
    mar_strength=1.10,
)
raw_long.parent.mkdir(parents=True, exist_ok=True)
long_df.to_csv(raw_long, index=False)

CleanDataset().fit_preprocess(input_path=raw_long, output_path=clean_long)
MissingImputation().fit_preprocess(
    technique="median_imputation",
    input_path=clean_long,
    output_path=imputed_long,
)
BuildTrajectories().fit_preprocess(
    mode="from_scratch",
    input_path=imputed_long,
    output_path=trajectories,
    id_col="subject_id",
    time_col="wave",
    value_cols=["depressive_score"],
    builder="clusterMLD",
    n_trajectories=4,
)
AggregateLongToCrossSectional().fit_preprocess(
    input_path=imputed_long,
    output_path=cross_sectional,
    subject_id_col="subject_id",
    numeric_columns=["depressive_score", "sleep_score", "anxiety_score", "age_baseline"],
    numeric_agg="mean",
)
CombineDatasetWithTrajectories().fit_preprocess(
    input_original_data_path=cross_sectional,
    input_trajectories_data_path=trajectories,
    output_path=model_ready,
    original_id_col="subject_id",
    trajectory_id_col="subject_id",
    merge_type="left",
    trajectory_columns=["trajectory_id", "trajectory_name"],
)

ml_result = StandardMachineLearning().fit_predict(
    technique="run_experiment",
    input_path=model_ready,
    target_column="trajectory_id",
    feature_columns="depressive_score,sleep_score,anxiety_score,age_baseline",
    estimator_key="random_forest",
    metric_keys="accuracy,f1_macro",
    cv_folds=5,
    validation_split="none",
    multiclass_mode="multiclass",
    random_seed=42,
    output_dir=str(root / "outputs/standard_ml"),
)
print(ml_result["mean_score"], ml_result["report_path"])
```

## E) Getting Started Example 4 - Longitudinal ML with Suffix-Inferred Feature Groups

API references: [`LongitudinalMachineLearning`](../api/longitudinal-machine-learning.md)

```python
from pathlib import Path

from ldt.machine_learning import LongitudinalMachineLearning

root = Path("/path/to/your/project")
wide_dataset = root / "data/model_ready_longitudinal.csv"

result = LongitudinalMachineLearning().fit_predict(
    technique="run_experiment",
    input_path=wide_dataset,
    target_column="depression_status",
    feature_columns="mood_w1,mood_w2,mood_w3,sleep_w1,sleep_w2,sleep_w3,sex",
    feature_groups_mode="suffix",
    feature_groups_suffix="_w",
    non_longitudinal_mode="auto",
    estimator_key="merwav_time_plus__lexico_random_forest",
    metric_keys="accuracy,f1_macro",
    cv_folds=5,
    validation_split="none",
    random_seed=42,
)
print(result["mean_score"], result["report_path"])
```

Manual and preset alternatives also work:

- Manual groups: `feature_groups_mode="manual", feature_groups="[[0,1,2],[3,4,5]]"` or `feature_groups="mood_w1,mood_w2,mood_w3;sleep_w1,sleep_w2,sleep_w3"`
- scikit-longitudinal preset: `feature_groups_mode="preset", feature_groups_preset="elsa"`
