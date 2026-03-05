# MCS by LEAP

<div class="mcs-hero" markdown>

Millennium Cohort Study (MCS) reproducibility study by LEAP, designed to move from raw wave files to an ML-ready longitudinal exploration with decomposed (1) preparation, (2) preprocessing and (3) modelling stages.

**This page is the practical guide for this preset.**

Follow the flow in order: `Prepare -> Preprocess -> Model`.

- `Prepare` ingests raw wave `.dta` cohorts and creates canonical long/wide dataset(s).
- `Preprocess` applies cleaning, harmonise, standardise, and other policy to the `wide` dataset to be ML-ready.
- `Model` entrypoint exists, but the LEAP modelling preset is currently marked as incoming.

</div>

## Process Overview Visualisation

<div class="mcs-stage-viz" data-mcs-viz>
  <div class="admonition info mcs-stage-hint" data-stage-hint>
    <p class="admonition-title">Pick a stage to visualise</p>
    <p>Click one card below to render its stage-specific flowchart, then click the displayed flowchart to expand it.</p>
  </div>

  <div class="mcs-stage-card-grid" role="tablist" aria-label="MCS stage visualisation selector">
    <button type="button" class="mcs-stage-card" data-stage-target="prepare" aria-pressed="false">
      <span class="mcs-stage-card__title">1) Prepare MCS by LEAP</span>
      <span class="mcs-stage-card__desc">Raw `.dta` wave inputs -> canonical long/wide outputs.</span>
    </button>
    <button type="button" class="mcs-stage-card" data-stage-target="preprocess" aria-pressed="false">
      <span class="mcs-stage-card__title">2) Preprocess MCS by LEAP</span>
      <span class="mcs-stage-card__desc">Policy-driven cleaning and encoding for ML-ready wide data.</span>
    </button>
    <button type="button" class="mcs-stage-card" data-stage-target="model" aria-pressed="false">
      <span class="mcs-stage-card__title">3) Model MCS by LEAP</span>
      <span class="mcs-stage-card__desc">Reserved modelling entrypoint for downstream LEAP workflow.</span>
      <code>incoming preset</code>
    </button>
  </div>

  <div class="mcs-stage-panel" data-stage-panel="prepare" hidden>
    <div class="mcs-stage-diagram" data-stage-diagram="prepare"></div>
    <script type="text/plain" data-stage-source="prepare">
flowchart LR
    W1["Wave path validation"] --> W2["Subject construction"]
    W2 --> W3["Feature preparation"]
    W3 --> W4["Composite construction"]
    W4 --> W5["Output formatting"]
    W5 --> L["Long output"]
    W5 --> V["Wide output (handoff)"]
    </script>
  </div>

  <div class="mcs-stage-panel" data-stage-panel="preprocess" hidden>
    <div class="mcs-stage-diagram" data-stage-diagram="preprocess"></div>
    <script type="text/plain" data-stage-source="preprocess">
flowchart LR
    P0["Target row policy"] --> P1["Structural cleanup"]
    P1 --> P2["Sentinel map + replacement"]
    P2 --> P3["Leakage policy"]
    P3 --> P4["Finalisation diagnostics"]
    P4 --> P5["Encoding policy"]
    P5 --> O["ML-ready wide output"]
    P5 --> A["Audit tables"]
    </script>
  </div>

  <div class="mcs-stage-panel" data-stage-panel="model" hidden>
    <p><code>Work in progress...</code></p>
  </div>
</div>

## Walk Step-by-Step Through (CLI & API)

=== "1) Prepare MCS by LEAP"

    **CLI**

    Use this stage to ingest selected raw waves and produce canonical outputs.

    Target lineage created in this stage: raw wave emotion columns -> `EMOTION` root -> wide target/history columns (`EMOTION_w<wave>`).

    ```bash
    ldt
    # Navigate: Data Preparation -> Presets / Reproducibility -> Prepare MCS by LEAP
    ```

    Provide one raw cohort path per selected wave:

    | Wave | Path example |
    |---|---|
    | W1 | `/path/to/MCS/W1/UKDA-4683-stata/stata` |
    | W2 | `/path/to/MCS/W2/UKDA-5350-stata/stata` |
    | W3 | `/path/to/MCS/W3/UKDA-5795-stata/stata` |
    | W4 | `/path/to/MCS/W4/UKDA-6411-stata/stata` |
    | W5 | `/path/to/MCS/W5/UKDA-7464-stata/stata` |
    | W6 | `/path/to/MCS/W6/UKDA-8156-stata/stata` |
    | W7 | `/path/to/MCS/W7/UKDA-8682-stata/stata` |

    Recommended runtime choices:

    | Parameter | Recommended value |
    |---|---|
    | `waves` | `ALL` (or explicit subset like `W1,W2,W3`) |
    | `output_format` | `wide` or `long_and_wide` |
    | `save_final_output` | `true` |
    | `wide_suffix_prefix` | `_w` |

    Expected handoff artefact:

    ```text
    data/processed/MCS/mcs_longitudinal_wide.csv
    ```

    **API**

    ```python
    from ldt.data_preparation import PrepareMCSByLEAP

    prepare = PrepareMCSByLEAP().fit_prepare(
        waves="ALL",
        wave_inputs={
            "W1": "/path/to/W1/stata",
            "W2": "/path/to/W2/stata",
            "W3": "/path/to/W3/stata",
            "W4": "/path/to/W4/stata",
            "W5": "/path/to/W5/stata",
            "W6": "/path/to/W6/stata",
            "W7": "/path/to/W7/stata",
        },
        output_format="wide",
        save_final_output=True,
    )

    print(prepare.wide_output_path)
    ```

=== "2) Preprocess MCS by LEAP"

    !!! warning "Mandatory dependency"

        `Preprocess MCS by LEAP` requires the **wide** artefact from `Prepare MCS by LEAP`.
        A long-only output cannot be used as preprocessing input for this preset.
        Final target for this preset is `EMOTION_w7`.

    **CLI**

    Use this stage to convert prepared wide data into ML-ready wide output with audit logs.

    ```bash
    ldt
    # Navigate: Data Preprocessing -> Presets / Reproducibility -> Preprocess MCS by LEAP
    ```

    Set preprocessing input to the wide output from step 1:

    ```text
    input_path = data/processed/MCS/mcs_longitudinal_wide.csv
    ```

    Expected outputs:

    - `data/processed/MCS/mcs_longitudinal_wide_preprocessed.csv`
    - `data/processed/MCS/preprocess_logs/` (audit tables + `pipeline_summary.json`)

    **API**

    ```python
    from ldt.data_preprocessing import PreprocessMCSByLEAP

    preprocess = PreprocessMCSByLEAP().fit_preprocess(
        input_path="data/processed/MCS/mcs_longitudinal_wide.csv",
    )

    print(preprocess.output_data.shape)
    print(preprocess.output_path)
    print(preprocess.audit_output_dir)
    ```

=== "3) Model MCS by LEAP"

    **CLI**

    `Work in progress...`

    **API**

    `Work in progress...`

## Stage-by-Stage Configuration

=== "PrepareMCSByLEAP (Config)"

    | Prepare stage | Default behaviour | Main policy/config |
    |---|---|---|
    | 1. Wave path validation | Validates selected wave folder against expected `.dta` dataset manifests | `stage_1_wave_paths/datasets.yaml` |
    | 2. Subject construction | Creates child-level anchors and merges child/family/parent/link roles with stable keys | `stage_2_subjects/subject_keys.yaml` |
    | 3. Feature preparation | Resolves canonical longitudinal and non-longitudinal feature mappings | `stage_3_features/longitudinal_features.yaml`, `stage_3_features/non_longitudinal_features.yaml` |
    | 4. Composite features | Applies configured composites (`sum`, `mean`, `median`, `min`, `max`, `coalesce`) | `stage_4_composites/composites.yaml` |
    | 5. Output formatting | Emits `long`, `wide`, or `long_and_wide` outputs with suffixing (`_w1.._w7`) | runtime + `defaults.yaml` |

    Target canonicalisation used by default:

    | Wave | Raw source column | Canonical root | Wide output column |
    |---|---|---|---|
    | W2 | `BEMOTION` | `EMOTION` | `EMOTION_w2` |
    | W3 | `CEMOTION` | `EMOTION` | `EMOTION_w3` |
    | W4 | `DDEMOTION` | `EMOTION` | `EMOTION_w4` |
    | W5 | `EEMOTION` | `EMOTION` | `EMOTION_w5` |
    | W6 | `FEMOTION` | `EMOTION` | `EMOTION_w6` |
    | W7 | `GEMOTION` | `EMOTION` | `EMOTION_w7` |

    Stage details (default behaviour):

    ??? abstract "Stage 1 - Wave path validation"

        - Validates selected wave directories against expected MCS `.dta` manifests.
        - Ensures required dataset groups for each selected wave are present before loading.
        - Prevents partial-stage execution when mandatory wave assets are missing.
        - Main config: `stage_1_wave_paths/datasets.yaml`.

    ??? abstract "Stage 2 - Subject construction"

        - Builds child-level anchor rows and stable join keys.
        - Merges child/family/parent/link roles into a subject-aligned dataset.
        - Preserves reproducible identifiers used downstream by feature mapping and output formatting.
        - Main config: `stage_2_subjects/subject_keys.yaml`.

    ??? abstract "Stage 3 - Feature preparation"

        - Maps raw wave-specific columns to canonical longitudinal roots.
        - Applies both longitudinal and non-longitudinal feature maps.
        - Canonicalises wave emotion sources (`BEMOTION`..`GEMOTION`) to root `EMOTION`.
        - Main config:
          - `stage_3_features/longitudinal_features.yaml`
          - `stage_3_features/non_longitudinal_features.yaml`

    ??? abstract "Stage 4 - Composite features"

        - Computes configured composite variables from prepared base features.
        - Uses deterministic reducers (`sum`, `mean`, `median`, `min`, `max`, `coalesce`) per policy.
        - Adds derived features before final long/wide formatting.
        - Main config: `stage_4_composites/composites.yaml`.

    ??? abstract "Stage 5 - Output formatting"

        - Produces `long`, `wide`, or `long_and_wide` artefacts according to runtime choice.
        - Uses configured wave suffixing convention (default `_w`, yielding columns like `EMOTION_w7` in wide output).
        - Persists final outputs and wave-level outputs according to runtime save settings.
        - Main config/runtime: `defaults.yaml` + runtime parameters.

=== "PreprocessMCSByLEAP (Config)"

    Stage-by-stage default policy:

    | Stage | What happens by default | Config file |
    |---|---|---|
    | 0. Target rows | Keeps rows where `EMOTION_w7` is observed, using sentinel-aware missingness | `stage_0_target_rows/policy.yaml` |
    | 1. Structural | Applies ID/target-history policy and drops unusable predictors by missingness/constant checks | `stage_1_structural/rules.yaml` |
    | 2. Sentinels | Builds final sentinel-to-NaN map per root/column and applies replacements | `stage_2_sentinels/sentinels.yaml` |
    | 3. Leakage | Removes configured leakage features while preserving final target | `stage_3_modelling_policy/policy.yaml` |
    | 4. Finalisation | Profiles predictor missingness and computes high-correlation pairs (dry-run by default) | `stage_4_finalisation/policy.yaml` |
    | 5. Encoding | Applies strict root-level binary/ordinal/continuous/nominal policy and cleanup | `stage_5_encoding_policy/policy.yaml` |

    Stage details (default behaviour):

    ??? abstract "Stage 0 - Target rows"

        - Target is fixed to `EMOTION_w7`.
        - `require_target_non_missing: true` keeps only rows with observed target values.
        - Missingness is sentinel-aware (for example, configured target sentinels are treated as missing).
        - Core audit output: `stage0_target_row_policy.csv`.

    ??? abstract "Stage 1 - Structural"

        - Keeps `CHID` as modelling identifier and removes secondary IDs (`MCSID`, `CNUM`) if present.
        - Removes configured target history predictors (`EMOTION_w1..EMOTION_w6`).
        - Drops baseline-only over-time columns when they are effectively missing.
        - Runs sentinel-aware column profiling and drops:
          - all-effectively-missing predictors,
          - single-valid-value predictors,
          - partial-missing predictors above threshold (`0.60`, enabled by default).
        - Core audit outputs:
          - `stage1_column_profile.csv`
          - `stage1_obvious_remove_features.csv`
          - `stage1_partial_missing_over_threshold.csv`

    ??? abstract "Stage 2 - Sentinels"

        - Starts from configured sentinel code space (`-9..-1`, `96..99`, `996..999`).
        - Uses dictionary label-term detection to identify missing-like codes.
        - Applies root-level overrides where configured (for example `ADOEDE00`, `ADACAQ00`).
        - Falls back to observed sentinel values when no label terms match.
        - Replaces final sentinel codes with `NaN` (`apply_nan_replacement: true`).
        - Core audit outputs:
          - `stage2_sentinel_final_map.csv`
          - `stage2_sentinel_final_map_compact.csv`
          - `stage2_sentinel_replacement_summary.csv`

    ??? abstract "Stage 3 - Leakage"

        - Applies leakage policy only (encoding is intentionally deferred to Stage 5).
        - Removes configured leakage columns if present.
        - Keeps the final target (`EMOTION_w7`) for downstream modelling.
        - Core audit outputs:
          - `stage3_leakage_policy.csv`
          - `stage3_leakage_dropped_features.csv`

    ??? abstract "Stage 4 - Finalisation"

        - Profiles predictor missingness on target-filtered rows.
        - Computes high-correlation predictor pairs with `spearman` and `abs_threshold: 0.98`.
        - Correlation pruning is disabled by default (`apply: false`), so this stage is diagnostic unless enabled.
        - Core audit outputs:
          - `stage4_predictor_missingness_target_filtered.csv`
          - `stage4_high_correlation_pairs.csv`

    ??? abstract "Stage 5 - Encoding"

        - Uses strict root-level policy (`fail_on_unconfigured_predictor: true`).
        - Applies one explicit encoding mode per configured root:
          - `binary_to_01`
          - `ordinal` (`identity` or `rank`)
          - `nominal_one_hot`
          - `continuous_numeric_keep`
        - Drops nominal source columns after one-hot expansion.
        - Keeps source missingness as `NaN` in generated one-hot columns.
        - Infers and reclassifies some single-wave nominal one-hot columns as non-longitudinal labels.
        - Drops constant predictors after encoding.
        - Core audit outputs:
          - `stage5_feature_policy_assignment.csv`
          - `stage5_nominal_one_hot_created_columns.csv`
          - `stage5_constant_predictor_dropped_features.csv`
          - `stage5_encoding_summary_counts.csv`

    Preprocess runtime defaults:

    | Parameter | Default |
    |---|---|
    | `input_path` | required |
    | `output_path` | `data/processed/MCS/mcs_longitudinal_wide_preprocessed.csv` |
    | `audit_output_dir` | `data/processed/MCS/preprocess_logs` |
    | `save_final_output` | `true` |
    | `save_audit_tables` | `true` |
    | `show_summary_logs` | `true` |

=== "ModelMCSByLEAP (Config)"

    `Work in progress...`

## Request Data

<div class="mcs-hero" markdown>

- Millennium Cohort Study (MCS): [https://cls.ucl.ac.uk/cls-studies/millennium-cohort-study/](https://cls.ucl.ac.uk/cls-studies/millennium-cohort-study/)
- Data used in this reproducibility study were acquired by [https://life-epi-psych.github.io](https://life-epi-psych.github.io).
- Data access can be requested personally from Dr Alex Kwong. See: [https://life-epi-psych.github.io/pages/people](https://life-epi-psych.github.io/pages/people)

</div>

## API Reference

=== "PrepareMCSByLEAP (API)"

    ::: ldt.data_preparation.presets.prepare_mcs_by_leap.tool.PrepareMCSByLEAP

=== "PreprocessMCSByLEAP (API)"

    ::: ldt.data_preprocessing.presets.preprocess_mcs_by_leap.tool.PreprocessMCSByLEAP

=== "ModelMCSByLEAP (API)"

    ::: ldt.machine_learning.presets.model_mcs_by_leap.tool.ModelMCSByLEAP
