# Data Preparation Tools

This page documents the Python tools used to create or reshape input data before preprocessing and modelling. Read each class docstring to understand accepted input formats, key arguments, and what artifact is produced at each step.

## Synthetic Data Generation

## ::: ldt.data_preparation.tools.synthetic_data_generation.generators.trend_patterns.TrendPatterns
    options:
      members:
        - prepare
        - default_class_specs
        - random_class_specs
        - default_feature_specs
        - default_class_proportions

## ::: ldt.data_preparation.tools.synthetic_data_generation.generators.event_shock_recovery.EventShockRecovery

## ::: ldt.data_preparation.tools.synthetic_data_generation.generators.piecewise_changepoint.PiecewiseChangepoint

## ::: ldt.data_preparation.tools.synthetic_data_generation.generators.missing_data_scenarios.MissingDataScenarios

## ::: ldt.data_preparation.tools.synthetic_data_generation.generators.harmonisation_challenge.HarmonisationChallenge

## Data Conversion

All converters below are part of the same **Data Conversion** family and each
one exposes `prepare(...)` (`run_mode="single"` for one file, `run_mode="folder"`
for batch conversion).

| Technique key | Converter class |
| --- | --- |
| `csv_to_parquet` | `CsvToParquet` |
| `csv_to_stata` | `CsvToStata` |
| `parquet_to_csv` | `ParquetToCsv` |
| `parquet_to_stata` | `ParquetToStata` |
| `stata_to_csv` | `StataToCsv` |
| `stata_to_parquet` | `StataToParquet` |

## ::: ldt.data_preparation.tools.data_conversion.converters.csv_to_parquet.CsvToParquet
    options:
      inherited_members: true
      members:
        - prepare

## ::: ldt.data_preparation.tools.data_conversion.converters.csv_to_stata.CsvToStata
    options:
      inherited_members: true
      members:
        - prepare

## ::: ldt.data_preparation.tools.data_conversion.converters.parquet_to_csv.ParquetToCsv
    options:
      inherited_members: true
      members:
        - prepare

## ::: ldt.data_preparation.tools.data_conversion.converters.parquet_to_stata.ParquetToStata
    options:
      inherited_members: true
      members:
        - prepare

## ::: ldt.data_preparation.tools.data_conversion.converters.stata_to_csv.StataToCsv
    options:
      inherited_members: true
      members:
        - prepare

## ::: ldt.data_preparation.tools.data_conversion.converters.stata_to_parquet.StataToParquet
    options:
      inherited_members: true
      members:
        - prepare
