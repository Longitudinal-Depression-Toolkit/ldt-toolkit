# LDT Toolkit Overview

## LDT-Toolkit, in a Nutshell

The `Longitudinal Depression Trajectories Toolkit (LDT-Toolkit)` is designed for social, medical, and clinical researchers working with repeated-measure data who need a stepping-stone path from raw cohort files to downstream modelling.

The initiative delivers two interconnected components:

- `ldt-toolkit` (this repository): the Python engine, with tools and reproducible workflows for longitudinal data preparation, preprocessing, and modelling.
- `ldt` (Go CLI): the no-code terminal interface that orchestrates the full toolkit experience end-to-end.

## Playground Tools

`Playground` tools are designed for fast experimentation on your own data. You can mix and match operations across:

- `Data preparation` (for example synthetic data generation and data conversion)
- `Data preprocessing` (for example trajectories building, harmonisation, imputation, aggregation)
- `Machine learning` (standard and longitudinal workflows, benchmarking, explainability)

This is the fastest path for custom studies where you want direct control over each step.

## Reproducibility Presets

`Presets` package stage-level reproducibility pipelines so studies can be rerun and compared consistently.

The long-term goal is a community-driven preset ecosystem. If you build a robust preset for your cohort or methodology, please submit it so others can reproduce and extend your workflow.

## Where to Get Started

- [Installation](installation.md): set up machine requirements and install both toolkit components.
- [Tutorials](../tutorials/index.md): learn the no-code CLI flow and advanced Python API flow.
- [API Reference](../api/index.md): browse tool docstrings and parameters by stage.
