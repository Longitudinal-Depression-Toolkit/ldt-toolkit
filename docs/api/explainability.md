# Explainability Tools

This page contains explainability tools for interpreting model behaviour after training. Use the class docstring below to check required model inputs, configurable explanation options, and how interpretation outputs are structured for downstream analysis.

!!! note
    `run_analysis` is the underlying explainability mode. In Python API usage, `SHAPAnalysis.fit_predict(...)` (or `fit(...)` + `predict(...)`) defaults to this mode.

## ::: ldt.machine_learning.tools.explainability.shap_analysis.run.SHAPAnalysis
