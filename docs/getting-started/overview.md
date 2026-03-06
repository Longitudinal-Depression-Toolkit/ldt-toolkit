---
hide:
  - navigation
---

# Getting Started

[> jump straight to installation](#installation){ .jump-install-link }

## LDT-Toolkit, in a Nutshell { .nutshell-heading }

<div class="mcs-hero" markdown>

The `Longitudinal Depression Trajectories Toolkit (LDT-Toolkit)` is designed for social, medical, and clinical researchers working with repeated-measure data who need a stepping-stone path from raw cohort files to downstream modelling.

The initiative delivers two interconnected components:

- `ldt-toolkit` (this repository): the Python engine, with tools and reproducible workflows for longitudinal data preparation, preprocessing, and modelling.
- `ldt` (Go CLI): the no-code terminal interface that orchestrates the full toolkit experience end-to-end.

</div>

### Playground Tools

`Playground` tools are designed for fast experimentation on your own data. You can mix and match operations across:

- `Data preparation` (for example synthetic data generation and data conversion)
- `Data preprocessing` (for example trajectories building, harmonisation, imputation, aggregation)
- `Machine learning` (standard and longitudinal workflows, benchmarking, explainability)

This is the fastest path for custom studies where you want direct control over each step.

### Reproducibility Presets

`Presets` package stage-level reproducibility pipelines so studies can be rerun and compared consistently.

The long-term goal is a community-driven preset ecosystem. If you build a robust preset for your cohort or methodology, please submit it so others can reproduce and extend your workflow.

## Installation

!!! tip "Highly recommended: use UV"
    We strongly recommend [uv](https://docs.astral.sh/uv/) as a state-of-the-art Python package manager for speed, reproducibility, and environment handling.

### Machine Requirements

You need `uv`/`Python`, `Go`, and `R` on your machine.

=== "macOS"

    ```bash
    # Install package managers and runtimes
    brew update
    brew install python@3.12 uv go r

    # Optional: verify
    python3 --version
    uv --version
    go version
    Rscript --version
    ```

=== "Linux (Ubuntu/Debian)"

    ```bash
    sudo apt-get update
    sudo apt-get install -y python3 python3-pip golang r-base curl
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Optional: verify
    python3 --version
    uv --version
    go version
    Rscript --version
    ```

`R` is required when using LCGA/GMM trajectory tooling via `lcmm`.

### Install Python Toolkit (`ldt-toolkit`)

=== "via UV (recommended)"

    ```bash
    uv add ldt-toolkit
    ```

=== "via Pip"

    ```bash
    pip install ldt-toolkit
    ```

Optional R dependencies for LCGA/GMM:

```bash
Rscript --vanilla -e "repos <- 'https://cloud.r-project.org'; required_packages <- c('lcmm'); missing <- setdiff(required_packages, rownames(installed.packages())); if (length(missing)) install.packages(missing, repos = repos) else message('All required R packages are already installed.')"
```

### Install Go CLI Toolkit (`ldt`)

=== "via Homebrew"

    ```bash
    brew tap Longitudinal-Depression-Toolkit/homebrew-tap
    brew install ldt
    ```

=== "via scoop"

    ```powershell
    scoop bucket add longitudinal-depression-toolkit https://github.com/Longitudinal-Depression-Toolkit/scoop-bucket
    scoop install ldt
    ```

### Launch

```bash
ldt
```

## Where to Continue

- [Tutorials](../tutorials/index.md): learn the no-code CLI flow and advanced Python API flow.
- [API Reference](../api/index.md): browse tool docstrings and parameters by stage.
- [Study Reproducibility](../study-reproducibility/index.md): access reproducible study pipelines and resources.
